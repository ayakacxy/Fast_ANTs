"""Ordinary FP32 CUDA sampling and Mattes MI with bounded host controls."""

import numpy as np

from fast_ants.backends.cuda.runtime import cp, DTYPES, _kernel
from fast_ants.backends.cuda.types import CUDAImage
from fast_ants.backends.cuda.metrics.mattes.histogram import fixed_bin_plan, histograms
from fast_ants.backends.cuda.metrics.mattes.finalize import finalize_device, finalize_registration, decode_control
from fast_ants.backends.cuda.metrics.mattes import sampling
from fast_ants.core.results import MetricEvaluation, NoValidSamplesError
from fast_ants.core.transforms import AffineTransform


class _MattesGraph:
    """Retain registration kernels and bounded pinned host control packets.

    Capturing warmed kernels leaves host transforms and optimizer decisions
    unchanged. Each independent trial owns a lane, including its result packet.
    The public PDF-producing path continues to allocate independent outputs.
    """

    def __init__(self,metric,fixed,count,words):
        ctx=self.ctx=metric.ctx;self.count=count
        self.lanes=words.shape[0];self.calls=[]
        self.host_control=np.frombuffer(cp.cuda.alloc_pinned_memory(words.nbytes),np.uint32,
                                        count=words.size).reshape(words.shape)
        np.copyto(self.host_control,words)
        self.control=ctx.upload(self.host_control,'mattes_graph_controls',np.uint32)
        shape=(self.lanes,count+3)
        self.host_packet=np.frombuffer(cp.cuda.alloc_pinned_memory(int(np.prod(shape))*4),
                                       np.uint32,count=int(np.prod(shape))).reshape(shape)
        with ctx.device,ctx.stream:self.packet=cp.empty(shape,np.uint32)
        for self.lane in range(self.lanes):
            metric._enqueue_prepared(self.control[self.lane],fixed,count,_context=self)
        stream=cp.cuda.Stream(non_blocking=True)
        with ctx.device,stream:
            stream.begin_capture()
            cp.cuda.runtime.memcpyAsync(self.control.data.ptr,self.host_control.ctypes.data,
                self.host_control.nbytes,cp.cuda.runtime.memcpyHostToDevice,stream.ptr)
            for _,kernel,arguments,count in self.calls:
                kernel(((count+255)//256,),(256,),arguments,stream=stream)
            cp.cuda.runtime.memcpyAsync(self.host_packet.ctypes.data,self.packet.data.ptr,
                self.host_packet.nbytes,cp.cuda.runtime.memcpyDeviceToHost,stream.ptr)
            self.graph=stream.end_capture()
        with ctx.device,ctx.stream:
            self.packet.get(out=self.host_packet,stream=ctx.stream,blocking=True)
        ctx._record('device_to_host','control_mattes_graph',self.host_packet)

    def __getattr__(self,name):
        return getattr(self.ctx,name)

    def call(self,name,source,inputs,shapes,count,dtypes=None):
        result=self.ctx.call(name,source,inputs,shapes,count,dtypes)
        outputs=[result] if len(shapes)==1 else list(result)
        if name.startswith('mattes_registration_finalize'):
            target=self.packet[self.lane]
            with self.ctx.device,self.ctx.stream:cp.copyto(target,result)
            result=target;outputs=[target]
        declarations=tuple((key,DTYPES[value.dtype]) for key,value in inputs.items())
        kernel=_kernel(name,declarations,source,tuple(DTYPES[value.dtype] for value in outputs))
        kernel.compile()
        arguments=tuple(inputs.values())+tuple(outputs)+(np.uint32(count),)
        self.calls.append((name,kernel,arguments,count))
        return result

    def replay(self,words):
        ctx=self.ctx
        np.copyto(self.host_control,words)
        with ctx.device,ctx.stream:
            self.graph.launch(ctx.stream)
            ctx.stream.synchronize()
        ctx._record('host_to_device','mattes_graph_controls',self.host_control)
        ctx._record('device_to_host','control_mattes_graph',self.host_packet)
        for name,_,_,_ in self.calls:ctx.kernels[name]+=1
        return self.host_packet


def index_points(ctx,shape):
    nz,ny,nx=shape;count=nx*ny*nz
    return ctx.call('mattes_index_points','''int3 p=xyz(i,make_int3(meta[0],meta[1],meta[2]));
      out0[3*i]=float(p.x);out0[3*i+1]=float(p.y);out0[3*i+2]=float(p.z);''',
      {'meta':ctx.meta(nx,ny,nz)},[(count,3)],count)


def affine(ctx,points,linear,offset,center):
    return ctx.call('mattes_point_affine','''for(int c=0;c<3;c++){
      float value=offset[c];for(int j=0;j<3;j++)value+=linear[c*3+j]*(points[3*i+j]-center[j]);
      out0[3*i+c]=value;}''',
      {'points':points,'linear':ctx.upload(linear),'offset':ctx.upload(offset),'center':ctx.upload(center)},
      [points.shape],points.shape[0])


def local_mapping(ctx,points,linear,offset,center):
    integer=np.rint(linear);integer_offset=np.rint(offset)
    if (not np.isfinite(integer).all() or not np.isfinite(integer_offset).all() or
            np.any(np.abs(integer)>2**30) or np.any(np.abs(integer_offset)>2**30)):
        raise ValueError('CUDA sample geometry exceeds integer index range')
    integer=integer.astype(np.int32);integer_offset=integer_offset.astype(np.int32)
    return ctx.call('mattes_local_mapping','''
      for(int j=0;j<3;j++){
        float q=points[3*i+j]-center[j];
        if(!isfinite(q)||fabsf(q)>1073741824.0f){
          for(int c=0;c<3;c++){out0[3*i+c]=1073741824;out1[3*i+c]=0;}return;}}
      for(int c=0;c<3;c++){
      long long base=bo[c];float value=offset[c];
      for(int j=0;j<3;j++){
        float q=points[3*i+j]-center[j];int whole=int(floorf(q));float fraction=q-float(whole);
        base+=static_cast<long long>(integer[3*c+j])*whole;
        value=__fmaf_rn(residual[3*c+j],float(whole),value);
        value=__fmaf_rn(float(integer[3*c+j])+residual[3*c+j],fraction,value);}
      out0[3*i+c]=base < -2147483647LL ? -2147483647 : (base > 2147483647LL ? 2147483647 : int(base));
      out1[3*i+c]=value;}''',
      {'points':points,'integer':ctx.upload(integer,dtype=np.int32),'bo':ctx.upload(integer_offset,dtype=np.int32),
       'residual':ctx.upload(linear-integer),'offset':ctx.upload(offset-integer_offset),'center':ctx.upload(center)},
      [points.shape,points.shape],points.shape[0],[np.int32,np.float32])


def sample(ctx,data,points,nearest=False):
    bases,offsets=points;nz,ny,nx=data.shape;count=bases.shape[0]
    return ctx.call('mattes_local_sample','''int3 n=make_int3(meta[0],meta[1],meta[2]);
      int3 b=make_int3(bases[3*i],bases[3*i+1],bases[3*i+2]);
      float3 p=make_float3(offsets[3*i],offsets[3*i+1],offsets[3*i+2]);
      bool valid=local_inside(b,p,n);float value=0;
      if(valid){
        if(meta[3]){
          int x=min(max(b.x+int(floorf(p.x+0.5f)),0),n.x-1);
          int y=min(max(b.y+int(floorf(p.y+0.5f)),0),n.y-1);
          int z=min(max(b.z+int(floorf(p.z+0.5f)),0),n.z-1);
          value=data[idx(x,y,z,n)];
        }else value=local_at(data,b,p,n);}
      out0[i]=value;out1[i]=valid;''',
      {'data':data,'bases':bases,'offsets':offsets,'meta':ctx.meta(nx,ny,nz,int(nearest))},
      [(count,),(count,)],count,[np.float32,np.bool_])


def gradient(ctx,data,points,shift,spacing):
    bases,offsets=points;nz,ny,nx=data.shape
    return ctx.call('mattes_local_gradient','''int3 n=make_int3(meta[0],meta[1],meta[2]);
      int3 b=make_int3(bases[3*i],bases[3*i+1],bases[3*i+2]);
      float3 p=make_float3(offsets[3*i],offsets[3*i+1],offsets[3*i+2]);
      float anchor=local_at(data,b,p,n);
      for(int c=0;c<3;c++){
        float3 a=make_float3(p.x+shift[c],p.y+shift[3+c],p.z+shift[6+c]);
        float3 m=make_float3(p.x-shift[c],p.y-shift[3+c],p.z-shift[6+c]);
        out0[3*i+c]=(local_inside(b,a,n)&&local_inside(b,m,n))?
          (local_at(data,b,a,n,anchor)-local_at(data,b,m,n,anchor))/spacing[c]:0.0f;}''',
      {'data':data,'bases':bases,'offsets':offsets,'shift':ctx.upload(shift),
       'spacing':ctx.upload(spacing),'meta':ctx.meta(nx,ny,nz)},[offsets.shape],bases.shape[0])


def derivatives(ctx,points,gradients,linear,translation):
    parameters=linear.shape[0];count=points.shape[0]
    return ctx.call('mattes_intensity_derivatives','''unsigned int P=meta[0];
      for(unsigned int p=0;p<P;p++){
        float total=0;for(int c=0;c<3;c++){
          float spatial=translation[p*3+c];
          for(int j=0;j<3;j++)spatial=__fmaf_rn(linear[p*9+c*3+j],points[3*i+j],spatial);
          total=__fmaf_rn(gradients[3*i+c],spatial,total);}
        out0[i*P+p]=total;}''',
      {'points':points,'gradients':gradients,'linear':ctx.upload(linear),
       'translation':ctx.upload(translation),'meta':ctx.meta(parameters)},[(count,parameters)],count)


def compact(ctx,valid,*arrays):
    with ctx.device,ctx.stream:
        positions=cp.cumsum(valid,dtype=cp.uint32)
    count=int(ctx.control(positions[-1:],'sample_count',1)[0])
    if not count:raise NoValidSamplesError('No valid CUDA samples')
    output=[]
    for data in arrays:
        width=int(np.prod(data.shape[1:]))
        output.append(ctx.call('mattes_compact_scatter','''
          if(valid[i])for(unsigned int c=0;c<meta[0];c++)out0[(positions[i]-1)*meta[0]+c]=data[i*meta[0]+c];''',
          {'data':data,'valid':valid,'positions':positions,'meta':ctx.meta(width)},
          [(count,*data.shape[1:])],valid.size,[data.dtype]))
    return output


def float1_bin_parameters(value_range,bins):
    """ITK float1 BinSize and NormalizedMin, retaining each FP32 rounding."""
    low,high=np.asarray(value_range,np.float32)
    width=np.float32(np.float32(high-low)/np.float32(bins-4))
    normalized_min=np.float32(np.float32(low/width)-np.float32(2))
    return np.asarray([width,normalized_min],np.float32)


def float1_bin_fractions(ctx,values,bins,parameters):
    return ctx.call('mattes_float1_bin_fractions','''
      float term=values[i]/params[0]-params[1];int index=min(max(int(term),2),int(meta[0])-3);
      out0[i]=index;out1[i]=term-float(index);''',
      {'values':values,'params':parameters,'meta':ctx.meta(bins)},
      [values.shape,values.shape],values.size,[np.int32,np.float32])


def fixed_bins(ctx,values,bins,value_range,*,parameters=None):
    if parameters is None:parameters=ctx.upload(float1_bin_parameters(value_range,bins))
    return ctx.call('mattes_fixed_bins','''
      out0[i]=min(max(int(values[i]/params[0]-params[1]),2),int(meta[0])-3);''',
      {'values':values,'params':parameters,'meta':ctx.meta(bins)},[values.shape],values.size,[np.int32])


def moments(ctx,data):
    from fast_ants.backends.cuda.moments import moments as parallel_moments
    return ctx.control(parallel_moments(ctx,data),'moments',4)


class CUDAMattesMetric:
    def __init__(self,ops,fixed,moving,virtual_grid=None,bins=32,points_lps=None,fixed_mask=None,moving_mask=None,
                 optimized=True):
        if not isinstance(bins,(int,np.integer)) or bins<5:
            raise ValueError('Mattes MI needs an integer bin count of at least five')
        self.ops,self.ctx=ops,ops.context;self.fixed,self.moving=fixed,moving;self.bins=int(bins)
        self.grid=fixed.grid if virtual_grid is None else virtual_grid
        self.moving_mask=(ops.image(moving_mask) if moving_mask is not None and not isinstance(moving_mask,CUDAImage)
                          else moving_mask)
        if fixed_mask is not None and not isinstance(fixed_mask,CUDAImage):fixed_mask=ops.image(fixed_mask)
        self.indices=(index_points(self.ctx,self.grid.shape_zyx) if points_lps is None else
                      self.ctx.upload(self.grid.physical_to_index(points_lps),'input_sample_indices'))
        if self.indices.shape[0]==0:raise NoValidSamplesError('No fixed CUDA MI samples')
        values,valid,_=self.sample(fixed,AffineTransform.identity())
        if fixed_mask is not None:valid=self.mask_and(valid,fixed_mask,AffineTransform.identity())
        self.indices,self.fixed_values=compact(self.ctx,valid,self.indices,values)
        self.fixed_range=self.image_range(fixed,fixed_mask)
        self.moving_range=self.image_range(moving,self.moving_mask)
        if points_lps is not None:
            values,valid,_=self.sample(fixed,AffineTransform.identity(),nearest=True)
            values,=compact(self.ctx,valid,values);self.fixed_range=self.range(values)
        if self.fixed_range[1]<=self.fixed_range[0] or self.moving_range[1]<=self.moving_range[0]:
            raise ValueError('MI needs nonconstant ranges')
        fixed_parameters=float1_bin_parameters(self.fixed_range,bins)
        moving_parameters=float1_bin_parameters(self.moving_range,bins)
        if (not np.isfinite(fixed_parameters).all() or fixed_parameters[0]<=0 or
                not np.isfinite(moving_parameters).all() or moving_parameters[0]<=0):
            raise ValueError('MI bin parameters must be finite with positive width')
        self.fixed_bin_parameters=self.ctx.upload(fixed_parameters)
        self.moving_bin_parameters=self.ctx.upload(moving_parameters)
        self._moving_parameters=moving_parameters
        self.moving_bin_width=float(moving_parameters[0])
        self.cached_bins=fixed_bins(self.ctx,self.fixed_values,bins,self.fixed_range,
                                    parameters=self.fixed_bin_parameters)
        self.grouped_plan=fixed_bin_plan(self.ctx,self.cached_bins,bins)
        self.control_evaluations=0;self.control_derivative_evaluations=0;self.control_bytes=0
        self._device_control_mode=False
        self._optimized=bool(optimized);self._sampling_constants=None
        self._batch_controls=True;self._preload_finalizer=True
        self._aggregate_histogram=True
        self._private_values=True
        self._graph_replay=True;self._graph_constants=None;self._graph_plans={}

    def mapping_parameters(self,image,transform):
        inverse=image.grid.lps_to_index[:3,:3]
        linear=inverse@transform.matrix[:3,:3]@self.grid.index_to_lps[:3,:3]
        center=np.floor((np.asarray(self.grid.shape_xyz)-1)*.5)
        offset=inverse@(transform.map_points(self.grid.index_to_physical(center))-image.grid.origin)
        return linear,offset,center

    def mapping(self,image,transform):
        return local_mapping(self.ctx,self.indices,*self.mapping_parameters(image,transform))

    def sample(self,image,transform,nearest=False):
        points=self.mapping(image,transform);values,valid=sample(self.ctx,image.data,points,nearest)
        return values,valid,points

    def mask_and(self,valid,mask,transform):
        values,inside,_=self.sample(mask,transform,nearest=True)
        return self.ctx.call('mattes_mask_and','out0[i]=valid[i]&&inside[i]&&values[i]!=0;',
            {'valid':valid,'inside':inside,'values':values},[valid.shape],valid.size,[np.bool_])

    def range(self,data):
        with self.ctx.device,self.ctx.stream:limits=cp.stack((cp.min(data),cp.max(data)))
        values=self.ctx.control(limits,'mattes_image_range',2)
        if not np.isfinite(values).all():raise ValueError('MI image ranges must be finite')
        return values

    def image_range(self,image,mask):
        if mask is None:return self.range(image.data)
        old_grid,old_indices=self.grid,self.indices
        try:
            self.grid=image.grid;self.indices=index_points(self.ctx,image.grid.shape_zyx)
            values,valid,_=self.sample(mask,AffineTransform.identity(),nearest=True)
            valid=self.ctx.call('mattes_range_mask','out0[i]=valid[i]&&values[i]!=0;',
                {'valid':valid,'values':values},[valid.shape],valid.size,[np.bool_])
            selected,=compact(self.ctx,valid,image.data.reshape(-1))
            return self.range(selected)
        finally:self.grid,self.indices=old_grid,old_indices

    def _enqueue(self,model,parameters,derivative):
        # Keep the separate kernels as an explicit fixed-state diagnostic path.
        # Mask grids may have independent geometry, so retain their public path.
        if not self._optimized or self.moving_mask is not None:
            return self._enqueue_unfused(model,parameters,derivative)
        words,fixed,count=self._prepare(model,parameters,derivative)
        control=self.ctx.upload(words,'mattes_evaluation_controls',np.uint32)
        return self._enqueue_prepared(control,fixed,count)

    def _prepare(self,model,parameters,derivative):
        transform=model.transform(parameters)
        fixed=sampling.constants(self,model)
        jacobian=model.derivatives(parameters) if derivative else None
        words=sampling.pack_controls(self.mapping_parameters(self.moving,transform),jacobian)
        return words,fixed,jacobian[0].shape[0] if derivative else 0

    def _graph_controls(self,words,fixed,count):
        if self._graph_constants is not fixed:
            self._graph_plans.clear();self._graph_constants=fixed
        key=(count,len(words),self._aggregate_histogram,self._preload_finalizer,self._private_values)
        plan=self._graph_plans.get(key)
        if plan is None:
            plan=_MattesGraph(self,fixed,count,words);self._graph_plans[key]=plan
            return plan.host_packet
        return plan.replay(words)

    def _enqueue_prepared(self,control,fixed,count,*,_context=None):
        ctx=self.ctx if _context is None else _context
        moving,valid,bi,bf,grad=sampling.sample(ctx,self.moving.data,self.indices,control,
                                             fixed,self.bins,count>0)
        histogram,counts=histograms(ctx,self.grouped_plan,moving,valid,None,bi,bf,self.moving_range,
            geometry=(self.indices,grad,control,fixed),parameters=count,range_device=fixed[30:32],
            aggregate=self._aggregate_histogram,private=self._private_values and self.bins<=32)
        if self._device_control_mode and self.bins<=32 and count<=12:
            return finalize_registration(ctx,histogram,counts,self.moving_bin_parameters[:1],
                                         preload=self._preload_finalizer)
        return finalize_device(ctx,histogram,counts,self.moving_bin_width)

    def _enqueue_unfused(self,model,parameters,derivative):
        transform=model.transform(parameters)
        moving,valid,indices=self.sample(self.moving,transform)
        if self.moving_mask is not None:valid=self.mask_and(valid,self.moving_mask,transform)
        if derivative:
            inverse=self.moving.grid.lps_to_index[:3,:3]
            grad=gradient(self.ctx,self.moving.data,indices,
                inverse@np.diag(self.moving.grid.spacing_xyz*.5),self.moving.grid.spacing_xyz)
            local=affine(self.ctx,self.indices,self.grid.index_to_lps[:3,:3],
                         self.grid.center-model.center,(np.asarray(self.grid.shape_xyz)-1)*.5)
            linear,translation=model.derivatives(parameters)
            intensity=derivatives(self.ctx,local,grad,linear,translation)
        else:
            with self.ctx.device,self.ctx.stream:intensity=cp.empty((moving.size,0),cp.float32)
        bi,bf=float1_bin_fractions(self.ctx,moving,self.bins,self.moving_bin_parameters)
        histogram,counts=histograms(self.ctx,self.grouped_plan,moving,valid,intensity,bi,bf,self.moving_range)
        return finalize_device(self.ctx,histogram,counts,self.moving_bin_width)

    @property
    def batch_value_evaluations(self):
        return self._device_control_mode

    def evaluate_values_many(self,model,trials):
        trials=tuple(trials)
        if (self._batch_controls and self._optimized and self.moving_mask is None and
                self._device_control_mode and self.bins<=32 and trials):
            fixed=sampling.constants(self,model)
            prepared=[]
            for trial in trials:
                try:
                    transform=model.transform(trial)
                    prepared.append(sampling.pack_controls(self.mapping_parameters(self.moving,transform)))
                except (NoValidSamplesError,ValueError):prepared.append(None)
            if self._graph_replay and len(trials)==2 and all(value is not None for value in prepared):
                words=self._graph_controls(np.stack(prepared),fixed,0).reshape(-1)
                return self._finish_values(words)
            # Every upload completes before any trial kernel is dispatched.
            # A later blocking H2D copy can otherwise wait for an earlier trial.
            words=[value for value in prepared if value is not None]
            controls=(self.ctx.upload(np.concatenate(words),'mattes_trial_controls',np.uint32)
                      if words else None)
            invalid=(self.ctx.upload([0,0,2],'invalid_mattes_control',np.uint32)
                     if any(value is None for value in prepared) else None)
            packets=[];offset=0
            for value in prepared:
                if value is None:packet=invalid
                else:
                    packet,_=self._enqueue_prepared(controls[offset:offset+value.size],fixed,0)
                    offset+=value.size
                packets.append(packet)
        else:
            packets=[]
            for trial in trials:
                try:packet,_=self._enqueue(model,trial,False)
                except (NoValidSamplesError,ValueError):packet=self.ctx.upload([0,0,2],dtype=np.uint32)
                packets.append(packet)
        if not packets:return ()
        with self.ctx.device,self.ctx.stream:pair=cp.concatenate(packets)
        words=self.ctx.control(pair,'mattes_final_batch',len(packets)*3)
        return self._finish_values(words)

    def _finish_values(self,words):
        maximum=np.float32(np.finfo(np.float32).max)
        values=[]
        for offset in range(0,len(words),3):
            try:values.append(np.float32(self._finish(words[offset:offset+3],None,False).value))
            except (NoValidSamplesError,ValueError):values.append(maximum)
        return tuple(values)

    def _finish(self,words,pdfs,derivative):
        value,descent,count=decode_control(words)
        details={'device_finalization':True,'software_expanded_precision':False,
                 'pdf_details_omitted':pdfs is None}
        transfer_bytes=words.nbytes
        if pdfs is not None:
            for key,data in zip(('joint_pdf','fixed_pdf','moving_pdf','pdf_derivative'),pdfs):
                details[key]=self.ctx.control(data,'mattes_'+key,self.bins*self.bins*12)
                transfer_bytes+=data.nbytes
        details['control_transfer_bytes']=int(transfer_bytes)
        self.control_evaluations+=1;self.control_derivative_evaluations+=int(derivative)
        self.control_bytes+=transfer_bytes
        return MetricEvaluation(value,descent,count,details)

    def evaluate(self,model,parameters,derivative=True):
        if (self._graph_replay and self._optimized and self.moving_mask is None and
                self._device_control_mode and self.bins<=32 and (not derivative or model.parameter_count<=12)):
            words,fixed,count=self._prepare(model,parameters,derivative)
            packet=self._graph_controls(words[None,:],fixed,count)[0]
            return self._finish(packet,None,derivative)
        packet,pdfs=self._enqueue(model,parameters,derivative)
        words=self.ctx.control(packet,'mattes_final',packet.size)
        return self._finish(words,None if self._device_control_mode else pdfs,derivative)
