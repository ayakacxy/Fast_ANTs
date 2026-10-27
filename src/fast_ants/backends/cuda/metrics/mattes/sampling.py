"""Fused local Mattes sampling with one changing host control packet.

Geometry and float1 bin constants are retained for a resolution/model center.
The mapping, interpolation and gradient expressions retain the unfused FP32
assignment order; no fast-math option or approximate texture sampling is used.
"""

import numpy as np


def pack_controls(mapping,derivatives=None):
    """Prepare owned host words before independent GPU trials are launched."""
    linear,offset,_=mapping
    integer=np.rint(linear);integer_offset=np.rint(offset)
    if (not np.isfinite(integer).all() or not np.isfinite(integer_offset).all() or
            np.any(np.abs(integer)>2**30) or np.any(np.abs(integer_offset)>2**30)):
        raise ValueError('CUDA sample geometry exceeds integer index range')
    integer=integer.astype(np.int32);integer_offset=integer_offset.astype(np.int32)
    pieces=[integer.reshape(-1).view(np.uint32),integer_offset.view(np.uint32),
            np.asarray(linear-integer,np.float32).reshape(-1).view(np.uint32),
            np.asarray(offset-integer_offset,np.float32).view(np.uint32)]
    if derivatives is not None:
        pieces.extend(np.asarray(value,np.float32).reshape(-1).view(np.uint32) for value in derivatives)
    return np.concatenate(pieces)


def controls(ctx,mapping,derivatives=None):
    return ctx.upload(pack_controls(mapping,derivatives),'mattes_evaluation_controls',np.uint32)


def constants(metric,model):
    """Cache only values independent of the current transform parameters."""
    key=np.asarray(model.center).tobytes()
    if metric._sampling_constants is None or metric._sampling_constants[0]!=key:
        grid=metric.grid;moving=metric.moving.grid
        inverse=moving.lps_to_index[:3,:3]
        values=np.concatenate((np.floor((np.asarray(grid.shape_xyz)-1)*.5),
            (inverse@np.diag(moving.spacing_xyz*.5)).reshape(-1),moving.spacing_xyz,
            grid.index_to_lps[:3,:3].reshape(-1),grid.center-model.center,
            (np.asarray(grid.shape_xyz)-1)*.5,metric.moving_range,
            metric._moving_parameters))
        metric._sampling_constants=(key,metric.ctx.upload(values,'mattes_level_constants'))
    return metric._sampling_constants[1]


def sample(ctx,data,points,packet,static,bins,derivative):
    nz,ny,nx=data.shape;count=points.shape[0]
    source=('constexpr bool derivative='+str(bool(derivative)).lower()+';\n'+'''
      const int* integer=reinterpret_cast<const int*>(control);
      const int* bo=reinterpret_cast<const int*>(control+9);
      const float* residual=reinterpret_cast<const float*>(control+12);
      const float* offset=reinterpret_cast<const float*>(control+21);
      int base[3];float local[3];bool safe=true;
      for(int j=0;j<3;j++){
        float q=points[3*i+j]-fixed[j];
        if(!isfinite(q)||fabsf(q)>1073741824.0f)safe=false;}
      for(int c=0;c<3;c++){
        long long anchor=bo[c];float value=offset[c];
        if(safe)for(int j=0;j<3;j++){
          float q=points[3*i+j]-fixed[j];int whole=int(floorf(q));float fraction=q-float(whole);
          anchor+=static_cast<long long>(integer[3*c+j])*whole;
          value=__fmaf_rn(residual[3*c+j],float(whole),value);
          value=__fmaf_rn(float(integer[3*c+j])+residual[3*c+j],fraction,value);}
        base[c]=safe?(anchor < -2147483647LL ? -2147483647 : (anchor > 2147483647LL ? 2147483647 : int(anchor))):1073741824;
        local[c]=safe?value:0.0f;}
      int3 n=make_int3(meta[0],meta[1],meta[2]);
      int3 b=make_int3(base[0],base[1],base[2]);
      float3 p=make_float3(local[0],local[1],local[2]);
      bool valid=local_inside(b,p,n);float value=valid?local_at(data,b,p,n):0.0f;
      float term=value/fixed[32]-fixed[33];int bin=min(max(int(term),2),int(meta[3])-3);
      out0[i]=value;out1[i]=valid;out2[i]=bin;out3[i]=term-float(bin);
      if(derivative){
        float anchor=valid?value:local_at(data,b,p,n);
        for(int c=0;c<3;c++){
          float3 a=make_float3(p.x+fixed[3+c],p.y+fixed[6+c],p.z+fixed[9+c]);
          float3 m=make_float3(p.x-fixed[3+c],p.y-fixed[6+c],p.z-fixed[9+c]);
          out4[3*i+c]=(local_inside(b,a,n)&&local_inside(b,m,n))?
            (local_at(data,b,a,n,anchor)-local_at(data,b,m,n,anchor))/fixed[12+c]:0.0f;}}
      ''')
    return ctx.call('mattes_fused_sample_d'+str(int(derivative)),source,
        {'data':data,'points':points,'control':packet,'fixed':static,'meta':ctx.meta(nx,ny,nz,bins)},
        [(count,),(count,),(count,),(count,),(count,3 if derivative else 0)],count,
        [np.float32,np.bool_,np.int32,np.float32,np.float32])
