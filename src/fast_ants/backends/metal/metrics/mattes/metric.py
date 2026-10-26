"""Ordinary FP32 Metal sampling and Mattes MI, without precision emulation."""

import math
import numpy as np

from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal import dense
from fast_ants.backends.metal import serial
from fast_ants.core.results import MetricEvaluation
from fast_ants.core.results import NoValidSamplesError
from fast_ants.pipeline.mattes32 import finalize_mattes_float1
from fast_ants.core.transforms import AffineTransform


def affine(ctx,points,linear,offset,center):
    return ctx.call('point_affine','''uint i=thread_position_in_grid.x;
      for(int c=0;c<3;c++){float v=offset[c];for(int j=0;j<3;j++)v+=linear[c*3+j]*(points[3*i+j]-center[j]);out0[3*i+c]=v;}''',
      {'points':points,'linear':ctx.upload(linear),'offset':ctx.upload(offset),'center':ctx.upload(center)},[points.shape],points.shape[0])


def local_mapping(ctx,points,linear,offset,center):
    integer=np.rint(linear).astype(np.int32);integer_offset=np.rint(offset).astype(np.int32)
    source="""uint i=thread_position_in_grid.x;
      for(int c=0;c<3;c++){int base=bo[c];float v=offset[c];
       for(int j=0;j<3;j++){float q=points[3*i+j]-center[j];int whole=int(metal::floor(q));float fraction=q-float(whole);
        base+=integer[3*c+j]*whole;v=metal::fma(residual[3*c+j],float(whole),v);
        v=metal::fma(float(integer[3*c+j])+residual[3*c+j],fraction,v);}
       out0[3*i+c]=base;out1[3*i+c]=v;}"""
    return ctx.call('local_point_mapping',source,{'points':points,'integer':ctx.upload(integer,dtype=np.int32),
        'bo':ctx.upload(integer_offset,dtype=np.int32),'residual':ctx.upload(linear-integer),
        'offset':ctx.upload(offset-integer_offset),'center':ctx.upload(center)},[points.shape]*2,points.shape[0],[mx.int32,mx.float32])


def sample(ctx,data,points,nearest=False):
    bases,offsets=points;nz,ny,nx=data.shape;N=bases.shape[0]
    source="""uint i=thread_position_in_grid.x;int3 n=int3(meta[0],meta[1],meta[2]);
      int3 b=int3(bases[3*i],bases[3*i+1],bases[3*i+2]);float3 p=float3(offsets[3*i],offsets[3*i+1],offsets[3*i+2]);
      bool valid=local_inside(b,p,n);float v=0;
      if(valid){if(meta[3]){int3 q=b+int3(metal::floor(p+0.5f));v=data[idx(q.x,q.y,q.z,n)];}else v=local_at(data,b,p,n);}
      out0[i]=v;out1[i]=valid;"""
    return ctx.call('local_point_sample',source,{'data':data,'bases':bases,'offsets':offsets,'meta':ctx.meta(nx,ny,nz,int(nearest))},
      [(N,),(N,)],N,[mx.float32,mx.bool_])


def gradient(ctx,data,points,shift,spacing):
    bases,offsets=points;nz,ny,nx=data.shape
    source="""uint i=thread_position_in_grid.x;int3 n=int3(meta[0],meta[1],meta[2]);
      int3 b=int3(bases[3*i],bases[3*i+1],bases[3*i+2]);float3 p=float3(offsets[3*i],offsets[3*i+1],offsets[3*i+2]);
      float anchor=local_at(data,b,p,n);
      for(int c=0;c<3;c++){float3 d=float3(shift[c],shift[3+c],shift[6+c]);float3 a=p+d,m=p-d;
       out0[3*i+c]=local_inside(b,a,n)&&local_inside(b,m,n)?(local_at(data,b,a,n,anchor)-local_at(data,b,m,n,anchor))/spacing[c]:0.0f;}"""
    return ctx.call('local_point_gradient',source,{'data':data,'bases':bases,'offsets':offsets,
      'shift':ctx.upload(shift),'spacing':ctx.upload(spacing),'meta':ctx.meta(nx,ny,nz)},[offsets.shape],bases.shape[0])


def bin_fractions(ctx,data,points,values,minimum,inverse_width,anchors,biases,bins):
    """Historical residual-sampling diagnostic; not the float-1 MI policy."""
    bases,offsets=points;nz,ny,nx=data.shape;N=values.size
    source="""uint i=thread_position_in_grid.x;int3 n=int3(meta[0],meta[1],meta[2]);
      int3 b=int3(bases[3*i],bases[3*i+1],bases[3*i+2]);float3 p=float3(offsets[3*i],offsets[3*i+1],offsets[3*i+2]);
      int k=metal::clamp(int((values[i]-params[0])*params[1]),0,int(meta[3])-5);float fraction=0;
      for(int attempt=0;attempt<3;attempt++){
       fraction=(local_at(data,b,p,n,anchors[k])-biases[k])*params[1];
       if(fraction<0&&k>0){k--;continue;}if(fraction>=1&&k<int(meta[3])-5){k++;continue;}break;}
      out0[i]=k+2;out1[i]=fraction;"""
    return ctx.call('local_bin_fractions',source,{'data':data,'bases':bases,'offsets':offsets,'values':values,
        'params':ctx.upload([minimum,inverse_width]),'anchors':ctx.upload(anchors),'biases':ctx.upload(biases),
        'meta':ctx.meta(nx,ny,nz,bins)},[(N,)]*2,N,[mx.int32,mx.float32])


def float1_bin_parameters(value_range,bins):
    """ITK float-1 BinSize/NormalizedMin, including each FP32 rounding.

    Algebraically replacing pixel/width - normalized_min with
    (pixel-minimum)/width + 2 moves integer bin boundaries in FP32.
    These two bounded constants are computed once per metric/image range.
    """
    low,high=np.asarray(value_range,np.float32)
    width=np.float32(np.float32(high-low)/np.float32(bins-4))
    normalized_min=np.float32(np.float32(low/width)-np.float32(2))
    return np.asarray([width,normalized_min],np.float32)


def float1_bin_fractions(ctx,values,bins,parameters):
    """Bin already-rounded FP32 interpolated intensities like ITK float-1.

    parameters is a device [width,normalized_min] array. The production path
    deliberately uses the stored sample, without reinterpolating a residual
    around a binary intensity anchor.
    """
    return ctx.call('float1_bin_fractions','''uint i=thread_position_in_grid.x;
      float term=values[i]/params[0]-params[1];
      int index=metal::clamp(int(term),2,int(meta[0])-3);
      out0[i]=index;out1[i]=term-float(index);''',
      {'values':values,'params':parameters,'meta':ctx.meta(bins)},
      [values.shape,values.shape],values.size,[mx.int32,mx.float32])


def derivatives(ctx,points,gradients,linear,translation):
    P=linear.shape[0];N=points.shape[0]
    return ctx.call('point_derivatives','''uint i=thread_position_in_grid.x,P=meta[0];
      for(uint p=0;p<P;p++){float total=0;
       for(int c=0;c<3;c++){float spatial=translation[p*3+c];
        for(int j=0;j<3;j++)spatial=metal::fma(linear[p*9+c*3+j],points[i*3+j],spatial);total=metal::fma(gradients[3*i+c],spatial,total);}
       out0[i*P+p]=total;}''',{'points':points,'gradients':gradients,'linear':ctx.upload(linear),
         'translation':ctx.upload(translation),'meta':ctx.meta(P)},[(N,P)],N)


def prefix_plan(ctx,valid):
    if 'parallel_prefix' in ctx.optimizations:
        tile=256;blocks=(valid.size+tile-1)//tile
        positions,block_counts=ctx.call('compact_prefix_tiles','''
          uint i=thread_position_in_grid.x,t=thread_index_in_threadgroup;
          threadgroup uint scratch[256];uint v=i<meta[0]?uint(valid[i]):0;
          scratch[t]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
          for(uint offset=1;offset<256;offset*=2){
            uint previous=t>=offset?scratch[t-offset]:0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scratch[t]+=previous;
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          if(i<meta[0])out0[i]=scratch[t]-v;
          if(t==255)out1[i/256]=scratch[t];''',
          {'valid':valid,'meta':ctx.meta(valid.size)},[valid.shape,(blocks,)],blocks*tile,[mx.uint32,mx.uint32])
        block_offsets,count=ctx.call('compact_prefix_block_offsets','''uint total=0;
          for(uint b=0;b<meta[0];b++){out0[b]=total;total+=counts[b];}out1[0]=total;''',
          {'counts':block_counts,'meta':ctx.meta(blocks)},[(blocks,),(1,)],1,[mx.uint32,mx.uint32])
    else:
        block_offsets=None
        positions,count=ctx.call('compact_prefix','''uint count=0;
          for(uint i=0;i<meta[0];i++){out0[i]=count;if(valid[i])count++;}out1[0]=count;''',
          {'valid':valid,'meta':ctx.meta(valid.size)},[valid.shape,(1,)],1,[mx.uint32,mx.uint32])
    return positions,block_offsets,count


def ordered_selection(ctx,valid):
    positions,blocks,count=prefix_plan(ctx,valid)
    inputs={'valid':valid,'positions':positions}
    offset='positions[i]'
    if blocks is not None:inputs['blocks']=blocks;offset+=' + blocks[i/256]'
    selected=ctx.call('mi_selection_scatter','uint i=thread_position_in_grid.x;if(valid[i])out0['+offset+']=i;',
                      inputs,[valid.shape],valid.size,[mx.uint32])
    return selected,count


def compact(ctx,valid,*arrays):
    positions,block_offsets,count=prefix_plan(ctx,valid)
    N=int(ctx.control(count,'sample_count')[0])
    if not N:raise NoValidSamplesError('No valid GPU samples')
    outputs=[]
    for data in arrays:
        width=int(np.prod(data.shape[1:]))
        inputs={'data':data,'valid':valid,'positions':positions,'meta':ctx.meta(width)}
        if block_offsets is None:
            source='''uint i=thread_position_in_grid.x,C=meta[0];
              if(valid[i])for(uint c=0;c<C;c++)out0[positions[i]*C+c]=data[i*C+c];'''
            name='compact_scatter'
        else:
            inputs['block_offsets']=block_offsets
            source='''uint i=thread_position_in_grid.x,C=meta[0];
              if(valid[i]){uint position=positions[i]+block_offsets[i/256];
               for(uint c=0;c<C;c++)out0[position*C+c]=data[i*C+c];}'''
            name='compact_scatter_tiles'
        outputs.append(ctx.call(name,source,inputs,[(N,*data.shape[1:])],valid.size,[data.dtype]))
    return outputs


def moments(ctx,data):
    if 'parallel_moments' in ctx.optimizations:
        from fast_ants.backends.metal.moments import moments as parallel_moments
        return ctx.control(parallel_moments(ctx,data),'moments',4)
    if data.size>serial.SLICE:return ctx.control(serial.moments(ctx,data),'moments',4)
    nz,ny,nx=data.shape
    value=ctx.call('moments','''float2 sums[4];for(int c=0;c<4;c++)sums[c]=float2(0);
      int3 n=int3(meta[0],meta[1],meta[2]);uint N=meta[0]*meta[1]*meta[2];
      for(uint i=0;i<N;i++){int3 p=xyz(i,n);float v=data[i];sums[0]=compensated(sums[0],v);
       for(int c=0;c<3;c++)sums[c+1]=compensated(sums[c+1],v*(float(p[c])-float(n[c]-1)*0.5f));}
      for(int c=0;c<4;c++)out0[c]=sums[c].x;''',
      {'data':data,'meta':ctx.meta(nx,ny,nz)},[(4,)],1)
    return ctx.control(value,'moments',4)


def fixed_bins(ctx,values,bins,value_range,*,parameters=None):
    if parameters is None:parameters=ctx.upload(float1_bin_parameters(value_range,bins))
    return ctx.call('fixed_bin_indices','''uint i=thread_position_in_grid.x;
      out0[i]=metal::clamp(int(values[i]/params[0]-params[1]),2,int(meta[0])-3);''',
      {'values':values,'params':parameters,'meta':ctx.meta(bins)},[values.shape],values.size,[mx.int32])


def cubic_tables(ctx,fractions,derivative):
    width=8 if derivative else 4
    return ctx.call('cubic_tables_derivative' if derivative else 'cubic_tables', '''uint i=thread_position_in_grid.x;
      for(int k=0;k<4;k++){float argument=float(k-1)-fractions[i];out0[i*meta[0]+k]=cubic(argument);
        if(meta[0]==8)out0[i*8+k+4]=cubic_derivative(argument);}''',
      {'fractions':fractions,'meta':ctx.meta(width)},[(fractions.size,width)],fractions.size)


def histograms(ctx,fixed,moving,intensity,bins,fixed_range,moving_range,bin_indices,bin_fractions,*,selection=None,sample_count=None,valid=None,cached_bins=None,weights=None,fixed_parameters=None,moving_width=None):
    """Accumulate bounded controls from indexed or directly masked samples.

    Direct validity partitions the original sample vector, avoiding a dynamic
    prefix/scatter pass. Its chunk grouping may change FP32 reduction rounding;
    valid support, intensity ranges and each sample's Parzen terms are shared.
    """
    N=fixed.size;P=intensity.shape[1];K=P+1;block=max(2048,math.ceil(N/128));chunks=math.ceil(N/block)
    if valid is not None:
        if selection is not None:
            raise ValueError('Choose indexed selection or direct validity, not both')
        if not isinstance(valid,mx.array) or valid.dtype!=mx.bool_ or valid.shape!=(N,):
            raise ValueError('Direct MI validity must be a device Boolean sample vector')
    shape=(chunks,bins,bins,K)
    simd=('simd_mi_histograms' in ctx.optimizations and N>=16384 and 5<=bins<=32 and 0<=P<=12)
    if simd:
        # Apple Metal SIMD groups contain 32 lanes. Runtime threadgroups are
        # 256 threads (or a smaller whole multiple of 32 here), so one task
        # always occupies a complete SIMD group, including the final group.
        name='histogram_private_simd32_b'+str(bins);work=chunks*bins*K*32
        source='constexpr uint NB='+str(bins)+''';
          uint gid=thread_position_in_grid.x,lane=gid%32u,task=gid/32u;
          uint B=meta[1],P=meta[2],K=P+1,channel=task%K,row=(task/K)%B,chunk=task/(B*K);
          uint N=meta[0],block=meta[3],S=B*B*K,base=chunk*S,cb=chunk*(B+1);
          float2 accum[NB];
          for(uint j=0;j<NB;j++)accum[j]=float2(0);
          uint row_count=0,total_count=0;
          for(uint n=chunk*block+lane;n<metal::min(N,(chunk+1)*block);n+=32u){
            if(moving[n]<ranges[2]||moving[n]>ranges[3])continue;
            int fi=metal::clamp(int(fixed[n]/fixed_params[0]-fixed_params[1]),2,int(B)-3);
            if(channel==0)total_count++;if(uint(fi)!=row)continue;if(channel==0)row_count++;
            int mi=bin_indices[n];
            for(int j=mi-1;j<mi+3;j++){
              float argument=float(j-mi)-bin_fractions[n];
              uint parameter=channel>0?channel-1:0;
              float value=channel==0?cubic(argument):intensity[n*P+parameter]*cubic_derivative(argument);
              accum[j]=compensated(accum[j],value);
            }
          }
          // Every lane participates, including lanes with no valid samples.
          // Per-lane compensated sums are combined with the SIMD reduction;
          // this changes grouping, not the per-sample PDF/derivative formula.
          for(uint j=0;j<NB;j++){
            float total=metal::simd_sum(accum[j].x);
            if(lane==0){uint q=base+(row*B+j)*K+channel;out0[q]=total;out1[q]=0.0f;}
          }
          uint selected=metal::simd_sum(row_count),all_selected=metal::simd_sum(total_count);
          if(lane==0&&channel==0){out2[cb+row]=selected;if(row==0)out2[cb+B]=all_selected;}'''
        if P==0:
            source=source.replace('float value=channel==0?cubic(argument):intensity[n*P+parameter]*cubic_derivative(argument);',
                                  'float value=cubic(argument);')
    elif 'histogram_channels' in ctx.optimizations and P>0:
        name='histogram_private_channels';work=chunks*bins*K
        source='''uint task=thread_position_in_grid.x,B=meta[1],P=meta[2],K=P+1,channel=task%K,row=(task/K)%B,chunk=task/(B*K);
          uint N=meta[0],block=meta[3],S=B*B*K,base=chunk*S,cb=chunk*(B+1);
          for(uint j=0;j<B;j++){out0[base+(row*B+j)*K+channel]=0;out1[base+(row*B+j)*K+channel]=0;}
          uint row_count=0,total_count=0;
          for(uint n=chunk*block;n<metal::min(N,(chunk+1)*block);n++){
            if(moving[n]<ranges[2]||moving[n]>ranges[3])continue;
            int fi=metal::clamp(int(fixed[n]/fixed_params[0]-fixed_params[1]),2,int(B)-3);
            if(channel==0)total_count++;if(uint(fi)!=row)continue;if(channel==0)row_count++;
            int mi=bin_indices[n];
            for(int j=mi-1;j<mi+3;j++){float argument=float(j-mi)-bin_fractions[n];uint q=base+(fi*B+j)*K+channel;
              uint parameter=channel>0?channel-1:0;
              float value=channel==0?cubic(argument):intensity[n*P+parameter]*cubic_derivative(argument);
              float2 total=compensated(float2(out0[q],out1[q]),value);out0[q]=total.x;out1[q]=total.y;}
          }
          if(channel==0){out2[cb+row]=row_count;if(row==0)out2[cb+B]=total_count;}'''
    elif 'histogram_rows' in ctx.optimizations:
        name='histogram_private_rows';work=chunks*bins
        source='''uint task=thread_position_in_grid.x,B=meta[1],row=task%B,chunk=task/B;
          uint N=meta[0],P=meta[2],K=P+1,block=meta[3],S=B*B*K,base=chunk*S,cb=chunk*(B+1);
          for(uint i=0;i<B*K;i++){out0[base+row*B*K+i]=0;out1[base+row*B*K+i]=0;}
          uint row_count=0,total_count=0;
          for(uint n=chunk*block;n<metal::min(N,(chunk+1)*block);n++){
            if(moving[n]<ranges[2]||moving[n]>ranges[3])continue;
            int fi=metal::clamp(int(fixed[n]/fixed_params[0]-fixed_params[1]),2,int(B)-3);
            total_count++;if(uint(fi)!=row)continue;row_count++;
            int mi=bin_indices[n];
            for(int j=mi-1;j<mi+3;j++){float argument=float(j-mi)-bin_fractions[n];uint q=base+(fi*B+j)*K;
              float2 v=compensated(float2(out0[q],out1[q]),cubic(argument));out0[q]=v.x;out1[q]=v.y;
              float weight=cubic_derivative(argument);
              for(uint p=0;p<P;p++){uint k=q+1+p;v=compensated(float2(out0[k],out1[k]),intensity[n*P+p]*weight);out0[k]=v.x;out1[k]=v.y;}}
          }
          out2[cb+row]=row_count;if(row==0)out2[cb+B]=total_count;'''
    else:
        name='histogram_private';work=chunks
        source='''uint chunk=thread_position_in_grid.x;
      uint N=meta[0],B=meta[1],P=meta[2],K=P+1,block=meta[3],S=B*B*K,base=chunk*S,cb=chunk*(B+1);
      for(uint i=0;i<S;i++){out0[base+i]=0;out1[base+i]=0;}
      for(uint i=0;i<=B;i++)out2[cb+i]=0;
      for(uint n=chunk*block;n<metal::min(N,(chunk+1)*block);n++){
       if(moving[n]<ranges[2]||moving[n]>ranges[3])continue;
       int fi=metal::clamp(int(fixed[n]/fixed_params[0]-fixed_params[1]),2,int(B)-3);
       int mi=bin_indices[n];
       out2[cb+fi]++;out2[cb+B]++;
       for(int j=mi-1;j<mi+3;j++){float argument=float(j-mi)-bin_fractions[n];uint q=base+(fi*B+j)*K;
        float2 v=compensated(float2(out0[q],out1[q]),cubic(argument));out0[q]=v.x;out1[q]=v.y;
        float weight=cubic_derivative(argument);
        for(uint p=0;p<P;p++){uint k=q+1+p;v=compensated(float2(out0[k],out1[k]),intensity[n*P+p]*weight);out0[k]=v.x;out1[k]=v.y;}}
      }'''
    if fixed_parameters is None:fixed_parameters=ctx.upload(float1_bin_parameters(fixed_range,bins))
    inputs={'fixed':fixed,'moving':moving,'intensity':intensity,'bin_indices':bin_indices,'bin_fractions':bin_fractions,
            'fixed_params':fixed_parameters,
            'ranges':ctx.upload([*fixed_range,*moving_range]),'meta':ctx.meta(N,bins,P,block)}
    if valid is not None:
        inputs['valid']=valid
        source=source.replace('if(moving[n]<ranges[2]||moving[n]>ranges[3])continue;',
                              'if(!valid[n])continue;\n            if(moving[n]<ranges[2]||moving[n]>ranges[3])continue;')
        name+='_direct_validity'
    if cached_bins is not None:
        inputs['cached_bins']=cached_bins
        source=source.replace('int fi=metal::clamp(int(fixed[n]/fixed_params[0]-fixed_params[1]),2,int(B)-3);','int fi=cached_bins[n];')
        name+='_cached_bins'
    if weights is not None:
        inputs['weights']=weights
        source=source.replace('cubic(argument)','weights[n*'+str(weights.shape[1])+'+j-mi+1]')
        source=source.replace('cubic_derivative(argument)','weights[n*8+j-mi+5]' if P else '0.0f')
        name+='_cached_cubic'
    merge_prefix='';active_chunks='meta[1]';merge_inputs={}
    if selection is not None:
        if sample_count is None:raise ValueError('Indexed histogram needs a device sample count')
        inputs.update(selection=selection,sample_count=sample_count)
        source=source.replace('N=meta[0]','N=sample_count[0]').replace('block=meta[3]','block=metal::max(2048u,(N+127u)/128u)')
        source=source.replace('for(uint n=chunk*block;n<metal::min(N,(chunk+1)*block);n++){',
                              'for(uint logical=chunk*block;logical<metal::min(N,(chunk+1)*block);logical++){uint n=selection[logical];')
        if simd:
            source=source.replace('for(uint n=chunk*block+lane;n<metal::min(N,(chunk+1)*block);n+=32u){',
                                  'for(uint logical=chunk*block+lane;logical<metal::min(N,(chunk+1)*block);logical+=32u){uint n=selection[logical];')
        name+='_indexed'
        # Extra allocated chunks must not enter the compensated merge: even a
        # zero add can consume a previous Kahan correction and change a bit.
        merge_prefix='uint N=sample_count[0],block=metal::max(2048u,(N+127u)/128u),chunks=(N+block-1)/block;'
        active_chunks='chunks';merge_inputs={'sample_count':sample_count}
    sums,corrections,counts=ctx.call(name,source,inputs,[shape,shape,(chunks,bins+1)],work,[mx.float32,mx.float32,mx.uint32])
    if ('packed_mi_controls' in ctx.optimizations and 5<=bins<=32 and P<=12):
        from fast_ants.backends.metal.metrics.mattes.histogram import merge_histogram_controls
        hist,fixed_counts,pdf,valid_count=merge_histogram_controls(
            ctx,sums,counts,bins,P,chunks,sample_count=sample_count if selection is not None else None)
        width=float(float1_bin_parameters(moving_range,bins)[0]) if moving_width is None else float(moving_width)
        return hist,fixed_counts,pdf,valid_count,width
    merged=ctx.call('histogram_merge'+('_indexed' if selection is not None else ''),merge_prefix+'''uint i=thread_position_in_grid.x,S=meta[0];float2 v=float2(0);
      for(uint c=0;c<'''+active_chunks+''';c++)v=compensated(v,data[c*S+i]);out0[i]=v.x;''',
      {'data':sums,'meta':ctx.meta(bins*bins*K,chunks),**merge_inputs},[(bins,bins,K)],bins*bins*K)
    merged_counts=ctx.call('histogram_counts'+('_indexed' if selection is not None else ''),merge_prefix+'''uint i=thread_position_in_grid.x,total=0;
      for(uint c=0;c<'''+active_chunks+''';c++)total+=counts[c*(meta[0]+1)+i];out0[i]=total;''',
      {'counts':counts,'meta':ctx.meta(bins,chunks),**merge_inputs},[(bins+1,)],bins+1,[mx.uint32])
    # Schedule both sibling reductions before the first host read; separate
    # control() calls would otherwise submit/wait on the count kernel later.
    # Only bounded histograms/counts cross the device boundary.
    mx.eval(merged,merged_counts)
    data=ctx.control(merged,'mattes_histogram',bins*bins*13)
    count=ctx.control(merged_counts,'mattes_counts',bins+1)
    width=float(float1_bin_parameters(moving_range,bins)[0]) if moving_width is None else float(moving_width)
    return data[:,:,0].copy(),count[:bins].astype(np.int64),data[:,:,1:].copy(),int(count[-1]),width


class MetalMattesMetric:
    def __init__(self,ops,fixed,moving,virtual_grid=None,bins=32,points_lps=None,fixed_mask=None,moving_mask=None):
        if not isinstance(bins,(int,np.integer)) or bins<5:raise ValueError('Mattes MI needs an integer bin count of at least five')
        self.ops,self.ctx=ops,ops.context;self.fixed,self.moving=fixed,moving;self.bins=bins
        self.grid=fixed.grid if virtual_grid is None else virtual_grid
        self.moving_mask=moving_mask
        self.indices=(dense.indices(self.ctx,self.grid.shape_zyx) if points_lps is None else
                      self.ctx.upload(self.grid.physical_to_index(points_lps),'input_sample_indices'))
        values,valid,_=self.sample(fixed,AffineTransform.identity())
        if fixed_mask is not None:valid=self.mask_and(valid,fixed_mask,AffineTransform.identity())
        self.indices,self.fixed_values=compact(self.ctx,valid,self.indices,values)
        self.fixed_range=self.image_range(fixed,fixed_mask)
        self.moving_range=self.image_range(moving,moving_mask)
        if points_lps is not None:
            values,valid,_=self.sample(fixed,AffineTransform.identity(),nearest=True)
            values,=compact(self.ctx,valid,values);self.fixed_range=dense.range32(self.ctx,values)
        if self.fixed_range[1]<=self.fixed_range[0] or self.moving_range[1]<=self.moving_range[0]:raise ValueError('MI needs nonconstant ranges')
        self.fixed_bin_parameters=self.ctx.upload(float1_bin_parameters(self.fixed_range,bins))
        moving_parameters=float1_bin_parameters(self.moving_range,bins)
        self.moving_bin_parameters=self.ctx.upload(moving_parameters)
        self.moving_bin_width=float(moving_parameters[0])
        self.cached_bins=(fixed_bins(self.ctx,self.fixed_values,bins,self.fixed_range,parameters=self.fixed_bin_parameters)
                          if ('cached_fixed_bins' in self.ctx.optimizations or 'grouped_fixed_bins' in self.ctx.optimizations) else None)
        self.grouped_plan=None
        self._stream_fixed_plan=None
        if 'grouped_fixed_bins' in self.ctx.optimizations:
            from fast_ants.backends.metal.metrics.mattes.histogram import fixed_bin_plan
            self.grouped_plan=fixed_bin_plan(self.ctx,self.cached_bins,bins)
        self.control_evaluations=0;self.control_derivative_evaluations=0;self.control_bytes=0
        # Full PDF details remain the public default. Registration can opt
        # into small lazy device controls, which its optimizer alone consumes.
        self._device_control_mode=False

    def mapping_parameters(self,image,transform):
        """Bounded host geometry shared by separate and fused GPU sampling."""
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
        return self.ctx.call('mi_mask_and','uint i=thread_position_in_grid.x;out0[i]=valid[i]&&inside[i]&&values[i]!=0;',
            {'valid':valid,'inside':inside,'values':values},[valid.shape],valid.size,[mx.bool_])

    def image_range(self,image,mask):
        if mask is None:return dense.range32(self.ctx,image.data)
        old_grid,old_indices=self.grid,self.indices
        try:
            self.grid=image.grid;self.indices=dense.indices(self.ctx,image.grid.shape_zyx)
            values,valid,_=self.sample(mask,AffineTransform.identity(),nearest=True)
            valid=self.ctx.call('range_mask','uint i=thread_position_in_grid.x;out0[i]=valid[i]&&values[i]!=0;',
                {'valid':valid,'values':values},[valid.shape],valid.size,[mx.bool_])
            selected,=compact(self.ctx,valid,image.data.reshape(-1))
            return dense.range32(self.ctx,selected)
        finally:self.grid,self.indices=old_grid,old_indices

    def _get_stream_fixed_plan(self):
        if self.cached_bins is None:
            self.cached_bins=fixed_bins(self.ctx,self.fixed_values,self.bins,self.fixed_range,
                                       parameters=self.fixed_bin_parameters)
        if self._stream_fixed_plan is None:
            from fast_ants.backends.metal.metrics.mattes.histogram import fixed_bin_plan
            self._stream_fixed_plan=fixed_bin_plan(self.ctx,self.cached_bins,self.bins)
        return self._stream_fixed_plan

    def _device_control_supported(self,derivative):
        flag='streamed_mi_derivative' if derivative else 'streamed_mi_fixed_plan'
        return (self._device_control_mode and 'device_mi_control' in self.ctx.optimizations
                and flag in self.ctx.optimizations and self.moving_mask is None
                and self.grouped_plan is None and self.indices.shape[0]>=16384
                and 5<=self.bins<=32)

    @property
    def batch_value_evaluations(self):
        """Whether registration may enqueue two independent value controls."""
        return ('batch_mi_search' in self.ctx.optimizations
                and self._device_control_supported(False))

    def _enqueue_device_control(self,model,parameters,derivative):
        """Return a lazy (control packet, parameter count), without a host read."""
        if not self._device_control_supported(derivative):return None
        from fast_ants.backends.metal.metrics.mattes.finalize import finalize_packed_device
        if 'fused_mi_finalize' in self.ctx.optimizations:
            from fast_ants.backends.metal.metrics.mattes.finalize_fused import finalize_packed_device
        transform=model.transform(parameters)
        mapping=self.mapping_parameters(self.moving,transform)
        if derivative:
            linear,translation=model.derivatives(parameters)
            if not 1<=linear.shape[0]<=12:return None
            from fast_ants.backends.metal.metrics.mattes.derivative import stream_derivative_histograms
            if self.cached_bins is None:
                self.cached_bins=fixed_bins(self.ctx,self.fixed_values,self.bins,self.fixed_range,
                                            parameters=self.fixed_bin_parameters)
            if 'streamed_mi_fixed_plan' in self.ctx.optimizations:
                self._get_stream_fixed_plan()
            inverse=self.moving.grid.lps_to_index[:3,:3]
            raw,width=stream_derivative_histograms(
                self.ctx,self.moving.data,self.indices,self.cached_bins,self.bins,
                self.moving_range,mapping,
                (inverse@np.diag(self.moving.grid.spacing_xyz*.5),self.moving.grid.spacing_xyz),
                (self.grid.index_to_lps[:3,:3],self.grid.center-model.center,
                 (np.asarray(self.grid.shape_xyz)-1)*.5),linear,translation,
                moving_parameters=self.moving_bin_parameters,fixed_plan=self._stream_fixed_plan,
                return_device=True)
        else:
            from fast_ants.backends.metal.metrics.mattes.sampling import stream_value_histograms
            raw,width=stream_value_histograms(
                self.ctx,self.moving.data,self.indices,self.bins,self.moving_range,mapping,
                self._get_stream_fixed_plan(),moving_parameters=self.moving_bin_parameters,
                return_device=True)
        packet=finalize_packed_device(self.ctx,raw.packed,raw.bins,raw.parameters,width,
                                      bad_blocks=raw.bad_blocks)
        return packet,raw.parameters

    def _finish_device_control(self,words,parameters,derivative):
        from fast_ants.backends.metal.metrics.mattes.finalize import decode_control
        value,descent,count=decode_control(words,parameters)
        transfer_bytes=(parameters+3)*np.dtype(np.uint32).itemsize
        self.control_evaluations+=1;self.control_derivative_evaluations+=int(derivative)
        self.control_bytes+=transfer_bytes
        # These private controls intentionally omit PDFs: they were finalized
        # on the GPU and never downloaded as arrays for this evaluation.
        details={'device_finalization':True,'pdf_details_omitted':True,
                 'control_transfer_bytes':transfer_bytes,'software_expanded_precision':False}
        details['fused_derivative' if derivative else 'fused_value_histogram']=True
        return MetricEvaluation(value,descent,count,details)

    def evaluate_values_many(self,model,trials):
        """Evaluate two value trials with one bounded read when the private path allows it.

        Invalid trial domains retain the optimizer's FLOAT1_MAX policy. Device
        execution failures propagate; a failed GPU is never an invalid cost.
        """
        trials=tuple(trials)
        maximum=np.float32(np.finfo(np.float32).max)
        def independent():
            values=[]
            for trial in trials:
                try:
                    value=np.float32(self.evaluate(model,trial,False).value)
                    values.append(value if np.isfinite(value) else maximum)
                except (NoValidSamplesError,ValueError):values.append(maximum)
            return tuple(values)
        if len(trials)!=2 or not self.batch_value_evaluations:return independent()
        from fast_ants.backends.metal.metrics.mattes.finalize import INVALID_DATA
        packets=[]
        for trial in trials:
            try:
                pending=self._enqueue_device_control(model,trial,False)
                if pending is None:return independent()
                packet,_=pending
            except (NoValidSamplesError,ValueError):
                # Preserve one read for the pair even when bounded host
                # geometry validation rejects one member before dispatch.
                packet=self.ctx.upload([0,0,INVALID_DATA],'invalid_mattes_control',np.uint32)
            packets.append(packet)
            if 'async_control_prefetch' in self.ctx.optimizations:
                self.ctx.submit(packet,reason='independent_mattes_trial')
        pair=self.ctx.call('mattes_control_pair32',
            'uint i=thread_position_in_grid.x;out0[i]=i<3u?first[i]:second[i-3u];',
            {'first':packets[0],'second':packets[1]},[(6,)],6,[mx.uint32])
        words=self.ctx.control(pair,'mattes_final_pair',6)
        values=[]
        for offset in (0,3):
            try:
                result=self._finish_device_control(words[offset:offset+3],0,False)
                values.append(np.float32(result.value))
            except (NoValidSamplesError,ValueError):values.append(maximum)
        return tuple(values)

    def evaluate(self,model,parameters,derivative=True):
        pending=self._enqueue_device_control(model,parameters,derivative)
        if pending is not None:
            packet,size=pending
            words=self.ctx.control(packet,'mattes_final',size+3)
            return self._finish_device_control(words,size,derivative)
        transform=model.transform(parameters)
        if (not derivative and 'streamed_mi_fixed_plan' in self.ctx.optimizations
                and self.moving_mask is None and self.grouped_plan is None
                and self.indices.shape[0]>=16384 and 5<=self.bins<=32):
            from fast_ants.backends.metal.metrics.mattes.sampling import stream_value_histograms
            hist,counts,pdf,count,width=stream_value_histograms(
                self.ctx,self.moving.data,self.indices,self.bins,self.moving_range,
                self.mapping_parameters(self.moving,transform),self._get_stream_fixed_plan(),
                moving_parameters=self.moving_bin_parameters)
            return self._finish(hist,counts,pdf,count,width,False,fused_value_histogram=True)
        if (derivative and 'streamed_mi_derivative' in self.ctx.optimizations
                and self.moving_mask is None and self.grouped_plan is None
                and self.indices.shape[0]>=16384 and 5<=self.bins<=32):
            linear,translation=model.derivatives(parameters)
            if 1<=linear.shape[0]<=12:
                from fast_ants.backends.metal.metrics.mattes.derivative import stream_derivative_histograms
                if self.cached_bins is None:
                    self.cached_bins=fixed_bins(self.ctx,self.fixed_values,self.bins,self.fixed_range,
                                                parameters=self.fixed_bin_parameters)
                if 'streamed_mi_fixed_plan' in self.ctx.optimizations:
                    self._get_stream_fixed_plan()
                inverse=self.moving.grid.lps_to_index[:3,:3]
                hist,counts,pdf,count,width=stream_derivative_histograms(
                    self.ctx,self.moving.data,self.indices,self.cached_bins,self.bins,
                    self.moving_range,self.mapping_parameters(self.moving,transform),
                    (inverse@np.diag(self.moving.grid.spacing_xyz*.5),self.moving.grid.spacing_xyz),
                    (self.grid.index_to_lps[:3,:3],self.grid.center-model.center,
                     (np.asarray(self.grid.shape_xyz)-1)*.5),linear,translation,
                    moving_parameters=self.moving_bin_parameters,fixed_plan=self._stream_fixed_plan)
                return self._finish(hist,counts,pdf,count,width,True,fused_derivative=True)
        direct_validity=('direct_mi_validity' in self.ctx.optimizations and self.grouped_plan is None)
        fused_value=(not derivative and 'fused_value_sampling' in self.ctx.optimizations and
                     (self.grouped_plan is not None or 'indexed_mi' in self.ctx.optimizations or direct_validity))
        if fused_value:
            from fast_ants.backends.metal.metrics.mattes.sampling import sample_value_bins
            moving,valid,bi,bf=sample_value_bins(self.ctx,self.moving.data,self.indices,
                *self.mapping_parameters(self.moving,transform),self.bins,self.moving_bin_parameters)
            indices=None
        else:
            moving,valid,indices=self.sample(self.moving,transform)
        if self.moving_mask is not None:valid=self.mask_and(valid,self.moving_mask,transform)
        selection=count=None
        if self.grouped_plan is not None:
            # Fixed CSR membership is reusable; moving validity remains
            # dynamic and is filtered by the grouped histogram each time.
            fixed,virtual=self.fixed_values,self.indices
        elif direct_validity:
            fixed,virtual=self.fixed_values,self.indices
        elif 'indexed_mi' in self.ctx.optimizations:
            selection,count=ordered_selection(self.ctx,valid);fixed,virtual=self.fixed_values,self.indices
        else:
            fixed,moving,bases,offsets,virtual=compact(self.ctx,valid,self.fixed_values,moving,indices[0],indices[1],self.indices)
            indices=(bases,offsets)
        if derivative:
            inverse=self.moving.grid.lps_to_index[:3,:3]
            grad=gradient(self.ctx,self.moving.data,indices,inverse@np.diag(self.moving.grid.spacing_xyz*.5),self.moving.grid.spacing_xyz)
            local=affine(self.ctx,virtual,self.grid.index_to_lps[:3,:3],self.grid.center-model.center,(np.asarray(self.grid.shape_xyz)-1)*.5)
            linear,translation=model.derivatives(parameters)
            intensity=derivatives(self.ctx,local,grad,linear,translation)
        else:intensity=mx.zeros((fixed.size,0),dtype=mx.float32)
        if not fused_value:
            bi,bf=float1_bin_fractions(self.ctx,moving,self.bins,self.moving_bin_parameters)
        # Derivative channels reuse each sample's weights. A value-only
        # histogram consumes them once; the opt-in candidate computes the
        # same cubic(argument) in that histogram instead of storing N*4 terms.
        cache_weights=('cached_cubic' in self.ctx.optimizations and
                       (derivative or 'value_inline_cubic' not in self.ctx.optimizations))
        weights=cubic_tables(self.ctx,bf,derivative) if cache_weights else None
        if self.grouped_plan is not None:
            from fast_ants.backends.metal.metrics.mattes.histogram import histograms as grouped_histograms
            hist,counts,pdf,count,width=grouped_histograms(self.ctx,self.grouped_plan,moving,valid,intensity,bi,bf,
                                                        self.moving_range,weights=weights,moving_width=self.moving_bin_width)
        else:
            hist,counts,pdf,count,width=histograms(self.ctx,fixed,moving,intensity,self.bins,self.fixed_range,self.moving_range,bi,bf,
                                                 selection=selection,sample_count=count,valid=valid if direct_validity else None,
                                                 cached_bins=self.cached_bins if selection is not None or direct_validity else None,weights=weights,
                                                 fixed_parameters=self.fixed_bin_parameters,moving_width=self.moving_bin_width)
        return self._finish(hist,counts,pdf,count,width,derivative)

    def _finish(self,hist,counts,pdf,count,width,derivative,**details):
        if not count:raise NoValidSamplesError('No valid GPU MI samples after range checks')
        value,descent,joint,fp,mp,pd=finalize_mattes_float1(hist,counts,pdf,count,np.float32(width))
        transfer_bytes=int(hist.nbytes+pdf.nbytes+counts.nbytes)
        self.control_evaluations+=1;self.control_derivative_evaluations+=int(derivative);self.control_bytes+=transfer_bytes
        return MetricEvaluation(float(value),descent,count,{'joint_pdf':joint,'fixed_pdf':fp,'moving_pdf':mp,
              'pdf_derivative':pd,'control_transfer_bytes':transfer_bytes,'software_expanded_precision':False,**details})
