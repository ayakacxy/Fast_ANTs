"""Fixed-bin GPU grouping and bounded shared-memory Mattes histograms.

The fixed sample order is prepared once per resolution. Each CUDA block owns
one fixed row and at most 1024 samples. Value-only tiles with at most 32 bins
use thread-private shared rows and a warp merge. Derivative tiles use shared
FP32 atomics; a compensated ordered merge combines tiles without global atomics.
Reduction grouping can differ from CPU and Metal, but every sample uses the
same float1 intensity bins and cubic Parzen/derivative terms.
"""

from dataclasses import dataclass
import numpy as np

from fast_ants.backends.cuda.runtime import cp


@dataclass
class FixedBinPlan:
    indices: object
    offsets: object
    task_offsets: object
    tasks: int
    bins: int
    tile: int = 1024


def fixed_bin_plan(ctx,fixed,bins):
    """Create a reusable device grouping, reading only the fixed marginal."""
    with ctx.device,ctx.stream:
        indices=cp.argsort(fixed).astype(cp.uint32)
        counts=cp.bincount(fixed,minlength=bins).astype(cp.uint32)
    marginal=ctx.control(counts,'mattes_plan_counts',bins).astype(np.uint64)
    offsets=np.concatenate(([0],np.cumsum(marginal))).astype(np.uint32)
    tasks=np.concatenate(([0],np.cumsum((marginal+1023)//1024))).astype(np.uint32)
    return FixedBinPlan(indices,ctx.upload(offsets,'mattes_plan',np.uint32),
                        ctx.upload(tasks,'mattes_plan',np.uint32),int(tasks[-1]),bins)


def histograms(ctx,plan,moving,valid,intensity,bin_indices,fractions,moving_range,*,
               geometry=None,parameters=None,range_device=None,aggregate=False,private=False):
    """Return device joint/derivative bins and exact uint32 fixed counts."""
    bins=plan.bins
    if parameters is None:parameters=intensity.shape[1]
    channels=parameters+1
    if bins*channels>8192:
        raise ValueError('CUDA Mattes shared histogram exceeds 8192 FP32 values per row')
    if geometry is None:
        inputs={'intensity':intensity}
        local=''
        component='intensity[n*P+p]'
    else:
        points,gradients,control,fixed=geometry
        inputs={'points':points,'gradients':gradients,'control':control,'fixed':fixed}
        local='''
        const float* linear=reinterpret_cast<const float*>(control+24);
        const float* translation=linear+P*9;
        float position[3],intensity[P?P:1];
        for(int c=0;c<3;c++){
          float value=fixed[24+c];
          for(int j=0;j<3;j++)value+=fixed[15+c*3+j]*(points[3*n+j]-fixed[27+j]);
          position[c]=value;}
        for(unsigned int p=0;p<P;p++){
          float total=0;for(int c=0;c<3;c++){
            float spatial=translation[p*3+c];
            for(int j=0;j<3;j++)spatial=__fmaf_rn(linear[p*9+c*3+j],position[j],spatial);
            total=__fmaf_rn(gradients[3*n+c],spatial,total);}
          intensity[p]=total;}
        ''' if parameters else ''
        component='intensity[p]' if parameters else '0.0f'
    source=('constexpr unsigned int B='+str(bins)+',P='+str(parameters)+',K=P+1;\n'+'''
      unsigned int t=threadIdx.x,task=blockIdx.x,row=0;
      while(task>=tasks[row+1])row++;
      unsigned int begin=offsets[row]+(task-tasks[row])*1024;
      unsigned int end=min(offsets[row+1],begin+1024);
      __shared__ float histogram[B*K];__shared__ unsigned int count;
      for(unsigned int q=t;q<B*K;q+=256)histogram[q]=0;
      if(t==0)count=0;__syncthreads();
      unsigned int selected=0;
      for(unsigned int q=begin+t;q<end;q+=256){
        unsigned int n=order[q];float value=moving[n];
        if(!valid[n]||value<range[0]||value>range[1])continue;
        selected++;int mi=bin_indices[n];float fraction=fractions[n];
        LOCAL_DERIVATIVES
        for(int k=0;k<4;k++){
          int column=mi+k-1;float argument=float(k-1)-fraction;
          VALUE_UPDATE
          float weight=cubic_derivative(argument);
          for(unsigned int p=0;p<P;p++)
            DERIVATIVE_UPDATE
        }
      }
      atomicAdd(&count,selected);__syncthreads();
      for(unsigned int q=t;q<B*K;q+=256)out0[task*B*K+q]=histogram[q];
      if(t==0)out1[task]=count;''').replace('LOCAL_DERIVATIVES',local).replace('COMPONENT',component)
    source=source.replace('VALUE_UPDATE',('''
          #if __CUDA_ARCH__ >= 700
          unsigned int active=__activemask();
          unsigned int exact=__ballot_sync(active,fraction==0.0f);
          if(fraction==0.0f){
            unsigned int peers=__match_any_sync(exact,mi);
            if((t&31)==__ffs(peers)-1)atomicAdd(&histogram[column*K],cubic(argument)*float(__popc(peers)));
          }else atomicAdd(&histogram[column*K],cubic(argument));
          #else
          atomicAdd(&histogram[column*K],cubic(argument));
          #endif
          ''') if aggregate else
          'atomicAdd(&histogram[column*K],cubic(argument));')
    source=source.replace('DERIVATIVE_UPDATE',
        ('{float contribution=COMPONENT*weight;if(contribution!=0)atomicAdd(&histogram[column*K+p+1],contribution);}'
         if aggregate and geometry is not None and parameters else
         'atomicAdd(&histogram[column*K+p+1],COMPONENT*weight);').replace('COMPONENT',component))
    name='mattes_grouped_local_tiles' if geometry is not None else 'mattes_grouped_tiles'
    if aggregate:name+='_warp'
    if private and not parameters:
        # Private lanes remove contended value atomics on background samples.
        # Warp sums change reduction grouping, retaining ordinary FP32 storage.
        name+='_private'
        source=('constexpr unsigned int B='+str(bins)+';\n'+'''
          unsigned int t=threadIdx.x,task=blockIdx.x,row=0;
          while(task>=tasks[row+1])row++;
          unsigned int begin=offsets[row]+(task-tasks[row])*1024;
          unsigned int end=min(offsets[row+1],begin+1024);
          __shared__ float histogram[B*256];__shared__ unsigned int count;
          for(unsigned int column=0;column<B;column++)histogram[column*256+t]=0;
          if(t==0)count=0;__syncthreads();
          unsigned int selected=0;
          for(unsigned int q=begin+t;q<end;q+=256){
            unsigned int n=order[q];float value=moving[n];
            if(!valid[n]||value<range[0]||value>range[1])continue;
            selected++;int mi=bin_indices[n];float fraction=fractions[n];
            for(int k=0;k<4;k++){
              int column=mi+k-1;float argument=float(k-1)-fraction;
              histogram[column*256+t]+=cubic(argument);}}
          atomicAdd(&count,selected);__syncthreads();
          unsigned int lane=t&31,warp=t/32;
          for(unsigned int column=warp;column<B;column+=8){
            float total=0;
            for(unsigned int q=lane;q<256;q+=32)total+=histogram[column*256+q];
            for(int shift=16;shift;shift/=2)total+=__shfl_down_sync(0xffffffff,total,shift);
            if(lane==0)out0[task*B+column]=total;}
          if(t==0)out1[task]=count;''')
    partial,counts=ctx.call(name+'_b'+str(bins)+'_p'+str(parameters),source,
        {'order':plan.indices,'offsets':plan.offsets,'tasks':plan.task_offsets,
         'moving':moving,'valid':valid,**inputs,'bin_indices':bin_indices,
         'fractions':fractions,'range':ctx.upload(moving_range) if range_device is None else range_device},
        [(plan.tasks,bins,channels),(plan.tasks,)],plan.tasks*256,[np.float32,np.uint32])
    merged,marginal=ctx.call('mattes_grouped_merge','''
      unsigned int B=meta[0],K=meta[1],row=i/(B*K),column=i%(B*K);
      float total=0,error=0;
      for(unsigned int task=tasks[row];task<tasks[row+1];task++){
        float a=partial[task*B*K+column]-error,s=total+a;error=(s-total)-a;total=s;}
      out0[i]=total;
      if(column==0){unsigned int count=0;
        for(unsigned int task=tasks[row];task<tasks[row+1];task++)count+=counts[task];out1[row]=count;}''',
      {'partial':partial,'counts':counts,'tasks':plan.task_offsets,'meta':ctx.meta(bins,channels)},
      [(bins,bins,channels),(bins,)],bins*bins*channels,[np.float32,np.uint32])
    return merged,marginal
