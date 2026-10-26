"""Single-group Mattes controls with shared, parallel log preparation.

The existing GPU finalizer defines normalization, support and FP32 reduction
order. This candidate retains those rules, including the final per-channel
i/j FMA chain. It combines the two dispatches and computes each joint/log
term once for all channels. No PDF or image data are read back to the CPU.
"""

import numpy as np

from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal.metrics.mattes.finalize import _layout
from fast_ants.pipeline.float1_policy import FLOAT1_PDF_EPS


def finalize_packed_device(ctx,packed,bins,parameters,bin_width,*,bad_blocks=0):
    """Return lazy uint32 [value bits, descent bits..., valid count, status].

    Supports the existing packed raw histogram layout for 5..32 bins and
    0..12 parameters. A 256-thread group owns all shared state; no inter-group
    synchronization, floating atomics or software precision emulation is used.
    """
    bins,parameters,width=_layout(bins,parameters,bin_width)
    if not isinstance(bad_blocks,(int,np.integer)) or bad_blocks<0:
        raise ValueError('GPU MI diagnostic block count must be nonnegative')
    bad_blocks=int(bad_blocks)
    size=bins*bins*(parameters+1)
    if (not isinstance(packed,mx.array) or packed.dtype!=mx.uint32 or
            packed.shape!=(size+bins+1+bad_blocks,)):
        raise ValueError('Invalid packed GPU MI histogram/control layout')
    epsilon=f'{float(FLOAT1_PDF_EPS):.24g}f'
    # Compile-time sizes bound shared storage (under 11 KiB at B=32), avoiding
    # a worst-case allocation for the smaller metric histograms.
    source=('constexpr uint B='+str(bins)+'u,P='+str(parameters)+'u,K=P+1u,C=B*B,S=C*K;\n'
            'constexpr uint BAD_BLOCKS='+str(bad_blocks)+'u;\n'+'''
      uint t=thread_index_in_threadgroup;
      threadgroup float totals[256];threadgroup uint flags[256];
      threadgroup float joint[C],ratios[C];
      threadgroup float fixed[B],moving[B],log_fixed[B];
      threadgroup float normalization,factor;
      threadgroup uint state,valid;
      float2 sum=float2(0);uint bad=0;
      // Same lane-local compensated sums and 256-lane positive tree as the
      // two-stage finalizer. Validate every derivative, including unused bins.
      for(uint cell=t;cell<C;cell+=256u){
        float h=as_type<float>(packed[cell*K]);
        if(!metal::isfinite(h))bad|=1u;
        else if(h<0)bad|=2u;
        else sum=compensated(sum,h);
        for(uint p=1;p<K;p++)
          if(!metal::isfinite(as_type<float>(packed[cell*K+p])))bad|=1u;
      }
      for(uint i=t;i<BAD_BLOCKS;i+=256u)
        if(packed[S+B+1u+i])bad|=1u;
      totals[t]=sum.x;flags[t]=bad;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128u;stride>0;stride/=2u){
        if(t<stride){totals[t]+=totals[t+stride];flags[t]|=flags[t+stride];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0u){
        valid=packed[S+B];state=flags[0];normalization=0;factor=0;
        float total=totals[0],denominator=params[0]*float(valid);
        if(!metal::isfinite(total))state|=4u;
        if(valid==0u||total<PDF_EPS)state|=8u;
        if(valid>0u&&(!metal::isfinite(denominator)||denominator<=0))state|=4u;
        if(state==0u){
          normalization=1.0f/total;
          if(P>0u)factor=-1.0f/denominator;
          if(!metal::isfinite(normalization)||normalization<=0||
             !metal::isfinite(factor))state|=4u;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint cell=t;cell<C;cell+=256u)
        joint[cell]=state==0u?as_type<float>(packed[cell*K])*normalization:0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(t<B){
        float f=0,m=0;
        if(state==0u){
          f=float(packed[S+t])*normalization;
          // Preserve each column's original serial i order and FP32 product
          // assignment. Only the normalized products are shared across uses.
          for(uint i=0;i<B;i++)m+=joint[i*B+t];
        }
        fixed[t]=f;moving[t]=m;
        log_fixed[t]=(state==0u&&f>PDF_EPS)?metal::log(f):0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint cell=t;cell<C;cell+=256u){
        uint i=cell/B,j=cell%B;float ratio=0;
        if(state==0u&&fixed[i]>PDF_EPS&&moving[j]>PDF_EPS&&joint[cell]>PDF_EPS)
          ratio=metal::log(joint[cell]/moving[j]);
        ratios[cell]=ratio;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      bad=state;
      if(t<K){
        float result=0;
        if(bad==0u){
          float accumulated=0;
          for(uint i=0;i<B;i++){
            if(fixed[i]<=PDF_EPS)continue;
            for(uint j=0;j<B;j++){
              uint cell=i*B+j;
              if(moving[j]>PDF_EPS&&joint[cell]>PDF_EPS){
                float ratio=ratios[cell];
                if(t==0u)accumulated=metal::fma(joint[cell],ratio-log_fixed[i],accumulated);
                else{
                  float derivative=as_type<float>(packed[cell*K+t])*factor;
                  accumulated=metal::fma(derivative,ratio,accumulated);
                }
              }
            }
          }
          result=t==0u?-accumulated:accumulated;
          if(!metal::isfinite(result))bad|=16u;
        }else result=t==0u?3.4028234663852886e38f:0.0f;
        out0[t]=as_type<uint>(result);
      }
      flags[t]=bad;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(t==0u){
        uint combined=0;
        for(uint i=0;i<256u;i++)combined|=flags[i];
        out0[K]=valid;out0[K+1u]=combined;
      }''').replace('PDF_EPS',epsilon)
    return ctx.call('metric_finalize_fused32',source,
        {'packed':packed,'params':ctx.upload([width])},
        [(parameters+3,)],256,[mx.uint32])
