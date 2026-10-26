"""Bounded GPU Mattes finalization with deferred, small control reads.

The raw PDF layout is the streaming histogram's packed uint32 transport:
float bit patterns for [fixed bin, moving bin, joint/parameter channel],
integer fixed counts, valid count, then optional diagnostic words. Arithmetic
uses ordinary FP32. A stable positive tree replaces the bounded CPU FP64 joint
total; no software-expanded precision or voxel work on the CPU is introduced.
"""

import numpy as np

from fast_ants.backends.metal.runtime import mx
from fast_ants.pipeline.float1_policy import FLOAT1_PDF_EPS
from fast_ants.core.results import NoValidSamplesError


INVALID_DATA=1
NEGATIVE_HISTOGRAM=2
INVALID_NORMALIZATION=4
EMPTY_DOMAIN=8
INVALID_RESULT=16


def _layout(bins,parameters,bin_width):
    if (not isinstance(bins,(int,np.integer)) or not 5<=bins<=32 or
            not isinstance(parameters,(int,np.integer)) or not 0<=parameters<=12):
        raise ValueError('GPU MI finalization supports 5..32 bins and 0..12 parameters')
    with np.errstate(over='ignore',invalid='ignore'):
        width=np.asarray(bin_width,dtype=np.float32)
    if width.ndim or not np.isfinite(width) or width<=0:
        raise ValueError('GPU MI bin width must be finite and positive in FP32')
    return int(bins),int(parameters),np.float32(width)


def finalize_packed_device(ctx,packed,bins,parameters,bin_width,*,bad_blocks=0):
    """Return a lazy uint32 [value, descent..., valid_count, status] packet.

    The first ``parameters+1`` words contain FP32 bit patterns. The last two
    words remain integer controls. No device-to-host read is performed here;
    independent metric trials can be submitted before reading their packets.
    """
    bins,parameters,width=_layout(bins,parameters,bin_width)
    if not isinstance(bad_blocks,(int,np.integer)) or bad_blocks<0:
        raise ValueError('GPU MI diagnostic block count must be nonnegative')
    bad_blocks=int(bad_blocks)
    channels=parameters+1
    cells=bins*bins
    size=cells*channels
    if (not isinstance(packed,mx.array) or packed.dtype!=mx.uint32 or
            packed.shape!=(size+bins+1+bad_blocks,)):
        raise ValueError('Invalid packed GPU MI histogram/control layout')
    epsilon=f'{float(FLOAT1_PDF_EPS):.24g}f'
    prepare='''
      uint t=thread_index_in_threadgroup,B=meta[0],P=meta[1],K=P+1u;
      uint S=B*B*K,C=B*B;
      threadgroup float totals[256];threadgroup uint flags[256];
      threadgroup float normalization,derivative_factor;
      threadgroup uint state,valid;
      float2 sum=float2(0);uint bad=0;
      for(uint cell=t;cell<C;cell+=256u){
        float h=as_type<float>(packed[cell*K]);
        if(!metal::isfinite(h))bad|=1u;
        else if(h<0)bad|=2u;
        else sum=compensated(sum,h);
        for(uint p=1;p<K;p++)
          if(!metal::isfinite(as_type<float>(packed[cell*K+p])))bad|=1u;
      }
      for(uint i=t;i<meta[2];i+=256u)
        if(packed[S+B+1u+i])bad|=1u;
      totals[t]=sum.x;flags[t]=bad;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128u;stride>0;stride/=2u){
        if(t<stride){totals[t]+=totals[t+stride];flags[t]|=flags[t+stride];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0){
        valid=packed[S+B];state=flags[0];normalization=0;derivative_factor=0;
        float total=totals[0],denominator=params[0]*float(valid);
        if(!metal::isfinite(total))state|=4u;
        if(valid==0u||total<PDF_EPS)state|=8u;
        if(valid>0u&&(!metal::isfinite(denominator)||denominator<=0))state|=4u;
        if(state==0u){
          normalization=1.0f/total;
          if(P>0u)derivative_factor=-1.0f/denominator;
          if(!metal::isfinite(normalization)||normalization<=0||
             !metal::isfinite(derivative_factor))state|=4u;
        }
        out0[0]=normalization;out0[1]=derivative_factor;
        out2[0]=valid;out2[1]=state;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(t<B){
        float fixed=0,moving=0;
        if(state==0u){
          fixed=float(packed[S+t])*normalization;
          for(uint i=0;i<B;i++)
            moving+=as_type<float>(packed[(i*B+t)*K])*normalization;
        }
        out1[t]=fixed;out1[B+t]=moving;
      }'''.replace('PDF_EPS',epsilon)
    scales,marginals,status=ctx.call('metric_finalize_prepare32',prepare,
        {'packed':packed,'params':ctx.upload([width]),
         'meta':ctx.meta(bins,parameters,bad_blocks)},
        [(2,),(2,bins),(2,)],256,[mx.float32,mx.float32,mx.uint32])
    finish='''
      uint t=thread_index_in_threadgroup,B=meta[0],P=meta[1],K=P+1u;
      threadgroup uint flags[32];uint bad=status[1];float result=0;
      if(t<K){
        if(bad==0u){
          float sum=0,normalization=scales[0],factor=scales[1];
          for(uint i=0;i<B;i++){
            float fixed=marginals[i];
            if(fixed<=PDF_EPS)continue;
            float log_fixed=t==0u?metal::log(fixed):0.0f;
            for(uint j=0;j<B;j++){
              uint cell=(i*B+j)*K;
              float moving=marginals[B+j];
              float joint=as_type<float>(packed[cell])*normalization;
              if(moving>PDF_EPS&&joint>PDF_EPS){
                float ratio=metal::log(joint/moving);
                if(t==0u)sum=metal::fma(joint,ratio-log_fixed,sum);
                else{
                  float derivative=as_type<float>(packed[cell+t])*factor;
                  sum=metal::fma(derivative,ratio,sum);
                }
              }
            }
          }
          result=t==0u?-sum:sum;
          if(!metal::isfinite(result))bad|=16u;
        }else result=t==0u?3.4028234663852886e38f:0.0f;
        out0[t]=as_type<uint>(result);
      }
      flags[t]=bad;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(t==0u){
        uint total=0;for(uint i=0;i<32u;i++)total|=flags[i];
        out0[K]=status[0];out0[K+1u]=total;
      }'''.replace('PDF_EPS',epsilon)
    return ctx.call('metric_finalize_controls32',finish,
        {'packed':packed,'scales':scales,'marginals':marginals,'status':status,
         'meta':ctx.meta(bins,parameters)},[(parameters+3,)],32,[mx.uint32])


def finalize_device(ctx,raw_histogram,counts,bin_width,*,badflags=None):
    """Compatibility wrapper for separate device raw PDFs and integer counts.

    Production streaming callers should use :func:`finalize_packed_device`
    directly and avoid this additional layout-pack dispatch.
    """
    if (not isinstance(raw_histogram,mx.array) or raw_histogram.dtype!=mx.float32
            or raw_histogram.ndim!=3 or raw_histogram.shape[0]!=raw_histogram.shape[1]):
        raise ValueError('GPU MI raw PDF must be square FP32 [bins,bins,channels]')
    bins,_,channels=raw_histogram.shape
    bins,parameters,_=_layout(bins,channels-1,bin_width)
    if (not isinstance(counts,mx.array) or counts.dtype!=mx.uint32 or counts.shape!=(bins+1,)):
        raise ValueError('GPU MI counts must be uint32 fixed counts plus valid count')
    if badflags is not None and (not isinstance(badflags,mx.array) or badflags.dtype!=mx.uint32):
        raise ValueError('GPU MI diagnostic flags must be uint32')
    bad_size=0 if badflags is None else badflags.size
    bad_blocks=(bad_size+255)//256
    size=bins*bins*channels
    inputs={'data':raw_histogram,'counts':counts,'meta':ctx.meta(size,bins,bad_size)}
    source='''uint i=thread_position_in_grid.x,S=meta[0],B=meta[1];
      if(i<S)out0[i]=as_type<uint>(data[i]);
      else if(i<S+B+1u)out0[i]=counts[i-S];'''
    if bad_size:
        inputs['badflags']=badflags
        source+='''else{uint begin=(i-S-B-1u)*256u,bad=0;
          for(uint n=begin;n<metal::min(meta[2],begin+256u);n++)bad|=badflags[n];
          out0[i]=bad;}'''
    packed_size=size+bins+1+bad_blocks
    packed=ctx.call('metric_finalize_pack_raw32',source,inputs,[(packed_size,)],packed_size,[mx.uint32])
    return finalize_packed_device(ctx,packed,bins,parameters,bin_width,bad_blocks=bad_blocks)


def decode_control(values,parameters):
    """Decode one materialized control packet without exposing borrowed arrays."""
    values=np.asarray(values)
    if (not isinstance(parameters,(int,np.integer)) or not 0<=parameters<=12 or
            values.dtype!=np.uint32 or values.shape!=(parameters+3,)):
        raise ValueError('Invalid GPU MI final control packet')
    parameters=int(parameters)
    valid=int(values[parameters+1]);status=int(values[parameters+2])
    if status&~EMPTY_DOMAIN:
        raise ValueError('Invalid or nonfinite GPU MI PDF/derivative (status '+str(status)+')')
    if status&EMPTY_DOMAIN or valid==0:
        raise NoValidSamplesError('No valid GPU MI samples or joint probability mass')
    floats=values[:parameters+1].view(np.float32)
    if not np.isfinite(floats).all():
        raise ValueError('GPU MI value and derivatives must be finite')
    return float(floats[0]),floats[1:].copy(),valid


def read_control(ctx,packet,parameters):
    """Read only value, descent and two integer controls from one evaluation."""
    return decode_control(ctx.control(packet,'mattes_final',parameters+3),parameters)
