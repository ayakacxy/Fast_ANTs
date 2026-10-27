"""Device Mattes PDF finalization with the float1 assignment boundaries.

Only the bounded histogram total uses native double compensated accumulation,
as in ITK. PDF storage, logs, normalization and the value/descent FMA chains
are FP32. Registration downloads a value, its derivatives and two status words.
"""

import numpy as np

from fast_ants.core.results import NoValidSamplesError
from fast_ants.pipeline.float1_policy import FLOAT1_PDF_EPS


def finalize_device(ctx,histogram,counts,width):
    bins,_,channels=histogram.shape
    parameters=channels-1
    epsilon=f'{float(FLOAT1_PDF_EPS):.24g}f'
    controls,state=ctx.call('mattes_normalization','''
      unsigned int B=meta[0],K=meta[1],count=0,bad=0;
      double total=0,correction=0;
      for(unsigned int row=0;row<B;row++){
        count+=counts[row];
        for(unsigned int column=0;column<B;column++){
          unsigned int at=(row*B+column)*K;float value=histogram[at];
          if(!isfinite(value)||value<0)bad|=1;
          double a=double(value)-correction,s=total+a;correction=(s-total)-a;total=s;
          for(unsigned int p=1;p<K;p++)if(!isfinite(histogram[at+p]))bad|=1;
        }
      }
      float sum=float(total),denominator=params[0]*float(count);
      if(count==0||sum<PDF_EPS)bad|=2;
      if(!isfinite(sum)||!isfinite(denominator)||denominator<=0)bad|=4;
      out0[0]=bad?0:float(1.0/double(sum));
      out0[1]=bad?0:float(-1.0/double(denominator));
      out1[0]=count;out1[1]=bad;'''.replace('PDF_EPS',epsilon),
      {'histogram':histogram,'counts':counts,'params':ctx.upload([width]),'meta':ctx.meta(bins,channels)},
      [(2,),(2,)],1,[np.float32,np.uint32])
    joint,pdf_derivative,fixed=ctx.call('mattes_normalize_bins','''
      unsigned int B=meta[0],P=meta[1]-1;
      out0[i]=histogram[i*(P+1)]*controls[0];
      for(unsigned int p=0;p<P;p++)out1[i*P+p]=histogram[i*(P+1)+p+1]*controls[1];
      if(i<B)out2[i]=float(counts[i])*controls[0];''',
      {'histogram':histogram,'counts':counts,'controls':controls,'meta':ctx.meta(bins,channels)},
      [(bins,bins),(bins,bins,parameters),(bins,)],bins*bins)
    moving=ctx.call('mattes_moving_pdf','''float total=0;
      for(unsigned int row=0;row<meta[0];row++)total+=joint[row*meta[0]+i];out0[i]=total;''',
      {'joint':joint,'meta':ctx.meta(bins)},[(bins,)],bins)
    packet=ctx.call('mattes_final_controls','''
      unsigned int B=meta[0],P=meta[1]-1;float total=0;
      if(state[1]==0){
        for(unsigned int row=0;row<B;row++){
          if(fixed[row]<=PDF_EPS)continue;
          float log_fixed=logf(fixed[row]);
          for(unsigned int column=0;column<B;column++){
            unsigned int at=row*B+column;
            if(moving[column]>PDF_EPS&&joint[at]>PDF_EPS){
              float ratio=logf(joint[at]/moving[column]);
              if(i==0)total=__fmaf_rn(joint[at],ratio-log_fixed,total);
              else total=__fmaf_rn(pdf[at*P+i-1],ratio,total);
            }
          }
        }
      }
      float value=i==0?-total:total;
      out0[i]=__float_as_uint(value);
      if(i==0){out0[P+1]=state[0];out0[P+2]=state[1];}'''.replace('PDF_EPS',epsilon),
      {'joint':joint,'pdf':pdf_derivative,'fixed':fixed,'moving':moving,'state':state,
       'meta':ctx.meta(bins,channels)},[(parameters+3,)],channels,[np.uint32])
    return packet,(joint,fixed,moving,pdf_derivative)


def finalize_registration(ctx,histogram,counts,parameters,*,preload=True):
    """Fuse bounded registration controls while retaining ordered FP32 sums.

    The public PDF-producing path remains available as an independent oracle.
    Shared PDFs/logs remove intermediate global arrays and four launches. One
    thread retains the exact serial native-double compensated histogram total;
    each result retains the original row/column FMA accumulation sequence.
    """
    bins,_,channels=histogram.shape
    if bins>32 or channels>13:
        raise ValueError('Fused Mattes finalization supports at most 32 bins and 12 parameters')
    epsilon=f'{float(FLOAT1_PDF_EPS):.24g}f'
    source=('constexpr unsigned int B='+str(bins)+',K='+str(channels)+',P=K-1;\n'+'''
      unsigned int t=threadIdx.x;
      __shared__ float joint[B*B],ratio[B*B],fixed[B],moving[B],log_fixed[B],scale[2];
      __shared__ unsigned int state[2];
      if(t==0){state[0]=0;state[1]=0;}__syncthreads();
      unsigned int bad=0;
      for(unsigned int at=t;at<B*B*K;at+=256){
        float value=histogram[at];
        if(!isfinite(value)||(at%K==0&&value<0))bad|=1;}
      if(bad)atomicOr(&state[1],bad);
      PRELOAD
      if(t==0){
        unsigned int count=0;double total=0,correction=0;
        for(unsigned int row=0;row<B;row++){
          count+=counts[row];
          for(unsigned int column=0;column<B;column++){
            float value=JOINT_VALUE;
            double a=double(value)-correction,s=total+a;correction=(s-total)-a;total=s;}}
        float sum=float(total),denominator=params[0]*float(count);unsigned int flags=0;
        if(count==0||sum<PDF_EPS)flags|=2;
        if(!isfinite(sum)||!isfinite(denominator)||denominator<=0)flags|=4;
        if(flags)atomicOr(&state[1],flags);
        scale[0]=float(1.0/double(sum));scale[1]=float(-1.0/double(denominator));state[0]=count;}
      __syncthreads();
      if(state[1]){if(t<K)out0[t]=__float_as_uint(0.0f);
        if(t==0){out0[P+1]=state[0];out0[P+2]=state[1];}return;}
      for(unsigned int at=t;at<B*B;at+=256)joint[at]=NORMALIZE_VALUE*scale[0];
      if(t<B){fixed[t]=float(counts[t])*scale[0];log_fixed[t]=logf(fixed[t]);}
      __syncthreads();
      if(t<B){float total=0;for(unsigned int row=0;row<B;row++)total+=joint[row*B+t];moving[t]=total;}
      __syncthreads();
      for(unsigned int at=t;at<B*B;at+=256){unsigned int row=at/B,column=at%B;
        ratio[at]=(fixed[row]>PDF_EPS&&moving[column]>PDF_EPS&&joint[at]>PDF_EPS)?
          logf(joint[at]/moving[column]):0.0f;}
      __syncthreads();
      if(t<K){float total=0;
        for(unsigned int row=0;row<B;row++){
          if(fixed[row]<=PDF_EPS)continue;
          for(unsigned int column=0;column<B;column++){
            unsigned int at=row*B+column;
            if(moving[column]>PDF_EPS&&joint[at]>PDF_EPS){
              if(t==0)total=__fmaf_rn(joint[at],ratio[at]-log_fixed[row],total);
              else {float derivative=histogram[at*K+t]*scale[1];
                total=__fmaf_rn(derivative,ratio[at],total);}}}}
        out0[t]=__float_as_uint(t==0?-total:total);}
      if(t==0){out0[P+1]=state[0];out0[P+2]=state[1];}
      ''').replace('PDF_EPS',epsilon)
    source=source.replace('PRELOAD',('for(unsigned int at=t;at<B*B;at+=256)joint[at]=histogram[at*K];'
        '__syncthreads();') if preload else '')
    source=source.replace('JOINT_VALUE','joint[row*B+column]' if preload else 'histogram[(row*B+column)*K]')
    source=source.replace('NORMALIZE_VALUE','joint[at]' if preload else 'histogram[at*K]')
    name='mattes_registration_finalize'+('_shared' if preload else '')
    packet=ctx.call(name+'_b'+str(bins)+'_p'+str(channels-1),source,
        {'histogram':histogram,'counts':counts,'params':parameters},[(channels+2,)],256,[np.uint32])
    return packet,None


def decode_control(words):
    state=int(words[-1]);count=int(words[-2])
    if state&2:raise NoValidSamplesError('No valid CUDA MI samples after range checks')
    if state:raise ValueError('Nonfinite or invalid CUDA Mattes histogram controls')
    values=words[:-2].view(np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite CUDA Mattes value or derivative')
    return float(values[0]),values[1:].copy(),count
