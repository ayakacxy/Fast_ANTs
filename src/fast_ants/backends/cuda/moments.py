"""Stable centered image moments with bounded FP32 parallel reductions."""

from fast_ants.backends.cuda.runtime import cp


def moments(ctx,data):
    """Return device [mass, centered_x, centered_y, centered_z] scalars."""
    if not isinstance(data,cp.ndarray) or data.dtype!=cp.float32:
        raise ValueError('Parallel moments require an FP32 device image')
    if data.ndim!=3 or min(data.shape)<1 or data.size>=2**32:
        raise ValueError('Parallel moments require a nonempty 3D uint32-sized image')
    nz,ny,nx=data.shape
    block=max(4096,((data.size+4095)//4096+255)//256*256)
    blocks=(data.size+block-1)//block
    partial,correction=ctx.call('moments_parallel_tiles32','''
      unsigned int t=threadIdx.x,b=blockIdx.x,begin=b*meta[4];
      unsigned int end=begin+min(meta[4],meta[3]-begin);
      int3 n=make_int3(meta[0],meta[1],meta[2]);
      float sums[4]={0,0,0,0},errors[4]={0,0,0,0};
      for(unsigned int q=begin+t;q<end;q+=256){
        int3 p=xyz(q,n);float v=data[q];
        float values[4]={v,v*(float(p.x)-float(n.x-1)*0.5f),
                         v*(float(p.y)-float(n.y-1)*0.5f),v*(float(p.z)-float(n.z-1)*0.5f)};
        for(int c=0;c<4;c++){float a=values[c]-errors[c],s=sums[c]+a;
          errors[c]=(s-sums[c])-a;sums[c]=s;}}
      __shared__ float totals[1024],corrections[1024];
      for(int c=0;c<4;c++){totals[c*256+t]=sums[c];corrections[c*256+t]=errors[c];}
      __syncthreads();
      for(unsigned int stride=128;stride;stride/=2){
        if(t<stride)for(int c=0;c<4;c++){
          unsigned int at=c*256+t;
          float a=totals[at+stride]-corrections[at],s=totals[at]+a;
          corrections[at]=(s-totals[at])-a;totals[at]=s;
          a=-corrections[at+stride]-corrections[at];s=totals[at]+a;
          corrections[at]=(s-totals[at])-a;totals[at]=s;}
        __syncthreads();}
      if(t==0)for(int c=0;c<4;c++){
        out0[b*4+c]=totals[c*256];out1[b*4+c]=corrections[c*256];}''',
      {'data':data,'meta':ctx.meta(nx,ny,nz,data.size,block)},[(blocks,4),(blocks,4)],blocks*256)
    return ctx.call('moments_parallel_merge32','''float total=0,error=0;
      for(unsigned int b=0;b<meta[0];b++){
        float a=sums[b*4+i]-error,s=total+a;error=(s-total)-a;total=s;
        a=-correction[b*4+i]-error;s=total+a;error=(s-total)-a;total=s;}
      out0[i]=total;''',{'sums':partial,'correction':correction,'meta':ctx.meta(blocks)},[(4,)],4)
