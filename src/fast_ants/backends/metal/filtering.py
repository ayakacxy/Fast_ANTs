"""Recursive image and discrete displacement-field Gaussian kernels."""
import numpy as np


def recursive(ctx,lines,coefficients):
    coeff=np.asarray(coefficients,np.float32);count,length=lines.shape
    outputs=ctx.call('recursive', '''uint line=thread_position_in_grid.x;int L=int(meta[0]);uint base=line*L;
      float anchor=data[base];
      for(int i=0;i<L;i++){float2 feed=float2(0),back=float2(0);
        for(int j=0;j<4;j++){
          feed=compensated(feed,coeff[j]*(data[base+metal::max(0,i-j)]-anchor));
          float v=i-j-1>=0?coeff[4+j]*out1[base+i-j-1]:coeff[12+j]*(data[base]-anchor);
          back=compensated(back,v);}
        out1[base+i]=feed.x-back.x;}
      for(int i=L-1;i>=0;i--){float2 feed=float2(0),back=float2(0);
        for(int j=0;j<4;j++){
          feed=compensated(feed,coeff[8+j]*(data[base+metal::min(L-1,i+j+1)]-anchor));
          float v=i+j+1<L?coeff[4+j]*out2[base+i+j+1]:coeff[16+j]*(data[base+L-1]-anchor);
          back=compensated(back,v);}
        out2[base+i]=feed.x-back.x;}
      for(int i=0;i<L;i++)out0[base+i]=out1[base+i]+out2[base+i]+anchor;''',
      {'data':lines,'coeff':ctx.upload(coeff),'meta':ctx.meta(length)},[lines.shape]*3,count)
    return outputs[0]


def convolve(ctx,lines,kernel):
    count,length=lines.shape
    return ctx.call('convolve', '''uint i=thread_position_in_grid.x;int L=int(meta[0]),K=int(meta[1]);
       uint base=(i/L)*L;int p=i%L;float2 total=float2(0);
       for(int k=0;k<K;k++) total=compensated(total,coeff[k]*data[base+metal::clamp(p+k-K/2,0,L-1)]);
       out0[i]=total.x;''',{'data':lines,'coeff':ctx.upload(kernel),'meta':ctx.meta(length,len(kernel))},[lines.shape],lines.size)


def convolve_field(ctx,values,kernel,dimension,original=None,weight=None):
    """Convolve native (component,z,y,x) storage without axis-copy buffers.

    Adjacent GPU threads write adjacent voxels for every axis. On the last
    axis, the optional original field folds in the existing boundary/mix pass.
    """
    _,nz,ny,nx=values.shape
    stride=(1,nx,nx*ny)[dimension]
    length=(nx,ny,nz)[dimension]
    inputs={'data':values,'coeff':ctx.upload(kernel),
            'meta':ctx.meta(length,len(kernel),stride,nx,ny,nz)}
    source='''uint i=thread_position_in_grid.x;
      int L=int(meta[0]),K=int(meta[1]),S=int(meta[2]);
      int p=int(i/uint(S))%L;uint base=i-uint(p*S);float2 total=float2(0);
      for(int k=0;k<K;k++){
        uint j=base+uint(metal::clamp(p+k-K/2,0,L-1)*S);
        total=compensated(total,coeff[k]*data[j]);
      }
    '''
    if original is None:
        source+='out0[i]=total.x;'
        name='convolve_field_axis'
    else:
        inputs.update(original=original,weight=ctx.upload([weight]))
        source+='''int3 n=int3(meta[3],meta[4],meta[5]);
          int3 p3=xyz(i%(meta[3]*meta[4]*meta[5]),n);
          out0[i]=edge(p3,n)?0.0f:total.x*weight[0]+original[i]*(1.0f-weight[0]);'''
        name='convolve_field_axis_mix'
    return ctx.call(name,source,inputs,[values.shape],values.size)


def mix_boundary(ctx,smoothed,original,weight):
    _,nz,ny,nx=original.shape
    return ctx.call('mix_boundary', '''uint i=thread_position_in_grid.x;int3 n=int3(meta[0],meta[1],meta[2]);
       int3 p=xyz(i%(meta[0]*meta[1]*meta[2]),n);
       out0[i]=edge(p,n)?0.0f:smoothed[i]*weight[0]+original[i]*(1.0f-weight[0]);''',
       {'smoothed':smoothed,'original':original,'weight':ctx.upload([weight]),'meta':ctx.meta(nx,ny,nz)},[original.shape],original.size)
