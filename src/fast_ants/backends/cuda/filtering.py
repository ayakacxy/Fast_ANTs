"""Recursive image and discrete displacement-field Gaussian kernels."""
import numpy as np
from fast_ants.backends.cuda.runtime import cp
from fast_ants.backends.cuda.types import CUDAImage


def recursive(ctx,lines,coefficients):
    coeff=np.asarray(coefficients,np.float32);count,length=lines.shape
    outputs=ctx.call('recursive', '''unsigned int line=i;int L=int(meta[0]);unsigned int base=line*L;
      float anchor=data[base];
      for(int i=0;i<L;i++){float2 feed=make_float2(0.0f,0.0f),back=make_float2(0.0f,0.0f);
        for(int j=0;j<4;j++){
          feed=compensated(feed,coeff[j]*(data[base+max(0,i-j)]-anchor));
          float v=i-j-1>=0?coeff[4+j]*out1[base+i-j-1]:coeff[12+j]*(data[base]-anchor);
          back=compensated(back,v);}
        out1[base+i]=feed.x-back.x;}
      for(int i=L-1;i>=0;i--){float2 feed=make_float2(0.0f,0.0f),back=make_float2(0.0f,0.0f);
        for(int j=0;j<4;j++){
          feed=compensated(feed,coeff[8+j]*(data[base+min(L-1,i+j+1)]-anchor));
          float v=i+j+1<L?coeff[4+j]*out2[base+i+j+1]:coeff[16+j]*(data[base+L-1]-anchor);
          back=compensated(back,v);}
        out2[base+i]=feed.x-back.x;}
      for(int i=0;i<L;i++)out0[base+i]=out1[base+i]+out2[base+i]+anchor;''',
      {'data':lines,'coeff':ctx.geometry(coeff),'meta':ctx.meta(length)},[lines.shape]*3,count)
    return outputs[0]


def convolve(ctx,lines,kernel):
    count,length=lines.shape
    return ctx.call('convolve', '''int L=int(meta[0]),K=int(meta[1]);
       unsigned int base=(i/L)*L;int p=i%L;float2 total=make_float2(0.0f,0.0f);
       for(int k=0;k<K;k++) total=compensated(total,coeff[k]*data[base+min(max(p+k-K/2,0),L-1)]);
       out0[i]=total.x;''',{'data':lines,'coeff':ctx.geometry(kernel),'meta':ctx.meta(length,len(kernel))},[lines.shape],lines.size)


def convolve_field(ctx,values,kernel,dimension,original=None,weight=None):
    """Convolve native (component,z,y,x) storage without axis-copy buffers.

    Adjacent GPU threads write adjacent voxels for every axis. On the last
    axis, the optional original field folds in the existing boundary/mix pass.
    """
    _,nz,ny,nx=values.shape
    stride=(1,nx,nx*ny)[dimension]
    length=(nx,ny,nz)[dimension]
    inputs={'data':values,'coeff':ctx.geometry(kernel),
            'meta':ctx.meta(length,len(kernel),stride,nx,ny,nz)}
    source='''
      int L=int(meta[0]),K=int(meta[1]),S=int(meta[2]);
      int p=int(i/static_cast<unsigned int>(S))%L;unsigned int base=i-static_cast<unsigned int>(p*S);float2 total=make_float2(0.0f,0.0f);
      for(int k=0;k<K;k++){
        unsigned int j=base+static_cast<unsigned int>(min(max(p+k-K/2,0),L-1)*S);
        total=compensated(total,coeff[k]*data[j]);
      }
    '''
    if original is None:
        source+='out0[i]=total.x;'
        name='convolve_field_axis'
    else:
        inputs.update(original=original,weight=ctx.geometry([weight]))
        source+='''int3 n=make_int3(meta[3],meta[4],meta[5]);
          int3 p3=xyz(i%(meta[3]*meta[4]*meta[5]),n);
          out0[i]=edge(p3,n)?0.0f:total.x*weight[0]+original[i]*(1.0f-weight[0]);'''
        name='convolve_field_axis_mix'
    return ctx.call(name,source,inputs,[values.shape],values.size)


def mix_boundary(ctx,smoothed,original,weight):
    _,nz,ny,nx=original.shape
    return ctx.call('mix_boundary', '''int3 n=make_int3(meta[0],meta[1],meta[2]);
       int3 p=xyz(i%(meta[0]*meta[1]*meta[2]),n);
       out0[i]=edge(p,n)?0.0f:smoothed[i]*weight[0]+original[i]*(1.0f-weight[0]);''',
       {'smoothed':smoothed,'original':original,'weight':ctx.geometry([weight]),'meta':ctx.meta(nx,ny,nz)},[original.shape],original.size)


def smooth_image(ctx,image,sigma,units='vox',*,method='sections'):
    """Filter image axes on the device in the established z/x/y order."""
    from fast_ants.backends.cpu.fp32.gaussian_sections import coefficients32
    from fast_ants.backends.cuda.gaussian_sections import section_axis32
    from fast_ants.pipeline.gaussian import recursive_gaussian_coefficients
    sigmas=np.full(3,float(sigma)) if np.ndim(sigma)==0 else np.asarray(sigma,np.float64)
    if sigmas.shape!=(3,) or not np.isfinite(sigmas).all() or np.any(sigmas<0):raise ValueError('Invalid Gaussian sigmas')
    if units=='mm':sigmas=sigmas/image.grid.spacing_xyz
    elif units!='vox':raise ValueError('Sigma units must be vox or mm')
    if method not in ('sections','recursive'):raise ValueError('Unknown CUDA Gaussian method')
    values=ctx.require(image.data)
    for dimension in (2,0,1):
        if sigmas[dimension]==0:continue
        axis=2-dimension
        if values.shape[axis]<4:raise ValueError('Gaussian filtering requires four pixels')
        if method=='sections':
            values=section_axis32(ctx,values,coefficients32(float(sigmas[dimension])),axis)
            continue
        with ctx.device,ctx.stream:
            moved=cp.moveaxis(values,axis,-1);shape=moved.shape
            lines=cp.ascontiguousarray(moved).reshape(-1,shape[-1])
        result=recursive(ctx,lines,recursive_gaussian_coefficients(float(sigmas[dimension])))
        with ctx.device,ctx.stream:
            values=cp.ascontiguousarray(cp.moveaxis(result.reshape(shape),-1,axis))
    return CUDAImage(values,image.grid,image.metadata)
