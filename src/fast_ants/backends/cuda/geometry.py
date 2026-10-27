"""Explicit FP32 physical-point transform components on the CUDA device."""

import numpy as np

from fast_ants.backends.cuda.runtime import cp


def map_world_components32(ctx,points32,matrices32):
    """Apply affine components in order, retaining each FP32 point boundary."""
    if not isinstance(points32,cp.ndarray) or points32.dtype!=cp.float32:
        raise TypeError('Physical points must be an FP32 CUDA device array')
    if points32.ndim!=2 or points32.shape[1]!=3:
        raise ValueError('Physical points must have shape (N,3)')
    if not isinstance(matrices32,(np.ndarray,cp.ndarray)) or matrices32.dtype!=np.float32:
        raise TypeError('Affine components must be an FP32 NumPy or CUDA array')
    if matrices32.ndim!=3 or matrices32.shape[1:]!=(3,4):
        raise ValueError('Affine components must have shape (K,3,4)')
    count,components=points32.shape[0],matrices32.shape[0]
    if count==0:return points32
    if components==0:
        return ctx.call('copy_world_points32','for(int c=0;c<3;c++)out0[3*i+c]=points[3*i+c];',
                        {'points':points32},[points32.shape],count)
    matrices=(ctx.upload(matrices32,'affine_components') if isinstance(matrices32,np.ndarray)
              else matrices32)
    return ctx.call('map_world_components32','''float p[3]={points[3*i],points[3*i+1],points[3*i+2]};
      for(unsigned int k=0;k<meta[0];k++){
        float q[3];for(int c=0;c<3;c++){
          float sum=0;for(int j=0;j<3;j++)sum=__fmaf_rn(matrices[k*12+c*4+j],p[j],sum);
          q[c]=sum+matrices[k*12+c*4+3];}
        for(int c=0;c<3;c++)p[c]=q[c];}
      for(int c=0;c<3;c++)out0[3*i+c]=p[c];''',
      {'points':points32,'matrices':matrices,'meta':ctx.meta(components)},[points32.shape],count)
