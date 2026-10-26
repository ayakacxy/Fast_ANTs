"""Explicit FP32 physical-point transform components on the Metal device.

Matrices are supplied in point-application order, not ITK queue-storage order.
Do not collapse their products: every component has a float point assignment
boundary. This module performs no image-grid conversion or interpolation.
"""

import numpy as np

from fast_ants.backends.metal.runtime import mx


_MAP_COMPONENTS_SOURCE = '''
  uint i=thread_position_in_grid.x;
  float3 p=float3(points[3*i],points[3*i+1],points[3*i+2]);
  for(uint k=0;k<meta[0];k++){
    float3 q;
    for(uint c=0;c<3;c++){
      float sum=0.0f;
      for(uint j=0;j<3;j++)sum=metal::fma(matrices[k*12+c*4+j],p[j],sum);
      q[c]=sum+matrices[k*12+c*4+3];
    }
    p=q;
  }
  for(uint c=0;c<3;c++)out0[3*i+c]=p[c];
'''


def map_world_components32(ctx, points32, matrices32):
    """Map device ``(N,3)`` points through ``(K,3,4)`` affine components.

    Points must already be an FP32 MLX array at the physical-point rounding
    boundary. Matrices must be an FP32 MLX or NumPy array; a NumPy table is
    explicitly uploaded and recorded as ``affine_components``. Neither input
    is silently downcast, and no points are downloaded or mapped on the CPU.

    Each row takes three native FP32 FMAs in column order, then separately
    adds the stored offset. All rows read the previous component's point.
    Nonempty outputs are independent device results. With K=0 the result is
    an exact device copy, including signed zero. With N=0 the empty input is
    returned without uploading matrices or dispatching a kernel. Shape/dtype
    checks still apply in both cases. Finite inputs are the caller's contract;
    this operator does not introduce a device synchronization to inspect them.
    """
    if not isinstance(points32, mx.array) or points32.dtype != mx.float32:
        raise TypeError('Physical points must be an FP32 MLX device array')
    if points32.ndim != 2 or points32.shape[1] != 3:
        raise ValueError('Physical points must have shape (N,3)')
    host_matrices = isinstance(matrices32, np.ndarray)
    if host_matrices:
        if matrices32.dtype != np.dtype('float32'):
            raise TypeError('Affine components must use FP32 storage')
    elif not isinstance(matrices32, mx.array) or matrices32.dtype != mx.float32:
        raise TypeError('Affine components must be an FP32 NumPy or MLX array')
    if matrices32.ndim != 3 or matrices32.shape[1:] != (3, 4):
        raise ValueError('Affine components must have shape (K,3,4)')

    count, components = points32.shape[0], matrices32.shape[0]
    if count == 0:
        return points32
    if components == 0:
        return ctx.call('copy_world_points32', '''uint i=thread_position_in_grid.x;
          for(uint c=0;c<3;c++)out0[3*i+c]=points[3*i+c];''',
          {'points': points32}, [points32.shape], count)
    matrices = ctx.upload(matrices32, 'affine_components') if host_matrices else matrices32
    return ctx.call('map_world_components32', _MAP_COMPONENTS_SOURCE,
                    {'points': points32, 'matrices': matrices, 'meta': ctx.meta(components)},
                    [points32.shape], count)
