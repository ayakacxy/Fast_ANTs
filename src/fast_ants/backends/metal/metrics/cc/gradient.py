"""Paired CC candidate with gradients computed inside active updates.

Window statistics and scalar reduction retain the existing Metal paired-CC
formulas. The only execution change is to avoid two complete gradient images:
centres with degenerate variance have zero updates and need no gradient work.
This module remains independently selectable until whole-pipeline measurement.
"""

import numpy as np

from fast_ants.backends.metal import dense
from fast_ants.backends.metal.runtime import mx


_SOURCE = dense._CC_FLOAT1_PREAMBLE + '''uint i=thread_position_in_grid.x;
  int3 n=int3(meta[0],meta[1],meta[2]),p=xyz(i,n);int r=int(meta[3]);
  uint N=meta[0]*meta[1]*meta[2];
  for(int c=0;c<3;c++){out0[c*N+i]=0.0f;out1[c*N+i]=0.0f;}out2[i]=0;
  if(!valid[i])return;
  float f0=fixed[i],m0=moving[i];float2 sf=float2(0),sm=float2(0);uint count=0;
  for(int z=metal::max(0,p.z-r);z<metal::min(n.z,p.z+r+1);z++)
   for(int y=metal::max(0,p.y-r);y<metal::min(n.y,p.y+r+1);y++)
    for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
     uint j=idx(x,y,z,n);if(valid[j]){
      sf=compensated(sf,fixed[j]-f0);sm=compensated(sm,moving[j]-m0);count++;}}
  float af=sf.x/float(count),am=sm.x/float(count);
  float2 ff=float2(0),mm=float2(0),fm=float2(0);
  for(int z=metal::max(0,p.z-r);z<metal::min(n.z,p.z+r+1);z++)
   for(int y=metal::max(0,p.y-r);y<metal::min(n.y,p.y+r+1);y++)
    for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
     uint j=idx(x,y,z,n);if(valid[j]){
      float f=(fixed[j]-f0)-af,m=(moving[j]-m0)-am;
      ff=compensated(ff,f*f);mm=compensated(mm,m*m);fm=compensated(fm,f*m);}}
  float product=ff.x*mm.x,eps=cc_epsilon;
  float value=metal::abs(product)>eps?fm.x*fm.x/product:1.0f;
  if(ff.x>eps&&mm.x>eps){
    float forward_weight=2.0f*fm.x/product*(-af+fm.x/mm.x*am);
    float reverse_weight=2.0f*fm.x/product*(-am+fm.x/ff.x*af);
    float3 point=float3(p);
    for(int c=0;c<3;c++){
      float3 d=float3(shift[c],shift[3+c],shift[6+c]);
      float3 a=point+d,b=point-d;
      float fixed_gradient=0.0f,moving_gradient=0.0f;
      if(inside(a,n)&&inside(b,n)){
        fixed_gradient=(sample1(fixed,a,n,f0)-sample1(fixed,b,n,f0))/spacing[c];
        moving_gradient=(sample1(moving,a,n,m0)-sample1(moving,b,n,m0))/spacing[c];
      }
      out0[c*N+i]=forward_weight*moving_gradient;
      out1[c*N+i]=reverse_weight*fixed_gradient;
    }
  }
  out2[i]=-value;'''


def cc_pair_gradient(ctx, fixed, moving, valid, radius, shift, spacing):
    """Return forward/reverse updates, shared CC value and valid-centre count.

    Device scalar images and Boolean validity occupy one z/y/x grid. ``shift``
    is the small host half-spacing-to-index matrix, and ``spacing`` is its LPS
    spacing vector. The stencil uses the same anchored ``sample1`` arithmetic
    as :func:`dense.gradient`, including oblique support and edge clamping.
    No full gradient volume is allocated or downloaded.
    """
    if fixed.dtype != mx.float32 or moving.dtype != mx.float32:
        raise ValueError('Paired CC needs FP32 device images')
    if fixed.ndim != 3 or fixed.shape != moving.shape:
        raise ValueError('Paired CC images must occupy one scalar grid')
    if valid.dtype != mx.bool_ or valid.shape != fixed.shape:
        raise ValueError('Paired CC validity must be Boolean on the image grid')
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('CC radius must be a nonnegative integer')
    shift = np.ascontiguousarray(shift, dtype=np.float32)
    spacing = np.ascontiguousarray(spacing, dtype=np.float32)
    if (shift.shape != (3, 3) or spacing.shape != (3,) or
            not np.isfinite(shift).all() or not np.isfinite(spacing).all() or
            np.any(spacing <= 0)):
        raise ValueError('CC gradient geometry must be finite with positive spacing')
    nz, ny, nx = fixed.shape
    forward, reverse, score = ctx.call(
        'cc_pair_gradient', _SOURCE,
        {'fixed': fixed, 'moving': moving, 'valid': valid,
         'shift': ctx.upload(shift), 'spacing': ctx.upload(spacing),
         'meta': ctx.meta(nx, ny, nz, radius)},
        [(3, nz, ny, nx), (3, nz, ny, nx), fixed.shape], fixed.size)
    value, count = dense._cc_score(ctx, score, valid)
    return forward, reverse, value, count
