"""Opt-in shared-halo CC pair, retaining centered ordinary FP32 statistics.

A 16 x 4 x 4 output tile cooperatively caches its image/mask halo. At radius
two this consumes 15 KiB of threadgroup memory. This saves repeated global
image reads, while deliberately retaining both centered compensated passes
and their traversal order. Arithmetic and gradient storage are unchanged.
Larger radii use the existing dense implementation.
"""

from fast_ants.backends.metal import dense
from fast_ants.backends.metal.runtime import mx


_TILE = (16, 4, 4)
_THREADS = 256

_SOURCE = '''
  uint gid=thread_position_in_grid.x,t=thread_index_in_threadgroup;
  uint group=gid/256u,gx=meta[4],gy=meta[5];
  int3 origin=int3((group%gx)*16u,((group/gx)%gy)*4u,(group/(gx*gy))*4u);
  int3 n=int3(meta[0],meta[1],meta[2]);int r=int(meta[3]);
  threadgroup float fixed_tile[$HS];
  threadgroup float moving_tile[$HS];
  threadgroup uint valid_tile[$HS];
  for(uint h=t;h<$HS;h+=256u){
    int3 hp=int3(h%$HX,(h/$HX)%$HY,h/($HX*$HY));
    int3 q=origin+hp-int3(r);
    uint take=0;float f=0.0f,m=0.0f;
    if(all(q>=int3(0))&&all(q<n)){
      uint j=idx(q.x,q.y,q.z,n);take=uint(valid[j]);
      if(take){f=fixed[j];m=moving[j];}
    }
    fixed_tile[h]=f;moving_tile[h]=m;valid_tile[h]=take;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Even lanes outside the image participate in the halo/barrier above.
  int3 local=int3(t%16u,(t/16u)%4u,t/64u);
  int3 p=origin+local;
  if(any(p>=n))return;
  uint i=idx(p.x,p.y,p.z,n),N=meta[0]*meta[1]*meta[2];
  for(int c=0;c<3;c++){out0[c*N+i]=0.0f;out1[c*N+i]=0.0f;}
  out2[i]=0.0f;
  int3 center=local+int3(r);
  uint ci=(center.z*$HY+center.y)*$HX+center.x;
  if(!valid_tile[ci])return;
  float f0=fixed_tile[ci],m0=moving_tile[ci];
  float2 sf=float2(0),sm=float2(0);uint count=0;
  for(int z=center.z-r;z<=center.z+r;z++)
    for(int y=center.y-r;y<=center.y+r;y++)
      for(int x=center.x-r;x<=center.x+r;x++){
        uint j=(z*$HY+y)*$HX+x;
        if(valid_tile[j]){
          sf=compensated(sf,fixed_tile[j]-f0);
          sm=compensated(sm,moving_tile[j]-m0);count++;
        }
      }
  float af=sf.x/float(count),am=sm.x/float(count);
  float2 ff=float2(0),mm=float2(0),fm=float2(0);
  for(int z=center.z-r;z<=center.z+r;z++)
    for(int y=center.y-r;y<=center.y+r;y++)
      for(int x=center.x-r;x<=center.x+r;x++){
        uint j=(z*$HY+y)*$HX+x;
        if(valid_tile[j]){
          float f=(fixed_tile[j]-f0)-af,m=(moving_tile[j]-m0)-am;
          ff=compensated(ff,f*f);mm=compensated(mm,m*m);fm=compensated(fm,f*m);
        }
      }
  float product=ff.x*mm.x,eps=cc_epsilon;
  float value=metal::abs(product)>eps?fm.x*fm.x/product:1.0f;
  if(ff.x>eps&&mm.x>eps){
    float forward_weight=2.0f*fm.x/product*(-af+fm.x/mm.x*am);
    float reverse_weight=2.0f*fm.x/product*(-am+fm.x/ff.x*af);
    for(int c=0;c<3;c++){
      out0[c*N+i]=forward_weight*moving_grad[c*N+i];
      out1[c*N+i]=reverse_weight*fixed_grad[c*N+i];
    }
  }
  out2[i]=-value;
'''


def cc_pair(ctx, fixed, moving, fixed_gradient, moving_gradient, valid, radius):
    """Return forward/reverse updates, CC value and number of valid centers.

    Out-of-image halo entries are invalid and never enter an accumulator.
    Consequently clipped and masked windows retain the dense z/y/x sample
    order, including singleton axes, zero radius and the flat-window rule.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('CC radius must be a nonnegative integer')
    images = (fixed, moving)
    if any(not isinstance(a, mx.array) or a.dtype != mx.float32 for a in images):
        raise ValueError('Tiled CC requires FP32 device images')
    if fixed.ndim != 3 or min(fixed.shape) < 1 or fixed.shape != moving.shape:
        raise ValueError('Tiled CC requires matching nonempty 3D images')
    if fixed.size >= 2**32:
        raise ValueError('Tiled CC requires uint32-sized images')
    if any(not isinstance(a, mx.array) or a.dtype != mx.float32 or
           a.shape != (3, *fixed.shape) for a in (fixed_gradient, moving_gradient)):
        raise ValueError('Tiled CC requires matching FP32 gradients')
    if not isinstance(valid, mx.array) or valid.dtype != mx.bool_ or valid.shape != fixed.shape:
        raise ValueError('Tiled CC validity grid/type mismatch')
    if radius > 2:
        return dense.cc_pair(ctx, fixed, moving, fixed_gradient, moving_gradient, valid, radius)

    nz, ny, nx = fixed.shape
    gx, gy, gz = ((n+t-1)//t for n, t in zip((nx, ny, nz), _TILE))
    hx, hy, hz = (t+2*radius for t in _TILE)
    source = _SOURCE.replace('$HS', str(hx*hy*hz)).replace('$HX', str(hx)).replace('$HY', str(hy))
    forward, reverse, score = ctx.call(
        'cc_tiled_pair32_r'+str(radius), dense._CC_FLOAT1_PREAMBLE+source,
        {'fixed': fixed, 'moving': moving, 'fixed_grad': fixed_gradient,
         'moving_grad': moving_gradient, 'valid': valid,
         'meta': ctx.meta(nx, ny, nz, radius, gx, gy)},
        [(3, nz, ny, nx), (3, nz, ny, nx), fixed.shape], gx*gy*gz*_THREADS)
    value, count = dense._cc_score(ctx, score, valid)
    return forward, reverse, value, count
