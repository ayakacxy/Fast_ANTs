"""Opt-in separable, centered ordinary-FP32 neighborhood CC statistics.

The x pass computes local means and centered second moments in two short
passes. The y and z passes merge disjoint groups with Chan's identities.
Means remain offsets from each original image value, avoiding subtraction
of raw squared intensity sums. These anchors are intensity origins, not
software-expanded precision. Each intermediate is five FP32 channels plus
one uint32 count per voxel; the last pass directly writes both updates.

This changes the reduction grouping. It preserves the CC objective and
cutoffs, rather than claiming identical last bits or optimizer trajectories.
"""

from fast_ants.backends.metal import dense
from fast_ants.backends.metal.runtime import mx


MAX_RADIUS = 4

_SETUP = '''
  uint i=thread_position_in_grid.x,N=meta[0]*meta[1]*meta[2];
  int3 n=int3(meta[0],meta[1],meta[2]),p=xyz(i,n);int r=int(meta[3]);
'''

_X_STATS = _SETUP + '''
  // Invalid centers still summarize valid neighbors for the next axes.
  float f0=fixed[i],m0=moving[i];
  float2 sf=float2(0),sm=float2(0);uint count=0;
  for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
    uint j=idx(x,p.y,p.z,n);
    if(valid[j]){
      sf=compensated(sf,fixed[j]-f0);
      sm=compensated(sm,moving[j]-m0);count++;
    }
  }
  float af=count?sf.x/float(count):0.0f,am=count?sm.x/float(count):0.0f;
  float2 ff=float2(0),mm=float2(0),fm=float2(0);
  for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
    uint j=idx(x,p.y,p.z,n);
    if(valid[j]){
      float f=(fixed[j]-f0)-af,m=(moving[j]-m0)-am;
      ff=compensated(ff,f*f);mm=compensated(mm,m*m);fm=compensated(fm,f*m);
    }
  }
  out0[i]=af;out0[N+i]=am;
  out0[2u*N+i]=ff.x;out0[3u*N+i]=mm.x;out0[4u*N+i]=fm.x;
  out1[i]=count;
'''

_MERGE = '''
  float f0=fixed[i],m0=moving[i];
  float2 af=float2(0),am=float2(0),ff=float2(0),mm=float2(0),fm=float2(0);
  uint count=0;
  for(int s=metal::max(0,p.$AXIS-r);s<metal::min(n.$AXIS,p.$AXIS+r+1);s++){
    int3 q=p;q.$AXIS=s;uint j=idx(q.x,q.y,q.z,n),other_count=counts[j];
    if(!other_count)continue;
    // Move the neighbor's local mean into this center's intensity origin.
    float other_f=(fixed[j]-f0)+stats[j];
    float other_m=(moving[j]-m0)+stats[N+j];
    if(!count){
      af=float2(other_f,0.0f);am=float2(other_m,0.0f);
      ff=float2(stats[2u*N+j],0.0f);mm=float2(stats[3u*N+j],0.0f);
      fm=float2(stats[4u*N+j],0.0f);count=other_count;
    }else{
      uint merged_count=count+other_count;
      float ratio=float(other_count)/float(merged_count);
      float weight=float(count)*ratio,df=other_f-af.x,dm=other_m-am.x;
      ff=compensated(ff,stats[2u*N+j]);ff=compensated(ff,df*df*weight);
      mm=compensated(mm,stats[3u*N+j]);mm=compensated(mm,dm*dm*weight);
      fm=compensated(fm,stats[4u*N+j]);fm=compensated(fm,df*dm*weight);
      af=compensated(af,df*ratio);am=compensated(am,dm*ratio);
      count=merged_count;
    }
  }
'''

_Y_STATS = _SETUP + _MERGE.replace('$AXIS', 'y') + '''
  out0[i]=af.x;out0[N+i]=am.x;
  out0[2u*N+i]=ff.x;out0[3u*N+i]=mm.x;out0[4u*N+i]=fm.x;
  out1[i]=count;
'''

_Z_PAIR = _SETUP + '''
  for(int c=0;c<3;c++){out0[c*N+i]=0.0f;out1[c*N+i]=0.0f;}
  out2[i]=0.0f;if(!valid[i])return;
''' + _MERGE.replace('$AXIS', 'z') + '''
  // A valid center contributes to all three passes, so count is nonzero.
  float product=ff.x*mm.x,eps=cc_epsilon;
  float value=metal::abs(product)>eps?fm.x*fm.x/product:1.0f;
  if(ff.x>eps&&mm.x>eps){
    float forward_weight=2.0f*fm.x/product*(-af.x+fm.x/mm.x*am.x);
    float reverse_weight=2.0f*fm.x/product*(-am.x+fm.x/ff.x*af.x);
    for(int c=0;c<3;c++){
      out0[c*N+i]=forward_weight*moving_grad[c*N+i];
      out1[c*N+i]=reverse_weight*fixed_grad[c*N+i];
    }
  }
  out2[i]=-value;
'''


def cc_pair(ctx, fixed, moving, fixed_gradient, moving_gradient, valid, radius):
    """Return forward/reverse updates, mean CC score and valid-center count.

    Radius 0--4 uses separable window statistics. Larger windows retain the
    existing dense implementation. Inputs and local anchors use ordinary
    FP32; masks gate samples in the first pass and centers in the last pass.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('CC radius must be a nonnegative integer')
    if any(not isinstance(a, mx.array) or a.dtype != mx.float32 for a in (fixed, moving)):
        raise ValueError('Window CC requires FP32 device images')
    if fixed.ndim != 3 or min(fixed.shape) < 1 or moving.shape != fixed.shape:
        raise ValueError('Window CC requires matching nonempty 3D images')
    if fixed.size >= 2**32:
        raise ValueError('Window CC requires uint32-sized images')
    if any(not isinstance(a, mx.array) or a.dtype != mx.float32 or
           a.shape != (3, *fixed.shape) for a in (fixed_gradient, moving_gradient)):
        raise ValueError('Window CC requires matching FP32 gradients')
    if not isinstance(valid, mx.array) or valid.dtype != mx.bool_ or valid.shape != fixed.shape:
        raise ValueError('Window CC validity grid/type mismatch')
    if radius > MAX_RADIUS:
        return dense.cc_pair(ctx, fixed, moving, fixed_gradient, moving_gradient, valid, radius)
    if fixed.size > (2**32-1)//5:
        raise ValueError('Window CC statistics exceed uint32 channel indexing')

    nz, ny, nx = fixed.shape
    metadata = ctx.meta(nx, ny, nz, radius)
    images = {'fixed': fixed, 'moving': moving}
    shapes = [(5, nz, ny, nx), fixed.shape]
    dtypes = [mx.float32, mx.uint32]
    stats_x, count_x = ctx.call('cc_window_x32', _X_STATS,
        {**images, 'valid': valid, 'meta': metadata}, shapes, fixed.size, dtypes)
    stats_y, count_y = ctx.call('cc_window_y32', _Y_STATS,
        {**images, 'stats': stats_x, 'counts': count_x, 'meta': metadata},
        shapes, fixed.size, dtypes)
    # Keep dependency edges, but let MLX release consumed intermediate arrays
    # instead of pinning both pairs through the bounded score host control.
    del stats_x, count_x
    forward, reverse, score = ctx.call('cc_window_z_pair32', dense._CC_FLOAT1_PREAMBLE+_Z_PAIR,
        {**images, 'stats': stats_y, 'counts': count_y, 'valid': valid,
         'fixed_grad': fixed_gradient, 'moving_grad': moving_gradient, 'meta': metadata},
        [(3, nz, ny, nx), (3, nz, ny, nx), fixed.shape], fixed.size)
    del stats_y, count_y
    value, count = dense._cc_score(ctx, score, valid)
    return forward, reverse, value, count
