"""Paired update controls and scale-before-interpolation field composition.

The maximum-vector reductions are the existing deterministic GPU operators.
Only their two three-component controls share a host read. The bounded host
FP64 norm and FP32 scale assignments match ``MetalOps.scale``. Composition
multiplies each sampled vertex before the original ``sample1`` interpolation,
so it does not allocate or scan a separate scaled update field.
"""

import numpy as np

from fast_ants.backends.metal import dense
from fast_ants.backends.metal import reductions


# Keep the same uncontracted lerps as common.metal sample1. The header disables
# implicit FMA/reassociation, and every vertex multiplication stores a float
# before either differences or interpolation weights are applied. In particular,
# sample1(unscaled)*factor would cross the old FP32 assignment boundaries.
_SCALED_COMPOSE = dense._COMPOSE_COORDINATES + '''
  for(int c=0;c<3;c++){
    float sampled=0.0f;
    if(valid){
      float3 point=metal::clamp(q,float3(0),float3(src-1));
      int3 lo=int3(metal::floor(point)),hi=metal::min(lo+1,src-1);
      float3 d=point-float3(lo);uint base=c*O;
      float a000=outer[base+idx(lo.x,lo.y,lo.z,src)]*factor[0];
      float a100=outer[base+idx(hi.x,lo.y,lo.z,src)]*factor[0];
      float a010=outer[base+idx(lo.x,hi.y,lo.z,src)]*factor[0];
      float a110=outer[base+idx(hi.x,hi.y,lo.z,src)]*factor[0];
      float a001=outer[base+idx(lo.x,lo.y,hi.z,src)]*factor[0];
      float a101=outer[base+idx(hi.x,lo.y,hi.z,src)]*factor[0];
      float a011=outer[base+idx(lo.x,hi.y,hi.z,src)]*factor[0];
      float a111=outer[base+idx(hi.x,hi.y,hi.z,src)]*factor[0];
      float aa=(a000-0.0f)+(a100-a000)*d.x;
      float bb=(a010-0.0f)+(a110-a010)*d.x;
      float cc=(a001-0.0f)+(a101-a001)*d.x;
      float dd=(a011-0.0f)+(a111-a011)*d.x;
      float low=aa+(bb-aa)*d.y,high=cc+(dd-cc)*d.y;
      sampled=low+(high-low)*d.z;
    }
    out0[c*N+i]=inner[c*N+i]+sampled;
  }'''


def _compose_scaled(ctx, outer, inner, factor):
    inverse=outer.grid.lps_to_index[:3,:3]
    linear=inverse@inner.grid.index_to_lps[:3,:3]
    offset=inverse@(inner.grid.origin-outer.grid.origin)
    _,nz,ny,nx=inner.data.shape
    _,oz,oy,ox=outer.data.shape
    return ctx.call('scaled_compose32',_SCALED_COMPOSE,
        {'outer':outer.data,'inner':inner.data,
         'linear':ctx.upload(linear),'offset':ctx.upload(offset),
         'inverse':ctx.upload(inverse),'factor':ctx.upload([factor]),
         'meta':ctx.meta(nx,ny,nz,ox,oy,oz)},
        [inner.data.shape],nx*ny*nz)


def scaled_compose_pair(ops, fields, inners, learning_rate=.1):
    """Return ``(first, second, first_scale, second_scale)`` without scaled fields.

    Each output lives on its own inner field's grid; source/output geometry
    need not match. The public scale detail dictionaries retain their existing
    meanings. Explicit configurations without ``parallel_max`` use the old
    independent scale/compose route, including serial selection and tie rules.
    Inputs are never mutated. Callers needing captured scaled intermediate
    fields should continue to use that old route.
    """
    from fast_ants.backends.metal.types import MetalField

    if len(fields)!=2 or len(inners)!=2:
        raise ValueError('Paired scaled composition needs two updates and two inner fields')
    if not all(isinstance(value,MetalField) for value in (*fields,*inners)):
        raise ValueError('Paired scaled composition requires Metal fields')
    if not np.isfinite(learning_rate) or learning_rate<0:
        raise ValueError('Invalid learning rate')
    ctx=ops.context
    if 'parallel_max' not in ctx.optimizations:
        first,first_scale=ops.scale(fields[0],learning_rate)
        second,second_scale=ops.scale(fields[1],learning_rate)
        return (ops.compose(first,inners[0]),ops.compose(second,inners[1]),
                first_scale,second_scale)

    candidates=[reductions.maximum_vector(ctx,field.data,field.grid.spacing_xyz)
                for field in fields]
    packed=ctx.call('scaled_compose_maximum_pair32',
        'uint i=thread_position_in_grid.x;out0[i]=i<3u?first[i]:second[i-3u];',
        {'first':candidates[0],'second':candidates[1]},[(6,)],6)
    vectors=ctx.control(packed,'scaled_compose_maxima',6).reshape(2,3)
    target=float(np.float32(learning_rate))
    details=[]
    outputs=[]
    for field,inner,vector in zip(fields,inners,vectors):
        values=vector.astype(np.float64)/np.asarray(field.grid.spacing_xyz,np.float64)
        maximum=float(np.sqrt(values@values))
        factor=np.float32(target/maximum) if maximum>0 else np.float32(target)
        details.append({'maximum_spacing_scaled_norm':maximum,'scale':float(factor),
                        'effective_target':target})
        outputs.append(MetalField(_compose_scaled(ctx,field,inner,factor),inner.grid))
    return outputs[0],outputs[1],details[0],details[1]
