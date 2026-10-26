"""Independent inverse fields advanced through one bounded control read.

Each active side owns its residual, norm and next-field arrays. Composition,
norm generation and tile reduction share one kernel; both sides are queued
before a small common control read. Finishing one side never advances it again.
"""

import numpy as np

from fast_ants.backends.metal import dense
from fast_ants.core.results import InversionResult


TILE = 256

_RESIDUAL_TILES = '''
  uint i=thread_position_in_grid.x,t=thread_index_in_threadgroup;
  uint count=meta[0]*meta[1]*meta[2];
  threadgroup float sums[256],maxima[256];
  float norm=0.0f;
  if(i<count){
''' + dense._COMPOSE_COORDINATES.replace('uint i=thread_position_in_grid.x;', '', 1) + '''
    float squared=0.0f;
    for(int c=0;c<3;c++){
      float residual=inner[c*N+i]+(valid?sample1(outer+c*O,q,src):0.0f);
      out0[c*N+i]=residual;
      float scaled=residual/spacing[c];squared+=scaled*scaled;
    }
    norm=metal::sqrt(squared);out1[i]=norm;
  }
  sums[t]=norm;maxima[t]=norm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=128u;stride>0u;stride/=2u){
    if(t<stride){sums[t]+=sums[t+stride];maxima[t]=metal::max(maxima[t],maxima[t+stride]);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(t==0u){out2[2u*(i/256u)]=sums[0];out2[2u*(i/256u)+1u]=maxima[0];}
'''


def residual_tiles(ctx, field, current):
    """Device residual/norm arrays and positive-sum/max tile statistics."""
    inverse=field.grid.lps_to_index[:3,:3]
    linear=inverse@current.grid.index_to_lps[:3,:3]
    offset=inverse@(current.grid.origin-field.grid.origin)
    _,nz,ny,nx=current.data.shape
    _,oz,oy,ox=field.data.shape
    count=nx*ny*nz
    blocks=(count+TILE-1)//TILE
    return ctx.call('inverse_residual_norm_tiles32',_RESIDUAL_TILES,
        {'outer':field.data,'inner':current.data,
         'linear':ctx.upload(linear),'offset':ctx.upload(offset),
         'inverse':ctx.upload(inverse),'spacing':ctx.upload(field.grid.spacing_xyz),
         'meta':ctx.meta(nx,ny,nz,ox,oy,oz)},
        [current.data.shape,current.data.shape[1:],(blocks,2)],blocks*TILE)


def _read_controls(ctx, pending, reason='inverse_pair_norms'):
    """Merge one or two queued tile arrays, then perform one 2/4-scalar read.

    Positive partial sums use lane-local compensation and a fixed tree. Near
    an inverse mean threshold the caller retains the existing serial guard.
    """
    if not 1<=len(pending)<=2:
        raise ValueError('Paired inverse controls need one or two active sides')
    inputs={};metadata=[];branches=[]
    for index,(_,_,norms,partial) in enumerate(pending):
        inputs['partial'+str(index)]=partial
        metadata.extend((norms.size,partial.shape[0]))
        branches.append(('if' if index==0 else 'else if')+'(side=='+str(index)+'''u){
          count=meta['''+str(2*index)+'''];
          for(uint b=t;b<meta['''+str(2*index+1)+'''];b+=256u){
            total=compensated(total,partial'''+str(index)+'''[2u*b]);
            maximum=metal::max(maximum,partial'''+str(index)+'''[2u*b+1u]);
          }
        }''')
    inputs['meta']=ctx.meta(*metadata)
    source='''
      uint t=thread_index_in_threadgroup,side=thread_position_in_grid.x/256u,count=0u;
      float2 total=float2(0);float maximum=0.0f;
      threadgroup float sums[256],maxima[256];
      '''+'\n'.join(branches)+'''
      sums[t]=total.x;maxima[t]=maximum;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128u;stride>0u;stride/=2u){
        if(t<stride){sums[t]+=sums[t+stride];maxima[t]=metal::max(maxima[t],maxima[t+stride]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0u){out0[2u*side]=sums[0]/float(count);out0[2u*side+1u]=maxima[0];}
    '''
    result=ctx.call('inverse_pair_controls32_'+str(len(pending)),source,inputs,
                    [(len(pending),2)],len(pending)*TILE)
    return ctx.control(result,reason,4)


def invert_pair(ops, fields, initials=(None,None), *, max_iterations=20,
                mean_tolerance=.001, max_tolerance=.1, enforce_boundary=True,
                _evaluate_final_residual=False):
    """Return two independently stopped inversions from shared control rounds.

    ``parallel_norms`` enables this fused implementation. Explicit serial-norm
    configurations retain the original independent operator, including its
    reference traversal. The ordinary public ``ops.invert`` is unchanged.
    """
    from fast_ants.backends.metal.types import MetalField

    if len(fields)!=2 or len(initials)!=2:
        raise ValueError('Paired inversion requires exactly two fields and initial values')
    if not isinstance(max_iterations,int) or max_iterations<0:
        raise ValueError('Invalid inverse iteration count')
    if (not np.isfinite([mean_tolerance,max_tolerance]).all()
            or min(mean_tolerance,max_tolerance)<0):
        raise ValueError('Invalid inverse tolerances')
    for field,initial in zip(fields,initials):
        if initial is not None and not initial.grid.same_geometry(field.grid):
            raise ValueError('Initial inverse grid mismatch')
    if 'parallel_norms' not in ops.context.optimizations:
        return tuple(ops.invert(field,initial,max_iterations=max_iterations,
            mean_tolerance=mean_tolerance,max_tolerance=max_tolerance,
            enforce_boundary=enforce_boundary,
            _evaluate_final_residual=_evaluate_final_residual)
            for field,initial in zip(fields,initials))

    ctx=ops.context
    current=[ops.zero(field.grid) if initial is None else initial
             for field,initial in zip(fields,initials)]
    histories=[[],[]];reasons=['max_iterations','max_iterations']
    active=[max_iterations>0,max_iterations>0]
    mean_limit,max_limit=float(mean_tolerance),float(max_tolerance)
    tiny=np.finfo(np.float32).tiny;eps=np.finfo(np.float32).eps
    while any(active):
        pending=[]
        for side in range(2):
            if active[side]:
                residual,norms,partial=residual_tiles(ctx,fields[side],current[side])
                pending.append((side,residual,norms,partial))
        controls=_read_controls(ctx,pending)
        for index,(side,residual,norms,_) in enumerate(pending):
            mean,maximum=(float(value) for value in controls[index])
            # Match dense.reduce_norms' existing near-threshold safeguard.
            scale=max(mean,mean_limit,tiny)
            if abs(mean-mean_limit)<=32*eps*scale:
                guarded=ctx.control(dense._serial_norms(ctx,norms,'norm_reduce_serial_guard'),
                                    'inverse_norms_guard',2)
                mean,maximum=(float(value) for value in guarded)
            mean_control,max_control=mean,maximum
            guard_mean=bool(abs(mean-mean_limit)<=8*eps*max(mean,mean_limit,tiny))
            guard_max=bool(abs(maximum-max_limit)<=8*eps*max(maximum,max_limit,tiny))
            if guard_max:
                max_control=dense.maximum_control(ctx,residual,fields[side].grid.spacing_xyz)
            iteration=len(histories[side])
            epsilon=np.float32(.75 if iteration==0 else .5)
            data=dense.inverse_step(ctx,current[side].data,residual,norms,maximum,epsilon,enforce_boundary)
            current[side]=MetalField(data,fields[side].grid)
            histories[side].append({'iteration':iteration+1,
                'mean_scaled_error_before_update':mean,'max_scaled_error_before_update':maximum,
                'epsilon':float(epsilon),'mean_control_error':mean_control,'max_control_error':max_control,
                'distance_to_mean_threshold':abs(mean_control-mean_limit),
                'distance_to_max_threshold':abs(max_control-max_limit),
                'native_scalar_control_guard':{'mean':guard_mean,'maximum':guard_max}})
            if max_control<=max_limit or mean_control<=mean_limit:
                reasons[side]='maximum_error_threshold' if max_control<=max_limit else 'mean_error_threshold'
                active[side]=False
            elif len(histories[side])>=max_iterations:
                active[side]=False
    if _evaluate_final_residual:
        pending=[]
        for side in range(2):
            residual,norms,partial=residual_tiles(ctx,fields[side],current[side])
            pending.append((side,residual,norms,partial))
        values=_read_controls(ctx,pending,'inverse_pair_final_norms')
        final=[{'mean_spacing_scaled_norm':float(values[side,0]),
                'max_spacing_scaled_norm':float(values[side,1])} for side in range(2)]
    else:
        final=[{'evaluated':False},{'evaluated':False}]
    return tuple(InversionResult(current[side],histories[side],reasons[side],final[side])
                 for side in range(2))
