"""Queue inverse updates before reading the independently stopped controls.

MLX async_eval accepts array trees and asynchronously submits their graphs
(local MLX core/__init__.pyi). Each update depends on the device maximum, so
both updates can be submitted before the CPU reads the common controls and
builds history. Actual overlap and speedup depend on MLX/runtime scheduling.
No subsequent inverse iteration is evaluated before its side's stop decision.
"""

import numpy as np

from fast_ants.backends.metal import dense
from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal.fields.inverse import TILE
from fast_ants.backends.metal.fields.inverse import residual_tiles
from fast_ants.backends.metal.fields.inverse import invert_pair as _synchronous_pair
from fast_ants.core.results import InversionResult


def _enqueue_controls(ctx, pending):
    """Same compensated tile merge as paired_inverse32, without a host read."""
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
    return ctx.call('inverse_overlap_controls32_'+str(len(pending)),source,inputs,
                    [(len(pending),2)],len(pending)*TILE)


def _device_inverse_step(ctx,current,residual,norms,controls,index,epsilon,enforce_boundary):
    """The original update expression, consuming its maximum on the GPU."""
    _,nz,ny,nx=current.shape
    return ctx.call('inverse_step_device_max32','''
      uint i=thread_position_in_grid.x,N=meta[0]*meta[1]*meta[2];
      int3 n=int3(meta[0],meta[1],meta[2]),p=xyz(i,n);
      float maximum=controls[2u*meta[4]+1u],eps=params[0];
      float factor=norms[i]>eps*maximum?eps*maximum/norms[i]:1.0f;
      for(int c=0;c<3;c++){float v=current[c*N+i]-residual[c*N+i]*factor*eps;
       out0[c*N+i]=meta[3]&&edge(p,n)?0.0f:v;}''',
      {'current':current,'residual':residual,'norms':norms,'controls':controls,
       'params':ctx.upload([epsilon]),'meta':ctx.meta(nx,ny,nz,int(enforce_boundary),index)},
      [current.shape],nx*ny*nz)


def _submit(ctx,*arrays):
    ctx.submit(*arrays,reason='inverse_update')
    ctx.control_guards['inverse_overlap_async_submissions']+=1


def invert_pair(ops, fields, initials=(None,None), *, max_iterations=20,
                mean_tolerance=.001, max_tolerance=.1, enforce_boundary=True,
                _evaluate_final_residual=False):
    """Submit each round's updates before its one shared control read.

    Update-then-stop semantics, independent iteration counts, threshold guards
    and optional post-update diagnostics match paired_inverse32. A serial mean
    guard can replace its side's prequeued output if it changes the update
    maximum. The native maximum guard changes only the stop control, as before.
    No host thread concurrently manipulates a context, and no global GPU
    synchronization is added. Serial configurations or missing async_eval
    retain the original paired implementation.
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
    ctx=ops.context
    if ('parallel_norms' not in ctx.optimizations or
            not callable(getattr(mx,'async_eval',None))):
        return _synchronous_pair(ops,fields,initials,max_iterations=max_iterations,
            mean_tolerance=mean_tolerance,max_tolerance=max_tolerance,
            enforce_boundary=enforce_boundary,
            _evaluate_final_residual=_evaluate_final_residual)

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
        device_controls=_enqueue_controls(ctx,pending)
        queued=[]
        for index,(side,residual,norms,_) in enumerate(pending):
            epsilon=np.float32(.75 if not histories[side] else .5)
            queued.append(_device_inverse_step(ctx,current[side].data,residual,norms,
                device_controls,index,epsilon,enforce_boundary))
        # Each next field depends on this round's controls. Submitting these
        # graphs first avoids leaving the updates out of a controls-only eval.
        _submit(ctx,*queued,device_controls)
        controls=ctx.control(device_controls,'inverse_pair_norms',4)
        for index,(side,residual,norms,_) in enumerate(pending):
            mean,maximum=(float(value) for value in controls[index])
            queued_maximum=maximum
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
            data=queued[index]
            if maximum!=queued_maximum:
                # Keep the original guarded update maximum if the mean guard
                # changes it. A prequeued array has no in-place side effects.
                data=dense.inverse_step(ctx,current[side].data,residual,norms,
                                        maximum,epsilon,enforce_boundary)
                _submit(ctx,data)
                ctx.control_guards['inverse_overlap_guard_recomputations']+=1
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
        values=ctx.control(_enqueue_controls(ctx,pending),'inverse_pair_final_norms',4)
        final=[{'mean_spacing_scaled_norm':float(values[side,0]),
                'max_spacing_scaled_norm':float(values[side,1])} for side in range(2)]
    else:
        final=[{'evaluated':False},{'evaluated':False}]
    return tuple(InversionResult(current[side],histories[side],reasons[side],final[side])
                 for side in range(2))
