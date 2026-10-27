"""Queue independent inverse updates before one bounded control read.

Each side retains the CPU row Kahan sums and fixed pairwise merge, owns its
output arrays, and advances only after its own stopping decision. Device
maximum controls avoid uploading the same scalar immediately after reading it.
"""

import numpy as np

from fast_ants.backends.cuda.fields import kernels
from fast_ants.backends.cuda import reductions
from fast_ants.backends.cuda.types import CUDAField
from fast_ants.core.results import InversionResult


def _controls(ctx,pending):
    if len(pending)==1:return reductions.row_norms(ctx,pending[0][2]).reshape(1,2)
    return reductions.row_norms_pair(ctx,pending[0][2],pending[1][2])


def invert_pair(ops,fields,initials=(None,None),*,max_iterations=20,mean_tolerance=.001,
                max_tolerance=.1,enforce_boundary=True,_evaluate_final_residual=True,
                _reference=False):
    """Preserve independent histories, guards and update-then-stop semantics.

    Both updates are queued from their device maxima before a shared control
    read. A serial mean guard may replace an output if its update maximum
    changes. No later iteration is queued before the current stopping decision.
    The original independent operators remain available for fixed-state A/B.
    """
    if len(fields)!=2 or len(initials)!=2:raise ValueError('Inverse pair requires two fields and initials')
    if not isinstance(max_iterations,int) or max_iterations<0:raise ValueError('Invalid inverse iteration count')
    if not np.isfinite([mean_tolerance,max_tolerance]).all() or min(mean_tolerance,max_tolerance)<0:
        raise ValueError('Invalid inverse tolerances')
    for field,initial in zip(fields,initials):
        if initial is not None and not initial.grid.same_geometry(field.grid):raise ValueError('Initial inverse grid mismatch')
    if _reference:
        return tuple(ops.invert(field,initial,max_iterations=max_iterations,
            mean_tolerance=mean_tolerance,max_tolerance=max_tolerance,enforce_boundary=enforce_boundary,
            _evaluate_final_residual=_evaluate_final_residual) for field,initial in zip(fields,initials))
    ctx=ops.context
    current=[ops.zero(field.grid) if initial is None else initial for field,initial in zip(fields,initials)]
    histories=[[],[]];reasons=['max_iterations','max_iterations'];active=[max_iterations>0]*2
    mean_limit,max_limit=float(mean_tolerance),float(max_tolerance)
    tiny=np.finfo(np.float32).tiny;eps=np.finfo(np.float32).eps
    while any(active):
        pending=[]
        for side in range(2):
            if active[side]:
                field=fields[side]
                residual,norms=kernels.residual_arrays(ctx,field.data,current[side].data,
                    field.grid.lps_to_index[:3,:3],field.grid.spacing_xyz)
                pending.append((side,residual,norms))
        device_controls=_controls(ctx,pending)
        queued=[]
        for index,(side,residual,norms) in enumerate(pending):
            epsilon=np.float32(.75 if not histories[side] else .5)
            queued.append(kernels.inverse_step_device(ctx,current[side].data,residual,norms,
                device_controls,index,epsilon,enforce_boundary))
        controls=ctx.control(device_controls,'inverse_pair_norms',4)
        for index,(side,residual,norms) in enumerate(pending):
            queued_maximum=float(controls[index,1])
            mean,maximum=kernels.guarded_norms(ctx,norms,*controls[index],mean_limit)
            mean_control,max_control=mean,maximum
            guard_mean=bool(abs(mean-mean_limit)<=8*eps*max(mean,mean_limit,tiny))
            guard_max=bool(abs(maximum-max_limit)<=8*eps*max(maximum,max_limit,tiny))
            if guard_max:max_control=kernels.maximum_control(ctx,residual,fields[side].grid.spacing_xyz)
            iteration=len(histories[side]);epsilon=np.float32(.75 if iteration==0 else .5)
            data=queued[index]
            if maximum!=queued_maximum:
                data=kernels.inverse_step(ctx,current[side].data,residual,norms,
                    maximum,epsilon,enforce_boundary)
            current[side]=CUDAField(data,fields[side].grid)
            histories[side].append({'iteration':iteration+1,'mean_scaled_error_before_update':mean,
                'max_scaled_error_before_update':maximum,'epsilon':float(epsilon),
                'mean_control_error':mean_control,'max_control_error':max_control,
                'distance_to_mean_threshold':abs(mean_control-mean_limit),
                'distance_to_max_threshold':abs(max_control-max_limit),
                'native_scalar_control_guard':{'mean':guard_mean,'maximum':guard_max}})
            if max_control<=max_limit or mean_control<=mean_limit:
                reasons[side]='maximum_error_threshold' if max_control<=max_limit else 'mean_error_threshold'
                active[side]=False
            elif len(histories[side])>=max_iterations:active[side]=False
    if _evaluate_final_residual:
        pending=[]
        for side,field in enumerate(fields):
            residual,norms=kernels.residual_arrays(ctx,field.data,current[side].data,
                field.grid.lps_to_index[:3,:3],field.grid.spacing_xyz)
            pending.append((side,residual,norms))
        controls=ctx.control(_controls(ctx,pending),'inverse_pair_final_norms',4)
        final=[]
        for index,(_,_,norms) in enumerate(pending):
            mean,maximum=kernels.guarded_norms(ctx,norms,*controls[index])
            final.append({'mean_spacing_scaled_norm':mean,'max_spacing_scaled_norm':maximum})
    else:final=[{'evaluated':False},{'evaluated':False}]
    return tuple(InversionResult(current[side],histories[side],reasons[side],final[side]) for side in range(2))
