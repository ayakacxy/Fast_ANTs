from fast_ants.core.results import SyNStepResult
from fast_ants.core.state import SyNState
"""Standard greedy SyN with two sides and four fields, CPU reference path.

Control flow is a modified translation of ITK's SyNImageRegistrationMethod,
Copyright NumFOCUS, Apache-2.0; upstream notices are retained in licenses/LICENSE-ITK.
"""

from dataclasses import dataclass
import numpy as np

from fast_ants.backends.cpu.fp64.metrics.cc import neighborhood_cc
from fast_ants.backends.cpu.fp64.metrics.cc import neighborhood_cc_pair
from fast_ants.backends.cpu.fp64.fields import compose_fields
from fast_ants.backends.cpu.fp64.fields import invert_field
from fast_ants.backends.cpu.fp64.fields import resample_field
from fast_ants.backends.cpu.fp64.fields import scale_update
from fast_ants.backends.cpu.fp64.fields import zero_field
from fast_ants.backends.cpu.fp64.fields import _InversionWorkspace
from fast_ants.backends.cpu.fp64.filtering import smooth_field
from fast_ants.backends.cpu.fp64.sampling import apply_image
from fast_ants.core.transforms import AffineTransform
from fast_ants.core.transforms import CompositeTransform




class _SyNWorkspace:
    """Owned by one sequential registration level, never shared with callbacks.

    Two output batches keep the current state readable while the next is
    written. The public syn_step default continues to return fresh storage.
    """
    def __init__(self,grid):
        self.inverse=_InversionWorkspace(grid.shape_zyx,packed=True)
        self.outputs=[[np.empty((3,*grid.shape_zyx),np.float64) for _ in range(4)] for _ in range(2)]
        self.smoothing=[np.empty((3,*grid.shape_zyx),np.float64) for _ in range(2)]
        self.valid=np.empty(grid.shape_zyx,np.bool_)
        self.iteration=0

    def next_outputs(self):
        outputs=self.outputs[self.iteration%2]
        self.iteration+=1
        return outputs




def syn_step(fixed, moving, state, *, linear_transform=None, radius=2, gradient_step=.1,
             update_variance=3., total_variance=0., fixed_mask=None, moving_mask=None,
             capture=False, _workspace=None):
    """Take one standard SyN update from an explicitly supplied common state.

    Each original smoothed image is pulled into the midpoint using the inverse
    side field. Moving additionally uses the frozen linear transform. Zero fill
    belongs to these new images; it does not invalidate their metric samples.
    """
    if not np.isfinite([gradient_step,update_variance,total_variance]).all() or min(gradient_step,update_variance,total_variance)<0:
        raise ValueError("SyN step and smoothing variances must be finite and nonnegative")
    gradient_step,update_variance,total_variance=(float(np.float32(v)) for v in (gradient_step,update_variance,total_variance))
    linear = AffineTransform.identity() if linear_transform is None else linear_transform
    fixed_pull = state.fixed_inverse
    moving_pull = CompositeTransform((state.moving_inverse,linear))
    fmid = apply_image(fixed,state.grid,fixed_pull)
    mmid = apply_image(moving,state.grid,moving_pull)
    fmask = None if fixed_mask is None else apply_image(fixed_mask,state.grid,fixed_pull,'NearestNeighbor')
    mmask = None if moving_mask is None else apply_image(moving_mask,state.grid,moving_pull,'NearestNeighbor')
    forward_metric, reverse_metric = neighborhood_cc_pair(fmid,mmid,radius,fixed_mask=fmask,moving_mask=mmask)
    forward_value,reverse_value=forward_metric.value,reverse_metric.value
    forward_count,reverse_count=forward_metric.valid_points,reverse_metric.valid_points
    if _workspace is not None and not capture:
        # Complete one independent side before reusing scratch for the other.
        # The consumed raw update buffer becomes the composition output.
        f_smoothed=smooth_field(forward_metric.update,update_variance,_buffers=_workspace.smoothing)
        f_update,fscale=scale_update(f_smoothed,gradient_step,_inplace=True)
        f_composed=compose_fields(f_update,state.fixed_to_middle,_output=forward_metric.update.data,_valid=_workspace.valid)
        del f_update,f_smoothed
        m_smoothed=smooth_field(reverse_metric.update,update_variance,_buffers=_workspace.smoothing)
        m_update,mscale=scale_update(m_smoothed,gradient_step,_inplace=True)
        m_composed=compose_fields(m_update,state.moving_to_middle,_output=reverse_metric.update.data,_valid=_workspace.valid)
        del m_update,m_smoothed,forward_metric,reverse_metric,fmid,mmid,fmask,mmask
    else:
        f_smoothed = smooth_field(forward_metric.update,update_variance)
        m_smoothed = smooth_field(reverse_metric.update,update_variance)
        forward_value,reverse_value=forward_metric.value,reverse_metric.value
        forward_count,reverse_count=forward_metric.valid_points,reverse_metric.valid_points
        if not capture:
            del forward_metric,reverse_metric,fmid,mmid,fmask,mmask
        f_update,fscale = scale_update(f_smoothed,gradient_step)
        m_update,mscale = scale_update(m_smoothed,gradient_step)
        if not capture:
            del f_smoothed,m_smoothed
        f_composed = compose_fields(f_update,state.fixed_to_middle)
        m_composed = compose_fields(m_update,state.moving_to_middle)
        if not capture:
            del f_update,m_update
    # These composed fields are private and are not mutated by inversion.
    # Capture retains the original independent intermediate-object contract.
    f_total = f_composed if total_variance == 0.0 and not capture else smooth_field(f_composed,total_variance)
    m_total = m_composed if total_variance == 0.0 and not capture else smooth_field(m_composed,total_variance)
    if not capture:
        del f_composed,m_composed
    # Preserve all four source inversions. Internal sequential registration
    # can reuse scratch and alternate output batches; public/captured calls
    # retain fresh arrays so earlier results remain independently owned.
    workspace=_InversionWorkspace(state.grid.shape_zyx) if _workspace is None else _workspace.inverse
    outputs=[None]*4 if _workspace is None or capture else _workspace.next_outputs()
    fi = invert_field(f_total,state.fixed_inverse,_workspace=workspace,_output=outputs[0])
    ff = invert_field(fi.field,f_total,_workspace=workspace,_output=outputs[1])
    mi = invert_field(m_total,state.moving_inverse,_workspace=workspace,_output=outputs[2])
    mf = invert_field(mi.field,m_total,_workspace=workspace,_output=outputs[3])
    next_state = SyNState(ff.field,fi.field,mf.field,mi.field)
    value = .5*(forward_value+reverse_value)
    details = {'forward_value':forward_value,'reverse_value':reverse_value,
               'fixed_scale':fscale,'moving_scale':mscale,'fixed_valid_centres':forward_count,
               'moving_valid_centres':reverse_count,
               'inversions':{name:{'history':result.history,'stop_reason':result.stop_reason,'final_residual':result.final_residual}
                             for name,result in (('fixed_inverse',fi),('fixed_forward',ff),('moving_inverse',mi),('moving_forward',mf))}}
    intermediates = {}
    if capture:
        intermediates = {'fixed_midpoint':fmid,'moving_midpoint':mmid,
                         'fixed_raw_update':forward_metric.update,'moving_raw_update':reverse_metric.update,
                         'fixed_smoothed_update':f_smoothed,'moving_smoothed_update':m_smoothed,
                         'fixed_scaled_update':f_update,'moving_scaled_update':m_update,
                         'fixed_composed':f_composed,'moving_composed':m_composed,
                         'fixed_total_before_inverse':f_total,'moving_total_before_inverse':m_total}
        intermediates.update(next_state.fields())
    return SyNStepResult(next_state,float(value),details,intermediates)


def run_syn_level(fixed,moving,state,iterations,*,callback=None,**step_options):
    """Fixed-count single-level iterator; convergence is added by P08's driver."""
    if not isinstance(iterations,int) or iterations<0:
        raise ValueError("SyN iteration count must be a nonnegative integer")
    history=[]
    for i in range(iterations):
        result = syn_step(fixed,moving,state,**step_options)
        state = result.state
        history.append({'iteration':i+1,'value':result.value,**result.details})
        if callback is not None:
            callback(i+1,result)
    return state,history
