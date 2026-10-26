"""Stage implementation extracted without changing numerical definitions."""
import numpy as np

from fast_ants.backends.cpu.fp32 import kernels as k

from fast_ants.backends.cpu.fp32.metrics.cc_queue import cc_pair_gradient32

from fast_ants.backends.cpu.fp32.norms import maximum_vector32

from fast_ants.backends.cpu.fp32.inverse import InverseWorkspace32
from fast_ants.backends.cpu.fp32.inverse import residual32_into
from fast_ants.backends.cpu.fp32.inverse import update32_into

from fast_ants.backends.cpu.fp32.dense_kernels import warp32 as image_warp32
from fast_ants.backends.cpu.fp32.dense_kernels import gradient32 as image_gradient32
from fast_ants.backends.cpu.fp32.dense_kernels import convolve_field_axis32

from fast_ants.core.results import CCEvaluation

from fast_ants.core.results import InversionResult

from fast_ants.pipeline.gaussian import discrete_gaussian_kernel
from fast_ants.pipeline.gaussian import recursive_gaussian_coefficients

from fast_ants.core.types import Image

from fast_ants.core.results import NoValidSamplesError

from fast_ants.core.fields import DisplacementField

from fast_ants.core.state import SyNState
from fast_ants.core.results import SyNStepResult

from fast_ants.core.transforms import AffineTransform

from fast_ants.pipeline.preprocessing import window_parameters

from fast_ants.backends.cpu.fp32.fields import compose_fields32
from fast_ants.backends.cpu.fp32.fields import invert_field32
from fast_ants.backends.cpu.fp32.fields import resample_field32
from fast_ants.backends.cpu.fp32.fields import scale_update32
from fast_ants.backends.cpu.fp32.images import image32
from fast_ants.backends.cpu.fp32.images import pull_image32
from fast_ants.backends.cpu.fp32.images import zero32
from fast_ants.backends.cpu.fp32.metrics.cc import neighborhood_cc_pair32
from fast_ants.backends.cpu.fp32.filtering import smooth_field32

def identity_state32(grid):
    return SyNState(*(zero32(grid) for i in range(4)))


def adapt_state32(state,grid):
    return SyNState(*(resample_field32(field,grid) for field in state.fields().values()))


def combine_state32(state):
    forward=compose_fields32(state.moving_inverse,state.fixed_to_middle)
    inverse=compose_fields32(state.fixed_inverse,state.moving_to_middle)
    forward.inverse_field=inverse;inverse.inverse_field=forward
    return forward,inverse


class _SyNWorkspace32:
    """Private sequential-level scratch; callback/public results stay owned.

    Alternate four-field output batches preserve the current state while the
    next is computed. Only the registration loop without callbacks uses this
    borrowed-storage path; earlier states can be invalidated two steps later.
    """
    def __init__(self,grid):
        self.grid=grid
        self.inverse=InverseWorkspace32(grid.shape_zyx)
        self.outputs=[[np.empty((3,*grid.shape_zyx),np.float32) for _ in range(4)] for _ in range(2)]
        self.smoothing=[np.empty((3,*grid.shape_zyx),np.float32) for _ in range(2)]
        self.valid=np.empty(grid.shape_zyx,np.bool_)
        self.iteration=0

    def next_outputs(self):
        outputs=self.outputs[self.iteration%2]
        self.iteration+=1
        return outputs


def syn_step32(fixed,moving,state,*,linear_transform=None,radius=2,gradient_step=.1,
               update_variance=3.,total_variance=0.,fixed_mask=None,moving_mask=None,capture=False,_inverse_workspace=None,_workspace=None):
    private=None if capture else _workspace
    if private is not None and not private.grid.same_geometry(state.grid):
        raise ValueError('SyN workspace must occupy the state grid')
    fmid=pull_image32(fixed,state.grid,state.fixed_inverse)
    mmid=pull_image32(moving,state.grid,state.moving_inverse,linear_transform)
    valid=None
    if fixed_mask is not None or moving_mask is not None:
        valid=np.ones(state.grid.shape_zyx,bool) if private is None else private.valid
        if private is not None:valid.fill(True)
        for mask,field,linear in ((fixed_mask,state.fixed_inverse,None),(moving_mask,state.moving_inverse,linear_transform)):
            if mask is not None:
                if not np.all((mask.data==0)|(mask.data==1)):raise ValueError("Mask must be binary")
                valid&=pull_image32(image32(mask),state.grid,field,linear,nearest=True).data!=0
    fm,mf=neighborhood_cc_pair32(fmid,mmid,radius,valid=valid)
    forward_value,reverse_value=fm.value,mf.value
    forward_count,reverse_count=fm.valid_points,mf.valid_points
    if private is not None:
        # Reuse both smoothing passes for one side at a time, then recycle the
        # consumed raw metric update as its independent composition output.
        fs=smooth_field32(fm.update,update_variance,_buffers=private.smoothing)
        fu,fscale=scale_update32(fs,gradient_step,_in_place=True)
        ft=compose_fields32(fu,state.fixed_to_middle,_output=fm.update.data)
        del fs,fu
        ms=smooth_field32(mf.update,update_variance,_buffers=private.smoothing)
        mu,mscale=scale_update32(ms,gradient_step,_in_place=True)
        mt=compose_fields32(mu,state.moving_to_middle,_output=mf.update.data)
        del ms,mu
    else:
        fs,ms=smooth_field32(fm.update,update_variance),smooth_field32(mf.update,update_variance)
        fu,fscale=scale_update32(fs,gradient_step);mu,mscale=scale_update32(ms,gradient_step)
        ft=compose_fields32(fu,state.fixed_to_middle)
        mt=compose_fields32(mu,state.moving_to_middle)
    if total_variance!=0:
        ft=smooth_field32(ft,total_variance);mt=smooth_field32(mt,total_variance)
    if not capture:
        # None of these arrays is needed by the inversion phase or returned
        # state. Release them before allocating the four new inverse fields.
        del fmid,mmid,fm,mf
        if private is None:del fs,ms,fu,mu
    # The zero-variance fields are already private outputs; another copy adds
    # no isolation. Public smooth_field32(variance=0) keeps its copy contract.
    workspace=(InverseWorkspace32(state.grid.shape_zyx) if _inverse_workspace is None else _inverse_workspace) if private is None else private.inverse
    outputs=[None]*4 if private is None else private.next_outputs()
    fi=invert_field32(ft,state.fixed_inverse,_workspace=workspace,_output=outputs[0],_evaluate_final_residual=private is None)
    ff=invert_field32(fi.field,ft,_workspace=workspace,_output=outputs[1],_evaluate_final_residual=private is None)
    mi=invert_field32(mt,state.moving_inverse,_workspace=workspace,_output=outputs[2],_evaluate_final_residual=private is None)
    mm=invert_field32(mi.field,mt,_workspace=workspace,_output=outputs[3],_evaluate_final_residual=private is None)
    next_state=SyNState(ff.field,fi.field,mm.field,mi.field)
    details={'forward_value':forward_value,'reverse_value':reverse_value,'fixed_scale':fscale,'moving_scale':mscale,
             'fixed_valid_centres':forward_count,'moving_valid_centres':reverse_count,
             'inversions':{name:{'history':r.history,'stop_reason':r.stop_reason,'final_residual':r.final_residual}
                          for name,r in (('fixed_inverse',fi),('fixed_forward',ff),('moving_inverse',mi),('moving_forward',mm))}}
    intermediate={}
    if capture:
        intermediate={'fixed_midpoint':fmid,'moving_midpoint':mmid,'fixed_raw_update':fm.update,
                      'moving_raw_update':mf.update,'fixed_smoothed_update':fs,'moving_smoothed_update':ms,
                      'fixed_scaled_update':fu,'moving_scaled_update':mu,'fixed_total_before_inverse':ft,'moving_total_before_inverse':mt}
        intermediate.update(next_state.fields())
    return SyNStepResult(next_state,float(np.float32(.5)*np.float32(forward_value+reverse_value)),details,intermediate)

