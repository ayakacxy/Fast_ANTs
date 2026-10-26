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

from fast_ants.backends.cpu.fp32.filtering import _field_output32
from fast_ants.backends.cpu.fp32.images import array32
from fast_ants.backends.cpu.fp32.images import require32
from fast_ants.backends.cpu.fp32.images import zero32

def compose_fields32(outer,inner,*,_output=None):
    require32(outer.data);require32(inner.data)
    inverse=outer.grid.lps_to_index[:3,:3]
    index_linear=inverse@inner.grid.index_to_lps[:3,:3]
    offset=inverse@(inner.grid.origin-outer.grid.origin)
    if _output is None:
        values=k.compose32(outer.data,inner.data,array32(index_linear),array32(offset),array32(inverse))
    else:
        _field_output32(_output,inner.data.shape)
        if np.shares_memory(_output,outer.data) or np.shares_memory(_output,inner.data):
            raise ValueError('Composition output cannot overwrite an input field')
        values=k.compose_into32(outer.data,inner.data,array32(index_linear),array32(offset),array32(inverse),_output)
    return DisplacementField(values,inner.grid)


def resample_field32(field,grid):
    return compose_fields32(field,zero32(grid))


def scale_update32(field,learning_rate,*,_in_place=False):
    require32(field.data)
    if not np.isfinite(learning_rate) or learning_rate<0:raise ValueError("Invalid learning rate")
    maximum=_maximum_control32(field)
    target=float(np.float32(learning_rate))
    factor=np.float32(target/maximum) if maximum>0 else np.float32(target)
    # The source SyN algorithm computes one norm and one scalar factor. A
    # rounded stored norm need not equal/exceed the requested step; rescanning
    # and increasing the factor changes that algorithm and adds dense passes.
    if _in_place:
        _field_output32(field.data,field.data.shape)
        values=k.multiply_into32(field.data,factor,field.data)
    else:
        values=k.multiply32(field.data,factor)
    return DisplacementField(values,field.grid),{'maximum_spacing_scaled_norm':maximum,'scale':float(factor),
             'effective_target':target}


def invert_field32(field,initial=None,*,max_iterations=20,mean_tolerance=.001,max_tolerance=.1,enforce_boundary=True,_workspace=None,_output=None,_evaluate_final_residual=True):
    require32(field.data)
    if not isinstance(max_iterations,int) or max_iterations<0:raise ValueError("Invalid inverse iteration count")
    if not np.isfinite([mean_tolerance,max_tolerance]).all() or min(mean_tolerance,max_tolerance)<0:raise ValueError("Invalid inverse tolerances")
    if initial is not None and not field.grid.same_geometry(initial.grid):raise ValueError("Initial inverse grid mismatch")
    workspace=InverseWorkspace32(field.grid.shape_zyx) if _workspace is None else _workspace
    # Shared scratch alone keeps public results fresh. Only an explicit private
    # output opts into borrowed storage owned by the sequential SyN level.
    owned=(workspace.output if _workspace is None else np.empty((3,*field.grid.shape_zyx),np.float32)) if _output is None else _field_output32(_output,field.data.shape)
    if (np.shares_memory(owned,field.data) or
        (initial is not None and np.shares_memory(owned,initial.data)) or
        np.shares_memory(owned,workspace.residual) or np.shares_memory(owned,workspace.norms)):
        raise ValueError('Inverse output must be separate from forward, initial and residual/norm arrays')
    if initial is None:
        owned.fill(0);current=owned
    else:
        current=require32(initial.data)
        if max_iterations==0:
            np.copyto(owned,current);current=owned
    history=[];reason='max_iterations'
    spacing=array32(field.grid.spacing_xyz)
    inverse_linear=array32(field.grid.lps_to_index[:3,:3])
    # Tolerances are CPU control quantities: quantizing .1 itself to FP32 can
    # change an equality into a different stopping condition.
    mean_limit,max_limit=float(mean_tolerance),float(max_tolerance)
    for i in range(max_iterations):
        residual,norms,mean,maximum=residual32_into(field.data,current,inverse_linear,
            spacing,workspace,mean_tolerance=mean_limit)
        mean_control,max_control=float(mean),float(maximum)
        guard_mean=abs(mean_control-mean_limit)<=8*np.finfo(np.float32).eps*max(mean_control,mean_limit,np.finfo(np.float32).tiny)
        guard_max=abs(max_control-max_limit)<=8*np.finfo(np.float32).eps*max(max_control,max_limit,np.finfo(np.float32).tiny)
        if guard_mean or guard_max:
            if guard_max:max_control=_maximum_control32(DisplacementField(residual,field.grid))
        epsilon=np.float32(.75 if i==0 else .5)
        current=update32_into(current,residual,norms,np.float32(maximum),epsilon,
            bool(enforce_boundary),workspace,output=owned)
        history.append({'iteration':i+1,'mean_scaled_error_before_update':float(mean),
                        'max_scaled_error_before_update':float(maximum),'epsilon':float(epsilon),
                        'mean_control_error':mean_control,'max_control_error':max_control,
                        'distance_to_mean_threshold':abs(mean_control-mean_limit),
                        'distance_to_max_threshold':abs(max_control-max_limit),
                        'native_scalar_control_guard':{'mean':bool(guard_mean),'maximum':bool(guard_max)}})
        if max_control<=max_limit or mean_control<=mean_limit:
            reason='maximum_error_threshold' if max_control<=max_limit else 'mean_error_threshold'
            break
    if _evaluate_final_residual:
        _,_,mean,maximum=residual32_into(field.data,current,inverse_linear,spacing,workspace,store_residual=False)
        final_residual={'mean_spacing_scaled_norm':float(mean),'max_spacing_scaled_norm':float(maximum)}
    else:
        # This additional diagnostic pass is not used by the inverse-update
        # algorithm or its stopping condition. Internal fitting can omit it.
        final_residual={'evaluated':False}
    return InversionResult(DisplacementField(current,field.grid),history,reason,final_residual)


def _maximum_control32(field):
    """One selected 3-vector is a native FP64 host control, not a dense pass."""
    candidate=maximum_vector32(field.data,array32(field.grid.spacing_xyz))
    scaled=candidate.astype(np.float64)/field.grid.spacing_xyz
    return float(np.sqrt(scaled@scaled))

