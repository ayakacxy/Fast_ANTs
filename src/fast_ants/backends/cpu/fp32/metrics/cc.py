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

from fast_ants.backends.cpu.fp32.images import array32
from fast_ants.backends.cpu.fp32.images import gradient_image32
from fast_ants.backends.cpu.fp32.images import require32

def neighborhood_cc32(fixed,moving,radius=2,*,valid=None):
    require32(fixed.data);require32(moving.data)
    if not fixed.grid.same_geometry(moving.grid):raise ValueError("CC images must occupy the same grid")
    if not isinstance(radius,int) or radius<0:raise ValueError("FP32 CC radius must be a nonnegative integer")
    valid=np.ones(fixed.grid.shape_zyx,bool) if valid is None else np.asarray(valid)
    if valid.shape!=fixed.grid.shape_zyx or valid.dtype!=np.bool_:raise ValueError("CC validity must be Boolean on the common grid")
    value,count,update=k.cc32(fixed.data,moving.data,gradient_image32(moving),np.ascontiguousarray(valid),radius)
    if not count:raise NoValidSamplesError("No valid FP32 CC centre")
    return CCEvaluation(float(value),DisplacementField(update,fixed.grid),int(count))


def neighborhood_cc_pair32(fixed,moving,radius=2,*,valid=None):
    """Both SyN directions from one centered neighborhood-statistics pass."""
    require32(fixed.data);require32(moving.data)
    if not fixed.grid.same_geometry(moving.grid):raise ValueError("CC images must occupy the same grid")
    if not isinstance(radius,int) or radius<0:raise ValueError("FP32 CC radius must be a nonnegative integer")
    valid=np.ones(fixed.grid.shape_zyx,bool) if valid is None else np.asarray(valid)
    if valid.shape!=fixed.grid.shape_zyx or valid.dtype!=np.bool_:raise ValueError("CC validity must be Boolean on the common grid")
    shift=fixed.grid.lps_to_index[:3,:3]@np.diag(fixed.grid.spacing_xyz*.5)
    value,count,forward,reverse=cc_pair_gradient32(fixed.data,moving.data,
        np.ascontiguousarray(valid),radius,array32(shift),array32(fixed.grid.spacing_xyz))
    if not count:raise NoValidSamplesError("No valid FP32 CC centre")
    return (CCEvaluation(float(value),DisplacementField(forward,fixed.grid),int(count)),
            CCEvaluation(float(value),DisplacementField(reverse,fixed.grid),int(count)))

