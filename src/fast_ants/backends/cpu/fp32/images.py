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


def array32(values):
    return np.ascontiguousarray(values,dtype=np.float32)


def require32(values):
    if values.dtype!=np.float32:
        raise ValueError("FP32 operator input must be explicitly quantized to float32")
    return values


def image32(image,normalize=False):
    values=array32(image.data)
    if normalize:
        low,high,scale,offset=window_parameters(*k.range32(values))
        values=k.window32(values,np.float32(low),np.float32(high),np.float32(scale))
    return Image(values,image.grid,{**image.metadata,'arithmetic':'FP32'})


def zero32(grid):
    return DisplacementField(np.zeros((3,*grid.shape_zyx),np.float32),grid)


def pull_image32(image,grid,field=None,linear=None,nearest=False):
    """Pull image at linear(x+field(x)), using local FP32 source indices."""
    require32(image.data)
    field=zero32(grid) if field is None else field
    require32(field.data)
    if not field.grid.same_geometry(grid):raise ValueError("Pull field must occupy the requested output grid")
    linear=AffineTransform.identity() if linear is None else linear
    a=linear.matrix[:3,:3]
    inverse=image.grid.lps_to_index[:3,:3]
    # The subtraction is performed once in small CPU matrices, before quantizing.
    index_linear=inverse@a@grid.index_to_lps[:3,:3]
    local_translation=linear.map_points(grid.origin)-image.grid.origin
    offset=inverse@local_translation
    output=image_warp32(image.data,field.data,array32(index_linear),array32(offset),array32(inverse@a),bool(nearest))
    return Image(output,grid,image.metadata)


def gradient_image32(image):
    require32(image.data)
    shift=image.grid.lps_to_index[:3,:3]@np.diag(image.grid.spacing_xyz*.5)
    return image_gradient32(image.data,array32(shift),array32(image.grid.spacing_xyz))

