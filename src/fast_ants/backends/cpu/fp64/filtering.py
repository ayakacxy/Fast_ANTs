from fast_ants.pipeline.gaussian import _bessel_n
from fast_ants.pipeline.gaussian import recursive_gaussian_coefficients
from fast_ants.pipeline.gaussian import _polynomial
from fast_ants.pipeline.gaussian import _bessel0
from fast_ants.pipeline.gaussian import _bessel1
from fast_ants.pipeline.gaussian import discrete_gaussian_kernel
"""Image recursive Gaussian and SyN discrete Gaussian are separate operators.

Coefficient calculations adapted from ITK 5.4.7 (Copyright NumFOCUS,
Apache-2.0): itkRecursiveGaussianImageFilter.hxx and itkGaussianOperator.hxx.
Python translation and array interfaces are modified implementations.
"""

import math
import numpy as np

from fast_ants.backends.cpu.fp64.filter_kernels import convolve_lines
from fast_ants.backends.cpu.fp64.filter_kernels import convolve_field_axis
from fast_ants.backends.cpu.fp64.filter_kernels import mix_field_and_zero_boundary
from fast_ants.backends.cpu.fp64.filter_kernels import recursive_lines
from fast_ants.core.types import Image
from fast_ants.core.fields import DisplacementField




def _axis_filter(values, axis, operation, *args):
    moved = np.moveaxis(values, axis, -1)
    lines = np.ascontiguousarray(moved.reshape(-1, moved.shape[-1]), dtype=np.float64)
    filtered = operation(lines, *args).reshape(moved.shape)
    return np.ascontiguousarray(np.moveaxis(filtered, -1, axis))


def smooth_image(image, sigma, units="vox", itk_float_intermediates=True):
    sigmas = np.full(3, float(sigma)) if np.ndim(sigma) == 0 else np.asarray(sigma, dtype=np.float64)
    if sigmas.shape != (3,) or not np.isfinite(sigmas).all() or np.any(sigmas < 0):
        raise ValueError("Smoothing requires one or three nonnegative finite sigmas, in x,y,z order")
    if units == "mm":
        sigmas = sigmas / image.grid.spacing_xyz
    elif units != "vox":
        raise ValueError("Sigma units must be vox or mm")
    values = np.asarray(image.data, dtype=np.float64)
    # ITK's mini-pipeline order is z, x, y, with FloatType intermediate images
    # even for double input. A full-FP64 option is retained for numerical audit.
    for dimension in (2, 0, 1):
        if sigmas[dimension] == 0:
            continue
        axis = 2-dimension
        if values.shape[axis] < 4:
            raise ValueError("ITK recursive smoothing needs at least four pixels per filtered dimension")
        values = _axis_filter(values, axis, recursive_lines, *recursive_gaussian_coefficients(sigmas[dimension]))
        if itk_float_intermediates:
            values = values.astype(np.float32).astype(np.float64)
    return Image(values.copy(), image.grid, {**image.metadata, "smoothing_units": units,
                                             "itk_float_intermediates": itk_float_intermediates})












def smooth_field(field, variance, *, _buffers=None):
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Field smoothing variance must be nonnegative and finite")
    values = np.asarray(field.data, dtype=np.float64)
    if variance == 0:
        return DisplacementField(values.copy(), field.grid)
    # The first axis pass allocates its own output; no input copy is needed.
    for dimension in range(3):
        kernel = discrete_gaussian_kernel(variance, field.grid.shape_xyz[dimension])
        if _buffers is None:
            values = convolve_field_axis(values, kernel, dimension)
        else:
            from fast_ants.backends.cpu.fp64.filter_kernels import convolve_field_into
            from numba import get_num_threads
            values = convolve_field_into(values,kernel,dimension,_buffers[dimension%2],values.size>=16384 and get_num_threads()>1)
    weight = 1-variance/.5 if variance < .5 else 1.0
    if _buffers is None:
        values = mix_field_and_zero_boundary(values, np.asarray(field.data, dtype=np.float64), weight)
    else:
        from fast_ants.backends.cpu.fp64.filter_kernels import mix_field_inplace
        values = mix_field_inplace(values,np.asarray(field.data,dtype=np.float64),weight,values.size>=16384 and get_num_threads()>1)
    return DisplacementField(values, field.grid)
