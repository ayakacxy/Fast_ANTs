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
from fast_ants.backends.cpu.fp32.images import require32

def _axis32(values,axis,function,*coefficients):
    moved=np.moveaxis(values,axis,-1)
    lines=array32(moved.reshape(-1,moved.shape[-1]))
    result=function(lines,*(array32(c) for c in coefficients)).reshape(moved.shape)
    return np.ascontiguousarray(np.moveaxis(result,-1,axis))


def smooth_image32(image,sigma,units='vox',*,method='sections'):
    require32(image.data)
    sigmas=np.full(3,float(sigma)) if np.ndim(sigma)==0 else np.asarray(sigma,dtype=np.float64)
    if sigmas.shape!=(3,) or not np.isfinite(sigmas).all() or np.any(sigmas<0):raise ValueError("Invalid Gaussian sigmas")
    if units=='mm':sigmas=sigmas/image.grid.spacing_xyz
    elif units!='vox':raise ValueError("Sigma units must be vox or mm")
    if method not in ('sections','recursive'):raise ValueError('Unknown FP32 Gaussian method')
    if method=='sections':
        from fast_ants.backends.cpu.fp32.gaussian_sections import smooth_sections32
        return smooth_sections32(image,sigmas,'vox')
    values=image.data.copy()
    for dimension in (2,0,1):
        if sigmas[dimension]==0:continue
        axis=2-dimension
        if values.shape[axis]<4:raise ValueError("Recursive Gaussian requires four pixels along filtered axes")
        values=_axis32(values,axis,k.recursive32,*recursive_gaussian_coefficients(sigmas[dimension]))
    return Image(values,image.grid,image.metadata)


def _field_output32(output,shape):
    if (output.shape!=shape or output.dtype!=np.float32 or
        not output.flags.c_contiguous or not output.flags.writeable):
        raise ValueError('Private field output must be writable contiguous FP32 on the same grid')
    return output


def smooth_field32(field,variance,*,_buffers=None):
    require32(field.data)
    if not np.isfinite(variance) or variance<0:raise ValueError("Variance must be finite and nonnegative")
    if _buffers is not None:
        if len(_buffers)!=2:raise ValueError('Field smoothing needs two private buffers')
        for output in _buffers:
            _field_output32(output,field.data.shape)
            if np.shares_memory(output,field.data):raise ValueError('Smoothing scratch cannot overwrite its input')
        if np.shares_memory(*_buffers):raise ValueError('Smoothing buffers must be separate')
    if variance==0:
        if _buffers is None:return DisplacementField(field.data.copy(),field.grid)
        np.copyto(_buffers[0],field.data)
        return DisplacementField(_buffers[0],field.grid)
    values=field.data
    weight=np.float32(1-variance/.5 if variance<.5 else 1.)
    for dimension in range(3):
        kernel=discrete_gaussian_kernel(variance,field.grid.shape_xyz[dimension])
        values=convolve_field_axis32(values,array32(kernel),dimension,
            original=field.data if dimension==2 else None,weight=weight if dimension==2 else None,
            output=None if _buffers is None else _buffers[dimension%2])
    return DisplacementField(values,field.grid)

