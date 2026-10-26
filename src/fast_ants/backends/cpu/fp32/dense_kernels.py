"""Thread independent FP32 output rows or slices without changing arithmetic.

Small arrays and single-thread execution use the existing serial kernels.
Each parallel task retains the serial operation and convolution tap order;
there are no shared accumulators or reductions across tasks.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from fast_ants.backends.cpu.fp32 import kernels as fp32
from fast_ants.backends.cpu.fp32.kernels import HALF
from fast_ants.backends.cpu.fp32.kernels import ONE
from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.kernels import inside32
from fast_ants.backends.cpu.fp32.kernels import linear32
from fast_ants.backends.cpu.fp32.kernels import linear_shift32


_PARALLEL_POINTS = 16384


@njit(cache=True, fastmath=False, parallel=True)
def _convolve32(lines, kernel):
    output = np.empty(lines.shape, np.float32)
    radius = kernel.size // 2
    for task in prange(lines.shape[0]):
        line = np.int64(task)
        for i in range(lines.shape[1]):
            total, correction = ZERO, ZERO
            for k in range(kernel.size):
                value = kernel[k] * lines[line, min(max(i + k - radius, 0), lines.shape[1] - 1)]
                total, correction = add32(total, correction, value)
            output[line, i] = total
    return output


@njit(cache=True, fastmath=False, inline='always')
def _field_tap32(output, correction, c, z, y, x, value):
    # Same Kahan update as add32, with independent x positions available to
    # vectorize. Both arrays and every arithmetic operand remain FP32.
    x = np.uint64(x)
    total = output[c, z, y, x]
    corrected = value - correction[x]
    updated = total + corrected
    correction[x] = (updated - total) - corrected
    output[c, z, y, x] = updated


@njit(cache=True, fastmath=False, inline='always')
def _field_axis_row32(values, kernel, dimension, original, weight, mix, c, z, y, output):
    _, nz, ny, nx = values.shape
    correction = np.zeros(nx, np.float32)
    for x in range(nx):
        output[c, z, y, x] = ZERO
    radius = kernel.size // 2
    if dimension == 0:
        for k in range(kernel.size):
            shift = k - radius
            left_end = min(max(-shift, 0), nx)
            interior_end = min(max(nx - shift, 0), nx)
            coefficient = kernel[k]
            for x in range(left_end):
                _field_tap32(output, correction, c, z, y, x, coefficient * values[c, z, y, 0])
            for x in range(left_end, interior_end):
                _field_tap32(output, correction, c, z, y, x, coefficient * values[c, z, y, np.uint64(x + shift)])
            for x in range(interior_end, nx):
                _field_tap32(output, correction, c, z, y, x, coefficient * values[c, z, y, nx - 1])
    elif dimension == 1:
        for k in range(kernel.size):
            source_y = min(max(y + k - radius, 0), ny - 1)
            coefficient = kernel[k]
            for x in range(nx):
                _field_tap32(output, correction, c, z, y, x, coefficient * values[c, z, source_y, np.uint64(x)])
    else:
        for k in range(kernel.size):
            source_z = min(max(z + k - radius, 0), nz - 1)
            coefficient = kernel[k]
            for x in range(nx):
                _field_tap32(output, correction, c, z, y, x, coefficient * values[c, source_z, y, np.uint64(x)])
    if mix:
        for x in range(nx):
            boundary = x == 0 or y == 0 or z == 0 or x == nx - 1 or y == ny - 1 or z == nz - 1
            output[c, z, y, x] = (ZERO if boundary else
                output[c, z, y, x] * weight + original[c, z, y, x] * (ONE - weight))


@njit(cache=True, fastmath=False)
def _convolve_field_axis32_serial(values, kernel, dimension, original, weight, mix, output):
    for c in range(values.shape[0]):
        for z in range(values.shape[1]):
            for y in range(values.shape[2]):
                _field_axis_row32(values, kernel, dimension, original, weight, mix, c, z, y, output)
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _convolve_field_axis32_parallel(values, kernel, dimension, original, weight, mix, output):
    nz, ny = values.shape[1:3]
    for task in prange(values.shape[0] * nz * ny):
        row = np.int64(task)
        c, z, y = row // (nz * ny), (row // ny) % nz, row % ny
        _field_axis_row32(values, kernel, dimension, original, weight, mix, c, z, y, output)
    return output


def convolve_field_axis32(values, kernel, dimension, original=None, weight=None, *, output=None):
    """Convolve an x/y/z axis directly in FP32 C/Z/Y/X field storage.

    Supplying ``original`` and ``weight`` folds the existing zero-boundary and
    small-variance mix into this axis, normally the final z pass. Inputs remain
    unmodified and may be strided. By default the result is fresh; private
    sequential callers can provide separate contiguous FP32 output scratch.
    """
    if dimension not in (0, 1, 2):
        raise ValueError('Field convolution dimension must be x/y/z = 0/1/2')
    if values.dtype != np.float32 or values.ndim != 4 or values.shape[0] != 3:
        raise ValueError('Field convolution needs FP32 [3,z,y,x] values')
    if kernel.dtype != np.float32 or kernel.ndim != 1:
        raise ValueError('Field convolution needs a one-dimensional FP32 kernel')
    mix = original is not None
    if mix:
        if original.dtype != np.float32 or original.shape != values.shape:
            raise ValueError('Original field must have the same shape and FP32 dtype')
        if weight is None:
            raise ValueError('Boundary mixing needs a finite FP32 weight')
        weight = np.float32(weight)
        if not np.isfinite(weight):
            raise ValueError('Boundary mixing needs a finite FP32 weight')
    else:
        original, weight = values, ZERO
    if output is None:
        output = np.empty(values.shape, np.float32)
    elif (output.shape != values.shape or output.dtype != np.float32
          or not output.flags.c_contiguous or not output.flags.writeable):
        raise ValueError('Field convolution output must be writable contiguous FP32 on the same grid')
    if np.shares_memory(output, values) or (mix and np.shares_memory(output, original)):
        raise ValueError('Field convolution output must not overwrite its inputs')
    function = (_convolve_field_axis32_parallel
                if values.size >= _PARALLEL_POINTS and get_num_threads() > 1
                else _convolve_field_axis32_serial)
    return function(values, kernel, dimension, original, weight, mix, output)


@njit(cache=True, fastmath=False, parallel=True)
def _gradient32(data, half_spacing_to_index, spacing):
    nz, ny, nx = data.shape
    output = np.zeros((3, nz, ny, nx), dtype=np.float32)
    for task in prange(nz):
        z = np.int64(task)
        for y in range(ny):
            for x in range(nx):
                for c in range(3):
                    dx, dy, dz = half_spacing_to_index[0, c], half_spacing_to_index[1, c], half_spacing_to_index[2, c]
                    px, py, pz = np.float32(x) + dx, np.float32(y) + dy, np.float32(z) + dz
                    mx, my, mz = np.float32(x) - dx, np.float32(y) - dy, np.float32(z) - dz
                    if inside32(px, py, pz, nx, ny, nz) and inside32(mx, my, mz, nx, ny, nz):
                        anchor = data[z, y, x]
                        output[c, z, y, x] = (linear_shift32(data, px, py, pz, anchor) - linear_shift32(data, mx, my, mz, anchor)) / spacing[c]
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _warp32(data, field, index_linear, index_offset, field_to_index, nearest=False):
    _, nz, ny, nx = field.shape
    output = np.zeros((nz, ny, nx), dtype=np.float32)
    iz, iy, ix = data.shape
    for task in prange(nz):
        z = np.int64(task)
        for y in range(ny):
            for x in range(nx):
                index = np.empty(3, np.float32)
                for c in range(3):
                    value = index_offset[c] + index_linear[c, 0] * np.float32(x) + index_linear[c, 1] * np.float32(y) + index_linear[c, 2] * np.float32(z)
                    for j in range(3):
                        value += field_to_index[c, j] * field[j, z, y, x]
                    index[c] = value
                if inside32(index[0], index[1], index[2], ix, iy, iz):
                    if nearest:
                        output[z, y, x] = data[int(np.floor(index[2] + HALF)), int(np.floor(index[1] + HALF)), int(np.floor(index[0] + HALF))]
                    else:
                        output[z, y, x] = linear32(data, index[0], index[1], index[2])
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _inverse_step32(inverse, residual, norms, maximum, epsilon, enforce_boundary):
    _, nz, ny, nx = inverse.shape
    output = np.empty(inverse.shape, np.float32)
    limit = epsilon * maximum
    for task in prange(nz * ny):
        row = np.int64(task)
        z, y = row // ny, row % ny
        for x in range(nx):
            factor = ONE
            if norms[z, y, x] > limit:
                factor = limit / norms[z, y, x]
            boundary = x == 0 or y == 0 or z == 0 or x == nx - 1 or y == ny - 1 or z == nz - 1
            for c in range(3):
                value = inverse[c, z, y, x] - residual[c, z, y, x] * factor * epsilon
                output[c, z, y, x] = ZERO if enforce_boundary and boundary else value
    return output


def convolve32(lines, kernel):
    """Apply the serial FP32 convolution formula, threading independent rows."""
    if lines.size >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _convolve32(lines, kernel)
    return fp32.convolve32(lines, kernel)


def gradient32(data, half_spacing_to_index, spacing):
    """Apply the serial FP32 gradient formula, threading independent z slices."""
    if data.size >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _gradient32(data, half_spacing_to_index, spacing)
    return fp32.gradient32(data, half_spacing_to_index, spacing)


def warp32(data, field, index_linear, index_offset, field_to_index, nearest=False):
    """Apply the serial FP32 pull formula, threading independent output slices."""
    if field.shape[1] * field.shape[2] * field.shape[3] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _warp32(data, field, index_linear, index_offset, field_to_index, nearest)
    return fp32.warp32(data, field, index_linear, index_offset, field_to_index, nearest)


def inverse_step32(inverse, residual, norms, maximum, epsilon, enforce_boundary):
    """Thread independent inverse updates after the shared scalar norm control."""
    if norms.size >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _inverse_step32(inverse, residual, norms, maximum, epsilon, enforce_boundary)
    return fp32.inverse_step32(inverse, residual, norms, maximum, epsilon, enforce_boundary)
