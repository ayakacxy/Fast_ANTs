"""Self-written stencil and recurrence loops.

Recursive Gaussian recurrence adapted to Python/Numba from ITK 5.4.7
itkRecursiveSeparableImageFilter.hxx (Copyright NumFOCUS, Apache-2.0).
This translation replaces ITK pipelines/iterators with arrays and explicit loops.
See licenses/LICENSE-ITK and licenses/NOTICE.md for the original notice and fixed source version.
"""

import numpy as np
from numba import get_num_threads, njit, prange
from fast_ants.backends.cpu.fp64.sampling_kernels import inside
from fast_ants.backends.cpu.fp64.sampling_kernels import linear_value
from fast_ants.core.arithmetic64 import fma64


# Parallelise only independent points/lines. Small requests and one-thread
# execution retain the original serial loops without a thread-pool launch.
# Serial and parallel functions have separate names and source locations so
# Numba disk caches cannot confuse their different compilation flags.
# Dispatch stays in Python: a jitted wrapper that references a parallel kernel
# embeds dynamic runtime globals and cannot itself be cached by Numba.
_PARALLEL_POINTS = 16384
_PARALLEL_VALUES = 16384


@njit(cache=True, fastmath=False)
def _physical_point_gradient_serial(data, points, origin, inverse_linear, spacing):
    output = np.zeros(points.shape, dtype=np.float64)
    nz, ny, nx = data.shape
    # A single serial worker can reuse these six temporary coordinates.
    negative, positive = np.empty(3, np.float64), np.empty(3, np.float64)
    for n in range(points.shape[0]):
        for c in range(3):
            offset = .5*spacing[c]
            for i in range(3):
                minus, plus = 0.0, 0.0
                for j in range(3):
                    # ITK offsets the physical neighbour before subtracting origin.
                    negative_coordinate = (points[n, j]-offset if c == j else points[n, j])-origin[j]
                    positive_coordinate = (points[n, j]+offset if c == j else points[n, j])-origin[j]
                    minus = fma64(inverse_linear[i, j], negative_coordinate, minus)
                    plus = fma64(inverse_linear[i, j], positive_coordinate, plus)
                negative[i], positive[i] = minus, plus
            delta = (points[n, c]+offset)-(points[n, c]-offset)
            if (inside(negative[0], negative[1], negative[2], nx, ny, nz) and
                    inside(positive[0], positive[1], positive[2], nx, ny, nz) and delta > 10*np.finfo(np.float64).eps):
                output[n, c] = (linear_value(data, positive[0], positive[1], positive[2])-
                                  linear_value(data, negative[0], negative[1], negative[2]))/delta
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _physical_point_gradient_parallel(data, points, origin, inverse_linear, spacing):
    output = np.zeros(points.shape, dtype=np.float64)
    nz, ny, nx = data.shape
    for n in prange(points.shape[0]):
        # Keep scratch private to each parallel iteration; Numba can hoist
        # these fixed-size allocations into worker-local storage.
        negative, positive = np.empty(3, np.float64), np.empty(3, np.float64)
        for c in range(3):
            offset = .5*spacing[c]
            for i in range(3):
                minus, plus = 0.0, 0.0
                for j in range(3):
                    # ITK offsets the physical neighbour before subtracting origin.
                    negative_coordinate = (points[n, j]-offset if c == j else points[n, j])-origin[j]
                    positive_coordinate = (points[n, j]+offset if c == j else points[n, j])-origin[j]
                    minus = fma64(inverse_linear[i, j], negative_coordinate, minus)
                    plus = fma64(inverse_linear[i, j], positive_coordinate, plus)
                negative[i], positive[i] = minus, plus
            delta = (points[n, c]+offset)-(points[n, c]-offset)
            if (inside(negative[0], negative[1], negative[2], nx, ny, nz) and
                    inside(positive[0], positive[1], positive[2], nx, ny, nz) and delta > 10*np.finfo(np.float64).eps):
                output[n, c] = (linear_value(data, positive[0], positive[1], positive[2])-
                                  linear_value(data, negative[0], negative[1], negative[2]))/delta
    return output


def physical_point_gradient(data, points, origin, inverse_linear, spacing):
    if points.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _physical_point_gradient_parallel(data, points, origin, inverse_linear, spacing)
    return _physical_point_gradient_serial(data, points, origin, inverse_linear, spacing)


@njit(cache=True, fastmath=False)
def _recursive_lines_serial(lines, numerator, denominator, anticausal, causal_boundary, anti_boundary):
    output = np.empty(lines.shape, dtype=np.float64)
    # Every line has the same length, so serial recurrence scratch is reusable.
    forward = np.empty(lines.shape[1], np.float64)
    backward = np.empty(lines.shape[1], np.float64)
    for line_number in range(lines.shape[0]):
        values = lines[line_number]
        length = values.size
        for i in range(length):
            feed = 0.0
            feedback = 0.0
            for k in range(4):
                feed += numerator[k] * values[max(0, i-k)]
                if i-k-1 >= 0:
                    feedback += denominator[k] * forward[i-k-1]
                else:
                    feedback += causal_boundary[k] * values[0]
            forward[i] = feed - feedback
        for i in range(length-1, -1, -1):
            feed = 0.0
            feedback = 0.0
            for k in range(4):
                feed += anticausal[k] * values[min(length-1, i+k+1)]
                if i+k+1 < length:
                    feedback += denominator[k] * backward[i+k+1]
                else:
                    feedback += anti_boundary[k] * values[length-1]
            backward[i] = feed - feedback
        for i in range(length):
            output[line_number, i] = forward[i] + backward[i]
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _recursive_lines_parallel(lines, numerator, denominator, anticausal, causal_boundary, anti_boundary):
    output = np.empty(lines.shape, dtype=np.float64)
    for line_number in prange(lines.shape[0]):
        values = lines[line_number]
        length = values.size
        forward, backward = np.empty(length, np.float64), np.empty(length, np.float64)
        for i in range(length):
            feed = 0.0
            feedback = 0.0
            for k in range(4):
                feed += numerator[k] * values[max(0, i-k)]
                if i-k-1 >= 0:
                    feedback += denominator[k] * forward[i-k-1]
                else:
                    feedback += causal_boundary[k] * values[0]
            forward[i] = feed - feedback
        for i in range(length-1, -1, -1):
            feed = 0.0
            feedback = 0.0
            for k in range(4):
                feed += anticausal[k] * values[min(length-1, i+k+1)]
                if i+k+1 < length:
                    feedback += denominator[k] * backward[i+k+1]
                else:
                    feedback += anti_boundary[k] * values[length-1]
            backward[i] = feed - feedback
        for i in range(length):
            output[line_number, i] = forward[i] + backward[i]
    return output


def recursive_lines(lines, numerator, denominator, anticausal, causal_boundary, anti_boundary):
    if lines.size >= _PARALLEL_VALUES and lines.shape[0] >= 8 and get_num_threads() > 1:
        return _recursive_lines_parallel(lines, numerator, denominator, anticausal, causal_boundary, anti_boundary)
    return _recursive_lines_serial(lines, numerator, denominator, anticausal, causal_boundary, anti_boundary)


@njit(cache=True, fastmath=False)
def _convolve_lines_serial(lines, kernel):
    output = np.empty(lines.shape, dtype=np.float64)
    radius = kernel.size // 2
    for line in range(lines.shape[0]):
        for i in range(lines.shape[1]):
            total = 0.0
            for k in range(kernel.size):
                total += kernel[k] * lines[line, min(max(i+k-radius, 0), lines.shape[1]-1)]
            output[line, i] = total
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _convolve_lines_parallel(lines, kernel):
    output = np.empty(lines.shape, dtype=np.float64)
    radius = kernel.size // 2
    for line in prange(lines.shape[0]):
        for i in range(lines.shape[1]):
            total = 0.0
            for k in range(kernel.size):
                total += kernel[k] * lines[line, min(max(i+k-radius, 0), lines.shape[1]-1)]
            output[line, i] = total
    return output


def convolve_lines(lines, kernel):
    if lines.size >= _PARALLEL_VALUES and lines.shape[0] >= 8 and get_num_threads() > 1:
        return _convolve_lines_parallel(lines, kernel)
    return _convolve_lines_serial(lines, kernel)


@njit(cache=True, fastmath=False, inline="always")
def _field_axis_row(data, kernel, dimension, c, z, y, output):
    """Keep tap order, but stream native x-contiguous field rows directly."""
    nz, ny, nx = data.shape[1:]
    radius = kernel.size//2
    for x in range(nx):
        output[c, z, y, np.uint64(x)] = 0.0
    if dimension == 0:
        for k in range(kernel.size):
            shift = k-radius
            left_end = min(max(-shift, 0), nx)
            interior_end = min(max(nx-shift, 0), nx)
            coefficient = kernel[k]
            for x in range(left_end):
                output[c, z, y, np.uint64(x)] += coefficient*data[c, z, y, 0]
            for x in range(left_end, interior_end):
                output[c, z, y, np.uint64(x)] += coefficient*data[c, z, y, np.uint64(x+shift)]
            for x in range(interior_end, nx):
                output[c, z, y, np.uint64(x)] += coefficient*data[c, z, y, nx-1]
    elif dimension == 1:
        for k in range(kernel.size):
            source_y = min(max(y+k-radius, 0), ny-1)
            coefficient = kernel[k]
            for x in range(nx):
                output[c, z, y, np.uint64(x)] += coefficient*data[c, z, source_y, np.uint64(x)]
    else:
        for k in range(kernel.size):
            source_z = min(max(z+k-radius, 0), nz-1)
            coefficient = kernel[k]
            for x in range(nx):
                output[c, z, y, np.uint64(x)] += coefficient*data[c, source_z, y, np.uint64(x)]


@njit(cache=True, fastmath=False)
def _convolve_field_axis_serial(data, kernel, dimension):
    output = np.empty(data.shape, np.float64)
    for c in range(data.shape[0]):
        for z in range(data.shape[1]):
            for y in range(data.shape[2]):
                _field_axis_row(data, kernel, dimension, c, z, y, output)
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _convolve_field_axis_parallel(data, kernel, dimension):
    output = np.empty(data.shape, np.float64)
    nz, ny = data.shape[1:3]
    for task in prange(data.shape[0]*nz*ny):
        row = np.int64(task)
        c, z, y = row//(nz*ny), (row//ny) % nz, row % ny
        _field_axis_row(data, kernel, dimension, c, z, y, output)
    return output


def convolve_field_axis(data, kernel, dimension):
    if dimension not in (0, 1, 2):
        raise ValueError("Field convolution dimension must be x/y/z = 0/1/2")
    function = (_convolve_field_axis_parallel if data.size >= _PARALLEL_VALUES
                and get_num_threads() > 1 else _convolve_field_axis_serial)
    return function(data, kernel, dimension)


@njit(cache=True, fastmath=False)
def _central_gradient_serial(data, inverse_index_to_lps):
    nz, ny, nx = data.shape
    gradient = np.zeros((3, nz, ny, nx), dtype=np.float64)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                gx, gy, gz = 0.0, 0.0, 0.0
                if 0 < x < nx-1:
                    gx = .5 * (data[z, y, x+1] - data[z, y, x-1])
                if 0 < y < ny-1:
                    gy = .5 * (data[z, y+1, x] - data[z, y-1, x])
                if 0 < z < nz-1:
                    gz = .5 * (data[z+1, y, x] - data[z-1, y, x])
                for c in range(3):
                    gradient[c, z, y, x] = (inverse_index_to_lps[0, c]*gx +
                                             inverse_index_to_lps[1, c]*gy + inverse_index_to_lps[2, c]*gz)
    return gradient


@njit(cache=True, fastmath=False, parallel=True)
def _central_gradient_parallel(data, inverse_index_to_lps):
    nz, ny, nx = data.shape
    gradient = np.zeros((3, nz, ny, nx), dtype=np.float64)
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                gx, gy, gz = 0.0, 0.0, 0.0
                if 0 < x < nx-1:
                    gx = .5 * (data[z, y, x+1] - data[z, y, x-1])
                if 0 < y < ny-1:
                    gy = .5 * (data[z, y+1, x] - data[z, y-1, x])
                if 0 < z < nz-1:
                    gz = .5 * (data[z+1, y, x] - data[z-1, y, x])
                for c in range(3):
                    gradient[c, z, y, x] = (inverse_index_to_lps[0, c]*gx +
                                             inverse_index_to_lps[1, c]*gy + inverse_index_to_lps[2, c]*gz)
    return gradient


def central_gradient(data, inverse_index_to_lps):
    if data.size >= _PARALLEL_VALUES and get_num_threads() > 1:
        return _central_gradient_parallel(data, inverse_index_to_lps)
    return _central_gradient_serial(data, inverse_index_to_lps)


@njit(cache=True, fastmath=False)
def _mix_field_and_zero_boundary_serial(smoothed, original, weight):
    output = np.empty(smoothed.shape, dtype=np.float64)
    _, nz, ny, nx = original.shape
    for c in range(3):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if x == 0 or y == 0 or z == 0 or x == nx-1 or y == ny-1 or z == nz-1:
                        output[c, z, y, x] = 0.0
                    else:
                        output[c, z, y, x] = smoothed[c, z, y, x]*weight + original[c, z, y, x]*(1-weight)
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _mix_field_and_zero_boundary_parallel(smoothed, original, weight):
    output = np.empty(smoothed.shape, dtype=np.float64)
    _, nz, ny, nx = original.shape
    for task in prange(3 * nz):
        c = np.int64(task) // nz
        z = np.int64(task) % nz
        for y in range(ny):
            for x in range(nx):
                if x == 0 or y == 0 or z == 0 or x == nx-1 or y == ny-1 or z == nz-1:
                    output[c, z, y, x] = 0.0
                else:
                    output[c, z, y, x] = smoothed[c, z, y, x]*weight + original[c, z, y, x]*(1-weight)
    return output


def mix_field_and_zero_boundary(smoothed, original, weight):
    if original.size >= _PARALLEL_VALUES and get_num_threads() > 1:
        return _mix_field_and_zero_boundary_parallel(smoothed, original, weight)
    return _mix_field_and_zero_boundary_serial(smoothed, original, weight)


@njit(cache=True,fastmath=False,parallel=True)
def convolve_field_into(data,kernel,dimension,output,use_parallel):
    nz,ny=data.shape[1:3]
    if use_parallel:
        for task in prange(data.shape[0]*nz*ny):
            row=np.int64(task)
            _field_axis_row(data,kernel,dimension,row//(nz*ny),(row//ny)%nz,row%ny,output)
    else:
        for c in range(data.shape[0]):
            for z in range(nz):
                for y in range(ny):
                    _field_axis_row(data,kernel,dimension,c,z,y,output)
    return output


@njit(cache=True,fastmath=False,inline="always")
def _mix_owned_row(data,original,weight,c,z,y,output):
    _,nz,ny,nx=data.shape
    for x in range(nx):
        if x==0 or y==0 or z==0 or x==nx-1 or y==ny-1 or z==nz-1:
            output[c,z,y,x]=0.0
        else:
            output[c,z,y,x]=data[c,z,y,x]*weight+original[c,z,y,x]*(1-weight)


@njit(cache=True,fastmath=False,parallel=True)
def mix_field_inplace(data,original,weight,use_parallel):
    _,nz,ny,nx=data.shape
    if use_parallel:
        for task in prange(3*nz*ny):
            row=np.int64(task)
            _mix_owned_row(data,original,weight,row//(nz*ny),(row//ny)%nz,row%ny,data)
    else:
        for c in range(3):
            for z in range(nz):
                for y in range(ny):
                    _mix_owned_row(data,original,weight,c,z,y,data)
    return data
