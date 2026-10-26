"""ITK pixel-centre support: [-.5, size-.5), edge pixels extend to half cells."""

import math
import numpy as np
from numba import get_num_threads, njit, prange


# Parallelise only independent points/lines. Small requests and one-thread
# execution retain the original serial loops without a thread-pool launch.
# Serial and parallel functions have separate names and source locations so
# Numba disk caches cannot confuse their different compilation flags.
# Dispatch stays in Python: a jitted wrapper that references a parallel kernel
# embeds dynamic runtime globals and cannot itself be cached by Numba.
_PARALLEL_POINTS = 16384


@njit(cache=True, fastmath=False)
def inside(x, y, z, nx, ny, nz):
    return (-0.5 <= x < nx - 0.5 and -0.5 <= y < ny - 0.5 and -0.5 <= z < nz - 0.5)


@njit(cache=True, fastmath={"contract"})
def linear_value(data, x, y, z):
    # ITK's optimized scalar interpolation contracts each lerp to FP64 FMA.
    # Disabling just this contraction can change fixed Mattes bin membership
    # for quantized, unsmoothed MRI intensities at exact bin boundaries.
    nz, ny, nx = data.shape
    # The caller checks the half-voxel support. At image edges ITK extends the
    # nearest pixel, rather than mixing a zero-valued outside neighbour.
    x, y, z = min(max(x, 0.0), nx - 1), min(max(y, 0.0), ny - 1), min(max(z, 0.0), nz - 1)
    x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
    dx, dy, dz = x - x0, y - y0, z - z0
    a = data[z0, y0, x0] + (data[z0, y0, x1] - data[z0, y0, x0]) * dx
    b = data[z0, y1, x0] + (data[z0, y1, x1] - data[z0, y1, x0]) * dx
    c = data[z1, y0, x0] + (data[z1, y0, x1] - data[z1, y0, x0]) * dx
    d = data[z1, y1, x0] + (data[z1, y1, x1] - data[z1, y1, x0]) * dx
    lower, upper = a + (b - a) * dy, c + (d - c) * dy
    return lower + (upper - lower) * dz


@njit(cache=True, fastmath=False)
def _sample_linear_serial(data, indices_xyz, default_value):
    output = np.empty(indices_xyz.shape[0], dtype=np.float64)
    valid = np.empty(indices_xyz.shape[0], dtype=np.bool_)
    nz, ny, nx = data.shape
    for n in range(indices_xyz.shape[0]):
        x, y, z = indices_xyz[n, 0], indices_xyz[n, 1], indices_xyz[n, 2]
        valid[n] = inside(x, y, z, nx, ny, nz)
        output[n] = linear_value(data, x, y, z) if valid[n] else default_value
    return output, valid


@njit(cache=True, fastmath=False, parallel=True)
def _sample_linear_parallel(data, indices_xyz, default_value):
    output = np.empty(indices_xyz.shape[0], dtype=np.float64)
    valid = np.empty(indices_xyz.shape[0], dtype=np.bool_)
    nz, ny, nx = data.shape
    for n in prange(indices_xyz.shape[0]):
        x, y, z = indices_xyz[n, 0], indices_xyz[n, 1], indices_xyz[n, 2]
        valid[n] = inside(x, y, z, nx, ny, nz)
        output[n] = linear_value(data, x, y, z) if valid[n] else default_value
    return output, valid


def sample_linear(data, indices_xyz, default_value):
    if indices_xyz.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _sample_linear_parallel(data, indices_xyz, default_value)
    return _sample_linear_serial(data, indices_xyz, default_value)


@njit(cache=True, fastmath=False)
def _sample_nearest_serial(data, indices_xyz, default_value):
    output = np.empty(indices_xyz.shape[0], dtype=data.dtype)
    valid = np.empty(indices_xyz.shape[0], dtype=np.bool_)
    nz, ny, nx = data.shape
    for n in range(indices_xyz.shape[0]):
        x, y, z = indices_xyz[n, 0], indices_xyz[n, 1], indices_xyz[n, 2]
        valid[n] = inside(x, y, z, nx, ny, nz)
        if valid[n]:
            output[n] = data[int(math.floor(z + .5)), int(math.floor(y + .5)), int(math.floor(x + .5))]
        else:
            output[n] = default_value
    return output, valid


@njit(cache=True, fastmath=False, parallel=True)
def _sample_nearest_parallel(data, indices_xyz, default_value):
    output = np.empty(indices_xyz.shape[0], dtype=data.dtype)
    valid = np.empty(indices_xyz.shape[0], dtype=np.bool_)
    nz, ny, nx = data.shape
    for n in prange(indices_xyz.shape[0]):
        x, y, z = indices_xyz[n, 0], indices_xyz[n, 1], indices_xyz[n, 2]
        valid[n] = inside(x, y, z, nx, ny, nz)
        if valid[n]:
            output[n] = data[int(math.floor(z + .5)), int(math.floor(y + .5)), int(math.floor(x + .5))]
        else:
            output[n] = default_value
    return output, valid


def sample_nearest(data, indices_xyz, default_value):
    if indices_xyz.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _sample_nearest_parallel(data, indices_xyz, default_value)
    return _sample_nearest_serial(data, indices_xyz, default_value)


@njit(cache=True, fastmath=False)
def _sample_vector_serial(field, indices_xyz):
    output = np.zeros((indices_xyz.shape[0], 3), dtype=np.float64)
    valid = np.empty(indices_xyz.shape[0], dtype=np.bool_)
    _, nz, ny, nx = field.shape
    for n in range(indices_xyz.shape[0]):
        x, y, z = indices_xyz[n, 0], indices_xyz[n, 1], indices_xyz[n, 2]
        valid[n] = inside(x, y, z, nx, ny, nz)
        if valid[n]:
            for c in range(3):
                output[n, c] = linear_value(field[c], x, y, z)
    return output, valid


@njit(cache=True, fastmath=False, parallel=True)
def _sample_vector_parallel(field, indices_xyz):
    output = np.zeros((indices_xyz.shape[0], 3), dtype=np.float64)
    valid = np.empty(indices_xyz.shape[0], dtype=np.bool_)
    _, nz, ny, nx = field.shape
    for n in prange(indices_xyz.shape[0]):
        x, y, z = indices_xyz[n, 0], indices_xyz[n, 1], indices_xyz[n, 2]
        valid[n] = inside(x, y, z, nx, ny, nz)
        if valid[n]:
            for c in range(3):
                output[n, c] = linear_value(field[c], x, y, z)
    return output, valid


def sample_vector(field, indices_xyz):
    if indices_xyz.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _sample_vector_parallel(field, indices_xyz)
    return _sample_vector_serial(field, indices_xyz)


@njit(cache=True, fastmath=False)
def _add_vectors_serial(points, vectors):
    output = np.empty(points.shape, dtype=np.float64)
    for n in range(points.shape[0]):
        for c in range(3):
            output[n, c] = points[n, c] + vectors[n, c]
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _add_vectors_parallel(points, vectors):
    output = np.empty(points.shape, dtype=np.float64)
    for n in prange(points.shape[0]):
        for c in range(3):
            output[n, c] = points[n, c] + vectors[n, c]
    return output


def add_vectors(points, vectors):
    if points.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _add_vectors_parallel(points, vectors)
    return _add_vectors_serial(points, vectors)
