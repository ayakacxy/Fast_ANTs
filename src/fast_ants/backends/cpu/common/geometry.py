import numpy as np
from numba import get_num_threads, njit, prange


# Parallelise only independent points/lines. Small requests and one-thread
# execution retain the original serial loops without a thread-pool launch.
# Serial and parallel functions have separate names and source locations so
# Numba disk caches cannot confuse their different compilation flags.
# Dispatch stays in Python: a jitted wrapper that references a parallel kernel
# embeds dynamic runtime globals and cannot itself be cached by Numba.
_PARALLEL_POINTS = 16384

# The four coordinate kernels below permit only contraction to hardware FMA.
# Reassociation, relaxed NaN/inf rules and approximate division stay disabled.
# This matches ITK's discrete-index and physical-point arithmetic: the ULPs
# matter when an interpolated fixed intensity lies on a Mattes bin boundary.


@njit(cache=True, fastmath=False)
def _affine_points_serial(points, linear, center, translation):
    output = np.empty(points.shape, dtype=np.float64)
    for n in range(points.shape[0]):
        for i in range(3):
            value = center[i] + translation[i]
            for j in range(3):
                value += linear[i, j] * (points[n, j] - center[j])
            output[n, i] = value
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _affine_points_parallel(points, linear, center, translation):
    output = np.empty(points.shape, dtype=np.float64)
    for n in prange(points.shape[0]):
        for i in range(3):
            value = center[i] + translation[i]
            for j in range(3):
                value += linear[i, j] * (points[n, j] - center[j])
            output[n, i] = value
    return output


def affine_points(points, linear, center, translation):
    if points.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _affine_points_parallel(points, linear, center, translation)
    return _affine_points_serial(points, linear, center, translation)


@njit(cache=True, fastmath={"contract"})
def _physical_to_indices_serial(points, inverse_linear, origin):
    output = np.empty(points.shape, dtype=np.float64)
    for n in range(points.shape[0]):
        for i in range(3):
            value = 0.0
            for j in range(3):
                value += inverse_linear[i, j] * (points[n, j] - origin[j])
            output[n, i] = value
    return output


@njit(cache=True, fastmath={"contract"}, parallel=True)
def _physical_to_indices_parallel(points, inverse_linear, origin):
    output = np.empty(points.shape, dtype=np.float64)
    for n in prange(points.shape[0]):
        for i in range(3):
            value = 0.0
            for j in range(3):
                value += inverse_linear[i, j] * (points[n, j] - origin[j])
            output[n, i] = value
    return output


def physical_to_indices(points, inverse_linear, origin):
    if points.shape[0] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _physical_to_indices_parallel(points, inverse_linear, origin)
    return _physical_to_indices_serial(points, inverse_linear, origin)


@njit(cache=True, fastmath={"contract"})
def _grid_points_serial(index_to_lps, shape_zyx, start_z, stop_z):
    nz, ny, nx = shape_zyx
    output = np.empty(((stop_z - start_z) * ny * nx, 3), dtype=np.float64)
    n = 0
    for z in range(start_z, stop_z):
        for y in range(ny):
            for x in range(nx):
                for i in range(3):
                    output[n, i] = (index_to_lps[i, 3] + index_to_lps[i, 0] * x +
                                    index_to_lps[i, 1] * y + index_to_lps[i, 2] * z)
                n += 1
    return output


@njit(cache=True, fastmath={"contract"}, parallel=True)
def _grid_points_parallel(index_to_lps, shape_zyx, start_z, stop_z):
    nz, ny, nx = shape_zyx
    output = np.empty(((stop_z - start_z) * ny * nx, 3), dtype=np.float64)
    for task in prange((stop_z - start_z) * ny * nx):
        n = np.int64(task)
        z = start_z + n // (ny * nx)
        y = (n // nx) % ny
        x = n % nx
        for i in range(3):
            output[n, i] = (index_to_lps[i, 3] + index_to_lps[i, 0] * x +
                            index_to_lps[i, 1] * y + index_to_lps[i, 2] * z)
    return output


def grid_points(index_to_lps, shape_zyx, start_z, stop_z):
    if (stop_z - start_z) * shape_zyx[1] * shape_zyx[2] >= _PARALLEL_POINTS and get_num_threads() > 1:
        return _grid_points_parallel(index_to_lps, shape_zyx, start_z, stop_z)
    return _grid_points_serial(index_to_lps, shape_zyx, start_z, stop_z)
