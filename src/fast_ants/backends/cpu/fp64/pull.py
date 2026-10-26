"""FP64 fused physical pull and transform rasterisation kernels.

Exactly matching output/field grids use their stored voxel displacements
directly. Other geometry conversion and interpolation retain the unfused
operation order. Only independent rows are parallelised; scratch is per row.
"""

import math

import numpy as np
from numba import get_num_threads, njit, prange

from fast_ants.backends.cpu.fp64.sampling_kernels import inside
from fast_ants.backends.cpu.fp64.sampling_kernels import linear_value


@njit(cache=True, fastmath=False, inline="always")
def _indices(point, inverse_linear, origin, index):
    for i in range(3):
        value = 0.0
        for j in range(3):
            value += inverse_linear[i, j] * (point[j] - origin[j])
        index[i] = value


@njit(cache=True, fastmath=False, inline="always")
def _affine(point, linear, center, translation, temporary):
    for i in range(3):
        value = center[i] + translation[i]
        for j in range(3):
            value += linear[i, j] * (point[j] - center[j])
        temporary[i] = value
    for i in range(3):
        point[i] = temporary[i]


@njit(cache=True, fastmath=False, inline="always")
def _field(point, field, inverse_linear, origin, index):
    _indices(point, inverse_linear, origin, index)
    _, nz, ny, nx = field.shape
    valid = inside(index[0], index[1], index[2], nx, ny, nz)
    if valid:
        # Reuse the same clamped coordinates and fractions for all components;
        # each component retains linear_value's exact interpolation expression.
        x = min(max(index[0], 0.0), nx - 1)
        y = min(max(index[1], 0.0), ny - 1)
        z = min(max(index[2], 0.0), nz - 1)
        x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
        dx, dy, dz = x - x0, y - y0, z - z0
        for c in range(3):
            a = field[c, z0, y0, x0] + (field[c, z0, y0, x1] - field[c, z0, y0, x0]) * dx
            b = field[c, z0, y1, x0] + (field[c, z0, y1, x1] - field[c, z0, y1, x0]) * dx
            lo = field[c, z1, y0, x0] + (field[c, z1, y0, x1] - field[c, z1, y0, x0]) * dx
            hi = field[c, z1, y1, x0] + (field[c, z1, y1, x1] - field[c, z1, y1, x0]) * dx
            lower, upper = a + (b - a) * dy, lo + (hi - lo) * dy
            point[c] = point[c] + (lower + (upper - lower) * dz)
    else:
        for c in range(3):
            # Retain the addition outside support: +0 is part of map_points.
            point[c] = point[c] + 0.0


@njit(cache=True, fastmath=False, inline="always")
def _finite_point(point):
    return np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2])


@njit(cache=True, fastmath=False, inline="always")
def _map(point, field, field_inverse, field_origin, linear, center,
         translation, mode, index, temporary):
    # Modes follow the actual application order; affine matrices are never
    # precomposed with geometry or displacement interpolation.
    if not _finite_point(point):
        return False
    if mode == 1 or mode == 4:
        _affine(point, linear, center, translation, temporary)
        if not _finite_point(point):
            return False
    if mode >= 2:
        _field(point, field, field_inverse, field_origin, index)
        if not _finite_point(point):
            return False
    if mode == 3:
        _affine(point, linear, center, translation, temporary)
    return _finite_point(point)


@njit(cache=True, fastmath=False, inline="always")
def _grid_point(matrix, x, y, z, point):
    for i in range(3):
        point[i] = (matrix[i, 3] + matrix[i, 0] * x +
                    matrix[i, 1] * y + matrix[i, 2] * z)


@njit(cache=True, fastmath=False)
def _pull_row(data, output, matrix, inverse, origin, field, field_inverse,
              field_origin, linear, center, translation, mode, default,
              nearest, z, y, direct_field=False):
    point = np.empty(3, np.float64)
    index = np.empty(3, np.float64)
    temporary = np.empty(3, np.float64)
    nz, ny, nx = data.shape
    finite = True
    for x in range(output.shape[2]):
        _grid_point(matrix, x, y, z, point)
        if direct_field:
            # The field is defined on this exact output grid and is the first
            # map. Integer voxel samples need no physical/index roundtrip or
            # trilinear interpolation. Keep world+u and any later affine in
            # their existing order; image sampling below is unchanged.
            mapped_finite = _finite_point(point)
            if mapped_finite:
                for c in range(3):
                    point[c] = point[c] + field[c, z, y, x]
                mapped_finite = _finite_point(point)
                if mapped_finite and mode == 3:
                    _affine(point, linear, center, translation, temporary)
                    mapped_finite = _finite_point(point)
        else:
            mapped_finite = _map(point, field, field_inverse, field_origin, linear,
                                 center, translation, mode, index, temporary)
        if not mapped_finite:
            finite = False
            output[z, y, x] = default
            continue
        _indices(point, inverse, origin, index)
        if inside(index[0], index[1], index[2], nx, ny, nz):
            if nearest:
                output[z, y, x] = data[int(math.floor(index[2] + .5)),
                                        int(math.floor(index[1] + .5)),
                                        int(math.floor(index[0] + .5))]
            else:
                output[z, y, x] = linear_value(data, index[0], index[1], index[2])
        else:
            output[z, y, x] = default
    return finite


@njit(cache=True, fastmath=False)
def _pull_serial(data, output, matrix, inverse, origin, field, field_inverse,
                 field_origin, linear, center, translation, mode, default,
                 nearest, finite, direct_field=False):
    nz, ny, nx = output.shape
    for z in range(nz):
        for y in range(ny):
            finite[z, y] = _pull_row(data, output, matrix, inverse, origin,
                                     field, field_inverse, field_origin, linear,
                                     center, translation, mode, default, nearest, z, y, direct_field)


@njit(cache=True, fastmath=False, parallel=True)
def _pull_parallel(data, output, matrix, inverse, origin, field, field_inverse,
                   field_origin, linear, center, translation, mode, default,
                   nearest, finite, direct_field=False):
    nz, ny, nx = output.shape
    for task in prange(nz * ny):
        row = np.int64(task)
        z, y = row // ny, row % ny
        finite[z, y] = _pull_row(data, output, matrix, inverse, origin,
                                 field, field_inverse, field_origin, linear,
                                 center, translation, mode, default, nearest, z, y, direct_field)


@njit(cache=True, fastmath=False)
def _field_row(output, matrix, field, field_inverse, field_origin, linear,
               center, translation, mode, z, y):
    point = np.empty(3, np.float64)
    original = np.empty(3, np.float64)
    index = np.empty(3, np.float64)
    temporary = np.empty(3, np.float64)
    finite = True
    for x in range(output.shape[3]):
        _grid_point(matrix, x, y, z, point)
        for c in range(3):
            original[c] = point[c]
        if not _map(point, field, field_inverse, field_origin, linear,
                    center, translation, mode, index, temporary):
            finite = False
        for c in range(3):
            output[c, z, y, x] = point[c] - original[c]
    return finite


@njit(cache=True, fastmath=False)
def _field_serial(output, matrix, field, field_inverse, field_origin, linear,
                  center, translation, mode, finite):
    _, nz, ny, nx = output.shape
    for z in range(nz):
        for y in range(ny):
            finite[z, y] = _field_row(output, matrix, field, field_inverse,
                                      field_origin, linear, center, translation, mode, z, y)


@njit(cache=True, fastmath=False, parallel=True)
def _field_parallel(output, matrix, field, field_inverse, field_origin, linear,
                    center, translation, mode, finite):
    _, nz, ny, nx = output.shape
    for task in prange(nz * ny):
        row = np.int64(task)
        z, y = row // ny, row % ny
        finite[z, y] = _field_row(output, matrix, field, field_inverse,
                                  field_origin, linear, center, translation, mode, z, y)


def _field_matches_output_grid(grid, parameters):
    """Exact geometry only: tolerance-based equality cannot skip sampling."""
    field, field_inverse, field_origin, _, _, _, mode = parameters
    return (mode in (2, 3) and field.shape[1:] == grid.shape_zyx and
            np.array_equal(field_inverse, grid.lps_to_index[:3, :3]) and
            np.array_equal(field_origin, grid.origin))


def pull_image64(data, grid, inverse, origin, parameters, default, nearest):
    output = np.empty(grid.shape_zyx, dtype=data.dtype if nearest else np.float64)
    finite = np.empty(grid.shape_zyx[:2], dtype=np.bool_)
    operation = _pull_parallel if output.size >= 16384 and get_num_threads() > 1 else _pull_serial
    operation(data, output, grid.index_to_lps, inverse, origin, *parameters,
              default, nearest, finite, _field_matches_output_grid(grid, parameters))
    if not finite.all():
        raise ValueError("Points must be finite (..., 3) physical or index coordinates")
    return output


def rasterize_transform64(grid, parameters):
    output = np.empty((3, *grid.shape_zyx), dtype=np.float64)
    finite = np.empty(grid.shape_zyx[:2], dtype=np.bool_)
    operation = _field_parallel if output.size >= 16384 and get_num_threads() > 1 else _field_serial
    operation(output, grid.index_to_lps, *parameters, finite)
    if not finite.all():
        raise ValueError("Points must be finite (..., 3) physical or index coordinates")
    return output
