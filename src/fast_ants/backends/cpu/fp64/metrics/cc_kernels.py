"""Dense ANTs neighbourhood CC, modified from ITK (NumFOCUS, Apache-2.0).

The x-plane queue follows the source accumulation order. This is the ITK
centre-local displacement update, not the derivative of all overlapping costs.
"""

import numpy as np
from numba import get_num_threads, njit, prange


@njit(cache=True, fastmath=False)
def _fill_plane(values, fixed, moving, valid, x, y, z, ry, rz):
    # Reuse a row's queue storage instead of allocating six values per voxel.
    for j in range(6):
        values[j] = 0.0
    nz, ny, nx = fixed.shape
    if x < 0 or x >= nx:
        return
    for zz in range(max(0, z-rz), min(nz, z+rz+1)):
        for yy in range(max(0, y-ry), min(ny, y+ry+1)):
            if valid[zz, yy, x]:
                f, m = fixed[zz, yy, x], moving[zz, yy, x]
                values[0] += f*f
                values[1] += m*m
                values[2] += f
                values[3] += m
                values[4] += f*m
                values[5] += 1.0


@njit(cache=True, fastmath=False)
def _cc_row(fixed, moving, gradient, valid, radius_xyz, keep_statistics,
            z, y, update, statistics, scores, evaluated):
    nz, ny, nx = fixed.shape
    rx, ry, rz = radius_xyz
    width = 2*rx+1
    queue = np.empty((width, 6), dtype=np.float64)
    totals = np.empty(6, dtype=np.float64)
    for i in range(width):
        _fill_plane(queue[i], fixed, moving, valid, i-rx, y, z, ry, rz)
    head = 0
    epsilon = np.finfo(np.float64).eps
    for x in range(nx):
        if x > 0:
            _fill_plane(queue[head], fixed, moving, valid, x+rx, y, z, ry, rz)
            head = (head+1) % width
        for j in range(6):
            totals[j] = 0.0
        for i in range(width):
            plane = (head+i) % width
            for j in range(6):
                totals[j] += queue[plane, j]
        count = totals[5]
        if count <= 0.0 or not valid[z, y, x]:
            continue
        evaluated[z, y, x] = True
        mean_f, mean_m = totals[2]/count, totals[3]/count
        ff = totals[0]-mean_f*totals[2]-mean_f*totals[2]+count*mean_f*mean_f
        mm = totals[1]-mean_m*totals[3]-mean_m*totals[3]+count*mean_m*mean_m
        fm = totals[4]-mean_m*totals[2]-mean_f*totals[3]+count*mean_m*mean_f
        product = ff*mm
        score = 1.0
        if abs(product) > epsilon:
            score = fm*fm/product
        if ff > epsilon and mm > epsilon:
            weight = 2.0*fm/product*(fixed[z, y, x]-mean_f-fm/mm*(moving[z, y, x]-mean_m))
            for c in range(3):
                update[c, z, y, x] = weight*gradient[c, z, y, x]
        scores[z, y, x] = score
        if keep_statistics:
            statistics[0, z, y, x] = count
            statistics[1, z, y, x] = mean_f
            statistics[2, z, y, x] = mean_m
            statistics[3, z, y, x] = ff
            statistics[4, z, y, x] = mm
            statistics[5, z, y, x] = fm
            statistics[6, z, y, x] = score


@njit(cache=True, fastmath=False)
def _cc_finish(scores, evaluated):
    # Preserve the original z/y/x scalar accumulation and stopping trajectory.
    values, flags = scores.ravel(), evaluated.ravel()
    value_sum, valid_centres = 0.0, 0
    for n in range(values.size):
        if flags[n]:
            value_sum -= values[n]
            valid_centres += 1
    return value_sum, valid_centres


@njit(cache=True, fastmath=False)
def _local_cc_serial(fixed, moving, gradient, valid, radius_xyz, keep_statistics):
    nz, ny, nx = fixed.shape
    update = np.zeros((3, nz, ny, nx), dtype=np.float64)
    # count, meanF, meanM, varF, varM, covFM, score; not allocated in normal runs.
    statistics = np.zeros((7, nz, ny, nx), dtype=np.float64) if keep_statistics else np.empty((0, 0, 0, 0))
    scores = np.empty((nz, ny, nx), dtype=np.float64)
    evaluated = np.zeros((nz, ny, nx), dtype=np.bool_)
    for z in range(nz):
        for y in range(ny):
            _cc_row(fixed, moving, gradient, valid, radius_xyz, keep_statistics,
                    z, y, update, statistics, scores, evaluated)
    value_sum, valid_centres = _cc_finish(scores, evaluated)
    return value_sum, valid_centres, update, statistics


@njit(cache=True, fastmath=False, parallel=True)
def _local_cc_parallel(fixed, moving, gradient, valid, radius_xyz, keep_statistics):
    nz, ny, nx = fixed.shape
    update = np.zeros((3, nz, ny, nx), dtype=np.float64)
    statistics = np.zeros((7, nz, ny, nx), dtype=np.float64) if keep_statistics else np.empty((0, 0, 0, 0))
    scores = np.empty((nz, ny, nx), dtype=np.float64)
    evaluated = np.zeros((nz, ny, nx), dtype=np.bool_)
    for task in prange(nz*ny):
        row = np.int64(task)
        _cc_row(fixed, moving, gradient, valid, radius_xyz, keep_statistics,
                row//ny, row%ny, update, statistics, scores, evaluated)
    value_sum, valid_centres = _cc_finish(scores, evaluated)
    return value_sum, valid_centres, update, statistics


def local_cc(fixed, moving, gradient, valid, radius_xyz, keep_statistics):
    if fixed.size >= 16384 and get_num_threads() > 1:
        return _local_cc_parallel(fixed, moving, gradient, valid, radius_xyz, keep_statistics)
    return _local_cc_serial(fixed, moving, gradient, valid, radius_xyz, keep_statistics)
