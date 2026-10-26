"""Cache image cells whose full half-spacing gradient support is zero."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _zero_support(data):
    nz, ny, nx = data.shape
    along_x = np.empty(data.shape, np.bool_)
    along_y = np.empty(data.shape, np.bool_)
    result = np.empty(data.shape, np.bool_)
    for task in prange(nz * ny):
        row = np.int64(task)
        z, y = row // ny, row % ny
        for x in range(nx):
            empty = True
            for xx in range(max(0, x - 1), min(nx, x + 3)):
                if data[z, y, xx] != np.float32(0):
                    empty = False
                    break
            along_x[z, y, x] = empty
    for task in prange(nz * ny):
        row = np.int64(task)
        z, y = row // ny, row % ny
        for x in range(nx):
            empty = True
            for yy in range(max(0, y - 1), min(ny, y + 3)):
                if not along_x[z, yy, x]:
                    empty = False
                    break
            along_y[z, y, x] = empty
    for task in prange(nz * ny):
        row = np.int64(task)
        z, y = row // ny, row % ny
        for x in range(nx):
            empty = True
            for zz in range(max(0, z - 1), min(nz, z + 3)):
                if not along_y[zz, y, x]:
                    empty = False
                    break
            result[z, y, x] = empty
    return result


def zero_support32(data):
    """Prepare once for finite gradient shifts of at most one voxel per axis.

    For a cell with lower corner i, every shifted trilinear support lies in
    [i-1,i+2] on each axis. Cropped edge ranges include the clamped samples.
    NaN and infinity count as nonzero, so they never enable this shortcut.
    """
    return _zero_support(data)
