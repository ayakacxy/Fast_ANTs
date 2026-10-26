"""Fixed-order reference reductions; no parallel reordering or fast-math."""

import math
import numpy as np
from numba import njit,prange,get_num_threads


@njit(cache=True, fastmath=False)
def sum_values(values, compensated=True):
    total, correction = 0.0, 0.0
    for i in range(values.size):
        value = values[i]
        updated = total + value
        if compensated:
            if abs(total) >= abs(value):
                correction += (total-updated)+value
            else:
                correction += (value-updated)+total
        total = updated
    return total + correction


@njit(cache=True, fastmath=False)
def maximum_norm(field, spacing_xyz):
    maximum = 0.0
    for z in range(field.shape[1]):
        for y in range(field.shape[2]):
            for x in range(field.shape[3]):
                norm = 0.0
                for c in range(3):
                    component = field[c, z, y, x]/spacing_xyz[c]
                    norm += component*component
                maximum = max(maximum, math.sqrt(norm))
    return maximum


@njit(cache=True, fastmath=False)
def minmax(values):
    minimum, maximum = values[0], values[0]
    for i in range(1, values.size):
        minimum, maximum = min(minimum, values[i]), max(maximum, values[i])
    return minimum, maximum


@njit(cache=True, fastmath=False)
def rescale(values):
    minimum, maximum = minmax(values.ravel())
    output = np.empty(values.shape, dtype=np.float64)
    flat_in, flat_out = values.ravel(), output.ravel()
    for i in range(values.size):
        flat_out[i] = (flat_in[i]-minimum)/(maximum-minimum) if maximum > minimum else 0.0
    return output


@njit(cache=True,fastmath=False,parallel=True)
def _finite_blocks(values):
    src=values.ravel()
    width=16384
    flags=np.empty((src.size+width-1)//width,np.bool_)
    for block in prange(flags.size):
        good=True
        for i in range(block*width,min((block+1)*width,src.size)):
            good=good & math.isfinite(src[i])
        flags[block]=good
    return flags


def all_finite(values):
    # Values arrive C-contiguous. Only boolean flags are reduced; no floating
    # arithmetic or error threshold changes, and no full boolean volume.
    if values.size >= 49152 and get_num_threads() > 1:
        return bool(_finite_blocks(values).all())
    return bool(np.isfinite(values).all())
