"""Ordinary FP32 bidirectional CC with shared centered window statistics.

Both directions use the same covariance, exchanged variances/means and the
same score. Keep the reference cc32 traversal and compensated accumulation
order, including its serial scalar reduction. Parallelize independent z
planes only; do not introduce FP64 dense arithmetic or fast-math reassociation.
"""

import numpy as np
from numba import njit, prange

from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import ONE
from fast_ants.backends.cpu.fp32.kernels import TWO
from fast_ants.backends.cpu.fp32.kernels import CC_EPS
from fast_ants.backends.cpu.fp32.kernels import add32


@njit(cache=True, fastmath=False, parallel=True)
def cc_pair32(fixed, moving, fixed_gradient, moving_gradient, valid, radius):
    """Return (value, centres, fixed_to_moving_update, moving_to_fixed_update).

    Inputs follow cc32: scalar images and a Boolean mask have shape (z,y,x),
    gradients have shape (3,z,y,x), and radius is one nonnegative integer.
    Images, gradients and both returned update fields use native FP32.
    """
    nz, ny, nx = fixed.shape
    forward = np.zeros((3, nz, ny, nx), np.float32)
    reverse = np.zeros((3, nz, ny, nx), np.float32)
    scores = np.zeros(fixed.shape, np.float32)
    selected = np.zeros(fixed.shape, np.bool_)
    for task in prange(nz):
        z = np.int64(task)
        for y in range(ny):
            for x in range(nx):
                if not valid[z, y, x]:
                    continue
                f0, m0 = fixed[z, y, x], moving[z, y, x]
                sf, sm, cf, cm = ZERO, ZERO, ZERO, ZERO
                count = 0
                for zz in range(max(0, z-radius), min(nz, z+radius+1)):
                    for yy in range(max(0, y-radius), min(ny, y+radius+1)):
                        for xx in range(max(0, x-radius), min(nx, x+radius+1)):
                            if valid[zz, yy, xx]:
                                sf, cf = add32(sf, cf, fixed[zz, yy, xx]-f0)
                                sm, cm = add32(sm, cm, moving[zz, yy, xx]-m0)
                                count += 1
                if count == 0:
                    continue
                mean_f, mean_m = sf/np.float32(count), sm/np.float32(count)
                ff, mm, fm, cff, cmm, cfm = ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
                for zz in range(max(0, z-radius), min(nz, z+radius+1)):
                    for yy in range(max(0, y-radius), min(ny, y+radius+1)):
                        for xx in range(max(0, x-radius), min(nx, x+radius+1)):
                            if valid[zz, yy, xx]:
                                f = (fixed[zz, yy, xx]-f0)-mean_f
                                m = (moving[zz, yy, xx]-m0)-mean_m
                                ff, cff = add32(ff, cff, f*f)
                                mm, cmm = add32(mm, cmm, m*m)
                                fm, cfm = add32(fm, cfm, f*m)
                product = ff*mm
                score = ONE
                if abs(product) > CC_EPS:
                    score = fm*fm/product
                if ff > CC_EPS and mm > CC_EPS:
                    forward_weight = TWO*fm/product*(-mean_f+fm/mm*mean_m)
                    reverse_weight = TWO*fm/product*(-mean_m+fm/ff*mean_f)
                    for c in range(3):
                        forward[c, z, y, x] = forward_weight*moving_gradient[c, z, y, x]
                        reverse[c, z, y, x] = reverse_weight*fixed_gradient[c, z, y, x]
                scores[z, y, x] = -score
                selected[z, y, x] = True
    total, total_correction = ZERO, ZERO
    centres = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if selected[z, y, x]:
                    total, total_correction = add32(total, total_correction, scores[z, y, x])
                    centres += 1
    value = total/np.float32(centres) if centres else ZERO
    return value, centres, forward, reverse
