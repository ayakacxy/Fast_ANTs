"""FP32 paired CC with stable x-plane queue statistics.

Each clipped yz plane uses its first valid sample as a local coordinate
anchor and computes compensated two-pass centered statistics. A bounded ring
queue retains those planes as x advances. The first nonempty plane anchors
each neighborhood; Chan merges in increasing x order use coordinate
differences without forming large absolute means or raw sums-minus-mean-squared.
All image/statistic arithmetic and stored states are ordinary FP32. Counts
remain integers. Independent rows run in parallel, followed by the existing
serial compensated score traversal. The ordinary-FP32 registration wrapper
uses this implementation; the direct window traversal remains a reference.
"""

import numpy as np
from numba import njit, prange

from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import ONE
from fast_ants.backends.cpu.fp32.kernels import TWO
from fast_ants.backends.cpu.fp32.kernels import CC_EPS
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.kernels import inside32


@njit(cache=True, fastmath=False)
def _plane_stats32(fixed, moving, valid, xx, z0, z1, y0, y1):
    """Count, local anchors, residual means and centered norms/cross product."""
    count = 0
    anchor_f, anchor_m = ZERO, ZERO
    sf, sm, cf, cm = ZERO, ZERO, ZERO, ZERO
    for zz in range(z0, z1):
        for yy in range(y0, y1):
            if valid[zz, yy, xx]:
                if count == 0:
                    anchor_f, anchor_m = fixed[zz, yy, xx], moving[zz, yy, xx]
                sf, cf = add32(sf, cf, fixed[zz, yy, xx] - anchor_f)
                sm, cm = add32(sm, cm, moving[zz, yy, xx] - anchor_m)
                count += 1
    if count == 0:
        return count, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
    mean_f, mean_m = sf / np.float32(count), sm / np.float32(count)
    ff, mm, fm, cff, cmm, cfm = ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
    for zz in range(z0, z1):
        for yy in range(y0, y1):
            if valid[zz, yy, xx]:
                f = (fixed[zz, yy, xx] - anchor_f) - mean_f
                m = (moving[zz, yy, xx] - anchor_m) - mean_m
                ff, cff = add32(ff, cff, f * f)
                mm, cmm = add32(mm, cmm, m * m)
                fm, cfm = add32(fm, cfm, f * m)
    return count, anchor_f, anchor_m, mean_f, mean_m, ff, mm, fm


@njit(cache=True, fastmath=False)
def _fill_plane32(fixed, moving, valid, xx, z0, z1, y0, y1,
                  counts, planes):
    slot = xx % counts.size
    count, anchor_f, anchor_m, mean_f, mean_m, ff, mm, fm = _plane_stats32(
        fixed, moving, valid, xx, z0, z1, y0, y1)
    counts[slot] = count
    planes[slot, 0], planes[slot, 1] = anchor_f, anchor_m
    planes[slot, 2], planes[slot, 3] = mean_f, mean_m
    planes[slot, 4], planes[slot, 5], planes[slot, 6] = ff, mm, fm


@njit(cache=True, fastmath=False, inline='always')
def _pair_shift32(fixed, moving, x, y, z, fixed_anchor, moving_anchor):
    """Two anchored interpolants sharing coordinates, with original FP32 order."""
    nz, ny, nx = fixed.shape
    x = min(max(x, ZERO), np.float32(nx - 1))
    y = min(max(y, ZERO), np.float32(ny - 1))
    z = min(max(z, ZERO), np.float32(nz - 1))
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
    dx, dy, dz = x - np.float32(x0), y - np.float32(y0), z - np.float32(z0)
    a = (fixed[z0, y0, x0] - fixed_anchor) + (fixed[z0, y0, x1] - fixed[z0, y0, x0]) * dx
    b = (fixed[z0, y1, x0] - fixed_anchor) + (fixed[z0, y1, x1] - fixed[z0, y1, x0]) * dx
    c = (fixed[z1, y0, x0] - fixed_anchor) + (fixed[z1, y0, x1] - fixed[z1, y0, x0]) * dx
    d = (fixed[z1, y1, x0] - fixed_anchor) + (fixed[z1, y1, x1] - fixed[z1, y1, x0]) * dx
    lo, hi = a + (b - a) * dy, c + (d - c) * dy
    fvalue = lo + (hi - lo) * dz
    a = (moving[z0, y0, x0] - moving_anchor) + (moving[z0, y0, x1] - moving[z0, y0, x0]) * dx
    b = (moving[z0, y1, x0] - moving_anchor) + (moving[z0, y1, x1] - moving[z0, y1, x0]) * dx
    c = (moving[z1, y0, x0] - moving_anchor) + (moving[z1, y0, x1] - moving[z1, y0, x0]) * dx
    d = (moving[z1, y1, x0] - moving_anchor) + (moving[z1, y1, x1] - moving[z1, y1, x0]) * dx
    lo, hi = a + (b - a) * dy, c + (d - c) * dy
    mvalue = lo + (hi - lo) * dz
    return fvalue, mvalue


@njit(cache=True, fastmath=False, inline='always')
def _gradient_updates32(fixed, moving, shift, spacing, z, y, x,
                        forward_weight, reverse_weight, forward, reverse):
    nz, ny, nx = fixed.shape
    fixed_anchor, moving_anchor = fixed[z, y, x], moving[z, y, x]
    for c in range(3):
        dx, dy, dz = shift[0, c], shift[1, c], shift[2, c]
        px, py, pz = np.float32(x) + dx, np.float32(y) + dy, np.float32(z) + dz
        mx, my, mz = np.float32(x) - dx, np.float32(y) - dy, np.float32(z) - dz
        gf, gm = ZERO, ZERO
        if inside32(px, py, pz, nx, ny, nz) and inside32(mx, my, mz, nx, ny, nz):
            pf, pm = _pair_shift32(fixed, moving, px, py, pz, fixed_anchor, moving_anchor)
            nf, nm = _pair_shift32(fixed, moving, mx, my, mz, fixed_anchor, moving_anchor)
            gf, gm = (pf - nf) / spacing[c], (pm - nm) / spacing[c]
        forward[c, z, y, x] = forward_weight * gm
        reverse[c, z, y, x] = reverse_weight * gf


@njit(cache=True, fastmath=False)
def _finish32(scores, selected):
    total, total_correction = ZERO, ZERO
    centres = 0
    for z in range(scores.shape[0]):
        for y in range(scores.shape[1]):
            for x in range(scores.shape[2]):
                if selected[z, y, x]:
                    total, total_correction = add32(total, total_correction, scores[z, y, x])
                    centres += 1
    value = total / np.float32(centres) if centres else ZERO
    return value, centres


@njit(cache=True, fastmath=False)
def _row32(fixed, moving, fixed_gradient, moving_gradient, valid, radius,
           shift, spacing, z, y, forward, reverse, scores, selected):
    nz, ny, nx = fixed.shape
    # Clamp to the actual axes before forming the queue. Radii larger than
    # an image dimension retain the same clipped-window semantics and do not
    # allocate a queue proportional to the requested radius.
    rx, ry, rz = min(radius, nx - 1), min(radius, ny - 1), min(radius, nz - 1)
    width = min(nx, 2 * rx + 1)
    counts = np.empty(width, np.int64)
    planes = np.empty((width, 7), np.float32)
    z0, z1 = max(0, z - rz), min(nz, z + rz + 1)
    y0, y1 = max(0, y - ry), min(ny, y + ry + 1)
    for xx in range(min(nx, rx + 1)):
        _fill_plane32(fixed, moving, valid, xx, z0, z1, y0, y1,
                      counts, planes)
    for x in range(nx):
        incoming = x + rx
        if x > 0 and incoming < nx:
            _fill_plane32(fixed, moving, valid, incoming, z0, z1, y0, y1,
                          counts, planes)
        if not valid[z, y, x]:
            continue
        count = 0
        anchor_f, anchor_m = ZERO, ZERO
        mean_f, mean_m, ff, mm, fm = ZERO, ZERO, ZERO, ZERO, ZERO
        for xx in range(max(0, x - rx), min(nx, x + rx + 1)):
            slot = xx % width
            plane_count = counts[slot]
            if plane_count == 0:
                continue
            if count == 0:
                count = plane_count
                anchor_f, anchor_m = planes[slot, 0], planes[slot, 1]
                mean_f, mean_m = planes[slot, 2], planes[slot, 3]
                ff, mm, fm = planes[slot, 4], planes[slot, 5], planes[slot, 6]
            else:
                merged_count = count + plane_count
                fraction = np.float32(plane_count) / np.float32(merged_count)
                cross_weight = np.float32(count) * np.float32(plane_count) / np.float32(merged_count)
                # Anchors are coordinate origins, not high/low parts of a
                # software-expanded number. Both terms stay small for a
                # narrow valid intensity range even when its DC is large.
                # Invalid samples never choose a plane/window anchor.
                delta_f = (planes[slot, 0] - anchor_f) + (planes[slot, 2] - mean_f)
                delta_m = (planes[slot, 1] - anchor_m) + (planes[slot, 3] - mean_m)
                mean_f += delta_f * fraction
                mean_m += delta_m * fraction
                ff = ff + planes[slot, 4] + delta_f * delta_f * cross_weight
                mm = mm + planes[slot, 5] + delta_m * delta_m * cross_weight
                fm = fm + planes[slot, 6] + delta_f * delta_m * cross_weight
                count = merged_count
        if count == 0:
            continue
        product = ff * mm
        score = ONE
        if abs(product) > CC_EPS:
            score = fm * fm / product
        if ff > CC_EPS and mm > CC_EPS:
            centre_f = (fixed[z, y, x] - anchor_f) - mean_f
            centre_m = (moving[z, y, x] - anchor_m) - mean_m
            forward_weight = TWO * fm / product * (centre_f - fm / mm * centre_m)
            reverse_weight = TWO * fm / product * (centre_m - fm / ff * centre_f)
            if fixed_gradient is None:
                # None is a separate Numba specialization, so this path never
                # creates or reads volume-sized gradient arrays. Geometry and
                # support checks are shared by both images at each centre.
                _gradient_updates32(fixed, moving, shift, spacing, z, y, x,
                                    forward_weight, reverse_weight, forward, reverse)
            else:
                for c in range(3):
                    forward[c, z, y, x] = forward_weight * moving_gradient[c, z, y, x]
                    reverse[c, z, y, x] = reverse_weight * fixed_gradient[c, z, y, x]
        scores[z, y, x] = -score
        selected[z, y, x] = True


@njit(cache=True, fastmath=False, parallel=True)
def _work32(fixed, moving, fixed_gradient, moving_gradient, valid, radius):
    """Internal worker also returns per-centre scores for independent tests."""
    nz, ny, nx = fixed.shape
    forward = np.zeros((3, nz, ny, nx), np.float32)
    reverse = np.zeros((3, nz, ny, nx), np.float32)
    scores = np.zeros(fixed.shape, np.float32)
    selected = np.zeros(fixed.shape, np.bool_)
    shift, spacing = np.empty((0, 0), np.float32), np.empty(0, np.float32)
    if nx > 0:
        for task in prange(nz * ny):
            row = np.int64(task)
            _row32(fixed, moving, fixed_gradient, moving_gradient, valid, radius,
                   shift, spacing, row // ny, row % ny, forward, reverse, scores, selected)
    value, centres = _finish32(scores, selected)
    return value, centres, forward, reverse, scores


def cc_queue_pair32(fixed, moving, fixed_gradient, moving_gradient, valid, radius):
    """Return the existing paired-CC (value, centres, forward, reverse) tuple.

    The input contract is unchanged: two scalar FP32 images, two FP32
    gradients shaped (3,z,y,x), a Boolean valid mask and a nonnegative integer
    radius. Statistics are regrouped by stable plane blocks, so the result
    need not be bit identical to the direct cubic traversal.
    """
    value, centres, forward, reverse, _ = _work32(
        fixed, moving, fixed_gradient, moving_gradient, valid, radius)
    return value, centres, forward, reverse


# Allows a source-selectable benchmark to keep the existing import name.
cc_pair32 = cc_queue_pair32


@njit(cache=True, fastmath=False, parallel=True)
def _work_gradient32(fixed, moving, valid, radius, shift, spacing):
    """Queue statistics and lazy physical gradients, without gradient volumes."""
    nz, ny, nx = fixed.shape
    forward = np.zeros((3, nz, ny, nx), np.float32)
    reverse = np.zeros((3, nz, ny, nx), np.float32)
    scores = np.zeros(fixed.shape, np.float32)
    selected = np.zeros(fixed.shape, np.bool_)
    if nx > 0:
        for task in prange(nz * ny):
            row = np.int64(task)
            _row32(fixed, moving, None, None, valid, radius, shift, spacing,
                   row // ny, row % ny, forward, reverse, scores, selected)
    value, centres = _finish32(scores, selected)
    return value, centres, forward, reverse, scores


def cc_pair_gradient32(fixed, moving, valid, radius, shift, spacing):
    """Paired CC with inline half-spacing LPS gradients at active centres.

    Images are FP32 [z,y,x] on one grid. ``shift`` is its FP32 3x3
    half-spacing-to-index matrix and ``spacing`` its FP32 three-vector.
    Statistics, gradients, updates and score reduction retain the existing
    FP32 formulas and order; only redundant gradient storage/work is removed.
    """
    value, centres, forward, reverse, _ = _work_gradient32(
        fixed, moving, valid, radius, shift, spacing)
    return value, centres, forward, reverse
