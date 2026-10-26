"""Unmasked CC with four independent compensated plane statistics per vector.

The two-pass centered statistics and Chan merge follow cc_queue.py. Vector
lanes correspond to adjacent x planes; each lane keeps the scalar yz order.
"""

import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.core import cgutils
from numba.extending import intrinsic

from fast_ants.backends.cpu.fp32.kernels import ZERO, ONE, TWO, CC_EPS, add32
from fast_ants.backends.cpu.fp32.metrics.cc_queue import _gradient_updates32
from fast_ants.backends.cpu.fp32.metrics.parzen import _VECTOR, _splat, _load, _store, _add


@intrinsic
def plane_stats4(typingctx, fixed, moving, first, z0, z1, y0, y1, planes):
    def codegen(context, builder, signature, arguments):
        images = [context.make_array(signature.args[i])(context, builder, arguments[i])
                  for i in (0, 1)]
        output = context.make_array(signature.args[7])(context, builder, arguments[7])
        _, ny, nx = cgutils.unpack_tuple(builder, images[0].shape, 3)
        x, za, zb, ya, yb = [context.cast(builder, arguments[i], signature.args[i], types.intp)
                            for i in (2, 3, 4, 5, 6)]
        def index(zz, yy):
            return builder.add(builder.mul(builder.add(builder.mul(zz, ny), yy), nx), x)
        anchors = [_load(builder, image.data, index(za, ya)) for image in images]
        zero = ir.Constant(_VECTOR, [0., 0., 0., 0.])
        sums = [cgutils.alloca_once(builder, _VECTOR) for _ in range(4)]
        for pointer in sums:
            builder.store(zero, pointer)
        for_sum = ((0, 1), (2, 3))
        with cgutils.for_range(builder, builder.sub(zb, za)) as iz:
            zz = builder.add(za, iz.index)
            with cgutils.for_range(builder, builder.sub(yb, ya)) as iy:
                yy = builder.add(ya, iy.index)
                for image, anchor, (s, c) in zip(images, anchors, for_sum):
                    value = builder.fsub(_load(builder, image.data, index(zz, yy)), anchor)
                    total, correction = _add(builder, builder.load(sums[s]), builder.load(sums[c]), value)
                    builder.store(total, sums[s])
                    builder.store(correction, sums[c])
        count = builder.mul(builder.sub(zb, za), builder.sub(yb, ya))
        denominator = _splat(builder, builder.sitofp(count, ir.FloatType()))
        means = [builder.fdiv(builder.load(sums[s]), denominator) for s in (0, 2)]
        stats = [cgutils.alloca_once(builder, _VECTOR) for _ in range(6)]
        for pointer in stats:
            builder.store(zero, pointer)
        with cgutils.for_range(builder, builder.sub(zb, za)) as iz:
            zz = builder.add(za, iz.index)
            with cgutils.for_range(builder, builder.sub(yb, ya)) as iy:
                yy = builder.add(ya, iy.index)
                f, m = [builder.fsub(builder.fsub(_load(builder, image.data, index(zz, yy)), anchor), mean)
                        for image, anchor, mean in zip(images, anchors, means)]
                for slot, term in enumerate((builder.fmul(f, f), builder.fmul(m, m), builder.fmul(f, m))):
                    total, correction = _add(builder, builder.load(stats[slot * 2]),
                                             builder.load(stats[slot * 2 + 1]), term)
                    builder.store(total, stats[slot * 2])
                    builder.store(correction, stats[slot * 2 + 1])
        values = [*anchors, *means, *(builder.load(stats[s]) for s in (0, 2, 4))]
        for channel, value in enumerate(values):
            offset = builder.add(builder.mul(ir.Constant(nx.type, channel), nx), x)
            _store(builder, output.data, offset, value)
        return context.get_dummy_value()

    return types.void(fixed, moving, first, z0, z1, y0, y1, planes), codegen


@njit(cache=True, fastmath=False)
def _plane_scalar(fixed, moving, x, z0, z1, y0, y1, planes):
    anchor_f, anchor_m = fixed[z0, y0, x], moving[z0, y0, x]
    sf, sm, cf, cm = ZERO, ZERO, ZERO, ZERO
    for z in range(z0, z1):
        for y in range(y0, y1):
            sf, cf = add32(sf, cf, fixed[z, y, x] - anchor_f)
            sm, cm = add32(sm, cm, moving[z, y, x] - anchor_m)
    count = (z1 - z0) * (y1 - y0)
    mean_f, mean_m = sf / np.float32(count), sm / np.float32(count)
    ff, mm, fm, cff, cmm, cfm = ZERO, ZERO, ZERO, ZERO, ZERO, ZERO
    for z in range(z0, z1):
        for y in range(y0, y1):
            f = (fixed[z, y, x] - anchor_f) - mean_f
            m = (moving[z, y, x] - anchor_m) - mean_m
            ff, cff = add32(ff, cff, f * f)
            mm, cmm = add32(mm, cmm, m * m)
            fm, cfm = add32(fm, cfm, f * m)
    planes[0, x], planes[1, x] = anchor_f, anchor_m
    planes[2, x], planes[3, x] = mean_f, mean_m
    planes[4, x], planes[5, x], planes[6, x] = ff, mm, fm


@njit(cache=True, fastmath=False)
def _row(fixed, moving, radius, shift, spacing, z, y, forward, reverse, scores):
    nz, ny, nx = fixed.shape
    rx, ry, rz = min(radius, nx - 1), min(radius, ny - 1), min(radius, nz - 1)
    z0, z1 = max(0, z - rz), min(nz, z + rz + 1)
    y0, y1 = max(0, y - ry), min(ny, y + ry + 1)
    planes = np.empty((7, nx), np.float32)
    stop = nx - nx % 4
    for x in range(0, stop, 4):
        plane_stats4(fixed, moving, x, z0, z1, y0, y1, planes)
    for x in range(stop, nx):
        _plane_scalar(fixed, moving, x, z0, z1, y0, y1, planes)
    # All planes in this row have the same valid count. Hoist the repeated
    # merge divisions without changing their original FP32 expression order.
    count = (z1 - z0) * (y1 - y0)
    width = min(nx, 2 * rx + 1)
    fractions, cross_weights = np.empty(width + 1, np.float32), np.empty(width + 1, np.float32)
    for k in range(2, width + 1):
        merged = k * count
        fractions[k] = np.float32(count) / np.float32(merged)
        cross_weights[k] = np.float32((k - 1) * count) * np.float32(count) / np.float32(merged)
    for x in range(nx):
        first, end = max(0, x - rx), min(nx, x + rx + 1)
        anchor_f, anchor_m = planes[0, first], planes[1, first]
        mean_f, mean_m = planes[2, first], planes[3, first]
        ff, mm, fm = planes[4, first], planes[5, first], planes[6, first]
        for xx in range(first + 1, end):
            k = xx - first + 1
            delta_f = (planes[0, xx] - anchor_f) + (planes[2, xx] - mean_f)
            delta_m = (planes[1, xx] - anchor_m) + (planes[3, xx] - mean_m)
            mean_f += delta_f * fractions[k]
            mean_m += delta_m * fractions[k]
            ff = ff + planes[4, xx] + delta_f * delta_f * cross_weights[k]
            mm = mm + planes[5, xx] + delta_m * delta_m * cross_weights[k]
            fm = fm + planes[6, xx] + delta_f * delta_m * cross_weights[k]
        product = ff * mm
        score = ONE
        if abs(product) > CC_EPS:
            score = fm * fm / product
        if ff > CC_EPS and mm > CC_EPS:
            centre_f = (fixed[z, y, x] - anchor_f) - mean_f
            centre_m = (moving[z, y, x] - anchor_m) - mean_m
            forward_weight = TWO * fm / product * (centre_f - fm / mm * centre_m)
            reverse_weight = TWO * fm / product * (centre_m - fm / ff * centre_f)
            _gradient_updates32(fixed, moving, shift, spacing, z, y, x,
                                forward_weight, reverse_weight, forward, reverse)
        scores[z, y, x] = -score


@njit(cache=True, fastmath=False, parallel=True)
def _work(fixed, moving, radius, shift, spacing):
    nz, ny, nx = fixed.shape
    forward, reverse = np.zeros((3, nz, ny, nx), np.float32), np.zeros((3, nz, ny, nx), np.float32)
    scores = np.empty(fixed.shape, np.float32)
    if nx:
        for task in prange(nz * ny):
            row = np.int64(task)
            _row(fixed, moving, radius, shift, spacing, row // ny, row % ny, forward, reverse, scores)
    total, correction = ZERO, ZERO
    for value in scores.ravel():
        total, correction = add32(total, correction, value)
    count = scores.size
    return total / np.float32(count) if count else ZERO, count, forward, reverse


def cc_unmasked_gradient32(fixed, moving, radius, shift, spacing):
    return _work(fixed, moving, radius, shift, spacing)
