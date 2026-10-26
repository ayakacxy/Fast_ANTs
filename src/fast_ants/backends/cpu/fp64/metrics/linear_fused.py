"""FP64 linear-Mattes fusion with deterministic block PDF accumulation.

The affine, physical sampling, central-difference, and parameter-Jacobian
expressions deliberately retain the established CPU reference operation order.
Heavily occupied fixed-bin rows are split into private sample blocks. Their
compensated merge permits load-balanced workers without data races or atomic
updates. This changes reduction rounding, not the Mattes objective/derivative.
"""

import math

import numpy as np
from numba import get_num_threads, njit, prange

from fast_ants.backends.cpu.fp64.metrics.mattes_kernels import _mattes_finish
from fast_ants.backends.cpu.fp64.metrics.mattes_kernels import cubic_bspline
from fast_ants.backends.cpu.fp64.metrics.mattes_kernels import cubic_bspline_derivative
from fast_ants.backends.cpu.fp64.sampling_kernels import inside
from fast_ants.backends.cpu.fp64.sampling_kernels import linear_value
from fast_ants.core.arithmetic64 import fma64


_PARALLEL_SAMPLES = 16384
_HISTOGRAM_TILE = 32768


@njit(cache=True, fastmath=False)
def fixed_bin_order(fixed, bins, fixed_min, fixed_max):
    """Cache a stable ordering; each PDF cell sees its original sample order."""
    fixed_width = (fixed_max-fixed_min)/(bins-4)
    fixed_normalized_min = fixed_min/fixed_width-2
    sample_bins = np.empty(fixed.size, np.int32)
    counts = np.zeros(bins, np.int64)
    for n in range(fixed.size):
        fbin = min(max(int(fixed[n]/fixed_width-fixed_normalized_min), 2), bins-3)
        sample_bins[n] = fbin
        counts[fbin] += 1
    offsets = np.zeros(bins+1, np.int64)
    for fbin in range(bins):
        offsets[fbin+1] = offsets[fbin]+counts[fbin]
    ordered = np.empty(fixed.size, np.int64)
    next_sample = offsets[:-1].copy()
    for n in range(fixed.size):
        fbin = sample_bins[n]
        ordered[next_sample[fbin]] = n
        next_sample[fbin] += 1
    return offsets, ordered


@njit(cache=True, fastmath=False, inline="always")
def _gradient_component(data, mapped_x, mapped_y, mapped_z, origin, inverse_linear, spacing, component):
    """ITK physical neighbours, explicit hardware FMA, scalar scratch only."""
    offset = .5*spacing[component]
    coordinate_x = mapped_x-origin[0]
    coordinate_y = mapped_y-origin[1]
    coordinate_z = mapped_z-origin[2]
    minus_x = (mapped_x-offset if component == 0 else mapped_x)-origin[0]
    plus_x = (mapped_x+offset if component == 0 else mapped_x)-origin[0]
    minus_y = (mapped_y-offset if component == 1 else mapped_y)-origin[1]
    plus_y = (mapped_y+offset if component == 1 else mapped_y)-origin[1]
    minus_z = (mapped_z-offset if component == 2 else mapped_z)-origin[2]
    plus_z = (mapped_z+offset if component == 2 else mapped_z)-origin[2]

    negative_x, positive_x = 0.0, 0.0
    negative_x = fma64(inverse_linear[0, 0], minus_x, negative_x)
    positive_x = fma64(inverse_linear[0, 0], plus_x, positive_x)
    negative_x = fma64(inverse_linear[0, 1], minus_y, negative_x)
    positive_x = fma64(inverse_linear[0, 1], plus_y, positive_x)
    negative_x = fma64(inverse_linear[0, 2], minus_z, negative_x)
    positive_x = fma64(inverse_linear[0, 2], plus_z, positive_x)
    negative_y, positive_y = 0.0, 0.0
    negative_y = fma64(inverse_linear[1, 0], minus_x, negative_y)
    positive_y = fma64(inverse_linear[1, 0], plus_x, positive_y)
    negative_y = fma64(inverse_linear[1, 1], minus_y, negative_y)
    positive_y = fma64(inverse_linear[1, 1], plus_y, positive_y)
    negative_y = fma64(inverse_linear[1, 2], minus_z, negative_y)
    positive_y = fma64(inverse_linear[1, 2], plus_z, positive_y)
    negative_z, positive_z = 0.0, 0.0
    negative_z = fma64(inverse_linear[2, 0], minus_x, negative_z)
    positive_z = fma64(inverse_linear[2, 0], plus_x, positive_z)
    negative_z = fma64(inverse_linear[2, 1], minus_y, negative_z)
    positive_z = fma64(inverse_linear[2, 1], plus_y, positive_z)
    negative_z = fma64(inverse_linear[2, 2], minus_z, negative_z)
    positive_z = fma64(inverse_linear[2, 2], plus_z, positive_z)

    mapped_component = mapped_x if component == 0 else (mapped_y if component == 1 else mapped_z)
    delta = (mapped_component+offset)-(mapped_component-offset)
    nz, ny, nx = data.shape
    if (inside(negative_x, negative_y, negative_z, nx, ny, nz) and
            inside(positive_x, positive_y, positive_z, nx, ny, nz) and
            delta > 10*np.finfo(np.float64).eps):
        return (linear_value(data, positive_x, positive_y, positive_z)-
                linear_value(data, negative_x, negative_y, negative_z))/delta
    return 0.0


@njit(cache=True, fastmath=False, inline="always")
def _sample_one(data, point, linear, center, translation, origin, inverse_linear, spacing, derivative):
    # Keep each original three-term sum, including its initial value, while
    # avoiding per-sample NRT allocations for mapped/index/gradient scratch.
    relative_x = point[0]-center[0]
    relative_y = point[1]-center[1]
    relative_z = point[2]-center[2]
    mapped_x = center[0]+translation[0]
    mapped_x += linear[0, 0]*relative_x
    mapped_x += linear[0, 1]*relative_y
    mapped_x += linear[0, 2]*relative_z
    if not math.isfinite(mapped_x):
        return 0.0, 2, 0.0, 0.0, 0.0
    mapped_y = center[1]+translation[1]
    mapped_y += linear[1, 0]*relative_x
    mapped_y += linear[1, 1]*relative_y
    mapped_y += linear[1, 2]*relative_z
    if not math.isfinite(mapped_y):
        return 0.0, 2, 0.0, 0.0, 0.0
    mapped_z = center[2]+translation[2]
    mapped_z += linear[2, 0]*relative_x
    mapped_z += linear[2, 1]*relative_y
    mapped_z += linear[2, 2]*relative_z
    if not math.isfinite(mapped_z):
        return 0.0, 2, 0.0, 0.0, 0.0

    coordinate_x = mapped_x-origin[0]
    coordinate_y = mapped_y-origin[1]
    coordinate_z = mapped_z-origin[2]
    index_x = 0.0
    index_x = fma64(inverse_linear[0, 0], coordinate_x, index_x)
    index_x = fma64(inverse_linear[0, 1], coordinate_y, index_x)
    index_x = fma64(inverse_linear[0, 2], coordinate_z, index_x)
    index_y = 0.0
    index_y = fma64(inverse_linear[1, 0], coordinate_x, index_y)
    index_y = fma64(inverse_linear[1, 1], coordinate_y, index_y)
    index_y = fma64(inverse_linear[1, 2], coordinate_z, index_y)
    index_z = 0.0
    index_z = fma64(inverse_linear[2, 0], coordinate_x, index_z)
    index_z = fma64(inverse_linear[2, 1], coordinate_y, index_z)
    index_z = fma64(inverse_linear[2, 2], coordinate_z, index_z)
    nz, ny, nx = data.shape
    if not inside(index_x, index_y, index_z, nx, ny, nz):
        return 0.0, 0, 0.0, 0.0, 0.0
    value = linear_value(data, index_x, index_y, index_z)
    if not math.isfinite(value):
        return value, 3, 0.0, 0.0, 0.0
    if not derivative:
        return value, 1, 0.0, 0.0, 0.0

    gradient_x = _gradient_component(data, mapped_x, mapped_y, mapped_z, origin, inverse_linear, spacing, 0)
    gradient_y = _gradient_component(data, mapped_x, mapped_y, mapped_z, origin, inverse_linear, spacing, 1)
    gradient_z = _gradient_component(data, mapped_x, mapped_y, mapped_z, origin, inverse_linear, spacing, 2)
    return value, 1, gradient_x, gradient_y, gradient_z


@njit(cache=True, fastmath=False)
def _values_serial(data, points, linear, center, translation, origin, inverse_linear, spacing):
    values, states = np.empty(points.shape[0], np.float64), np.empty(points.shape[0], np.int8)
    for n in range(points.shape[0]):
        value, state, _, _, _ = _sample_one(data, points[n], linear, center, translation,
                                           origin, inverse_linear, spacing, False)
        values[n], states[n] = value, state
    return values, states


@njit(cache=True, fastmath=False, parallel=True)
def _values_parallel(data, points, linear, center, translation, origin, inverse_linear, spacing):
    values, states = np.empty(points.shape[0], np.float64), np.empty(points.shape[0], np.int8)
    for n in prange(points.shape[0]):
        value, state, _, _, _ = _sample_one(data, points[n], linear, center, translation,
                                           origin, inverse_linear, spacing, False)
        values[n], states[n] = value, state
    return values, states


@njit(cache=True, fastmath=False)
def _gradients_serial(data, points, linear, center, translation, origin, inverse_linear, spacing):
    values, states = np.empty(points.shape[0], np.float64), np.empty(points.shape[0], np.int8)
    gradients = np.empty((points.shape[0], 3), np.float64)
    for n in range(points.shape[0]):
        value, state, gx, gy, gz = _sample_one(data, points[n], linear, center, translation,
                                             origin, inverse_linear, spacing, True)
        values[n], states[n] = value, state
        gradients[n, 0], gradients[n, 1], gradients[n, 2] = gx, gy, gz
    return values, states, gradients


@njit(cache=True, fastmath=False, parallel=True)
def _gradients_parallel(data, points, linear, center, translation, origin, inverse_linear, spacing):
    values, states = np.empty(points.shape[0], np.float64), np.empty(points.shape[0], np.int8)
    gradients = np.empty((points.shape[0], 3), np.float64)
    for n in prange(points.shape[0]):
        value, state, gx, gy, gz = _sample_one(data, points[n], linear, center, translation,
                                             origin, inverse_linear, spacing, True)
        values[n], states[n] = value, state
        gradients[n, 0], gradients[n, 1], gradients[n, 2] = gx, gy, gz
    return values, states, gradients


@njit(cache=True, fastmath=False)
def _sample_status(states, fixed_values):
    count, bad_mapped, bad_values = 0, False, False
    for n in range(states.size):
        count += states[n] == 1
        bad_mapped |= states[n] == 2
        bad_values |= states[n] == 3
        if states[n] == 1 and not math.isfinite(fixed_values[n]):
            bad_values = True
    return count, bad_mapped, bad_values


@njit(cache=True, fastmath=False, parallel=True)
def _sample_status_parallel(states, fixed_values):
    # Only integer counts and flags are combined. No FP64 sum is reordered.
    chunk = 16384
    summaries = np.zeros(((states.size+chunk-1)//chunk, 3), np.int64)
    for block in prange(summaries.shape[0]):
        count, bad_mapped, bad_values = 0, False, False
        for n in range(block*chunk, min((block+1)*chunk, states.size)):
            count += states[n] == 1
            bad_mapped |= states[n] == 2
            bad_values |= states[n] == 3
            if states[n] == 1 and not math.isfinite(fixed_values[n]):
                bad_values = True
        summaries[block, 0] = count
        summaries[block, 1] = bad_mapped
        summaries[block, 2] = bad_values
    count, bad_mapped, bad_values = 0, False, False
    for block in range(summaries.shape[0]):
        count += summaries[block, 0]
        bad_mapped |= summaries[block, 1] != 0
        bad_values |= summaries[block, 2] != 0
    return count, bad_mapped, bad_values


@njit(cache=True, fastmath=False, inline="always")
def _value_row(fbin, offsets, ordered, values, states, bins, moving_min, moving_max, joint, fixed_pdf):
    moving_width = (moving_max-moving_min)/(bins-4)
    moving_normalized_min = moving_min/moving_width-2
    valid = 0
    for position in range(offsets[fbin], offsets[fbin+1]):
        n = ordered[position]
        if states[n] != 1 or values[n] < moving_min or values[n] > moving_max:
            continue
        moving_term = values[n]/moving_width-moving_normalized_min
        mbin = min(max(int(moving_term), 2), bins-3)
        fixed_pdf[fbin] += 1
        valid += 1
        for j in range(mbin-1, mbin+3):
            joint[fbin, j] += cubic_bspline(j-moving_term)
    return valid


@njit(cache=True, fastmath=False)
def _value_histogram_serial(offsets, ordered, values, states, bins, moving_min, moving_max):
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    valid = 0
    for fbin in range(bins):
        valid += _value_row(fbin, offsets, ordered, values, states, bins, moving_min, moving_max, joint, fixed_pdf)
    return joint, fixed_pdf, valid


@njit(cache=True, fastmath=False, parallel=True)
def _value_histogram_parallel(offsets, ordered, values, states, bins, moving_min, moving_max):
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    counts = np.empty(bins, np.int64)
    for fbin in prange(bins):
        counts[fbin] = _value_row(fbin, offsets, ordered, values, states, bins, moving_min, moving_max, joint, fixed_pdf)
    valid = 0
    for fbin in range(bins):
        valid += counts[fbin]
    return joint, fixed_pdf, valid


@njit(cache=True, fastmath=False, inline="always")
def _zero_coefficients_safe(linear_derivatives, translation_derivatives):
    # With |coefficient| and |point-center| <= 1e100, each spatial
    # intermediate is bounded by 3e200+1e100, far below FP64 overflow.
    # This is an eligibility bound, never a threshold for treating a nonzero
    # gradient as zero. Extreme/nonfinite inputs retain the original loop.
    for parameter in range(linear_derivatives.shape[0]):
        for c in range(3):
            value = translation_derivatives[parameter, c]
            if not math.isfinite(value) or abs(value) > 1e100:
                return False
            for j in range(3):
                value = linear_derivatives[parameter, c, j]
                if not math.isfinite(value) or abs(value) > 1e100:
                    return False
    return True


@njit(cache=True, fastmath=False, inline="always")
def _zero_sample_safe(gradients, points, center, n, safe_coefficients):
    if (not safe_coefficients or gradients[n, 0] != 0.0 or
            gradients[n, 1] != 0.0 or gradients[n, 2] != 0.0):
        return False
    for j in range(3):
        coordinate = points[n, j]-center[j]
        if not math.isfinite(coordinate) or abs(coordinate) > 1e100:
            return False
    return True


@njit(cache=True, fastmath=False, inline="always")
def _derivative_row(fbin, offsets, ordered, values, states, gradients, points, center,
                    linear_derivatives, translation_derivatives, bins, moving_min, moving_max,
                    joint, fixed_pdf, pdf_derivatives):
    moving_width = (moving_max-moving_min)/(bins-4)
    moving_normalized_min = moving_min/moving_width-2
    parameter_count = linear_derivatives.shape[0]
    parameter_gradient = np.empty(parameter_count, np.float64)
    safe_coefficients = _zero_coefficients_safe(linear_derivatives, translation_derivatives)
    valid, bad_derivative = 0, False
    for position in range(offsets[fbin], offsets[fbin+1]):
        n = ordered[position]
        if states[n] != 1:
            continue
        skip_zero = _zero_sample_safe(gradients, points, center, n, safe_coefficients)
        if not skip_zero:
            # Preserve the complete sample Jacobian even for an intensity
            # outside the allowed moving range. In particular an exact-zero
            # gradient with extreme spatial terms must still expose 0*inf.
            for parameter in range(parameter_count):
                total = 0.0
                for c in range(3):
                    spatial = translation_derivatives[parameter, c]
                    for j in range(3):
                        spatial += linear_derivatives[parameter, c, j]*(points[n, j]-center[j])
                    total += gradients[n, c]*spatial
                parameter_gradient[parameter] = total
                if not math.isfinite(total):
                    bad_derivative = True
        if values[n] < moving_min or values[n] > moving_max:
            continue
        moving_term = values[n]/moving_width-moving_normalized_min
        mbin = min(max(int(moving_term), 2), bins-3)
        fixed_pdf[fbin] += 1
        valid += 1
        for j in range(mbin-1, mbin+3):
            argument = j-moving_term
            joint[fbin, j] += cubic_bspline(argument)
            if not skip_zero:
                derivative = cubic_bspline_derivative(argument)
                for p in range(parameter_count):
                    pdf_derivatives[fbin, j, p] += parameter_gradient[p]*derivative
    return valid, bad_derivative


@njit(cache=True, fastmath=False)
def _derivative_histogram_serial(offsets, ordered, values, states, gradients, points, center,
                                 linear_derivatives, translation_derivatives, bins, moving_min, moving_max):
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    pdf_derivatives = np.zeros((bins, bins, linear_derivatives.shape[0]), np.float64)
    valid, bad_derivative = 0, False
    for fbin in range(bins):
        count, bad = _derivative_row(fbin, offsets, ordered, values, states, gradients, points, center,
                                     linear_derivatives, translation_derivatives, bins, moving_min, moving_max,
                                     joint, fixed_pdf, pdf_derivatives)
        valid += count
        bad_derivative |= bad
    return joint, fixed_pdf, pdf_derivatives, valid, bad_derivative


@njit(cache=True, fastmath=False, parallel=True)
def _derivative_histogram_parallel(offsets, ordered, values, states, gradients, points, center,
                                   linear_derivatives, translation_derivatives, bins, moving_min, moving_max):
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    pdf_derivatives = np.zeros((bins, bins, linear_derivatives.shape[0]), np.float64)
    counts, bad_derivatives = np.empty(bins, np.int64), np.empty(bins, np.bool_)
    for fbin in prange(bins):
        count, bad = _derivative_row(fbin, offsets, ordered, values, states, gradients, points, center,
                                     linear_derivatives, translation_derivatives, bins, moving_min, moving_max,
                                     joint, fixed_pdf, pdf_derivatives)
        counts[fbin], bad_derivatives[fbin] = count, bad
    valid, bad_derivative = 0, False
    for fbin in range(bins):
        valid += counts[fbin]
        bad_derivative |= bad_derivatives[fbin]
    return joint, fixed_pdf, pdf_derivatives, valid, bad_derivative


@njit(cache=True, fastmath=False, inline="always")
def _derivative_channel_row(fbin, first_parameter, last_parameter, write_value,
                            offsets, ordered, values, states, gradients, points, center,
                            linear_derivatives, translation_derivatives, bins, moving_min,
                            moving_max, joint, fixed_pdf, pdf_derivatives):
    moving_width = (moving_max-moving_min)/(bins-4)
    moving_normalized_min = moving_min/moving_width-2
    parameter_gradient = np.empty(linear_derivatives.shape[0], np.float64)
    safe_coefficients = _zero_coefficients_safe(linear_derivatives, translation_derivatives)
    valid, bad_derivative = 0, False
    for position in range(offsets[fbin], offsets[fbin+1]):
        n = ordered[position]
        if states[n] != 1:
            continue
        skip_zero = _zero_sample_safe(gradients, points, center, n, safe_coefficients)
        if not skip_zero:
            for parameter in range(first_parameter, last_parameter):
                total = 0.0
                for c in range(3):
                    spatial = translation_derivatives[parameter, c]
                    for j in range(3):
                        spatial += linear_derivatives[parameter, c, j]*(points[n, j]-center[j])
                    total += gradients[n, c]*spatial
                parameter_gradient[parameter] = total
                if not math.isfinite(total):
                    bad_derivative = True
        if values[n] < moving_min or values[n] > moving_max:
            continue
        moving_term = values[n]/moving_width-moving_normalized_min
        mbin = min(max(int(moving_term), 2), bins-3)
        if write_value:
            fixed_pdf[fbin] += 1
            valid += 1
        for j in range(mbin-1, mbin+3):
            argument = j-moving_term
            if write_value:
                joint[fbin, j] += cubic_bspline(argument)
            if not skip_zero:
                derivative = cubic_bspline_derivative(argument)
                for p in range(first_parameter, last_parameter):
                    pdf_derivatives[fbin, j, p] += parameter_gradient[p]*derivative
    return valid, bad_derivative


@njit(cache=True, fastmath=False, parallel=True)
def _derivative_histogram_channels(offsets, ordered, values, states, gradients, points, center,
                                   linear_derivatives, translation_derivatives, bins,
                                   moving_min, moving_max):
    """Each PDF cell/parameter has one owner and its original sample order.

    A heavily occupied fixed bin can use several workers. Lane-private planes
    avoid false sharing; assembling the result only copies cells, never sums
    partial floating-point accumulators.
    """
    parameters = linear_derivatives.shape[0]
    lanes = 3 if parameters == 6 else max(1, min(4, parameters))
    width = (parameters+lanes-1)//lanes
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    private = np.zeros((lanes, bins, bins, parameters), np.float64)
    counts, bad = np.zeros(lanes*bins, np.int64), np.zeros(lanes*bins, np.bool_)
    for task in prange(lanes*bins):
        index = np.int64(task)
        fbin, lane = index % bins, index//bins
        first, last = lane*width, min((lane+1)*width, parameters)
        counts[index], bad[index] = _derivative_channel_row(
            fbin, first, last, lane == 0, offsets, ordered, values, states, gradients,
            points, center, linear_derivatives, translation_derivatives, bins,
            moving_min, moving_max, joint, fixed_pdf, private[lane])
    pdf_derivatives = np.empty((bins, bins, parameters), np.float64)
    valid, bad_derivative = 0, False
    for fbin in range(bins):
        valid += counts[fbin]
        for j in range(bins):
            for p in range(parameters):
                pdf_derivatives[fbin, j, p] = private[p//width, fbin, j, p]
    for task in range(lanes*bins):
        bad_derivative |= bad[task]
    return joint, fixed_pdf, pdf_derivatives, valid, bad_derivative


@njit(cache=True, fastmath=False)
def _histogram_tasks(offsets):
    """Fixed sample partitions, independent of worker count or scheduling."""
    bins = offsets.size-1
    tasks = 0
    for fbin in range(bins):
        tasks += (offsets[fbin+1]-offsets[fbin]+_HISTOGRAM_TILE-1)//_HISTOGRAM_TILE
    starts, owners = np.empty((tasks, 2), np.int64), np.empty(tasks, np.int64)
    task = 0
    for fbin in range(bins):
        for first in range(offsets[fbin], offsets[fbin+1], _HISTOGRAM_TILE):
            starts[task, 0] = first
            starts[task, 1] = min(first+_HISTOGRAM_TILE, offsets[fbin+1])
            owners[task] = fbin
            task += 1
    return starts, owners


@njit(cache=True, fastmath=False, inline="always")
def _merge_add64(total, correction, value):
    """Neumaier correction for the short merge of signed derivative blocks."""
    updated = total+value
    if abs(total) >= abs(value):
        correction += (total-updated)+value
    else:
        correction += (value-updated)+total
    return updated, correction


@njit(cache=True, fastmath=False, parallel=True)
def _value_histogram_blocks(offsets, ordered, values, states, bins, moving_min, moving_max):
    starts, owners = _histogram_tasks(offsets)
    # Each task needs a single fixed-bin row, not a full bins-by-bins PDF.
    private = np.zeros((owners.size, 1, bins), np.float64)
    counts = np.zeros((owners.size, 1), np.float64)
    valid = np.empty(owners.size, np.int64)
    for task in prange(owners.size):
        valid[task] = _value_row(0, starts[task], ordered, values, states, bins,
                                moving_min, moving_max, private[task], counts[task])
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    correction = np.zeros((bins, bins), np.float64)
    total = 0
    for task in range(owners.size):
        fbin = owners[task]
        total += valid[task]
        fixed_pdf[fbin] += counts[task, 0]
        for j in range(bins):
            joint[fbin, j], correction[fbin, j] = _merge_add64(
                joint[fbin, j], correction[fbin, j], private[task, 0, j])
    for fbin in range(bins):
        for j in range(bins):
            joint[fbin, j] += correction[fbin, j]
    return joint, fixed_pdf, total


@njit(cache=True, fastmath=False, parallel=True)
def _derivative_histogram_blocks(offsets, ordered, values, states, gradients, points, center,
                                  linear_derivatives, translation_derivatives, bins,
                                  moving_min, moving_max):
    starts, owners = _histogram_tasks(offsets)
    parameters = linear_derivatives.shape[0]
    private = np.zeros((owners.size, 1, bins), np.float64)
    counts = np.zeros((owners.size, 1), np.float64)
    derivatives = np.zeros((owners.size, 1, bins, parameters), np.float64)
    valid, bad = np.empty(owners.size, np.int64), np.empty(owners.size, np.bool_)
    for task in prange(owners.size):
        # Reuse the established one-sample formula and invalid-derivative
        # handling. Each sample's Parzen terms and full Jacobian are computed
        # once, instead of scanning a heavy row separately for each lane.
        valid[task], bad[task] = _derivative_row(
            0, starts[task], ordered, values, states, gradients, points, center,
            linear_derivatives, translation_derivatives, bins, moving_min, moving_max,
            private[task], counts[task], derivatives[task])
    joint, fixed_pdf = np.zeros((bins, bins), np.float64), np.zeros(bins, np.float64)
    pdf_derivatives = np.zeros((bins, bins, parameters), np.float64)
    joint_correction = np.zeros((bins, bins), np.float64)
    derivative_correction = np.zeros((bins, bins, parameters), np.float64)
    total, bad_derivative = 0, False
    for task in range(owners.size):
        fbin = owners[task]
        total += valid[task]
        bad_derivative |= bad[task]
        fixed_pdf[fbin] += counts[task, 0]
        for j in range(bins):
            joint[fbin, j], joint_correction[fbin, j] = _merge_add64(
                joint[fbin, j], joint_correction[fbin, j], private[task, 0, j])
            for p in range(parameters):
                pdf_derivatives[fbin, j, p], derivative_correction[fbin, j, p] = _merge_add64(
                    pdf_derivatives[fbin, j, p], derivative_correction[fbin, j, p],
                    derivatives[task, 0, j, p])
    for fbin in range(bins):
        for j in range(bins):
            joint[fbin, j] += joint_correction[fbin, j]
            for p in range(parameters):
                pdf_derivatives[fbin, j, p] += derivative_correction[fbin, j, p]
    return joint, fixed_pdf, pdf_derivatives, total, bad_derivative


def evaluate_linear_mattes(data, points, transform, origin, inverse_linear, spacing,
                           fixed_values, offsets, ordered, bins, moving_range, derivative_center=None,
                           linear_derivatives=None, translation_derivatives=None):
    """Python dispatch keeps the independent compiled kernels disk-cacheable."""
    parallel = len(points) >= _PARALLEL_SAMPLES and get_num_threads() > 1
    largest_row = int(np.max(np.diff(offsets))) if parallel else 0
    use_blocks = parallel and largest_row > _HISTOGRAM_TILE
    split_heavy_bin = parallel and len(points) >= 32768 and largest_row >= len(points)/4
    status_function = _sample_status_parallel if parallel else _sample_status
    args = (data, points, transform.matrix[:3, :3], transform.center, transform.translation,
            origin, inverse_linear, spacing)
    if linear_derivatives is None:
        values, states = (_values_parallel if parallel else _values_serial)(*args)
        geometric_count, bad_mapped, bad_values = status_function(states, fixed_values)
        if bad_mapped:
            raise ValueError("Points must be finite (..., 3) physical or index coordinates")
        if bad_values:
            raise ValueError("Invalid Mattes sample values or histogram bin count")
        histogram = (_value_histogram_blocks if use_blocks else
                     (_value_histogram_parallel if parallel else _value_histogram_serial))
        joint, fixed_pdf, valid = histogram(
            offsets, ordered, values, states, bins, *moving_range)
        pdf_derivatives = np.zeros((bins, bins, 0), np.float64)
    else:
        values, states, gradients = (_gradients_parallel if parallel else _gradients_serial)(*args)
        geometric_count, bad_mapped, bad_values = status_function(states, fixed_values)
        if bad_mapped:
            raise ValueError("Points must be finite (..., 3) physical or index coordinates")
        if bad_values:
            raise ValueError("Invalid Mattes sample values or histogram bin count")
        histogram = (_derivative_histogram_blocks if use_blocks else
                     _derivative_histogram_channels if split_heavy_bin else
                     (_derivative_histogram_parallel if parallel else _derivative_histogram_serial))
        joint, fixed_pdf, pdf_derivatives, valid, bad_derivative = histogram(
                offsets, ordered, values, states, gradients, points, derivative_center,
                linear_derivatives, translation_derivatives, bins, *moving_range)
        if bad_derivative:
            raise ValueError("Intensity derivatives must have shape [samples, parameters]")
    return _mattes_finish(joint, fixed_pdf, pdf_derivatives, valid, bins, *moving_range), geometric_count
