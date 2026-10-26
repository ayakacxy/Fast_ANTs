"""Mattes box/cubic Parzen PDFs and global-transform derivatives.

Algorithm and derivative sign follow ITK 5.4.7 Mattes metric, Copyright
NumFOCUS, Apache-2.0. Modified independent Python/Numba implementation.
"""

import math
import numpy as np
from numba import get_num_threads, njit, prange

from fast_ants.backends.cpu.common.reductions import sum_values


@njit(cache=True, fastmath=False)
def cubic_bspline(argument):
    value = abs(argument)
    if value < 1:
        return (4-6*value*value+3*value*value*value)/6
    if value < 2:
        return (2-value)**3/6
    return 0.0


@njit(cache=True, fastmath=False)
def cubic_bspline_derivative(argument):
    value = abs(argument)
    sign = 1.0 if argument >= 0 else -1.0
    if value < 1:
        return sign*(-2*value+1.5*value*value)
    if value < 2:
        return -.5*sign*(2-value)**2
    return 0.0


@njit(cache=True, fastmath=False)
def _mattes_histograms_serial(fixed, moving, intensity_derivatives, bins, fixed_min, fixed_max, moving_min, moving_max):
    fixed_width, moving_width = (fixed_max-fixed_min)/(bins-4), (moving_max-moving_min)/(bins-4)
    fixed_normalized_min, moving_normalized_min = fixed_min/fixed_width-2, moving_min/moving_width-2
    joint = np.zeros((bins, bins), np.float64)
    fixed_marginal = np.zeros(bins, np.float64)
    parameter_count = intensity_derivatives.shape[1]
    pdf_derivatives = np.zeros((bins, bins, parameter_count), np.float64)
    valid = 0
    for n in range(fixed.size):
        if moving[n] < moving_min or moving[n] > moving_max:
            continue
        fbin = min(max(int(fixed[n]/fixed_width-fixed_normalized_min), 2), bins-3)
        moving_term = moving[n]/moving_width-moving_normalized_min
        mbin = min(max(int(moving_term), 2), bins-3)
        fixed_marginal[fbin] += 1
        valid += 1
        for j in range(mbin-1, mbin+3):
            argument = j-moving_term
            joint[fbin, j] += cubic_bspline(argument)
            derivative = cubic_bspline_derivative(argument)
            for p in range(parameter_count):
                pdf_derivatives[fbin, j, p] += intensity_derivatives[n, p]*derivative
    return joint, fixed_marginal, pdf_derivatives, valid


@njit(cache=True, fastmath=False, parallel=True)
def _mattes_histograms_parallel(fixed, moving, intensity_derivatives, bins, fixed_min, fixed_max, moving_min, moving_max):
    """Parallelize independent fixed-bin rows without reordering any sum.

    Thread-local histograms would require merging partial sums and would change
    FP64 rounding. A stable counting-sort instead assigns each fixed bin its
    original ascending sample indices; its PDF entries then receive exactly the
    same additions, in the same order, as the serial implementation.
    """
    fixed_width, moving_width = (fixed_max-fixed_min)/(bins-4), (moving_max-moving_min)/(bins-4)
    fixed_normalized_min, moving_normalized_min = fixed_min/fixed_width-2, moving_min/moving_width-2
    sample_bins = np.empty(fixed.size, np.int32)
    counts = np.zeros(bins, np.int64)
    for n in range(fixed.size):
        if moving[n] < moving_min or moving[n] > moving_max:
            sample_bins[n] = -1
            continue
        fbin = min(max(int(fixed[n]/fixed_width-fixed_normalized_min), 2), bins-3)
        sample_bins[n] = fbin
        counts[fbin] += 1
    offsets = np.zeros(bins+1, np.int64)
    for fbin in range(bins):
        offsets[fbin+1] = offsets[fbin]+counts[fbin]
    valid = offsets[bins]
    ordered_samples = np.empty(valid, np.int64)
    next_sample = offsets[:-1].copy()
    for n in range(fixed.size):
        fbin = sample_bins[n]
        if fbin >= 0:
            ordered_samples[next_sample[fbin]] = n
            next_sample[fbin] += 1

    joint = np.zeros((bins, bins), np.float64)
    fixed_marginal = np.zeros(bins, np.float64)
    parameter_count = intensity_derivatives.shape[1]
    pdf_derivatives = np.zeros((bins, bins, parameter_count), np.float64)
    for fbin in prange(bins):
        for position in range(offsets[fbin], offsets[fbin+1]):
            n = ordered_samples[position]
            moving_term = moving[n]/moving_width-moving_normalized_min
            mbin = min(max(int(moving_term), 2), bins-3)
            fixed_marginal[fbin] += 1
            for j in range(mbin-1, mbin+3):
                argument = j-moving_term
                joint[fbin, j] += cubic_bspline(argument)
                derivative = cubic_bspline_derivative(argument)
                for p in range(parameter_count):
                    pdf_derivatives[fbin, j, p] += intensity_derivatives[n, p]*derivative
    return joint, fixed_marginal, pdf_derivatives, valid


def mattes_samples(fixed, moving, intensity_derivatives, bins, fixed_min, fixed_max, moving_min, moving_max):
    # Small arrays and value-only line-search evaluations do not amortize stable
    # bucketing plus the parallel launch. Their original loop stays unchanged.
    if fixed.size >= 32768 and intensity_derivatives.shape[1] > 0 and get_num_threads() > 1:
        joint, fixed_marginal, pdf_derivatives, valid = _mattes_histograms_parallel(
            fixed, moving, intensity_derivatives, bins, fixed_min, fixed_max, moving_min, moving_max)
    else:
        joint, fixed_marginal, pdf_derivatives, valid = _mattes_histograms_serial(
            fixed, moving, intensity_derivatives, bins, fixed_min, fixed_max, moving_min, moving_max)
    return _mattes_finish(joint, fixed_marginal, pdf_derivatives, valid, bins, moving_min, moving_max)


@njit(cache=True, fastmath=False)
def _mattes_finish(joint, fixed_marginal, pdf_derivatives, valid, bins, moving_min, moving_max):
    """Keep normalization and every scalar reduction in the reference order."""
    parameter_count = pdf_derivatives.shape[2]
    moving_width = (moving_max-moving_min)/(bins-4)
    total = sum_values(joint.ravel())
    if valid == 0 or total <= np.finfo(np.float64).eps:
        return np.finfo(np.float64).max, np.zeros(parameter_count), joint, fixed_marginal, np.zeros(bins), pdf_derivatives, valid
    normalization_factor = 1.0/total
    joint *= normalization_factor
    fixed_marginal *= normalization_factor
    moving_marginal = np.zeros(bins)
    for i in range(bins):
        for j in range(bins):
            moving_marginal[j] += joint[i, j]
    pdf_derivatives *= -1/(moving_width*valid)
    mutual_information = 0.0
    descent = np.zeros(parameter_count)
    epsilon = np.finfo(np.float64).eps
    for i in range(bins):
        if fixed_marginal[i] <= epsilon:
            continue
        log_fixed = math.log(fixed_marginal[i])
        for j in range(bins):
            if joint[i, j] > epsilon and moving_marginal[j] > epsilon:
                ratio = math.log(joint[i, j]/moving_marginal[j])
                mutual_information += joint[i, j]*(ratio-log_fixed)
                for p in range(parameter_count):
                    descent[p] += pdf_derivatives[i, j, p]*ratio
    return -mutual_information, descent, joint, fixed_marginal, moving_marginal, pdf_derivatives, valid


@njit(cache=True, fastmath=False)
def mean_squares_samples(fixed, moving, intensity_derivatives):
    value = 0.0
    descent = np.zeros(intensity_derivatives.shape[1])
    for n in range(fixed.size):
        residual = fixed[n]-moving[n]
        value += residual*residual
        for p in range(descent.size):
            descent[p] += 2*residual*intensity_derivatives[n, p]
    return value/fixed.size, descent/fixed.size
