"""Bounded Mattes finalization with the official float1 assignment boundaries.

Adapted from ITK (NumFOCUS, Apache-2.0), MattesMutualInformationImageToImageMetricv4
ComputeResults/GetValueCommonAfterThreadedExecution and its threader's
AfterThreadedExecution. The relevant formulas were checked against v5.4.1 and
v5.4.7; no difference was found in these routines. See their official sources:
https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.7/Modules/Registration/Metricsv4/include/itkMattesMutualInformationImageToImageMetricv4.hxx
https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.1/Modules/Registration/Metricsv4/include/itkMattesMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx

Float histogram bins are summed using native CPU double Kahan accumulation
(CompensatedSummation<float>::AccumulateType is double), then stored as float.
PDF normalization, marginals, logs, value and derivative accumulation have
float state. Double literals in the two reciprocal expressions are retained.
This bounded CPU work uses no expanded floating-point representation.

Input histograms must already represent the desired sampling/binning policy.
This routine does not reproduce their upstream thread accumulation order.
The installed ANTs 2.6.5 ARM64 float ComputeResults uses native fused
multiply-add for value and global derivative accumulation. Those two sites
use explicit FP32 FMA here, with reassociation otherwise disabled. Histogram
construction and platform logf agreement still require an independent oracle.
"""

import numpy as np
from numba import njit

from fast_ants.pipeline.float1_policy import FLOAT1_PDF_EPS
from fast_ants.core.arithmetic32 import fma32


@njit(cache=True, fastmath=False)
def _joint_total_float1(histogram):
    total, compensation = np.float64(0), np.float64(0)
    for value in histogram.ravel():
        adjusted = np.float64(value)-compensation
        updated = total+adjusted
        compensation = (updated-total)-adjusted
        total = updated
    return np.float32(total)


@njit(cache=True, fastmath=False)
def _finalize_mattes_float1(histogram, counts, derivatives, valid, bin_width):
    bins = histogram.shape[0]
    parameters = derivatives.shape[2]
    joint = histogram.copy()
    pdf_derivative = derivatives.copy()
    fixed_pdf = np.zeros(bins, np.float32)
    moving_pdf = np.zeros(bins, np.float32)
    descent = np.zeros(parameters, np.float32)
    total = _joint_total_float1(joint)
    if valid == 0 or total < FLOAT1_PDF_EPS:
        # Public project convention: invalid evaluations return a sentinel.
        # Upstream throws for these cases; callers must not accept this as MI.
        return np.float32(np.finfo(np.float32).max), descent, joint, fixed_pdf, moving_pdf, pdf_derivative

    # C++: -1.0 / (float_width * uint_count), assigned to PDFValueType.
    denominator = np.float32(bin_width*np.float32(valid))
    derivative_factor = np.float32(np.float64(-1)/np.float64(denominator))
    for i in range(bins):
        for j in range(bins):
            for p in range(parameters):
                pdf_derivative[i, j, p] = np.float32(pdf_derivative[i, j, p]*derivative_factor)

    # C++: 1.0 / float_total, assigned to float; normalization is multiply,
    # not per-bin division. Counts represent the already merged fixed marginal.
    normalization = np.float32(np.float64(1)/np.float64(total))
    for i in range(bins):
        fixed_pdf[i] = np.float32(counts[i]*normalization)
        for j in range(bins):
            joint[i, j] = np.float32(joint[i, j]*normalization)
    for j in range(bins):
        for i in range(bins):
            moving_pdf[j] = np.float32(moving_pdf[j]+joint[i, j])

    information = np.float32(0)
    for i in range(bins):
        if fixed_pdf[i] <= FLOAT1_PDF_EPS:
            continue
        log_fixed = np.float32(np.log(fixed_pdf[i]))
        for j in range(bins):
            if moving_pdf[j] > FLOAT1_PDF_EPS and joint[i, j] > FLOAT1_PDF_EPS:
                ratio = np.float32(np.log(np.float32(joint[i, j]/moving_pdf[j])))
                # Installed ARM64 ComputeResults: fsub at 0x8ef7a4,
                # fmadd at 0x8ef7a8. Preserve the intervening FP32 subtraction.
                information = fma32(joint[i, j], np.float32(ratio-log_fixed), information)
                for p in range(parameters):
                    # Global-support derivative: scalar fmadd at 0x8ef8d8;
                    # the vectorized parameter branch uses fmla as well.
                    descent[p] = fma32(pdf_derivative[i, j, p], ratio, descent[p])
    return np.float32(-np.float64(1)*np.float64(information)), descent, joint, fixed_pdf, moving_pdf, pdf_derivative


def finalize_mattes_float1(histogram, counts, derivatives, valid, bin_width):
    """Finalize raw float32 bins; return the existing six-item metric tuple.

Counts can be integer or float arrays. Integer counts are converted at the
float histogram-state boundary. Large counts above float's exact integer
range cannot reconstruct an upstream per-sample float accumulator's history.
``bin_width`` must be the float width used when constructing the histogram.
Only global-support Rigid/Affine derivative histograms are supported.
"""
    histogram, counts, derivatives = np.asarray(histogram), np.asarray(counts), np.asarray(derivatives)
    if histogram.dtype != np.float32 or derivatives.dtype != np.float32:
        raise ValueError("Float1 finalization requires float32 histogram and derivative storage")
    if histogram.ndim != 2 or histogram.shape[0] != histogram.shape[1] or histogram.shape[0] < 5:
        raise ValueError("Mattes histogram must be square with at least five bins per axis")
    if counts.shape != (histogram.shape[0],) or derivatives.ndim != 3 or derivatives.shape[:2] != histogram.shape:
        raise ValueError("Fixed marginal and derivative histogram shapes must match joint histogram")
    if not np.isfinite(histogram).all() or np.any(histogram < 0) or not np.isfinite(derivatives).all():
        raise ValueError("Histogram must be finite and nonnegative; derivatives must be finite")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Fixed histogram counts must be finite and nonnegative")
    if int(valid) != valid or valid < 0:
        raise ValueError("Valid sample count must be a nonnegative integer")
    with np.errstate(over="ignore", invalid="ignore"):
        width = np.float32(bin_width)
        marginal = np.ascontiguousarray(counts, dtype=np.float32)
        total_bound = np.sum(histogram, dtype=np.float64)
        denominator = np.float32(width*np.float32(valid))
    if not np.isfinite(width) or width <= 0 or not np.isfinite(marginal).all():
        raise ValueError("Bin width and fixed marginal must be representable in float32")
    if total_bound > np.finfo(np.float32).max or (valid and (not np.isfinite(denominator) or denominator <= 0)):
        raise ValueError("Histogram total or derivative denominator is outside float32 range")
    return _finalize_mattes_float1(np.ascontiguousarray(histogram), marginal,
                                  np.ascontiguousarray(derivatives), int(valid), width)
