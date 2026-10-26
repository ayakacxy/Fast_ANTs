"""Compare SIMD bin updates with independent scalar FP32 operations."""
import unittest

import numpy as np
from numba import njit

from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import cubic32, cubic_derivative32
from fast_ants.backends.cpu.fp32.metrics.parzen import parzen_add4, derivative_add4


@njit(cache=True)
def scalar_bins(joint, correction, fi, mi, fraction):
    weights = np.empty(4, np.float32)
    for slot in range(4):
        j = mi - 1 + slot
        argument = np.float32(j - mi) - fraction
        joint[fi, j], correction[fi, j] = add32(
            joint[fi, j], correction[fi, j], cubic32(argument))
        weights[slot] = cubic_derivative32(argument)
    return weights


@njit(cache=True)
def vector_bins(joint, correction, fi, mi, fraction):
    return parzen_add4(joint, correction, fi, mi, fraction)


@njit(cache=True)
def scalar_channels(histogram, correction, intensity, fi, mi, first, weight):
    for p in range(first, first + 4):
        histogram[fi, mi, p], correction[fi, mi, p] = add32(
            histogram[fi, mi, p], correction[fi, mi, p], intensity[p] * weight)


@njit(cache=True)
def vector_channels(histogram, correction, intensity, fi, mi, first, weight):
    derivative_add4(histogram, correction, intensity, fi, mi, first, weight)


class ParzenSIMD32Tests(unittest.TestCase):
    def test_four_bins_and_derivatives_keep_scalar_rounding_at_knots(self):
        rng = np.random.default_rng(3007)
        fractions = np.concatenate((np.linspace(-1, 2, 19, dtype=np.float32),
            np.array([-np.inf, np.inf, np.nan, -0., 0.,
                      np.nextafter(np.float32(0), np.float32(1)),
                      np.nextafter(np.float32(1), np.float32(0)),
                      np.nextafter(np.float32(1), np.float32(2))], np.float32)))
        for mi in (2, 8):
            total = rng.normal(size=(2, 11)).astype(np.float32)
            correction = rng.uniform(-1e-5, 1e-5, size=total.shape).astype(np.float32)
            scalar, sc = total.copy(), correction.copy()
            vector, vc = total.copy(), correction.copy()
            for fraction in fractions:
                wanted = scalar_bins(scalar, sc, 1, mi, fraction)
                actual = np.asarray(vector_bins(vector, vc, 1, mi, fraction), np.float32)
                np.testing.assert_array_equal(actual.view(np.uint32), wanted.view(np.uint32))
                np.testing.assert_array_equal(vector.view(np.uint32), scalar.view(np.uint32))
                np.testing.assert_array_equal(vc.view(np.uint32), sc.view(np.uint32))

    def test_parameter_groups_and_unaligned_rows_keep_scalar_rounding(self):
        rng = np.random.default_rng(113)
        for parameters in (6, 12):
            values = rng.normal(size=(3, 7, parameters)).astype(np.float32)
            corrections = rng.uniform(-1e-5, 1e-5, values.shape).astype(np.float32)
            intensity = rng.normal(size=parameters).astype(np.float32)
            scalar, sc = values.copy(), corrections.copy()
            vector, vc = values.copy(), corrections.copy()
            for weight in np.array([-1e-4, 0., .37, -2.1], np.float32):
                for first in range(0, parameters - 3, 4):
                    scalar_channels(scalar, sc, intensity, 1, 2, first, weight)
                    vector_channels(vector, vc, intensity, 1, 2, first, weight)
                np.testing.assert_array_equal(vector.view(np.uint32), scalar.view(np.uint32))
                np.testing.assert_array_equal(vc.view(np.uint32), sc.view(np.uint32))


if __name__ == '__main__':
    unittest.main()
