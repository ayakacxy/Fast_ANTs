"""Bounded float1 PDF contracts; these do not establish registration accuracy."""

import unittest
import ctypes
import ctypes.util
import math
import numpy as np

from fast_ants.pipeline.mattes32 import finalize_mattes_float1, _joint_total_float1


class MattesFloat1FinalizationTests(unittest.TestCase):
    def test_value_and_derivative_follow_native_fmaf_accumulation(self):
        # Independent C-library scalar oracle, including platform logf.
        library = ctypes.CDLL(ctypes.util.find_library('m') or None)
        native_fma = library.fmaf
        native_fma.argtypes = [ctypes.c_float]*3
        native_fma.restype = ctypes.c_float
        native_log = library.logf
        native_log.argtypes = [ctypes.c_float]
        native_log.restype = ctypes.c_float
        f = np.float32
        rng = np.random.default_rng(147)
        raw = rng.integers(1, 100, size=(5, 5)).astype(np.float32)
        raw_derivative = rng.normal(size=(5, 5, 3)).astype(np.float32)
        counts = raw.sum(axis=1)
        valid, width = int(raw.sum()), f(.137)
        normalization = f(1/float(f(math.fsum(float(v) for v in raw.ravel()))))
        joint, fixed = raw*normalization, counts*normalization
        moving = np.zeros(5, np.float32)
        for j in range(5):
            for i in range(5):
                moving[j] = f(moving[j]+joint[i, j])
        derivative = raw_derivative*f(-1/float(f(width*f(valid))))
        value, separate_value = f(0), f(0)
        descent, separate_descent = np.zeros(3, np.float32), np.zeros(3, np.float32)
        for i in range(5):
            log_fixed = f(native_log(fixed[i]))
            for j in range(5):
                ratio = f(native_log(f(joint[i, j]/moving[j])))
                difference = f(ratio-log_fixed)
                value = f(native_fma(joint[i, j], difference, value))
                separate_value = f(separate_value+f(joint[i, j]*difference))
                for p in range(3):
                    descent[p] = f(native_fma(derivative[i, j, p], ratio, descent[p]))
                    separate_descent[p] = f(separate_descent[p]+f(derivative[i, j, p]*ratio))
        actual = finalize_mattes_float1(raw, counts, raw_derivative, valid, width)
        self.assertEqual(f(actual[0]).view(np.uint32), f(-value).view(np.uint32))
        np.testing.assert_array_equal(actual[1].view(np.uint32), descent.view(np.uint32))
        np.testing.assert_array_equal(actual[2], joint)
        np.testing.assert_array_equal(actual[4], moving)
        # Ensure this fixture detects reverting to separate product-plus-sum.
        self.assertTrue(value != separate_value or np.any(descent != separate_descent))

    def test_binary_perfect_dependence_and_derivative_cancellation(self):
        joint = np.zeros((5, 5), np.float32)
        joint[1, 1] = joint[3, 3] = 2
        derivative = np.zeros((5, 5, 1), np.float32)
        derivative[1, 1, 0] = 1
        derivative[3, 3, 0] = -1
        result = finalize_mattes_float1(joint, [0, 2, 0, 2, 0], derivative, 4, .25)
        self.assertEqual(result[0], -np.float32(np.log(np.float32(2))))
        np.testing.assert_array_equal(result[1], [0])
        np.testing.assert_array_equal(result[2], joint*.25)
        for value in result[1:]:
            self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(joint[[1, 3], [1, 3]], [2, 2])

    def test_total_retains_small_float_bins_in_double_before_float_assignment(self):
        joint = np.zeros((5, 5), np.float32)
        joint.flat[:3] = [2**24, 1, 1]
        # Sequential FP32 loses both ones. The official mixed accumulator does not.
        self.assertEqual(_joint_total_float1(joint), np.float32(2**24+2))

    def test_normalization_multiplies_the_rounded_reciprocal(self):
        joint = np.zeros((5, 5), np.float32)
        joint[1, 1], joint[3, 3] = 3, 4
        result = finalize_mattes_float1(joint, [0, 3, 0, 4, 0], np.empty((5, 5, 0), np.float32), 7, 1)
        self.assertEqual(result[2][1, 1], np.float32(3)*np.float32(1/7))
        self.assertNotEqual(result[2][1, 1], np.float32(3/7))
        np.testing.assert_array_equal(result[3], result[4])
        self.assertEqual(result[1].shape, (0,))

    def test_pdf_epsilon_is_strict_and_derivative_normalization_is_separate(self):
        eps = np.float32(np.finfo(np.float32).eps)
        results = []
        for probability in (eps, np.nextafter(eps, np.float32(np.inf))):
            joint = np.zeros((5, 5), np.float32)
            joint[1, 1], joint[3, 1] = probability, np.float32(1)-probability
            derivative = np.zeros((5, 5, 1), np.float32)
            derivative[1, 1, 0] = 1
            results.append(finalize_mattes_float1(joint, joint.sum(axis=1), derivative, 1, 1))
        self.assertEqual(results[0][1][0], 0)
        self.assertGreater(results[1][1][0], 0)
        self.assertEqual(results[0][5][1, 1, 0], -1)

    def test_derivative_denominator_rounds_count_and_product_to_float(self):
        joint = np.zeros((5, 5), np.float32); joint[1, 1] = 1
        derivative = np.ones((5, 5, 1), np.float32)
        result = finalize_mattes_float1(joint, [0, 1, 0, 0, 0], derivative, 2**24+1, .1)
        rounded_denominator = np.float32(.1)*np.float32(2**24)
        expected = np.float32(-1/float(rounded_denominator))
        np.testing.assert_array_equal(result[5], np.full_like(derivative, expected))
        self.assertNotEqual(expected, np.float32(-1/(float(np.float32(.1))*(2**24+1))))

    def test_invalid_histograms_and_empty_valid_domain(self):
        joint = np.zeros((5, 5), np.float32)
        derivative = np.zeros((5, 5, 0), np.float32)
        result = finalize_mattes_float1(joint, np.zeros(5), derivative, 0, 1)
        self.assertEqual(result[0], np.finfo(np.float32).max)
        with self.assertRaises(ValueError):
            finalize_mattes_float1(joint.astype(np.float64), np.zeros(5), derivative, 1, 1)
        joint[0, 0] = -1
        with self.assertRaises(ValueError):
            finalize_mattes_float1(joint, np.zeros(5), derivative, 1, 1)


if __name__ == '__main__':
    unittest.main()
