"""Sampling batch and cached-zero-support checks against scalar execution."""
import unittest

import numpy as np
from numba import njit

from fast_ants.backends.cpu.fp32.metrics import mattes_kernels as k
from fast_ants.backends.cpu.fp32.metrics.background import zero_support32
from fast_ants.backends.cpu.fp32.metrics.coordinates import map_integer4, integer_plan_kind
from fast_ants.backends.cpu.fp32.metrics.derivative import derivative_histogram32


@njit(cache=True)
def scalar_mapping(points, mapping):
    integer, residual, integer_offset, residual_offset, origin = mapping
    bases = np.empty(points.shape, np.int32)
    offsets = np.empty(points.shape, np.float32)
    for n in range(len(points)):
        k._local_affine_integer_point32(points, n, integer, residual, integer_offset,
                                      residual_offset, origin, bases[n], offsets[n])
    return bases, offsets


@njit(cache=True)
def vector_mapping(points, mapping):
    integer, residual, integer_offset, residual_offset, origin = mapping
    bases = np.empty(points.shape, np.int32)
    offsets = np.empty(points.shape, np.float32)
    b, f = np.empty((3, 4), np.int32), np.empty((3, 4), np.float32)
    for n in range(0, len(points), 4):
        map_integer4(points, n, integer, residual, integer_offset, residual_offset, origin, b, f)
        for lane in range(4):
            for c in range(3):
                bases[n + lane, c], offsets[n + lane, c] = b[c, lane], f[c, lane]
    return bases, offsets


class MattesSIMDSampling32Tests(unittest.TestCase):
    def test_mapping_signed_coordinates_and_integer_roundtrip(self):
        rng = np.random.default_rng(703)
        points = rng.integers(-200, 200, (12, 3)).astype(np.float32)
        points[-1] = [2.**34, -(2.**33), 2.**32]
        mapping = (rng.integers(-2, 3, (3, 3), dtype=np.int32),
                   rng.uniform(-.1, .1, (3, 3)).astype(np.float32),
                   np.array([-3, 7, 2], np.int32),
                   np.array([.13, -.21, .07], np.float32),
                   np.array([3, -7, 1], np.float32))
        expected, actual = scalar_mapping(points, mapping), vector_mapping(points, mapping)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))
        self.assertEqual(integer_plan_kind(points, mapping, True), 2)
        self.assertEqual(integer_plan_kind(np.asfortranarray(points), mapping, True), 1)
        self.assertEqual(integer_plan_kind(points, mapping, False), 0)

    def test_zero_support_matches_cropped_neighborhoods_and_rejects_nan(self):
        data = np.zeros((6, 8, 10), np.float32)
        data[2:4, 3:5, 4:7] = 1
        data[0, 0, 0] = np.nan
        actual = zero_support32(data)
        expected = np.empty(data.shape, np.bool_)
        for z, y, x in np.ndindex(data.shape):
            block = data[max(0, z-1):z+3, max(0, y-1):y+3, max(0, x-1):x+3]
            expected[z, y, x] = not np.any(block != np.float32(0))
        np.testing.assert_array_equal(actual, expected)

    def test_zero_support_preserves_derivatives_and_unsafe_geometry_checks(self):
        shape = (7, 9, 11)
        data = np.zeros(shape, np.float32)
        z, y, x = np.indices(shape, dtype=np.float32)
        data[2:5, 3:6, 3:8] = (.1*x+.02*y+.04*z)[2:5, 3:6, 3:8]
        points = k.index_points32(shape)
        mapping = (np.eye(3, dtype=np.int32), np.zeros((3, 3), np.float32),
                   np.zeros(3, np.int32), np.array([.1, -.1, .2], np.float32), np.zeros(3, np.float32))
        bins, parameters = 16, 6
        fixed = np.full(len(points), 2, np.int32)
        shift = np.array([[.47, -.19, .08], [.16, .53, -.11], [-.07, .09, .49]], np.float32)
        spacing = np.ones(3, np.float32)
        local = (np.eye(3, dtype=np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32))
        linear = np.ones((parameters, 3, 3), np.float32)
        translation = np.zeros((parameters, 3), np.float32)
        support = zero_support32(data)
        width = np.float32(2/(bins-4))
        args = (data, points, fixed, mapping, bins, (np.float32(0), np.float32(2)),
                width, np.float32(-2), (shift, spacing), local, linear, translation)
        expected = derivative_histogram32(*args, points_are_integer=True)
        actual = derivative_histogram32(*args, points_are_integer=True, zero_support=support)
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        self.assertEqual(actual[3:], expected[3:])
        unsafe_local = (local[0], np.full(3, 1e30, np.float32), local[2])
        unsafe = (*args[:9], unsafe_local, np.full(linear.shape, 1e30, np.float32), translation)
        a = derivative_histogram32(*unsafe, points_are_integer=True)
        b = derivative_histogram32(*unsafe, points_are_integer=True, zero_support=support)
        self.assertTrue(a[-1]); self.assertTrue(b[-1])
        self.assertEqual(a[3:], b[3:])


if __name__ == '__main__':
    unittest.main()
