"""Opt-in device checks for the shared-statistics SyN CC candidate.

Run explicitly after GPU testing is resumed:
    FAST_ANTS_TEST_METAL=1 python -m unittest discover -s tests -p test_metal_cc_pair.py

The module does not import MLX or initialize a GPU during ordinary discovery.
The separate-direction implementation remains the per-operator reference;
end-to-end registration must additionally pass against ANTs --float 1.
"""

import os
import unittest

import numpy as np

from fast_ants.core.types import Grid, Image
from fast_ants.core.results import NoValidSamplesError


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalCCPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.ops_type, cls.context_type = MetalOps, MetalContext

    def fixture(self, kind):
        matrix = np.eye(4)
        angle = .23
        matrix[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]) @ np.diag([.7, 1.3, 2.1])
        matrix[:3, 3] = [17, -33, 9]
        grid = Grid((5, 7, 9), matrix)
        z, y, x = np.indices(grid.shape_zyx)
        fixed = np.sin(.3*x) + np.cos(.4*y) + .12*z
        moving = np.sin(.3*x + .09) + np.cos(.4*y - .12) + .1*z + .02*x*y
        if kind == 'flat':
            fixed, moving = np.ones_like(fixed), np.full_like(moving, 3)
        elif kind == 'low_contrast':
            fixed, moving = 1 + fixed*1e-5, 2 + moving*1e-5
        elif kind == 'large_offset':
            fixed, moving = 4096 + fixed, -8192 + moving
        elif kind == 'negative_correlation':
            moving = -moving
        return Image(fixed.astype(np.float32), grid), Image(moving.astype(np.float32), grid)

    def assert_same(self, ctx, actual, expected):
        self.assertEqual(actual.value, expected.value)
        self.assertEqual(actual.valid_points, expected.valid_points)
        actual_values = ctx.download(actual.update.data, 'paired_cc_test')
        expected_values = ctx.download(expected.update.data, 'paired_cc_test')
        self.assertTrue(np.isfinite(actual_values).all())
        # Include signed zeros and every stored FP32 bit in the comparison.
        np.testing.assert_array_equal(actual_values.view(np.uint32), expected_values.view(np.uint32))

    def test_pair_matches_independent_directions(self):
        for options in ((), ('parallel_cc',)):
            ctx = self.context_type(options)
            ops = self.ops_type(ctx)
            for kind in ('normal', 'flat', 'low_contrast', 'large_offset', 'negative_correlation'):
                fixed, moving = self.fixture(kind)
                fixed, moving = ops.image(fixed), ops.image(moving)
                valid = np.ones(fixed.data.shape, dtype=bool)
                valid[1:3, 2:5, 3:7] = False
                sparse = np.zeros_like(valid)
                sparse[0, 0, 0] = sparse[-1, -1, -1] = True
                for radius in (0, 1, 2, 9):
                    for mask in (None, valid, sparse):
                        with self.subTest(options=options, kind=kind, radius=radius,
                                          masked=mask is not None):
                            device_mask = None if mask is None else ctx.upload(mask, dtype=np.bool_)
                            reference_f = ops.cc(fixed, moving, radius, device_mask)
                            reference_m = ops.cc(moving, fixed, radius, device_mask)
                            forward, reverse = ops.cc_pair(fixed, moving, radius, device_mask)
                            self.assert_same(ctx, forward, reference_f)
                            self.assert_same(ctx, reverse, reference_m)

    def test_pair_handles_guarded_reduction_and_empty_domain(self):
        ctx = self.context_type(('parallel_cc', 'paired_cc'))
        ctx.serial_cc_guard = True
        ops = self.ops_type(ctx)
        fixed, moving = (ops.image(value) for value in self.fixture('normal'))
        reference_f = ops.cc(fixed, moving)
        reference_m = ops.cc(moving, fixed)
        forward, reverse = ops.cc_pair(fixed, moving)
        self.assert_same(ctx, forward, reference_f)
        self.assert_same(ctx, reverse, reference_m)
        empty = ctx.upload(np.zeros(fixed.data.shape, dtype=bool), dtype=np.bool_)
        with self.assertRaises(NoValidSamplesError):
            ops.cc_pair(fixed, moving, valid=empty)

    def test_pair_reduces_window_and_scalar_dispatches(self):
        ctx = self.context_type(('parallel_cc', 'paired_cc'))
        ops = self.ops_type(ctx)
        fixed, moving = (ops.image(value) for value in self.fixture('normal'))
        before = ctx.kernels.copy()
        ops.cc_pair(fixed, moving)
        calls = ctx.kernels - before
        self.assertEqual(calls['cc_pair'], 1)
        self.assertEqual(calls['cc'], 0)
        self.assertEqual(calls['gradient'], 2)
        self.assertEqual(calls['valid_ones'], 1)
        self.assertEqual(calls['cc_reduce_tiles'], 1)
        self.assertEqual(calls['cc_reduce_merge'], 1)


if __name__ == '__main__':
    unittest.main()
