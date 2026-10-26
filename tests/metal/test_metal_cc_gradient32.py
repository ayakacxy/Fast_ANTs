"""Device checks for the opt-in CC gradient-fusion candidate."""

import os
import unittest

import numpy as np

from fast_ants.core.types import Grid


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalCCGradient32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal import dense
        from fast_ants.backends.metal.metrics.cc.gradient import cc_pair_gradient
        from fast_ants.backends.metal.runtime import MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.dense, cls.context_type, cls.fused = dense, MetalContext, staticmethod(cc_pair_gradient)

    def fixture(self, shape=(5, 7, 9)):
        angle = .23
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]]) @ np.diag([.7, 1.3, 2.1])
        matrix[:3, 3] = [17, -33, 9]
        grid = Grid(shape, matrix)
        z, y, x = np.indices(shape, dtype=np.float32)
        fixed = np.asarray(np.sin(.3*x) + np.cos(.4*y) + .12*z, np.float32)
        moving = np.asarray(np.sin(.3*x + .09) + np.cos(.4*y - .12) + .1*z + .02*x*y, np.float32)
        shift = grid.lps_to_index[:3, :3] @ np.diag(grid.spacing_xyz * .5)
        return fixed, moving, shift, grid.spacing_xyz

    def compare(self, ctx, fixed, moving, valid, radius, shift, spacing):
        f, m = ctx.upload(fixed), ctx.upload(moving)
        mask = ctx.upload(valid, dtype=np.bool_)
        reference = self.dense.cc_pair(
            ctx, f, m, self.dense.gradient(ctx, f, shift, spacing),
            self.dense.gradient(ctx, m, shift, spacing), mask, radius)
        before = ctx.kernels.copy()
        actual = self.fused(ctx, f, m, mask, radius, shift, spacing)
        dispatches = ctx.kernels - before
        self.assertEqual(dispatches['cc_pair_gradient'], 1)
        self.assertEqual(dispatches['gradient'], 0)
        self.assertEqual(dispatches['cc_pair'], 0)
        self.assertEqual(actual[2:], reference[2:])
        for got, wanted in zip(actual[:2], reference[:2]):
            values = ctx.download(got, 'cc_gradient_test')
            expected = ctx.download(wanted, 'cc_gradient_test')
            self.assertTrue(np.isfinite(values).all())
            np.testing.assert_allclose(values, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(ctx.download(f, 'cc_gradient_input_test'), fixed)
        np.testing.assert_array_equal(ctx.download(m, 'cc_gradient_input_test'), moving)
        return actual

    def test_oblique_anisotropic_mask_edges_and_large_dc(self):
        fixed, moving, shift, spacing = self.fixture()
        ctx = self.context_type(('parallel_cc',))
        valid = np.ones(fixed.shape, np.bool_)
        valid[1:4, 2:5, 3:6] = False
        valid[:, :, 0] = False
        for dc in (False, True):
            f = fixed + np.float32(4096) if dc else fixed
            m = moving + np.float32(-8192) if dc else moving
            for radius in (1, 2):
                with self.subTest(dc=dc, radius=radius):
                    self.compare(ctx, f, m, valid, radius, shift, spacing)

    def test_flat_singleton_empty_and_guarded_reduction(self):
        ctx = self.context_type(('parallel_cc',))
        ctx.serial_cc_guard = True
        for shape in ((1, 1, 7), (1, 5, 9), (5, 7, 9)):
            fixed, moving, shift, spacing = self.fixture(shape)
            if shape == (5, 7, 9):
                fixed.fill(np.float32(1e6)); moving.fill(np.float32(-2e6))
            for empty in (False, True):
                valid = np.full(shape, not empty, np.bool_)
                with self.subTest(shape=shape, empty=empty):
                    result = self.compare(ctx, fixed, moving, valid, 3, shift, spacing)
                    if empty:
                        self.assertEqual(result[2:], (0., 0))
                    if empty or shape == (5, 7, 9):
                        for update in result[:2]:
                            np.testing.assert_array_equal(ctx.download(update, 'cc_gradient_zero_test'), 0)


if __name__ == '__main__':
    unittest.main()
