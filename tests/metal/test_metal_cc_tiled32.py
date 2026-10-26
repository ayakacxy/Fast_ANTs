"""Bounded opt-in checks for shared-halo CC; ordinary discovery uses no GPU."""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalCCTiled32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal import dense
        from fast_ants.backends.metal.metrics.cc.tiled import cc_pair
        from fast_ants.backends.metal.runtime import MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.dense, cls.tiled = dense, staticmethod(cc_pair)
        cls.context_type = MetalContext

    def check_pair(self, fixed, moving, mask, radius, guarded=False):
        ctx = self.context_type(('parallel_cc',))
        ctx.serial_cc_guard = guarded
        fixed, moving = (np.asarray(a, np.float32) for a in (fixed, moving))
        # Independent, spatially varying covectors exercise both update
        # directions without changing the precomputed-gradient contract.
        z, y, x = np.indices(fixed.shape, dtype=np.float32)
        fg = np.stack((1+.03*x, -.4+.02*y, .7+.04*z)).astype(np.float32)
        mg = np.stack((-.6+.04*y, .8-.02*z, .3+.01*x)).astype(np.float32)
        arrays = [ctx.upload(a, 'cc_tiled_test') for a in (fixed, moving, fg, mg)]
        valid = ctx.upload(mask, 'cc_tiled_test', np.bool_)
        expected = self.dense.cc_pair(ctx, *arrays, valid, radius)
        before = ctx.kernels.copy()
        actual = self.tiled(ctx, *arrays, valid, radius)
        calls = ctx.kernels-before
        self.assertEqual(calls['cc_tiled_pair32_r'+str(radius)], int(radius <= 2))
        self.assertEqual(calls['cc_pair'], int(radius > 2))
        self.assertEqual(actual[3], expected[3])
        self.assertEqual(actual[3], int(mask.sum()))
        self.assertAlmostEqual(actual[2], expected[2], delta=3e-7)
        outputs = []
        for result, reference in zip(actual[:2], expected[:2]):
            values = ctx.download(result, 'cc_tiled_test_result')
            self.assertTrue(np.isfinite(values).all())
            np.testing.assert_allclose(values, ctx.download(reference, 'cc_tiled_test_reference'),
                                       rtol=3e-6, atol=3e-6)
            np.testing.assert_array_equal(values[:, ~mask], 0)
            outputs.append(values)
        return actual, outputs

    def test_masked_tile_tails_and_centered_large_offsets(self):
        # All axes cross a tile boundary, with partial tiles at every edge.
        z, y, x = np.indices((5, 9, 19), dtype=np.float32)
        fixed = np.sin(.3*x)+np.cos(.4*y)+.12*z
        moving = np.sin(.3*x+.09)+np.cos(.4*y-.12)+.1*z+.02*x*y
        mask = np.ones(fixed.shape, dtype=bool)
        mask[1:4, 2:6, 7:12] = False
        mask[0, :, 0] = False
        for radius in (1, 2):
            for offsets in ((0., 0.), (4096., -8192.)):
                with self.subTest(radius=radius, offsets=offsets):
                    self.check_pair(fixed+offsets[0], moving+offsets[1], mask, radius)

    def test_singleton_flat_empty_and_large_radius_fallback(self):
        for shape, radius, kind in (((1, 1, 7), 0, 'all'),
                                    ((1, 5, 17), 2, 'all'),
                                    ((1, 5, 17), 2, 'empty'),
                                    ((1, 5, 17), 3, 'sparse')):
            with self.subTest(shape=shape, radius=radius, mask=kind):
                mask = np.ones(shape, dtype=bool)
                if kind == 'empty':
                    mask[:] = False
                elif kind == 'sparse':
                    mask[:] = False
                    mask[0, 0, 0] = mask[0, -1, -1] = True
                actual, outputs = self.check_pair(np.full(shape, 1e6, np.float32),
                                                  np.full(shape, -2e6, np.float32),
                                                  mask, radius, guarded=True)
                self.assertEqual(actual[2], -1. if mask.any() else 0.)
                for values in outputs:
                    np.testing.assert_array_equal(values, 0)


if __name__ == '__main__':
    unittest.main()
