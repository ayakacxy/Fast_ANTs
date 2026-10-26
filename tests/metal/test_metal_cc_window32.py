"""Bounded opt-in checks for separable centered CC window statistics."""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalCCWindow32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal import dense
        from fast_ants.backends.metal.metrics.cc.window import cc_pair
        from fast_ants.backends.metal.runtime import MetalContext, mx
        from fast_ants.pipeline.float1_policy import FLOAT1_CC_EPS
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.dense, cls.window = dense, staticmethod(cc_pair)
        cls.context_type, cls.epsilon = MetalContext, float(FLOAT1_CC_EPS)

    def oracle(self, fixed, moving, fg, mg, mask, radius):
        """Direct centered FP64 math on already-quantized FP32 inputs."""
        forward = np.zeros(fg.shape, dtype=np.float64)
        reverse = np.zeros(mg.shape, dtype=np.float64)
        total = 0.
        for p in np.ndindex(fixed.shape):
            if not mask[p]:
                continue
            window = tuple(slice(max(0, k-radius), min(n, k+radius+1))
                           for k, n in zip(p, fixed.shape))
            take = mask[window]
            f = fixed[window][take].astype(np.float64)-float(fixed[p])
            m = moving[window][take].astype(np.float64)-float(moving[p])
            af, am = f.mean(), m.mean()
            f, m = f-af, m-am
            ff, mm, fm = np.dot(f, f), np.dot(m, m), np.dot(f, m)
            product = ff*mm
            total -= fm*fm/product if abs(product) > self.epsilon else 1.
            if ff > self.epsilon and mm > self.epsilon:
                forward[(slice(None), *p)] = (2*fm/product*(-af+fm/mm*am))*mg[(slice(None), *p)]
                reverse[(slice(None), *p)] = (2*fm/product*(-am+fm/ff*af))*fg[(slice(None), *p)]
        count = int(mask.sum())
        return forward, reverse, total/count if count else 0., count

    def check_pair(self, fixed, moving, mask, radius, guarded=False):
        fixed, moving = (np.asarray(a, np.float32) for a in (fixed, moving))
        z, y, x = np.indices(fixed.shape, dtype=np.float32)
        fg = np.stack((1+.03*x, -.4+.02*y, .7+.04*z)).astype(np.float32)
        mg = np.stack((-.6+.04*y, .8-.02*z, .3+.01*x)).astype(np.float32)
        oracle = self.oracle(fixed, moving, fg, mg, mask, radius)
        ctx = self.context_type(('parallel_cc',))
        ctx.serial_cc_guard = guarded
        arrays = [ctx.upload(a, 'cc_window_test') for a in (fixed, moving, fg, mg)]
        valid = ctx.upload(mask, 'cc_window_test', np.bool_)
        reference = self.dense.cc_pair(ctx, *arrays, valid, radius)
        before = ctx.kernels.copy()
        actual = self.window(ctx, *arrays, valid, radius)
        calls = ctx.kernels-before
        for name in ('cc_window_x32', 'cc_window_y32', 'cc_window_z_pair32'):
            self.assertEqual(calls[name], int(radius <= 4))
        self.assertEqual(calls['cc_pair'], int(radius > 4))
        self.assertEqual(actual[3], oracle[3])
        self.assertEqual(actual[3], reference[3])
        self.assertAlmostEqual(actual[2], oracle[2], delta=2e-6)
        self.assertAlmostEqual(actual[2], reference[2], delta=2e-6)
        outputs = []
        for output, expected, math_expected in zip(actual[:2], reference[:2], oracle[:2]):
            result = ctx.download(output, 'cc_window_test_result')
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result, math_expected, rtol=8e-5, atol=2e-5)
            np.testing.assert_allclose(result, ctx.download(expected, 'cc_window_test_reference'),
                                       rtol=8e-5, atol=2e-5)
            np.testing.assert_array_equal(result[:, ~mask], 0)
            outputs.append(result)
        return actual, outputs

    def test_centered_masked_windows_and_large_dc(self):
        z, y, x = np.indices((5, 7, 9), dtype=np.float32)
        fixed = np.sin(.3*x)+np.cos(.4*y)+.12*z
        moving = np.sin(.3*x+.09)+np.cos(.4*y-.12)+.1*z+.02*x*y
        mask = np.ones(fixed.shape, dtype=bool)
        mask[1:4, 2:5, 3:6] = False
        mask[:, 0, ::2] = False
        for radius in (2, 4):
            for offsets in ((0., 0.), (4096., -8192.)):
                with self.subTest(radius=radius, offsets=offsets):
                    self.check_pair(fixed+offsets[0], moving+offsets[1], mask, radius)

    def test_degenerate_singleton_zero_radius_and_dense_fallback(self):
        for shape, radius, kind in (((1, 5, 9), 0, 'varying'),
                                    ((1, 1, 7), 2, 'flat'),
                                    ((1, 5, 9), 2, 'low_variance'),
                                    ((1, 5, 9), 2, 'empty'),
                                    ((1, 5, 9), 5, 'sparse')):
            with self.subTest(shape=shape, radius=radius, kind=kind):
                z, y, x = np.indices(shape, dtype=np.float32)
                fixed, moving = np.full(shape, 1., np.float32), np.full(shape, 2., np.float32)
                mask = np.ones(shape, dtype=bool)
                if kind == 'varying':
                    fixed, moving = fixed+.2*x+.1*y, moving+.13*x-.11*y
                elif kind == 'low_variance':
                    fixed, moving = fixed+1e-6*x, moving+1e-6*y
                elif kind == 'empty':
                    mask[:] = False
                elif kind == 'sparse':
                    mask[:] = False
                    mask[0, 0, 0] = mask[0, -1, -1] = True
                actual, outputs = self.check_pair(fixed, moving, mask, radius, guarded=True)
                self.assertEqual(actual[2], -1. if mask.any() else 0.)
                for values in outputs:
                    np.testing.assert_array_equal(values, 0)


if __name__ == '__main__':
    unittest.main()
