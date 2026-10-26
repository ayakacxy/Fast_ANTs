"""Opt-in fixed-state contracts for the value-only inline cubic candidate.

No MLX import or GPU initialization occurs during ordinary test discovery.
The candidate must also pass whole-registration quality and timing A/B before
it becomes a default optimization; these small tests do not establish speed.
"""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalValueInlineCubicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import metric as mattes
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import AVAILABLE_OPTIMIZATIONS, DEFAULT_OPTIMIZATIONS, MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.mattes, cls.ops_type, cls.context_type = mattes, MetalOps, MetalContext
        cls.defaults, cls.available = DEFAULT_OPTIMIZATIONS, AVAILABLE_OPTIMIZATIONS

    def fixture(self):
        from fast_ants.core.types import Grid, Image
        matrix = np.eye(4)
        angle = .21
        matrix[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]) @ np.diag([.8, 1.3, 2.1])
        matrix[:3, 3] = [17, -33, 9]
        grid = Grid((9, 7, 5), matrix)
        z, y, x = np.indices(grid.shape_zyx)
        fixed = .5 + .17*np.sin(.6*x) + .11*np.cos(.5*y) + .03*z
        moving = .4 + .15*np.sin(.6*x+.09) + .13*np.cos(.5*y-.12) + .025*z + .004*x*y
        mask = np.ones(grid.shape_zyx, np.float32)
        mask[1:3, 2:4, 3:5] = 0
        return [Image(a.astype(np.float32), grid) for a in (fixed, moving, mask)]

    def metrics(self, options, masked):
        ctx = self.context_type(options)
        ops = self.ops_type(ctx)
        fixed, moving, mask = (ops.image(a) for a in self.fixture())
        kwargs = {'fixed_mask': mask, 'moving_mask': mask} if masked else {}
        metric = self.mattes.MetalMattesMetric(ops, fixed, moving, bins=16, **kwargs)
        return ctx, metric

    def test_candidate_is_explicit_and_not_default(self):
        self.assertIn('value_inline_cubic', self.available)
        self.assertNotIn('value_inline_cubic', self.defaults)
        self.assertIn('cached_cubic', self.defaults)

    def test_fixed_parameters_preserve_metric_and_derivative_policy(self):
        from fast_ants.pipeline.models32 import LinearModel32
        for kind in ('Rigid', 'Affine'):
            for masked in (False, True):
                baseline_ctx, baseline = self.metrics(self.defaults, masked)
                candidate_ctx, candidate = self.metrics((*self.defaults, 'value_inline_cubic'), masked)
                model = LinearModel32(kind, baseline.grid.center)
                parameters = model.identity_parameters()
                if kind == 'Rigid':
                    parameters[:3] = [.015, -.012, .008]
                else:
                    parameters[0] += np.float32(.01)
                    parameters[1] += np.float32(.005)
                parameters[-3:] = [.09, -.07, .08]
                for derivative in (False, True):
                    with self.subTest(kind=kind, masked=masked, derivative=derivative):
                        before_baseline, before_candidate = baseline_ctx.kernels.copy(), candidate_ctx.kernels.copy()
                        expected = baseline.evaluate(model, parameters, derivative)
                        actual = candidate.evaluate(model, parameters, derivative)
                        baseline_calls = baseline_ctx.kernels - before_baseline
                        candidate_calls = candidate_ctx.kernels - before_candidate
                        self.assertEqual(actual.valid_points, expected.valid_points)
                        self.assertTrue(np.isfinite(actual.value))
                        np.testing.assert_allclose(actual.value, expected.value, rtol=2e-6, atol=2e-7)
                        np.testing.assert_allclose(actual.derivative, expected.derivative, rtol=3e-6, atol=3e-7)
                        for key in ('joint_pdf', 'fixed_pdf', 'moving_pdf', 'pdf_derivative'):
                            self.assertTrue(np.isfinite(actual.details[key]).all())
                            np.testing.assert_allclose(actual.details[key], expected.details[key],
                                                       rtol=3e-6, atol=3e-7)
                        name = 'cubic_tables_derivative' if derivative else 'cubic_tables'
                        self.assertEqual(baseline_calls[name], 1)
                        self.assertEqual(candidate_calls[name], int(derivative))
                        self.assertEqual(candidate_calls['cubic_tables'], 0)
                        histogram_names = [name for name in candidate_calls if name.startswith('histogram_private')]
                        self.assertEqual(len(histogram_names), 1)
                        self.assertEqual('_cached_cubic' in histogram_names[0], derivative)

    def test_existing_uncached_mode_stays_uncached(self):
        from fast_ants.pipeline.models32 import LinearModel32
        options = tuple(x for x in self.defaults if x != 'cached_cubic')
        ctx, metric = self.metrics((*options, 'value_inline_cubic'), False)
        model = LinearModel32('Affine', metric.grid.center)
        for derivative in (False, True):
            before = ctx.kernels.copy()
            result = metric.evaluate(model, model.identity_parameters(), derivative)
            calls = ctx.kernels - before
            self.assertTrue(np.isfinite(result.value))
            self.assertEqual(calls['cubic_tables'], 0)
            self.assertEqual(calls['cubic_tables_derivative'], 0)


if __name__ == '__main__':
    unittest.main()
