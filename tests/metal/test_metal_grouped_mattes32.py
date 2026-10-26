"""Opt-in contracts for the fixed-bin CSR Metal MI candidate.

These tests check a static device index plan, independently accumulated
histograms, and changing transformed validity. They do not assert optimizer
trajectory identity or whole-registration performance. GPU imports occur only
after the explicit test opt-in.
"""

import os
import unittest

import numpy as np


def cubic_oracle(argument):
    """Double-precision spline formula, independent of the device source."""
    a = abs(float(argument))
    if a < 1.:
        return (4. - 6.*a*a + 3.*a*a*a)/6.
    if a < 2.:
        return (2.-a)**3/6.
    return 0.


def cubic_derivative_oracle(argument):
    a = abs(float(argument))
    sign = 1. if argument >= 0 else -1.
    if a < 1.:
        return sign*(-2.*a + 1.5*a*a)
    if a < 2.:
        return -.5*sign*(2.-a)**2
    return 0.


def histogram_oracle(fixed_bins, moving, valid, intensity, moving_bins, fractions,
                     bins, moving_range):
    """Direct double accumulation of the already-quantized FP32 samples."""
    joint = np.zeros((bins, bins), np.float64)
    derivative = np.zeros((bins, bins, intensity.shape[1]), np.float64)
    counts = np.zeros(bins, np.int64)
    for n in range(moving.size):
        if not valid[n] or moving[n] < moving_range[0] or moving[n] > moving_range[1]:
            continue
        fi, mi = int(fixed_bins[n]), int(moving_bins[n])
        counts[fi] += 1
        for column in range(mi-1, mi+3):
            argument = float(column-mi)-float(fractions[n])
            joint[fi, column] += cubic_oracle(argument)
            derivative[fi, column] += intensity[n].astype(np.float64)*cubic_derivative_oracle(argument)
    return joint, counts, derivative, int(counts.sum())


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalGroupedMattes32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import histogram as grouped_mattes32, metric as mattes
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import AVAILABLE_OPTIMIZATIONS, DEFAULT_OPTIMIZATIONS, MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.grouped, cls.mattes = grouped_mattes32, mattes
        cls.ops_type, cls.context_type, cls.mx = MetalOps, MetalContext, mx
        cls.defaults, cls.available = DEFAULT_OPTIMIZATIONS, AVAILABLE_OPTIMIZATIONS

    def test_candidate_is_explicit_and_not_default(self):
        self.assertIn('grouped_fixed_bins', self.available)
        self.assertNotIn('grouped_fixed_bins', self.defaults)

    def test_device_plan_is_stable_bin_grouping_across_multiple_tiles(self):
        rng = np.random.default_rng(20261024)
        for bins in (5, 16, 32):
            with self.subTest(bins=bins):
                ctx = self.context_type((*self.defaults, 'grouped_fixed_bins'))
                values = rng.integers(2, bins-2, size=4301, dtype=np.int32)
                plan = self.grouped.fixed_bin_plan(ctx, ctx.upload(values, dtype=np.int32), bins)
                ctx.finish(plan.indices, plan.offsets)
                # Construction and synchronization must not transfer samples
                # or the plan to the host. The explicit downloads below are QA.
                self.assertFalse(any(key.startswith('device_to_host:') for key in ctx.transfers))
                indices = ctx.download(plan.indices, 'test_grouped_plan')
                offsets = ctx.download(plan.offsets, 'test_grouped_plan')
                expected_indices = np.concatenate([np.flatnonzero(values == row) for row in range(bins)]).astype(np.uint32)
                expected_offsets = np.concatenate(([0], np.cumsum(np.bincount(values, minlength=bins)))).astype(np.uint32)
                self.assertEqual(plan.size, values.size)
                self.assertEqual(plan.bins, bins)
                self.assertEqual(plan.tile, 2048)
                self.assertEqual(indices.dtype, np.uint32)
                self.assertEqual(offsets.dtype, np.uint32)
                np.testing.assert_array_equal(indices, expected_indices)
                np.testing.assert_array_equal(offsets, expected_offsets)
                plan_calls = {key: count for key, count in ctx.kernels.items() if key.startswith('grouped_fixed_bin_')}
                self.assertEqual(len(plan_calls), 4)
                self.assertTrue(all(count == 1 for count in plan_calls.values()))

    def test_grouped_histograms_match_independent_oracle_with_dynamic_validity(self):
        rng = np.random.default_rng(9201)
        n, bins = 4301, 16
        moving_range = np.array([.1, 1.3], np.float32)
        fixed_bins = rng.integers(2, bins-2, size=n, dtype=np.int32)
        moving = rng.uniform(-.2, 1.6, size=n).astype(np.float32)
        moving[:2] = moving_range
        moving[2:4] = np.nextafter(moving_range, np.array([-np.inf, np.inf], np.float32))
        parameters = self.mattes.float1_bin_parameters(moving_range, bins)
        term = np.asarray(moving/parameters[0]-parameters[1], np.float32)
        moving_bins = np.clip(np.trunc(term).astype(np.int32), 2, bins-3)
        fractions = np.asarray(term-moving_bins.astype(np.float32), np.float32)
        valid = rng.random(n) > .22
        valid[:4] = True
        for p in (0, 6, 12):
            ctx = self.context_type((*self.defaults, 'grouped_fixed_bins'))
            plan = self.grouped.fixed_bin_plan(ctx, ctx.upload(fixed_bins, dtype=np.int32), bins)
            intensity = rng.uniform(-.3, .3, size=(n, p)).astype(np.float32)
            device_moving, device_intensity = ctx.upload(moving), ctx.upload(intensity)
            bi, bf = self.mattes.float1_bin_fractions(ctx, device_moving, bins, ctx.upload(parameters))
            initial_plan_calls = {key: count for key, count in ctx.kernels.items() if key.startswith('grouped_fixed_bin_')}
            for cached, current_valid in ((False, valid), (True, ~valid), (False, np.zeros(n, bool))):
                with self.subTest(parameters=p, cached=cached, valid_points=int(current_valid.sum())):
                    weights = self.mattes.cubic_tables(ctx, bf, p > 0) if cached else None
                    actual = self.grouped.histograms(
                        ctx, plan, device_moving, ctx.upload(current_valid, dtype=np.bool_), device_intensity,
                        bi, bf, moving_range, weights=weights, moving_width=parameters[0])
                    expected = histogram_oracle(fixed_bins, moving, current_valid, intensity,
                                                moving_bins, fractions, bins, moving_range)
                    self.assertEqual(actual[3], expected[3])
                    self.assertEqual(actual[4], float(parameters[0]))
                    self.assertEqual(actual[0].dtype, np.float32)
                    self.assertEqual(actual[1].dtype, np.int64)
                    self.assertEqual(actual[2].dtype, np.float32)
                    self.assertEqual(actual[2].shape, (bins, bins, p))
                    np.testing.assert_array_equal(actual[1], expected[1])
                    np.testing.assert_allclose(actual[0], expected[0], atol=4e-5, rtol=4e-6)
                    np.testing.assert_allclose(actual[2], expected[2], atol=4e-5, rtol=4e-6)
                    self.assertTrue(np.isfinite(actual[0]).all())
                    self.assertTrue(np.isfinite(actual[2]).all())
            self.assertEqual(initial_plan_calls, {key: count for key, count in ctx.kernels.items()
                                                 if key.startswith('grouped_fixed_bin_')})

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

    def metric(self, options, masked):
        ctx = self.context_type(options)
        ops = self.ops_type(ctx)
        fixed, moving, mask = (ops.image(a) for a in self.fixture())
        kwargs = {'fixed_mask': mask, 'moving_mask': mask} if masked else {}
        return ctx, self.mattes.MetalMattesMetric(ops, fixed, moving, bins=16, **kwargs)

    def test_metric_reuses_plan_and_tracks_moving_validity_for_both_models(self):
        from fast_ants.pipeline.models32 import LinearModel32
        for kind in ('Rigid', 'Affine'):
            for masked in (False, True):
                for inline in (False, True):
                    base_ctx, baseline = self.metric(self.defaults, masked)
                    options = (*self.defaults, 'grouped_fixed_bins', *(('value_inline_cubic',) if inline else ()))
                    ctx, candidate = self.metric(options, masked)
                    plan = candidate.grouped_plan
                    self.assertIsNotNone(plan)
                    initial_plan_calls = {key: count for key, count in ctx.kernels.items()
                                          if key.startswith('grouped_fixed_bin_')}
                    model = LinearModel32(kind, candidate.grid.center)
                    valid_counts = []
                    for translation in ((.09, -.07, .08), (1.6, .9, .8)):
                        parameters = model.identity_parameters()
                        if kind == 'Rigid':
                            parameters[:3] = [.015, -.012, .008]
                        else:
                            parameters[0] += np.float32(.01)
                            parameters[1] += np.float32(.005)
                        parameters[-3:] = translation
                        for derivative in (False, True):
                            with self.subTest(kind=kind, masked=masked, inline=inline,
                                              translation=translation, derivative=derivative):
                                before = ctx.kernels.copy()
                                expected = baseline.evaluate(model, parameters, derivative)
                                actual = candidate.evaluate(model, parameters, derivative)
                                calls = ctx.kernels-before
                                self.assertIs(candidate.grouped_plan, plan)
                                self.assertEqual(actual.valid_points, expected.valid_points)
                                valid_counts.append(actual.valid_points)
                                self.assertTrue(np.isfinite(actual.value))
                                np.testing.assert_allclose(actual.value, expected.value, atol=5e-6, rtol=5e-6)
                                np.testing.assert_allclose(actual.derivative, expected.derivative, atol=5e-6, rtol=1e-5)
                                for key in ('joint_pdf', 'fixed_pdf', 'moving_pdf', 'pdf_derivative'):
                                    self.assertTrue(np.isfinite(actual.details[key]).all())
                                    np.testing.assert_allclose(actual.details[key], expected.details[key], atol=5e-6, rtol=1e-5)
                                self.assertEqual(calls['mi_selection_scatter'], 0)
                                self.assertFalse(any(key.startswith('compact_') for key in calls))
                                name = 'cubic_tables_derivative' if derivative else 'cubic_tables'
                                self.assertEqual(calls[name], int(derivative or not inline))
                    self.assertGreater(len(set(valid_counts)), 1)
                    self.assertEqual(initial_plan_calls, {key: count for key, count in ctx.kernels.items()
                                                         if key.startswith('grouped_fixed_bin_')})
                    self.assertEqual(base_ctx.kernels['grouped_histogram_merge'], 0)

    def test_constant_ranges_and_all_invalid_transforms_are_rejected(self):
        from fast_ants.core.types import Image
        from fast_ants.pipeline.models32 import LinearModel32
        from fast_ants.core.results import NoValidSamplesError
        ctx = self.context_type((*self.defaults, 'grouped_fixed_bins'))
        ops = self.ops_type(ctx)
        fixed, moving, _ = self.fixture()
        constant = Image(np.ones(fixed.grid.shape_zyx, np.float32), fixed.grid)
        for a, b in ((constant, moving), (fixed, constant)):
            with self.subTest(constant_fixed=a is constant):
                with self.assertRaisesRegex(ValueError, 'nonconstant'):
                    self.mattes.MetalMattesMetric(ops, ops.image(a), ops.image(b), bins=16)
        _, metric = self.metric((*self.defaults, 'grouped_fixed_bins'), False)
        model = LinearModel32('Affine', metric.grid.center)
        parameters = model.identity_parameters()
        parameters[-3:] = [1e6, -1e6, 1e6]
        for derivative in (False, True):
            with self.subTest(derivative=derivative):
                with self.assertRaises(NoValidSamplesError):
                    metric.evaluate(model, parameters, derivative)


if __name__ == '__main__':
    unittest.main()
