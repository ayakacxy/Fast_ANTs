"""Value-only fusion contracts against the separate GPU operators.

The primitive checks cover exact half-voxel and float1 bin boundaries.
Fixed-state metric checks keep objective/derivatives, noncommuting transforms,
different image/mask grids and the nonindexed fallback explicit. No GPU import
occurs during ordinary test discovery; these tests do not establish speed.
"""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalFusedValue32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import sampling as fused_value32, metric as mattes
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import AVAILABLE_OPTIMIZATIONS, DEFAULT_OPTIMIZATIONS, MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.fused, cls.mattes = fused_value32, mattes
        cls.ops_type, cls.context_type = MetalOps, MetalContext
        cls.production_defaults = DEFAULT_OPTIMIZATIONS
        cls.defaults = tuple(flag for flag in DEFAULT_OPTIMIZATIONS if flag != 'fused_value_sampling')
        cls.available, cls.mx = AVAILABLE_OPTIMIZATIONS, mx

    def compare_primitive(self, data, points, linear, offset, center, bins, bounds):
        ctx = self.context_type(())
        device_data, device_points = ctx.upload(data), ctx.upload(points)
        params = ctx.upload(self.mattes.float1_bin_parameters(bounds, bins))
        positions = self.mattes.local_mapping(ctx, device_points, linear, offset, center)
        values, valid = self.mattes.sample(ctx, device_data, positions)
        bi, bf = self.mattes.float1_bin_fractions(ctx, values, bins, params)
        expected = (values, valid, bi, bf)
        actual = self.fused.sample_value_bins(ctx, device_data, device_points,
                                             linear, offset, center, bins, params)
        ctx.finish(*expected, *actual)
        # Both construction and evaluation remain on device; only explicit QA
        # downloads below are allowed to expose these per-point arrays.
        self.assertFalse(any(key.startswith('device_to_host:') for key in ctx.transfers))
        host = []
        for index, (result, reference) in enumerate(zip(actual, expected)):
            a = ctx.download(result, 'test_fused_value')
            b = ctx.download(reference, 'test_fused_value')
            self.assertEqual(a.dtype, b.dtype)
            if a.dtype == np.float32:
                self.assertTrue(np.isfinite(a).all())
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32),
                                              err_msg='separate/fused output '+str(index))
            else:
                np.testing.assert_array_equal(a, b)
            host.append(a)
        return host

    def test_validated_fusion_is_available_by_default(self):
        self.assertIn('fused_value_sampling', self.available)
        self.assertIn('fused_value_sampling', self.production_defaults)

    def test_default_context_fuses_value_only_and_preserves_derivatives(self):
        _, reference = self.metric(self.defaults, False)
        ctx, actual = self.metric(None, False)
        model, parameters = self.model_and_parameters('Affine', actual.grid.center)
        for derivative in (False, True):
            before = ctx.kernels.copy()
            self.compare_metric(actual.evaluate(model, parameters, derivative),
                                reference.evaluate(model, parameters, derivative))
            calls = ctx.kernels-before
            self.assertEqual(calls['fused_value_sampling32'], int(not derivative))
            self.assertEqual(calls['float1_bin_fractions'], int(derivative))

    def test_pixel_bin_and_half_voxel_boundaries_match_separate_operators(self):
        for bins in (5, 16, 32):
            for bounds in ((0., 1.), (.1, 1.3), (-4.3, -.7), (4096., 4097.), (-8192., -8190.)):
                with self.subTest(bins=bins, bounds=bounds):
                    edges = np.linspace(*bounds, bins-3).astype(np.float32)
                    pixels = np.concatenate((edges, np.nextafter(edges, np.float32(-np.inf)),
                                             np.nextafter(edges, np.float32(np.inf))))
                    data = np.broadcast_to(pixels, (2, 2, pixels.size)).copy()
                    nx = pixels.size
                    integer_points = np.stack((np.arange(nx), np.zeros(nx), np.zeros(nx)), axis=1)
                    boundaries = np.array([[-.5001, .25, .25], [-.5, .25, .25], [-.4999, .25, .25],
                                           [nx-.5001, .25, .25], [nx-.5, .25, .25], [nx-.4999, .25, .25],
                                           [.3, -.5, 0], [.3, 1.5, 0], [.3, .2, -.5], [.3, .2, 1.5]])
                    points = np.concatenate((integer_points, boundaries)).astype(np.float32)
                    center = np.array([np.floor((nx-1)*.5), 0., 0.])
                    values, valid, bi, bf = self.compare_primitive(
                        data, points, np.eye(3), center, center, bins, bounds)
                    expected_valid = np.all(points >= np.float32(-.5), axis=1) & np.all(
                        points < np.array([nx-.5, 1.5, 1.5], np.float32), axis=1)
                    np.testing.assert_array_equal(valid, expected_valid)
                    # The existing separate GPU interpolation path flushes
                    # this fixture's subnormal pixels to +0. Fusion must
                    # match that path (checked bitwise above), rather than
                    # promise CPU subnormal preservation. Keep exact checks
                    # for all normal pixels and their neighboring bin edges.
                    sampled_pixels = pixels.copy()
                    subnormal = ((np.abs(sampled_pixels) > np.float32(0)) &
                                 (np.abs(sampled_pixels) < np.finfo(np.float32).tiny))
                    sampled_pixels[subnormal] = np.float32(0)
                    np.testing.assert_array_equal(values[:nx].view(np.uint32), sampled_pixels.view(np.uint32))
                    width, normalized_min = self.mattes.float1_bin_parameters(bounds, bins)
                    term = np.asarray(sampled_pixels/width-normalized_min, np.float32)
                    expected_i = np.clip(np.trunc(term).astype(np.int32), 2, bins-3)
                    expected_f = np.asarray(term-expected_i.astype(np.float32), np.float32)
                    np.testing.assert_array_equal(bi[:nx], expected_i)
                    np.testing.assert_array_equal(bf[:nx].view(np.uint32), expected_f.view(np.uint32))
                    np.testing.assert_array_equal(values[~valid], np.zeros(int((~valid).sum()), np.float32))

    def test_fractional_points_and_coupled_local_affine_match_separate_operators(self):
        z, y, x = np.indices((4, 5, 7), dtype=np.float32)
        data = np.asarray(.2 + .04*x + .03*y + .02*z + .001*x*y, np.float32)
        points = np.array([[-.5, 0, 0], [0, 0, 0], [2.3, 1.7, .6], [6.4999, 4, 3],
                           [6.5, 4, 3], [-3.2, 2, 1], [3, 1.2, 2.9]], np.float32)
        linear = np.array([[.97, .08, .02], [-.05, 1.02, .03], [.01, -.04, 1.01]])
        offset = np.array([2.1, 1.9, 1.6])
        self.compare_primitive(data, points, linear, offset, np.array([3., 2., 1.]), 32, (.1, 1.3))

    def fixtures(self):
        from fast_ants.core.types import Grid, Image

        def centered_grid(shape, basis, center):
            matrix = np.eye(4)
            matrix[:3, :3] = basis
            matrix[:3, 3] = center-basis@((np.asarray(shape)-1)*.5)
            return Grid(shape, matrix)

        def rz(angle):
            return np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])

        center = np.array([17., -33., 9.])
        fixed_grid = centered_grid((9, 7, 5), rz(.21)@np.diag([.8, 1.3, 2.1]), center)
        moving_grid = centered_grid((9, 7, 5), rz(-.14)@np.diag([.85, 1.25, 2.]),
                                    center+np.array([.2, -.1, .1]))
        mask_grid = centered_grid((11, 9, 7), rz(.08)@np.diag([.9, 1.2, 2.]), center)
        z, y, x = np.indices(fixed_grid.shape_zyx)
        fixed = .5 + .17*np.sin(.6*x) + .11*np.cos(.5*y) + .03*z
        moving = .4 + .15*np.sin(.6*x+.09) + .13*np.cos(.5*y-.12) + .025*z + .004*x*y
        fixed_mask = np.ones(fixed_grid.shape_zyx, np.float32)
        fixed_mask[1:3, 2:4, 3:5] = 0
        moving_mask = np.ones(mask_grid.shape_zyx, np.float32)
        moving_mask[1:4, 2:5, 4:7] = 0
        moving_mask[:, :1, :] = 0
        return (Image(fixed.astype(np.float32), fixed_grid), Image(moving.astype(np.float32), moving_grid),
                Image(fixed_mask, fixed_grid), Image(moving_mask, mask_grid))

    def metric(self, options, masked):
        ctx = self.context_type(options)
        ops = self.ops_type(ctx)
        fixed, moving, fixed_mask, moving_mask = (ops.image(a) for a in self.fixtures())
        kwargs = {'fixed_mask': fixed_mask, 'moving_mask': moving_mask} if masked else {}
        return ctx, self.mattes.MetalMattesMetric(ops, fixed, moving, bins=16, **kwargs)

    def model_and_parameters(self, kind, center):
        from fast_ants.pipeline.models32 import ComposedLinearModel32, LinearModel32
        from fast_ants.core.transforms import AffineTransform
        base = AffineTransform.centered([[1.01, .02, 0], [0, .99, .01], [.01, 0, 1.02]],
                                        [.1, -.08, .03], center+np.array([.2, -.1, .2]))
        model = ComposedLinearModel32(kind, center, base)
        parameters = model.identity_parameters()
        if kind == 'Rigid':
            parameters[:3] = [.015, -.012, .008]
        else:
            parameters[0] += np.float32(.01)
            parameters[1] += np.float32(.005)
            parameters[5] += np.float32(.008)
        parameters[-3:] = [.09, -.07, .08]
        local = LinearModel32(kind, center).transform(parameters)
        # Verify the fixture actually distinguishes composition order.
        self.assertGreater(float(np.max(np.abs(base.matrix@local.matrix-local.matrix@base.matrix))), 1e-5)
        return model, parameters

    def compare_metric(self, actual, expected):
        self.assertEqual(actual.valid_points, expected.valid_points)
        self.assertTrue(np.isfinite(actual.value))
        np.testing.assert_allclose(actual.value, expected.value, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(actual.derivative, expected.derivative, atol=3e-7, rtol=3e-6)
        for name in ('joint_pdf', 'fixed_pdf', 'moving_pdf', 'pdf_derivative'):
            self.assertTrue(np.isfinite(actual.details[name]).all())
            np.testing.assert_allclose(actual.details[name], expected.details[name], atol=3e-7, rtol=3e-6)

    def test_rigid_affine_distinct_grids_masks_and_derivative_dispatch(self):
        for grouped in (False, True):
            for masked in (False, True):
                for kind in ('Rigid', 'Affine'):
                    for inline in (False, True):
                        options = (*self.defaults, *(('grouped_fixed_bins',) if grouped else ()),
                                   *(('value_inline_cubic',) if inline else ()))
                        _, baseline = self.metric(options, masked)
                        ctx, candidate = self.metric((*options, 'fused_value_sampling'), masked)
                        model, parameters = self.model_and_parameters(kind, candidate.grid.center)
                        for derivative in (False, True):
                            with self.subTest(grouped=grouped, masked=masked, kind=kind,
                                              inline=inline, derivative=derivative):
                                before = ctx.kernels.copy()
                                expected = baseline.evaluate(model, parameters, derivative)
                                actual = candidate.evaluate(model, parameters, derivative)
                                self.compare_metric(actual, expected)
                                calls = ctx.kernels-before
                                self.assertEqual(calls['fused_value_sampling32'], int(not derivative))
                                self.assertEqual(calls['local_point_mapping'], int(derivative)+int(masked))
                                self.assertEqual(calls['local_point_sample'], int(derivative)+int(masked))
                                self.assertEqual(calls['float1_bin_fractions'], int(derivative))
                                self.assertEqual(calls['local_point_gradient'], int(derivative))
                                self.assertEqual(calls['point_derivatives'], int(derivative))
                                self.assertEqual(calls['mi_mask_and'], int(masked))
                                self.assertEqual(calls['cubic_tables'], int(not derivative and not inline))
                                self.assertEqual(calls['cubic_tables_derivative'], int(derivative))
                        for key, entry in ctx.transfers.items():
                            if key.startswith('device_to_host:'):
                                self.assertLessEqual(entry['largest_array_bytes'], 16*16*13*4)

    def test_nonindexed_fallback_does_not_fuse(self):
        from fast_ants.pipeline.models32 import LinearModel32
        options = tuple(flag for flag in self.defaults if flag != 'indexed_mi')
        _, baseline = self.metric(options, False)
        ctx, candidate = self.metric((*options, 'fused_value_sampling'), False)
        model = LinearModel32('Affine', candidate.grid.center)
        for derivative in (False, True):
            before = ctx.kernels.copy()
            expected = baseline.evaluate(model, model.identity_parameters(), derivative)
            actual = candidate.evaluate(model, model.identity_parameters(), derivative)
            self.compare_metric(actual, expected)
            calls = ctx.kernels-before
            self.assertEqual(calls['fused_value_sampling32'], 0)
            self.assertEqual(calls['local_point_mapping'], 1)
            self.assertEqual(calls['float1_bin_fractions'], 1)

    def test_all_invalid_transform_keeps_no_valid_samples_error(self):
        from fast_ants.pipeline.models32 import LinearModel32
        from fast_ants.core.results import NoValidSamplesError
        for grouped in (False, True):
            options = (*self.defaults, *(('grouped_fixed_bins',) if grouped else ()), 'fused_value_sampling')
            _, metric = self.metric(options, False)
            model = LinearModel32('Affine', metric.grid.center)
            parameters = model.identity_parameters()
            parameters[-3:] = [1e6, -1e6, 1e6]
            for derivative in (False, True):
                with self.subTest(grouped=grouped, derivative=derivative):
                    with self.assertRaises(NoValidSamplesError):
                        metric.evaluate(model, parameters, derivative)


if __name__ == '__main__':
    unittest.main()
