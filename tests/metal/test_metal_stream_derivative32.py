"""Streaming Metal derivative PDFs against the existing operator composition."""
import os
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1', 'Explicit GPU test opt-in')
class MetalStreamDerivative32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import metric as mattes
        from fast_ants.backends.metal.metrics.mattes.derivative import stream_derivative_histograms
        from fast_ants.backends.metal.runtime import MetalContext
        cls.mattes, cls.context_type = mattes, MetalContext
        cls.stream = staticmethod(stream_derivative_histograms)

    def reference(self, ctx, data, points, fixed, bins, moving_range,
                  mapping, gradient_geometry, local_geometry, linear, translation):
        m = self.mattes
        mapped = m.local_mapping(ctx, points, *mapping)
        values, valid = m.sample(ctx, data, mapped)
        gradients = m.gradient(ctx, data, mapped, *gradient_geometry)
        local = m.affine(ctx, points, *local_geometry)
        intensity = m.derivatives(ctx, local, gradients, linear, translation)
        parameters = ctx.upload(m.float1_bin_parameters(moving_range, bins))
        bi, bf = m.float1_bin_fractions(ctx, values, bins, parameters)
        fixed_bins = m.fixed_bins(ctx, fixed, bins, (0., 1.))
        weights = m.cubic_tables(ctx, bf, True)
        expected = m.histograms(ctx, fixed, values, intensity, bins, (0., 1.),
            moving_range, bi, bf, valid=valid, cached_bins=fixed_bins, weights=weights)
        return expected, fixed_bins, parameters

    def test_oblique_fractional_boundaries_unit_and_generic_jacobians(self):
        from fast_ants.pipeline.mattes32 import finalize_mattes_float1
        from fast_ants.backends.metal.metrics.mattes.histogram import fixed_bin_plan
        rng = np.random.default_rng(582)
        shape = (17, 29, 41)
        z, y, x = np.indices(shape, dtype=np.float32)
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        points[::13] += np.asarray([.17, -.09, .13], np.float32)
        points = np.vstack((points, np.asarray([[-.5, .2, .3], [-.5001, .2, .3],
                                               [40.5, .2, .3], [40.4999, .2, .3]], np.float32)))
        floor_center = np.floor((np.asarray(shape[::-1])-1)*.5)
        mapping = (np.eye(3)+np.asarray([[.003, .002, -.001], [-.002, -.004, .003],
                                        [.001, -.002, .005]]),
                   floor_center+np.asarray([.13, -.21, .09]), floor_center)
        angle = .23
        direction = np.asarray([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        spacing = np.asarray([.7, 1.3, 2.1], np.float32)
        shift = np.diag(1/spacing)@direction.T@np.diag(spacing*.5)
        local_geometry = (direction@np.diag(spacing), np.asarray([.12, -.07, .11]),
                          (np.asarray(shape[::-1])-1)*.5)
        base = np.asarray([[.97, .11, -.07], [-.12, 1.03, .06],
                           [.02, -.08, .99]], np.float32)
        for p, dc in ((6, 0.), (12, 4096.)):
            with self.subTest(parameters=p, dc=dc):
                ctx = self.context_type()
                image = (dc+.5+.17*np.sin(.3*x)+.11*np.cos(.2*y)+.004*z).astype(np.float32)
                fixed = rng.uniform(0, 1, len(points)).astype(np.float32)
                linear = np.zeros((p, 3, 3), np.float32)
                translation = np.zeros((p, 3), np.float32)
                if p == 12:
                    # A previous-stage affine transforms the unit local
                    # Jacobian into a three-vector in one coordinate column.
                    for j in range(9):linear[j, :, j%3] = base[:, j//3]
                    for j in range(3):translation[j+9] = base[:, j]
                    # Retain all plan types and a generic parameter, with no
                    # reliance on the built-in model's parameter ordering.
                    linear[0] = 0;linear[0, 1, 2] = 1
                    translation[11] = [0, 1, 0]
                    linear[2] = rng.normal(0, .05, (3, 3)).astype(np.float32)
                    translation[2] = [.01, -.02, .03]
                else:
                    linear[:3] = rng.normal(0, .05, (3, 3, 3)).astype(np.float32)
                    translation[:3] = rng.normal(0, .02, (3, 3)).astype(np.float32)
                    for j in range(3):translation[j+3] = base[:, j]
                    translation[5] = [0, 1, 0]
                from fast_ants.backends.metal.metrics.mattes.derivative import _jacobian_plan
                kinds = set(_jacobian_plan(linear, translation)[:, 0])
                self.assertTrue({0, 2, 4} <= kinds)
                if p == 12:self.assertTrue({1, 3} <= kinds)
                moving_range = (np.float32(dc+.41), np.float32(dc+.71))
                uploaded = ctx.upload(image), ctx.upload(points), ctx.upload(fixed)
                expected, fixed_bins, bin_parameters = self.reference(
                    ctx, *uploaded, 16, moving_range, mapping, (shift, spacing),
                    local_geometry, linear, translation)
                for fixed_plan in (None, fixed_bin_plan(ctx, fixed_bins, 16)):
                    with self.subTest(fixed_rows=fixed_plan is not None):
                        before = ctx.kernels.copy()
                        actual = self.stream(ctx, uploaded[0], uploaded[1], fixed_bins, 16,
                            moving_range, mapping, (shift, spacing), local_geometry,
                            linear, translation, moving_parameters=bin_parameters,
                            fixed_plan=fixed_plan)
                        self.assertEqual(actual[3:], expected[3:])
                        np.testing.assert_array_equal(actual[1], expected[1])
                        np.testing.assert_allclose(actual[0], expected[0], atol=2e-4, rtol=5e-6)
                        np.testing.assert_allclose(actual[2], expected[2], atol=3e-3, rtol=3e-5)
                        a = finalize_mattes_float1(*actual[:3], actual[3], actual[4])
                        b = finalize_mattes_float1(*expected[:3], expected[3], expected[4])
                        np.testing.assert_allclose(a[0], b[0], atol=5e-6, rtol=5e-6)
                        np.testing.assert_allclose(a[1], b[1], atol=1e-5, rtol=2e-5)
                        calls = ctx.kernels-before
                        for name in ('local_point_mapping', 'local_point_sample', 'local_point_gradient',
                                     'point_affine', 'point_derivatives', 'cubic_tables_derivative'):
                            self.assertEqual(calls[name], 0)
                        self.assertEqual(calls['stream_derivative_sample32'], 1)
                        self.assertTrue(any(name.startswith('stream_derivative_histogram_simd32') for name in calls))
                        if fixed_plan is not None:
                            self.assertTrue(any(name.endswith('_fixed_rows') for name in calls))

    def test_unsafe_derivatives_remain_visible_and_rejected_ranges_skip_them(self):
        from fast_ants.backends.metal.metrics.mattes.histogram import fixed_bin_plan
        ctx = self.context_type()
        shape = (5, 7, 9)
        z, y, x = np.indices(shape, dtype=np.float32)
        data = ctx.upload((.1*x+.2*y+.3*z).astype(np.float32))
        points = ctx.upload(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
        fixed_bins = ctx.upload(np.full(x.size, 2, np.int32), dtype=np.int32)
        mapping = np.eye(3), np.zeros(3), np.zeros(3)
        shift = np.eye(3)*.5
        linear = np.zeros((1, 3, 3), np.float32)
        translation = np.asarray([[1, 0, 0]], np.float32)
        local = np.eye(3), np.zeros(3), np.zeros(3)
        def run(spacing, local_geometry=local, bounds=(0., 5.), map_parameters=mapping,
                fixed_plan=None):
            return self.stream(ctx, data, points, fixed_bins, 16, bounds, map_parameters,
                               (shift, spacing), local_geometry, linear, translation,
                               fixed_plan=fixed_plan)
        unsafe_local = np.eye(3), np.full(3, np.inf), np.zeros(3)
        outside = np.eye(3), np.full(3, 1e6), np.zeros(3)
        # All fixed samples belong to row 2, so row 0 has no CSR members.
        # A nonfinite moving sentinel must be checked in its actual row,
        # rather than silently disappearing in an empty row-zero check.
        for fixed_plan in (None, fixed_bin_plan(ctx, fixed_bins, 16)):
            with self.subTest(fixed_rows=fixed_plan is not None):
                # Inf in an unselected gradient axis must still expose 0*inf.
                with self.assertRaises(ValueError):
                    run(np.asarray([1, 1, 1e-39], np.float32), fixed_plan=fixed_plan)
                with self.assertRaises(ValueError):
                    run(np.ones(3), unsafe_local, fixed_plan=fixed_plan)
                rejected = run(np.asarray([1, 1, 1e-39], np.float32), unsafe_local,
                               (10., 11.), fixed_plan=fixed_plan)
                self.assertEqual(rejected[3], 0)
                np.testing.assert_array_equal(rejected[0], 0)
                np.testing.assert_array_equal(rejected[2], 0)
                empty = run(np.ones(3), map_parameters=outside, fixed_plan=fixed_plan)
                self.assertEqual(empty[3], 0)
                np.testing.assert_array_equal(empty[0], 0)

    def test_metric_integration_and_moving_mask_fallback(self):
        from fast_ants.core.types import Grid, Image
        from fast_ants.pipeline.models32 import ComposedLinearModel32
        from fast_ants.core.transforms import AffineTransform
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import DEFAULT_OPTIMIZATIONS
        options=tuple(flag for flag in DEFAULT_OPTIMIZATIONS if flag not in
                      ('streamed_mi_derivative','streamed_mi_fixed_plan',
                       'packed_mi_controls','inline_cc_gradient'))
        matrix=np.diag([.8,1.3,2.1,1.]);matrix[:3,3]=[17.,-33.,9.]
        grid=Grid((17,29,37),matrix)
        z,y,x=np.indices(grid.shape_zyx)
        fixed=(.5+.17*np.sin(.6*x)+.11*np.cos(.5*y)+.03*z).astype(np.float32)
        moving=(.4+.15*np.sin(.6*x+.09)+.13*np.cos(.5*y-.12)+.025*z+.004*x*y).astype(np.float32)
        metrics=[]
        for flags in (options,(*options,'streamed_mi_derivative','packed_mi_controls'),
                      (*options,'streamed_mi_derivative','packed_mi_controls','streamed_mi_fixed_plan')):
            ops=MetalOps(self.context_type(flags))
            metrics.append(self.mattes.MetalMattesMetric(
                ops,ops.image(Image(fixed,grid)),ops.image(Image(moving,grid)),bins=16))
        base=AffineTransform.centered(np.asarray([[.97,.11,-.07],[-.12,1.03,.06],
                                                  [.02,-.08,.99]]),
                                      [.2,-.1,.15],grid.center)
        model=ComposedLinearModel32('Affine',grid.center,base)
        p=model.identity_parameters();p[1]=np.float32(.007);p[-3:]=[.3,-.21,.17]
        for derivative in (True,False):
            expected=metrics[0].evaluate(model,p,derivative)
            for metric in metrics[1:]:
                actual=metric.evaluate(model,p,derivative)
                self.assertEqual(actual.valid_points,expected.valid_points)
                np.testing.assert_allclose(actual.value,expected.value,atol=5e-6,rtol=5e-6)
                np.testing.assert_allclose(actual.derivative,expected.derivative,atol=5e-6,rtol=1e-5)
                self.assertEqual(bool(actual.details.get('fused_derivative')),derivative)
        candidate=metrics[-1]
        self.assertEqual(candidate.ctx.kernels['grouped_fixed_bin_counts'],1)
        candidate.evaluate(model,p,True)
        self.assertEqual(candidate.ctx.kernels['grouped_fixed_bin_counts'],1)
        candidate.moving_mask=candidate.ops.image(Image(np.ones(grid.shape_zyx,np.float32),grid))
        before=candidate.ctx.kernels.copy()
        fallback=candidate.evaluate(model,p,True)
        self.assertNotIn('fused_derivative',fallback.details)
        calls=candidate.ctx.kernels-before
        self.assertEqual(calls['stream_derivative_sample32'],0)
        self.assertEqual(calls['point_derivatives'],1)


if __name__ == '__main__':
    unittest.main()
