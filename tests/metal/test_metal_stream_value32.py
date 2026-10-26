"""Fixed-CSR direct MI values versus the existing Metal sampling operators."""
import os
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1', 'Explicit GPU test opt-in')
class MetalStreamValue32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import metric as mattes
        from fast_ants.backends.metal.metrics.mattes.histogram import fixed_bin_plan
        from fast_ants.backends.metal.runtime import MetalContext, mx
        from fast_ants.backends.metal.metrics.mattes.sampling import stream_value_histograms
        cls.mattes, cls.context_type, cls.mx = mattes, MetalContext, mx
        cls.plan = staticmethod(fixed_bin_plan)
        cls.stream = staticmethod(stream_value_histograms)

    def test_direct_sampling_histogram_matches_separate_oblique_dc_path(self):
        from fast_ants.pipeline.mattes32 import finalize_mattes_float1
        rng = np.random.default_rng(673)
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
        for dc in (0., 4096.):
            with self.subTest(dc=dc):
                ctx = self.context_type()
                data = ctx.upload((dc+.5+.17*np.sin(.3*x)+.11*np.cos(.2*y)+.004*z).astype(np.float32))
                indices = ctx.upload(points)
                fixed = rng.uniform(0, 1, len(points)).astype(np.float32)
                fixed[::3] = np.float32(.03)
                f = ctx.upload(fixed)
                bins, bounds = 16, (np.float32(dc+.41), np.float32(dc+.71))
                fixed_bins = self.mattes.fixed_bins(ctx, f, bins, (0., 1.))
                plan = self.plan(ctx, fixed_bins, bins)
                parameters = ctx.upload(self.mattes.float1_bin_parameters(bounds, bins))
                mapped = self.mattes.local_mapping(ctx, indices, *mapping)
                values, valid = self.mattes.sample(ctx, data, mapped)
                bi, bf = self.mattes.float1_bin_fractions(ctx, values, bins, parameters)
                weights = self.mattes.cubic_tables(ctx, bf, False)
                expected = self.mattes.histograms(ctx, f, values,
                    self.mx.zeros((len(points), 0), dtype=self.mx.float32), bins,
                    (0., 1.), bounds, bi, bf, valid=valid, cached_bins=fixed_bins, weights=weights)
                before = ctx.kernels.copy()
                actual = self.stream(ctx, data, indices, bins, bounds, mapping, plan,
                                     moving_parameters=parameters)
                self.assertEqual(actual[3:], expected[3:])
                np.testing.assert_array_equal(actual[1], expected[1])
                self.assertEqual(actual[2].shape, (bins, bins, 0))
                self.assertEqual(actual[0].dtype, np.float32)
                self.assertEqual(actual[1].dtype, np.int64)
                self.assertTrue(np.isfinite(actual[0]).all())
                np.testing.assert_allclose(actual[0], expected[0], atol=2e-4, rtol=1e-5)
                a = finalize_mattes_float1(*actual[:3], actual[3], actual[4])
                b = finalize_mattes_float1(*expected[:3], expected[3], expected[4])
                np.testing.assert_allclose(a[0], b[0], atol=5e-6, rtol=5e-6)
                calls = ctx.kernels-before
                for name in ('local_point_mapping', 'local_point_sample', 'float1_bin_fractions',
                             'cubic_tables', 'fused_value_sampling32'):
                    self.assertEqual(calls[name], 0)
                self.assertEqual(calls['stream_value_fixed_simd32_b16'], 1)
                self.assertEqual(sum(name.startswith('grouped_fixed_bin_') for name in calls), 0)

    def test_dynamic_domain_empty_ranges_and_nonfinite_nonzero_row(self):
        ctx = self.context_type()
        shape = (5, 7, 9)
        z, y, x = np.indices(shape, dtype=np.float32)
        array = (.1*x+.2*y+.3*z).astype(np.float32)
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        data, indices = ctx.upload(array), ctx.upload(points)
        bins = 16
        # All membership is in row 8, so diagnostics cannot depend on row 0
        # visiting all samples or providing the total-count column.
        fixed_bins = ctx.upload(np.full(x.size, 8, np.int32), dtype=np.int32)
        plan = self.plan(ctx, fixed_bins, bins)
        mapping = np.eye(3), np.zeros(3), np.zeros(3)
        def run(bounds=(0., 5.), geometry=mapping, image=data, samples=indices):
            return self.stream(ctx, image, samples, bins, bounds, geometry, plan)
        ordinary = run()
        self.assertEqual(ordinary[3], x.size)
        self.assertEqual(ordinary[1][8], x.size)
        self.assertEqual(int(ordinary[1].sum()), x.size)
        saved = ordinary[0].copy()
        for actual in (run((10., 11.)), run(geometry=(np.eye(3), np.full(3, 1e6), np.zeros(3)))):
            self.assertEqual(actual[3], 0)
            np.testing.assert_array_equal(actual[0], 0)
            np.testing.assert_array_equal(actual[1], 0)
        np.testing.assert_array_equal(ordinary[0], saved)
        bad_points = points.copy();bad_points[7, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            run(samples=ctx.upload(bad_points))
        bad_image = array.copy();bad_image[2, 3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            run(image=ctx.upload(bad_image))


if __name__ == '__main__':
    unittest.main()
