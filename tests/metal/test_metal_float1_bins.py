"""Device checks for ITK float-1 bin arithmetic on rounded samples."""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalFloat1BinsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import metric as mattes
        from fast_ants.backends.metal.runtime import MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.mattes, cls.context_type = mattes, MetalContext

    def reference(self, values, value_range, bins):
        low, high = np.asarray(value_range, np.float32)
        width = np.float32(np.float32(high-low)/np.float32(bins-4))
        normalized_min = np.float32(np.float32(low/width)-np.float32(2))
        term = np.asarray(values/width-normalized_min, np.float32)
        indices = np.clip(np.trunc(term).astype(np.int32), 2, bins-3)
        fractions = np.asarray(term-indices.astype(np.float32), np.float32)
        return np.array([width, normalized_min], np.float32), indices, fractions

    def test_rounded_samples_match_float1_at_bin_boundaries(self):
        ctx = self.context_type(())
        for bins in (5, 8, 32, 64):
            for bounds in ((0, 1), (.1, 1.3), (-4.3, -.7), (-2.5, 3.1), (4096, 4097), (-8192, -8190)):
                low, high = np.asarray(bounds, np.float32)
                edges = np.linspace(float(low), float(high), bins-3).astype(np.float32)
                values = np.concatenate((edges, np.nextafter(edges, np.float32(-np.inf)),
                                         np.nextafter(edges, np.float32(np.inf))))
                parameters, expected_i, expected_f = self.reference(values, bounds, bins)
                actual_parameters = self.mattes.float1_bin_parameters(bounds, bins)
                with self.subTest(bins=bins, bounds=bounds):
                    np.testing.assert_array_equal(actual_parameters.view(np.uint32), parameters.view(np.uint32))
                    device_values, device_parameters = ctx.upload(values), ctx.upload(actual_parameters)
                    indices, fractions = self.mattes.float1_bin_fractions(ctx, device_values, bins, device_parameters)
                    actual_i = ctx.download(indices, 'test_float1_bins')
                    actual_f = ctx.download(fractions, 'test_float1_bins')
                    np.testing.assert_array_equal(actual_i, expected_i)
                    np.testing.assert_array_equal(actual_f.view(np.uint32), expected_f.view(np.uint32))
                    cached_fixed = self.mattes.fixed_bins(ctx, device_values, bins, bounds, parameters=device_parameters)
                    np.testing.assert_array_equal(ctx.download(cached_fixed, 'test_float1_bins'), expected_i)

    def test_histogram_variants_use_same_fixed_bin_formula(self):
        bins, bounds = 16, (.1, 1.3)
        edges = np.linspace(*bounds, bins-3).astype(np.float32)
        fixed = np.concatenate((edges, np.nextafter(edges, np.float32(-np.inf)),
                                np.nextafter(edges, np.float32(np.inf))))
        moving = np.linspace(.2, 1.2, fixed.size).astype(np.float32)
        parameters, expected_i, _ = self.reference(fixed, bounds, bins)
        expected_counts = np.bincount(expected_i, minlength=bins)
        intensity = np.tile(np.array([.2, -.3], np.float32), (fixed.size, 1))
        baseline = None
        for options in ((), ('histogram_rows',), ('histogram_channels',)):
            ctx = self.context_type(options)
            f, m, d, params = (ctx.upload(a) for a in (fixed, moving, intensity, parameters))
            bi, bf = self.mattes.float1_bin_fractions(ctx, m, bins, params)
            for cached in (False, True):
                with self.subTest(options=options, cached=cached):
                    fb = self.mattes.fixed_bins(ctx, f, bins, bounds, parameters=params) if cached else None
                    result = self.mattes.histograms(ctx, f, m, d, bins, bounds, bounds, bi, bf,
                        cached_bins=fb, fixed_parameters=params, moving_width=parameters[0])
                    joint, counts, derivative, count, width = result
                    np.testing.assert_array_equal(counts, expected_counts)
                    self.assertEqual(count, fixed.size)
                    self.assertEqual(width, float(parameters[0]))
                    if baseline is None:
                        baseline = joint, derivative
                    else:
                        np.testing.assert_array_equal(joint.view(np.uint32), baseline[0].view(np.uint32))
                        np.testing.assert_array_equal(derivative.view(np.uint32), baseline[1].view(np.uint32))


if __name__ == '__main__':
    unittest.main()
