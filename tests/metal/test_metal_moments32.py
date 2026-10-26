"""Parallel image moments against mathematical and existing GPU oracles."""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Explicit Metal device test opt-in required')
class MetalMoments32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.moments import moments
        from fast_ants.backends.metal.runtime import MetalContext, mx
        from fast_ants.backends.metal import serial
        cls.parallel = staticmethod(moments)
        cls.context_type, cls.mx, cls.serial = MetalContext, mx, serial

    @staticmethod
    def reference(values):
        z, y, x = np.indices(values.shape, dtype=np.float64)
        nz, ny, nx = values.shape
        terms = (values.astype(np.float64),
                 values.astype(np.float64)*(x-(nx-1)*.5),
                 values.astype(np.float64)*(y-(ny-1)*.5),
                 values.astype(np.float64)*(z-(nz-1)*.5))
        exact = np.asarray([a.sum(dtype=np.float64) for a in terms])
        # Bound practical FP32 product/reduction rounding by the input's
        # condition, including nearly cancelling signed first moments.
        bound = np.asarray([8*np.finfo(np.float32).eps*np.abs(a).sum(dtype=np.float64)
                            for a in terms])
        return exact, bound

    def check_image(self, values):
        ctx = self.context_type(())
        original = values.copy();values.setflags(write=False)
        device = ctx.upload(values, 'test_input')
        before = ctx.kernels.copy()
        result = self.parallel(ctx, device)
        self.assertEqual(result.shape, (4,))
        self.assertEqual(result.dtype, self.mx.float32)
        self.assertEqual(ctx.kernels['moments_parallel_tiles32']-before['moments_parallel_tiles32'], 1)
        self.assertEqual(ctx.kernels['moments_parallel_merge32']-before['moments_parallel_merge32'], 1)
        self.assertEqual(sum((ctx.kernels-before).values()), 2)
        self.assertEqual(ctx.control_guards['serial_watchdog_slices'], 0)
        self.assertFalse(any(k.startswith('device_to_host:') for k in ctx.transfers))
        actual = ctx.control(result, 'test_moments', 4)
        expected, bound = self.reference(values)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_less(np.abs(actual.astype(np.float64)-expected),
                                     np.maximum(bound, 1e-7))
        if abs(expected[0]) > 1:
            np.testing.assert_allclose(actual[1:].astype(np.float64)/float(actual[0]),
                                       expected[1:]/expected[0], rtol=0, atol=2e-5)
        np.testing.assert_array_equal(values, original)
        return actual, device, ctx

    def test_non_aligned_tiles_single_axes_and_existing_serial_reference(self):
        rng = np.random.default_rng(619)
        for shape in ((1, 1, 1), (1, 3, 259), (9, 17, 31)):
            with self.subTest(shape=shape):
                values = rng.uniform(.1, 2., shape).astype(np.float32)
                actual, device, ctx = self.check_image(values)
                serial = ctx.control(self.serial.moments(ctx, device), 'test_serial_moments', 4)
                # Both implement the same centered moments; their reduction
                # trees differ, so compare the resulting center rather than
                # demanding the serial accumulator's last bits.
                np.testing.assert_allclose(actual[1:].astype(np.float64)/float(actual[0]),
                                           serial[1:].astype(np.float64)/float(serial[0]),
                                           rtol=0, atol=2e-5)

    def test_zero_signed_cancellation_and_large_dc_center(self):
        zero = np.zeros((3, 5, 257), np.float32)
        actual, _, _ = self.check_image(zero)
        np.testing.assert_array_equal(actual, 0)
        signed = np.broadcast_to(np.where(np.arange(258) % 2, 1., -1.), (3, 5, 258)).astype(np.float32).copy()
        actual, _, _ = self.check_image(signed)
        expected, _ = self.reference(signed)
        self.assertEqual(actual[0], 0)
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        z, y, x = np.indices((9, 17, 263), dtype=np.float32)
        dc = (np.float32(1e6)+np.float32(.125)*x+np.float32(.25)*y+np.float32(.5)*z).astype(np.float32)
        self.check_image(dc)

    def test_invalid_storage_and_shapes_do_not_dispatch(self):
        ctx = self.context_type(())
        malformed = (np.ones((2, 3, 4), np.float32),
                     ctx.upload(np.ones((2, 3, 4), np.int32), dtype=np.int32),
                     ctx.upload(np.ones((2, 3), np.float32)),
                     ctx.upload(np.empty((0, 3, 4), np.float32)))
        before = ctx.kernels.copy()
        for data in malformed:
            with self.subTest(shape=data.shape):
                with self.assertRaises(ValueError):
                    self.parallel(ctx, data)
        self.assertEqual(ctx.kernels, before)


if __name__ == '__main__':
    unittest.main()
