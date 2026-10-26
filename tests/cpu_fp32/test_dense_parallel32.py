"""Threaded FP32 dense kernels preserve each serial output calculation."""

import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from fast_ants.backends.cpu.fp32 import dense_kernels as dense_parallel32, kernels as fp32


class DenseParallel32Tests(unittest.TestCase):
    def setUp(self):
        self.threads = get_num_threads()

    def tearDown(self):
        set_num_threads(self.threads)

    def assert_same_bits(self, actual, expected):
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def compare_threads(self, parallel, serial, *args):
        expected = serial(*args)
        for threads in (1, 2):
            set_num_threads(threads)
            with self.subTest(threads=threads):
                self.assert_same_bits(parallel(*args), expected)

    def test_convolution_edges_tap_order_and_strided_readonly_input(self):
        rng = np.random.default_rng(42)
        # More than the dispatch threshold, with noncontiguous rows and both
        # signs to exercise compensated summation at the repeated boundaries.
        source = rng.normal(size=(37, 898)).astype(np.float32)
        lines = source[:, ::2]
        lines[:, 0] = np.float32(1e4)
        lines[:, -1] = np.float32(-1e4)
        kernel = np.array([.015625, -.03125, .125, .25, .28125, .25, .125, -.03125, .015625], np.float32)
        saved = source.copy()
        lines.setflags(write=False)
        kernel.setflags(write=False)
        self.compare_threads(dense_parallel32.convolve32, fp32.convolve32, lines, kernel)
        np.testing.assert_array_equal(source, saved)

    def test_gradient_anisotropy_oblique_steps_and_readonly_input(self):
        rng = np.random.default_rng(43)
        source = (np.float32(1e5) + rng.normal(size=(17, 29, 74)).astype(np.float32))
        data = source[:, :, ::2]
        data.setflags(write=False)
        saved = source.copy()
        spacing = np.array([.7, 1.3, 2.1], np.float32)
        angle = np.float32(.23)
        direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
        half_spacing_to_index = np.asarray(np.diag(np.float32(1) / spacing) @ direction.T @ np.diag(spacing * np.float32(.5)), np.float32)
        half_spacing_to_index.setflags(write=False)
        spacing.setflags(write=False)
        self.compare_threads(dense_parallel32.gradient32, fp32.gradient32, data, half_spacing_to_index, spacing)
        np.testing.assert_array_equal(source, saved)

    def test_warp_linear_and_nearest_oblique_geometry_and_boundaries(self):
        rng = np.random.default_rng(44)
        source = rng.normal(size=(17, 29, 74)).astype(np.float32)
        data = source[:, :, ::2]
        fields = rng.uniform(-.4, .4, size=(3, 17, 29, 74)).astype(np.float32)
        field = fields[:, :, :, ::2]
        data.setflags(write=False)
        field.setflags(write=False)
        data_saved, field_saved = source.copy(), fields.copy()
        linear = np.array([[1.02, .015, -.01], [-.02, 1.03, .025], [.01, -.02, 1.04]], np.float32)
        offset = np.array([-.5, -.25, .125], np.float32)
        inverse = np.array([[1.25, .1, 0], [-.05, .7, .02], [0, 0, .5]], np.float32)
        for nearest in (False, True):
            with self.subTest(nearest=nearest):
                self.compare_threads(dense_parallel32.warp32, fp32.warp32, data, field, linear, offset, inverse, nearest)
        # The half-voxel support edge and nearest-neighbor half ties are
        # sensitive to coordinate arithmetic even with a zero displacement.
        field = np.zeros((3, 17, 29, 37), np.float32)
        for nearest in (False, True):
            for half_offset in (-.5, .5):
                with self.subTest(nearest=nearest, half_offset=half_offset):
                    self.compare_threads(dense_parallel32.warp32, fp32.warp32, data, field,
                                         np.eye(3, dtype=np.float32), np.full(3, half_offset, np.float32), inverse, nearest)
        np.testing.assert_array_equal(source, data_saved)
        np.testing.assert_array_equal(fields, field_saved)

    def test_small_and_empty_outputs_preserve_serial_behavior(self):
        lines = np.array([[1, -2, 3], [0, .25, -.5]], np.float32)
        kernel = np.array([.25, .5, .25], np.float32)
        self.compare_threads(dense_parallel32.convolve32, fp32.convolve32, lines, kernel)
        self.compare_threads(dense_parallel32.convolve32, fp32.convolve32, np.empty((0, 5), np.float32), kernel)
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        half = np.eye(3, dtype=np.float32) * np.float32(.5)
        spacing = np.array([.5, 1.5, 2.], np.float32)
        self.compare_threads(dense_parallel32.gradient32, fp32.gradient32, data, half, spacing)
        self.compare_threads(dense_parallel32.gradient32, fp32.gradient32, np.empty((0, 3, 4), np.float32), half, spacing)
        field = np.zeros((3, 2, 3, 4), np.float32)
        self.compare_threads(dense_parallel32.warp32, fp32.warp32, data, field,
                             np.eye(3, dtype=np.float32), np.zeros(3, np.float32), half)
        self.compare_threads(dense_parallel32.warp32, fp32.warp32, data, np.empty((3, 0, 3, 4), np.float32),
                             np.eye(3, dtype=np.float32), np.zeros(3, np.float32), half)

    def test_inverse_update_clipping_boundary_and_small_dispatch(self):
        rng = np.random.default_rng(45)
        for shape in ((1, 2, 3), (17, 29, 37)):
            inverse = rng.normal(size=(3, *shape)).astype(np.float32)
            residual = rng.normal(size=(3, *shape)).astype(np.float32)
            norms = rng.uniform(0, 2, size=shape).astype(np.float32)
            norms.ravel()[:3] = np.array([0, .5, 1.5], np.float32)
            inverse.setflags(write=False)
            residual.setflags(write=False)
            norms.setflags(write=False)
            for enforce_boundary in (False, True):
                for epsilon in (.5, .75):
                    with self.subTest(shape=shape, boundary=enforce_boundary, epsilon=epsilon):
                        self.compare_threads(dense_parallel32.inverse_step32, fp32.inverse_step32,
                                             inverse, residual, norms, np.float32(2),
                                             np.float32(epsilon), enforce_boundary)

    def field_axis_reference(self, values, kernel, dimension):
        """Existing move-axis/line convolution, independent of the new layout."""
        moved = np.moveaxis(values, 3-dimension, -1)
        lines = np.ascontiguousarray(moved.reshape(-1, moved.shape[-1]), dtype=np.float32)
        result = fp32.convolve32(lines, kernel).reshape(moved.shape)
        return np.ascontiguousarray(np.moveaxis(result, -1, 3-dimension))

    def test_native_field_axes_preserve_clamping_kahan_and_strided_inputs(self):
        rng = np.random.default_rng(46)
        kernel = np.array([.015625, -.03125, .125, .25, .28125, .25, .125, -.03125, .015625], np.float32)
        kernel.setflags(write=False)
        for shape in ((1, 3, 5), (17, 29, 37)):
            source = rng.normal(size=(3, shape[0], shape[1], 2*shape[2])).astype(np.float32)
            values = source[:, :, :, ::2]
            values[:, :, :, 0] = np.float32(1e4)
            values[:, :, :, -1] = np.float32(-1e4)
            before = source.copy()
            values.setflags(write=False)
            for dimension in range(3):
                reference = self.field_axis_reference(values, kernel, dimension)
                for threads in (1, 2):
                    set_num_threads(threads)
                    with self.subTest(shape=shape, dimension=dimension, threads=threads):
                        actual = dense_parallel32.convolve_field_axis32(values, kernel, dimension)
                        self.assert_same_bits(actual, reference)
                        self.assertTrue(actual.flags.c_contiguous)
                        self.assertFalse(np.shares_memory(actual, values))
            np.testing.assert_array_equal(source, before)

    def test_native_field_final_axis_fuses_small_variance_mix_and_boundary(self):
        from fast_ants.pipeline.gaussian import discrete_gaussian_kernel
        rng = np.random.default_rng(47)
        for shape in ((1, 3, 5), (17, 29, 37)):
            original = rng.normal(size=(3, *shape)).astype(np.float32)
            original.setflags(write=False)
            for variance in (.1, 3.):
                kernels = [np.asarray(discrete_gaussian_kernel(variance, shape[2-d]), np.float32)
                           for d in range(3)]
                weight = np.float32(1-variance/.5 if variance < .5 else 1.)
                reference = original
                for dimension, kernel in enumerate(kernels):
                    reference = self.field_axis_reference(reference, kernel, dimension)
                reference = fp32.mix_boundary32(reference, original, weight)
                for threads in (1, 2):
                    set_num_threads(threads)
                    actual = original
                    for dimension, kernel in enumerate(kernels):
                        actual = dense_parallel32.convolve_field_axis32(actual, kernel, dimension,
                            original=original if dimension == 2 else None, weight=weight)
                    with self.subTest(shape=shape, variance=variance, threads=threads):
                        self.assert_same_bits(actual, reference)


if __name__ == '__main__':
    unittest.main()
