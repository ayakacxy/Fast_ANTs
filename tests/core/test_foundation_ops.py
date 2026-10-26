import math
import unittest

import numpy as np

from fast_ants import DisplacementField, Grid, Image, apply_image
from fast_ants.backends.cpu.common.reductions import maximum_norm, sum_values
from fast_ants.pipeline.gaussian import discrete_gaussian_kernel
from fast_ants.backends.cpu.fp64.filtering import smooth_field, smooth_image
from fast_ants.backends.cpu.fp64.gradients import gradient_at_points, gradient_image
from fast_ants.backends.cpu.fp64.preprocessing import prepare_level, preprocess_image
from fast_ants.pipeline.pyramid import shrink_grid
from fast_ants.pipeline.masks import sample_mask


class FoundationTests(unittest.TestCase):
    def setUp(self):
        angle = .31
        matrix = np.array([[np.cos(angle), -2*np.sin(angle), 0, 30],
                           [np.sin(angle), 2*np.cos(angle), 0, -60], [0, 0, 3, 100], [0, 0, 0, 1]])
        self.grid = Grid((9, 10, 11), matrix)
        z, y, x = np.indices(self.grid.shape_zyx, dtype=float)
        self.points = self.grid.index_to_physical(np.stack([x, y, z], axis=-1))

    def test_linear_physical_gradient_and_boundary(self):
        gradient = np.array([.2, -.3, .4])
        image = Image(self.points @ gradient, self.grid)
        actual = gradient_image(image)
        np.testing.assert_allclose(actual[:, 1:-1, 1:-1, 1:-1], np.broadcast_to(gradient[:, None, None, None], actual[:, 1:-1, 1:-1, 1:-1].shape), atol=2e-14)
        probes = self.grid.index_to_physical([[3.25, 4.5, 3.75], [5, 5, 5]])
        np.testing.assert_allclose(gradient_at_points(image, probes), np.broadcast_to(gradient, (2, 3)), atol=3e-14)
        identity_grid = Grid((6, 6, 6), np.eye(4))
        z, y, x = np.indices(identity_grid.shape_zyx, dtype=float)
        actual = gradient_image(Image(x+2*y+3*z, identity_grid))
        np.testing.assert_array_equal(actual[0, :, :, 0], 0)
        np.testing.assert_array_equal(actual[1, :, 0, :], 0)
        np.testing.assert_array_equal(actual[2, 0, :, :], 0)

    def test_recursive_constant_impulse_and_zero_sigma(self):
        constant = Image(np.ones(self.grid.shape_zyx), self.grid)
        np.testing.assert_allclose(smooth_image(constant, 1).data, 1, atol=1e-7)
        np.testing.assert_array_equal(smooth_image(constant, 0).data, constant.data)
        # Leave enough support for the recursive approximation's noncompact tail.
        grid = Grid((51, 51, 51), np.eye(4))
        values = np.zeros(grid.shape_zyx); values[25, 25, 25] = 1
        result = smooth_image(Image(values, grid), 1.2, itk_float_intermediates=False).data
        np.testing.assert_allclose(result, result[::-1, ::-1, ::-1], atol=2e-16)
        self.assertAlmostEqual(float(result.sum()), 1, places=5)

    def test_discrete_kernel_and_field_boundary_rules(self):
        for variance in (.1, .49, .5, 3, 8):
            kernel = discrete_gaussian_kernel(variance)
            np.testing.assert_array_equal(kernel, kernel[::-1])
            self.assertAlmostEqual(float(kernel.sum()), 1, places=14)
            self.assertTrue(np.all(kernel >= 0))
        values = np.ones((3, *self.grid.shape_zyx))
        field = DisplacementField(values, self.grid)
        for variance in (.1, .5, 3):
            output = smooth_field(field, variance).data
            np.testing.assert_allclose(output[:, 1:-1, 1:-1, 1:-1], 1, atol=1e-14)
            np.testing.assert_array_equal(output[:, :, :, 0], 0)
            np.testing.assert_array_equal(output[:, 0], 0)
        np.testing.assert_array_equal(smooth_field(field, 0).data, values)
        self.assertFalse(np.shares_memory(smooth_field(field, 0).data, values))

    def test_centered_shrink_and_field_millimetres(self):
        coarse = shrink_grid(self.grid, [2, 3, 2])
        self.assertEqual(coarse.shape_xyz, (5, 3, 4))
        np.testing.assert_allclose(coarse.center, self.grid.center, atol=1e-13)
        np.testing.assert_allclose(coarse.spacing_xyz, self.grid.spacing_xyz*[2, 3, 2])
        field_component = Image(np.full(self.grid.shape_zyx, 2.5), self.grid)
        np.testing.assert_allclose(apply_image(field_component, coarse).data, 2.5)

    def test_full_resolution_metric_images_and_preprocessing(self):
        image = Image(self.points[..., 0], self.grid)
        level = prepare_level(image, image, 2, 1)
        self.assertEqual(level.fixed.data.shape, image.data.shape)
        self.assertNotEqual(level.virtual_grid.shape_zyx, image.grid.shape_zyx)
        normalized = preprocess_image(image,profile="minmax")
        self.assertEqual(normalized.data.min(), 0)
        self.assertEqual(normalized.data.max(), 1)
        with self.assertRaises(ValueError):
            preprocess_image(image, histogram_matching=True)

    def test_ants_default_histogram_endpoint_margin(self):
        values=np.linspace(0,1,np.prod(self.grid.shape_zyx)).reshape(self.grid.shape_zyx)
        result=preprocess_image(Image(values,self.grid))
        upper=float(np.float32(1+1/2560))
        self.assertEqual(result.data.min(),0.)
        self.assertEqual(result.data.max(),1/upper)
        self.assertLess(result.data.max(),1.)
        np.testing.assert_allclose(result.data,values/upper,atol=2e-16,rtol=0)

    def test_compensated_reduction_and_spacing_norm(self):
        data = np.array([1e16, 1, -1e16, 3, -2, 1e-6])
        self.assertEqual(sum_values(data), math.fsum(data))
        field = np.zeros((3, 2, 3, 4)); field[:, 1, 1, 1] = [1, 4, 6]
        self.assertAlmostEqual(maximum_norm(field, np.array([1, 2, 3])), 3)

    def test_binary_masks_use_their_own_physical_grid(self):
        values = np.zeros(self.grid.shape_zyx, np.uint8)
        values[2:5, 3:7, 4:8] = 1
        mask = Image(values, self.grid)
        points = self.grid.index_to_physical([[5, 4, 3], [0, 0, 0], [-1, 4, 3]])
        np.testing.assert_array_equal(sample_mask(mask, points), [True, False, False])
        empty = Image(np.zeros_like(values), self.grid)
        self.assertFalse(sample_mask(empty, points).any())


if __name__ == "__main__":
    unittest.main()
