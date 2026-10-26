import unittest

import numpy as np

from fast_ants import Grid, Image
from fast_ants.pipeline.models64 import LinearModel, ComposedLinearModel
from fast_ants import AffineTransform
from fast_ants.backends.cpu.fp64.metrics.mattes import MeanSquaresMetric, evaluate_mattes_samples


class MattesModelTests(unittest.TestCase):
    def test_fused_sampling_preserves_public_geometry_and_physical_gradient(self):
        """Fusion must retain coordinate rounding on an oblique physical grid."""
        from numba import get_num_threads, set_num_threads
        from fast_ants.backends.cpu.fp64.metrics.linear_fused import _gradients_serial, _gradients_parallel
        from fast_ants.backends.cpu.fp64.gradients import gradient_at_points
        from fast_ants.backends.cpu.fp64.sampling import sample_image

        rng = np.random.default_rng(64335)
        angle = .123562678
        direction = np.array([[0., 0., -1.],
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.]])
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag([.9375, .9375, 1.1999969482421875])
        affine[:3, 3] = [88.6399, -116.532, -112.114]
        grid = Grid((27, 29, 31), affine)
        image = Image(rng.integers(0, 4096, grid.shape_zyx).astype(np.float64)/4095, grid)
        indices = rng.uniform(-1., np.asarray(grid.shape_xyz), size=(17003, 3))
        points = np.ascontiguousarray(grid.index_to_physical(indices))
        transform = AffineTransform.centered(
            [[1.001, .002, -.003], [.001, .998, .002], [-.002, .001, 1.002]],
            [.13, -.17, .19], grid.center)
        mapped = transform.map_points(points)
        expected_values, valid = sample_image(image, mapped)
        expected_gradients = gradient_at_points(image, mapped)
        original = get_num_threads()
        try:
            for threads in sorted({1, min(4, original)}):
                set_num_threads(threads)
                function = _gradients_serial if threads == 1 else _gradients_parallel
                values, states, gradients = function(
                    image.data, points, transform.matrix[:3, :3], transform.center,
                    transform.translation, grid.origin, grid.lps_to_index[:3, :3], grid.spacing_xyz)
                np.testing.assert_array_equal(states, valid.astype(np.int8))
                np.testing.assert_array_equal(values.view(np.uint64), expected_values.view(np.uint64))
                np.testing.assert_array_equal(gradients.view(np.uint64), expected_gradients.view(np.uint64))
        finally:
            set_num_threads(original)

    def test_parzen_pdf_matches_independent_truncated_power_basis(self):
        fixed = np.array([.1, .2, .4, .5, .7, .9])
        moving = np.array([.12, .24, .35, .52, .69, .86])
        result = evaluate_mattes_samples(fixed, moving, bins=16, fixed_range=(0, 1), moving_range=(0, 1))
        reference = np.zeros((16, 16))
        for f, m in zip(fixed, moving):
            i = min(max(int(12*f+2), 2), 13)
            for j in range(16):
                a = abs(j-(12*m+2))
                reference[i, j] += (max(0, 2-a)**3-4*max(0, 1-a)**3)/6
        reference /= reference.sum()
        np.testing.assert_allclose(result.details['joint_pdf'], reference, atol=2e-16)
        np.testing.assert_allclose(result.details['fixed_pdf'], reference.sum(axis=1), atol=2e-16)
        np.testing.assert_allclose(result.details['moving_pdf'], reference.sum(axis=0), atol=2e-16)
        positive = reference > 0
        ratio = reference / np.maximum(reference.sum(axis=1)[:, None]*reference.sum(axis=0)[None, :], 1e-300)
        expected = -np.sum(reference[positive]*np.log(ratio[positive]))
        self.assertAlmostEqual(result.value, expected, places=14)

    def test_mattes_global_derivative_multistep_finite_difference(self):
        rng = np.random.default_rng(23)
        fixed, moving = rng.uniform(.1, .9, (2, 200))
        derivatives = rng.normal(size=(200, 5))*.03
        result = evaluate_mattes_samples(fixed, moving, derivatives, 24, (0, 1), (0, 1))
        errors = []
        for step in (1e-3, 1e-4, 1e-5):
            numerical = []
            for p in range(5):
                plus = evaluate_mattes_samples(fixed, moving+step*derivatives[:, p], bins=24, fixed_range=(0, 1), moving_range=(0, 1)).value
                minus = evaluate_mattes_samples(fixed, moving-step*derivatives[:, p], bins=24, fixed_range=(0, 1), moving_range=(0, 1)).value
                numerical.append((plus-minus)/(2*step))
            errors.append(np.max(np.abs(np.array(numerical)-result.gradient)))
        self.assertLess(errors[-1], 1e-8)
        self.assertLess(errors[-1], errors[0])

    def test_rigid_and_affine_jacobian_parameter_directions(self):
        points = np.array([[12, -15, 23], [7, -22, 35], [9, -16, 28.]])
        gradients = np.array([[.1, .3, -.2], [.2, -.1, .4], [-.3, .2, .1]])
        for kind in ('Rigid', 'Affine'):
            model = LinearModel(kind, [10, -20, 30])
            parameters = model.identity_parameters()
            parameters += np.linspace(.003, .009, len(parameters))
            actual = model.intensity_derivatives(parameters, points, gradients)
            for p in range(len(parameters)):
                delta = np.zeros(len(parameters)); delta[p] = 1e-6
                numerical = np.sum((model.transform(parameters+delta).map_points(points)-
                                    model.transform(parameters-delta).map_points(points))*gradients, axis=1)/(2e-6)
                np.testing.assert_allclose(actual[:, p], numerical, atol=2e-9)

    def test_incremental_affine_jacobian_through_frozen_initial_map(self):
        base = AffineTransform.centered([[1.1, .1, 0], [0, .9, .05], [0, 0, 1.02]], [1, -2, 3], [2, 4, 6])
        model = ComposedLinearModel('Affine', [2, 4, 6], base)
        p = model.identity_parameters()
        points = np.array([[4, 7, 9], [-2, 3, 8.]])
        gradients = np.array([[.2, -.3, .4], [-.1, .2, .3]])
        actual = model.intensity_derivatives(p, points, gradients)
        for k in range(12):
            delta = np.zeros(12); delta[k] = 1e-5
            expected = np.sum((model.transform(p+delta).map_points(points)-model.transform(p-delta).map_points(points))*gradients,axis=1)/(2e-5)
            np.testing.assert_allclose(actual[:, k], expected, atol=2e-10)

    def test_mse_parametric_derivative_on_analytic_ramp(self):
        grid = Grid((12, 13, 14), np.eye(4))
        z, y, x = np.indices(grid.shape_zyx, dtype=float)
        moving = Image(.03*x+.02*y+.01*z, grid)
        fixed = Image(moving.data+.025, grid)
        probes = np.array([[3.2, 4.3, 5.4], [7.1, 6.2, 4.3], [5.2, 5.5, 6.6]])
        metric = MeanSquaresMetric(fixed, moving, points_lps=probes)
        model = LinearModel('Rigid', grid.center)
        parameters = model.identity_parameters()
        evaluation = metric.evaluate(model, parameters)
        for p in range(6):
            delta = np.zeros(6); delta[p] = 1e-5
            numerical = (metric.evaluate(model, parameters+delta, False).value-
                         metric.evaluate(model, parameters-delta, False).value)/(2e-5)
            self.assertAlmostEqual(evaluation.gradient[p], numerical, places=10)

    def test_constant_intensity_range_and_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_mattes_samples(np.ones(10), np.arange(10.))
        with self.assertRaises(ValueError):
            evaluate_mattes_samples(np.array([]), np.array([]))


if __name__ == '__main__':
    unittest.main()
