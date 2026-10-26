import sys
from pathlib import Path
import unittest
import numpy as np

from fast_ants.core.types import Grid, Image
from fast_ants.backends.cpu.fp64.metrics.cc import neighborhood_cc
from fast_ants.backends.cpu.fp64.gradients import gradient_at_points
from fast_ants.core.results import NoValidSamplesError
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'validation'))
from validation.common.cc_reference import naive_cc


class CCTests(unittest.TestCase):
    def fixture(self):
        angle = .23
        direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        matrix = np.eye(4)
        matrix[:3, :3] = direction@np.diag([.7, 1.3, 2.1])
        matrix[:3, 3] = [11, -21, 7]
        grid = Grid((5, 6, 7), matrix)
        z, y, x = np.indices(grid.shape_zyx)
        f = np.sin(.3*x)+np.cos(.4*y)+.12*z
        m = np.sin(.3*x+.09)+np.cos(.4*y-.12)+.1*z+.02*x*y
        return Image(f, grid), Image(m, grid)

    def test_queue_matches_independent_centered_windows_and_masks(self):
        fixed, moving = self.fixture()
        valid = np.ones(fixed.data.shape, dtype=bool)
        valid[1:3, 2:4, 3:6] = False
        for radius in ((1, 2, 1), (0, 0, 0), (9, 9, 9)):
            actual = neighborhood_cc(fixed, moving, radius, valid=valid, return_statistics=True)
            points = fixed.grid.index_to_physical(np.indices(fixed.data.shape)[::-1].reshape(3, -1).T)
            gradient = gradient_at_points(moving, points).T.reshape((3, *fixed.data.shape))
            value, update, stats = naive_cc(fixed.data, moving.data, gradient, valid, np.array(radius))
            self.assertAlmostEqual(actual.value, value, places=12)
            np.testing.assert_allclose(actual.update.data, update, atol=2e-12, rtol=1e-10)
            for i, name in enumerate(actual.statistics):
                np.testing.assert_allclose(actual.statistics[name], stats[i], atol=4e-12, rtol=1e-10)
            self.assertEqual(actual.valid_points, int(valid.sum()))

    def test_constant_low_contrast_and_empty_domain(self):
        fixed, moving = self.fixture()
        constant = Image(np.ones(fixed.data.shape)*3, fixed.grid)
        actual = neighborhood_cc(constant, moving)
        self.assertEqual(actual.value, -1)
        self.assertEqual(np.max(np.abs(actual.update.data)), 0)
        low = Image(1e-7*fixed.data, fixed.grid)
        evaluation = neighborhood_cc(low, Image(1e-7*moving.data, moving.grid))
        self.assertTrue(np.isfinite(evaluation.value))
        self.assertTrue(np.isfinite(evaluation.update.data).all())
        with self.assertRaises(NoValidSamplesError):
            neighborhood_cc(fixed, moving, valid=np.zeros(fixed.data.shape, dtype=bool))

    def test_independent_intensity_affine_invariance_and_role_exchange(self):
        fixed, moving = self.fixture()
        forward = neighborhood_cc(fixed, moving)
        reverse = neighborhood_cc(moving, fixed)
        self.assertAlmostEqual(forward.value, reverse.value, places=13)
        scaled = neighborhood_cc(Image(2*fixed.data+4, fixed.grid), Image(-3*moving.data-1, moving.grid))
        self.assertAlmostEqual(forward.value, scaled.value, places=12)
        np.testing.assert_allclose(forward.update.data, scaled.update.data, atol=1e-12)
        self.assertGreater(np.max(np.abs(forward.update.data-reverse.update.data)), 1e-6)

    def test_single_window_intensity_derivative_finite_differences(self):
        f = np.array([1.2, -.2, 3., 1., .1])
        m = np.array([1., -.7, 2., .9, .2])
        def score(a):
            fc, mc = f-f.mean(), a-a.mean()
            return np.dot(fc, mc)**2/(np.dot(fc, fc)*np.dot(mc, mc))
        fc, mc = f-f.mean(), m-m.mean()
        ff, mm, fm = np.dot(fc, fc), np.dot(mc, mc), np.dot(fc, mc)
        analytic = 2*fm/(ff*mm)*(fc-fm/mm*mc)
        for h in (1e-3, 1e-4, 1e-5):
            numerical = []
            for i in range(m.size):
                delta = np.zeros_like(m); delta[i] = h
                numerical.append((score(m+delta)-score(m-delta))/(2*h))
            np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=1e-5)
