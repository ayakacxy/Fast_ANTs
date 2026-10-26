import unittest

import numpy as np

from fast_ants import AffineTransform, Grid
from fast_ants.core.results import MetricEvaluation
from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.pipeline.optimizer64 import conjugate_gradient, golden_section_search, parameter_scales


class TranslationModel:
    def transform(self, parameters):
        return AffineTransform.centered(np.eye(3), parameters, np.zeros(3))


class QuadraticMetric:
    def evaluate(self, model, parameters, derivative=True):
        residual = parameters-np.array([.2, -.1, .3])
        weighted = residual*np.array([1, 2, 3])
        return MetricEvaluation(float(.5*residual@weighted), -weighted if derivative else np.zeros(0), 8, {})


class OptimizerTests(unittest.TestCase):
    def test_translation_physical_scales(self):
        grid = Grid((6, 7, 8), np.diag([1.2, 2, 3, 1]))
        np.testing.assert_allclose(parameter_scales(TranslationModel(), np.zeros(3), grid), 1, atol=1e-12)

    def test_golden_section_bracket_reduces_known_cost(self):
        rate, evaluations = golden_section_search(lambda x:(x-1.3)**2, 0, 1, 2)
        self.assertLess(abs(rate-1.3), .2)
        self.assertGreater(evaluations, 0)

    def test_conjugate_gradient_known_minimum(self):
        grid = Grid((6, 7, 8), np.eye(4))
        result = conjugate_gradient(QuadraticMetric(), TranslationModel(), np.zeros(3), grid,
                                    iterations=80, convergence_threshold=1e-10, convergence_window=8)
        np.testing.assert_allclose(result.parameters, [.2, -.1, .3], atol=1e-6)
        self.assertGreater(len(result.history), 1)
        self.assertLess(result.history[-1]['value'], result.history[0]['value']*1e-8)

    def test_window_is_scale_invariant_and_uses_accumulated_energy(self):
        first, second = WindowConvergence(8), WindowConvergence(8)
        for value in np.linspace(-1, -2, 20):
            a, b = first.add(value), second.add(30*value)
        self.assertAlmostEqual(a, b, places=14)
        self.assertGreater(a, 0)
        current = WindowConvergence(8)
        for value in np.linspace(-1, -2, 20)[-8:]:
            fresh = current.add(value)
        self.assertNotEqual(a, fresh)


if __name__ == '__main__':
    unittest.main()
