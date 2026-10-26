"""Small scalar/vector controls only; these are not float1 registration oracles."""

import unittest
from types import SimpleNamespace

import numpy as np

from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.pipeline.optimizer32 import FLOAT1_EPSILON, FLOAT1_MAX, WindowConvergence32, apply_parameter_step32, conjugate_direction32, golden_section_search32, initial_learning_rate32, polak_ribiere_gamma32, parameter_scales32, step_scale32, conjugate_gradient32, _inner_product32, _point_shift32
from fast_ants.core.types import Grid
from fast_ants.pipeline.models32 import AffineTransform32, ComposedLinearModel32
from fast_ants.pipeline.optimizer64 import corner_points


class TranslationModel32:
    def transform(self, parameters):
        return SimpleNamespace(map_points=lambda points: np.add(
            np.asarray(points, np.float32), parameters, dtype=np.float32))


class QuadraticMetric32:
    def evaluate(self, model, parameters, derivative=True):
        if parameters.dtype != np.float32:
            raise AssertionError("Optimizer leaked double parameters into metric")
        residual = parameters-np.array([.2, -.1, .3], np.float32)
        weighted = residual*np.array([1, 2, 3], np.float32)
        return SimpleNamespace(value=np.float32(.5)*np.sum(residual*weighted, dtype=np.float32),
                               derivative=-weighted, valid_points=8)


class Optimizer32ControlTests(unittest.TestCase):
    def test_gamma_float_epsilon_and_restart_boundaries(self):
        # A double epsilon would incorrectly retain a nonzero conjugate term.
        self.assertEqual(polak_ribiere_gamma32(2e-8, 1e-8), np.float32(0))
        self.assertEqual(polak_ribiere_gamma32(FLOAT1_EPSILON, FLOAT1_EPSILON), 0)
        above = np.nextafter(FLOAT1_EPSILON, np.float32(np.inf))
        self.assertEqual(polak_ribiere_gamma32(above, above), 1)
        self.assertEqual(polak_ribiere_gamma32(-1, 1), 0)
        self.assertEqual(polak_ribiere_gamma32(5, 1), 5)
        self.assertEqual(polak_ribiere_gamma32(np.nextafter(np.float32(5), np.float32(6)), 1), 0)

    def test_learning_rate_float_epsilon(self):
        self.assertEqual(initial_learning_rate32(.1, 1e-8), np.float32(1))
        rate = initial_learning_rate32(.1, .3)
        self.assertIsInstance(rate, np.float32)
        self.assertEqual(rate, np.float32(.1) / np.float32(.3))

    def test_parameter_and_direction_storage_are_owned_float32(self):
        base = np.array([2**24, 1, -1], np.float64)
        direction = np.array([1, 1e-8, .25], np.float64)
        result = apply_parameter_step32(base, direction, 1)
        np.testing.assert_array_equal(result, [2**24, 1, -.75])
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(np.shares_memory(result, base))
        conjugate = conjugate_direction32(direction, direction, 1)
        np.testing.assert_array_equal(conjugate, 2 * direction.astype(np.float32))
        self.assertEqual(conjugate.dtype, np.float32)
        self.assertFalse(np.shares_memory(conjugate, direction))
        np.testing.assert_array_equal(base, [2**24, 1, -1])

    def test_parameter_step_rounds_multiply_before_add(self):
        # Product rounds to 1 in FP32 before addition: a fused update would
        # instead retain the residual -2**-46.
        plus = np.float32(1 + 2**-23)
        minus = np.float32(1 - 2**-23)
        result = apply_parameter_step32([-1], [plus], minus)
        self.assertEqual(result[0], 0)

    def test_golden_phi_assignment_and_recursion_limit(self):
        visited = []

        def cost(rate):
            self.assertIsInstance(rate, np.float32)
            visited.append(rate)
            return np.float32(0)

        rate, count = golden_section_search32(cost, 0, 1, 2, maximum_iterations=0)
        # The first candidate is m_Phi - 1; rounding 2 - 1.618034 once is wrong.
        self.assertEqual(visited[0].view(np.uint32), 0x3F1E377A)
        self.assertEqual(visited[1], 1)
        self.assertEqual(count, 2)
        self.assertEqual(rate, np.float32((visited[0] + np.float32(2)) / np.float32(2)))

    def test_golden_minimizes_known_cost(self):
        rate, count = golden_section_search32(lambda x: (x - np.float32(1.3))**2, 0, 1, 2)
        self.assertIsInstance(rate, np.float32)
        self.assertLess(abs(float(rate) - 1.3), .2)
        self.assertGreater(count, 0)

    def test_golden_float_metric_rounding_changes_tie_decision(self):
        # Both cost values round to 1 in float. The tie must take the branch
        # containing the old middle; a double comparison chooses the left.
        rate, count = golden_section_search32(lambda x: 1 + float(x) * 1e-8,
                                             0, 1, 2, maximum_iterations=0)
        self.assertGreater(rate, 1)
        self.assertEqual(count, 2)

    def test_golden_invalid_sample_sentinel_keeps_lower_bracket(self):
        rate, count = golden_section_search32(lambda x: FLOAT1_MAX, 0, 1, 2,
                                             maximum_iterations=0)
        self.assertEqual(rate, np.float32(.5))
        self.assertEqual(count, 2)

    def test_golden_pair_protocol_preserves_scalar_visits_and_decisions(self):
        quadratic = lambda rate: (rate-np.float32(1.3))**2
        cases = (
            ('quadratic', quadratic, 20),
            # The callback returns distinct double values which become a
            # plateau at the optimizer's float32 comparison boundary.
            ('rounded_plateau', lambda rate: 1+float(rate)*1e-8, 20),
            ('invalid_sentinel', lambda rate: FLOAT1_MAX, 20),
            ('zero_recursion_budget', quadratic, 0),
        )
        for name, objective, maximum in cases:
            with self.subTest(cost=name, maximum_iterations=maximum):
                def recorded_cost(with_pairs):
                    visits, pairs = [], []

                    def record(rate):
                        self.assertIsInstance(rate, np.float32)
                        visits.append(int(rate.view(np.uint32)))
                        return objective(rate)

                    def cost(rate):
                        return record(rate)

                    if with_pairs:
                        def evaluate_pair(first, second):
                            pairs.append((int(first.view(np.uint32)),
                                          int(second.view(np.uint32))))
                            return record(first), record(second)
                        cost.evaluate_pair = evaluate_pair
                    return cost, visits, pairs

                scalar, scalar_visits, _ = recorded_cost(False)
                paired, paired_visits, pairs = recorded_cost(True)
                expected_rate, expected_count = golden_section_search32(
                    scalar, 0, 1, 2, maximum_iterations=maximum)
                actual_rate, actual_count = golden_section_search32(
                    paired, 0, 1, 2, maximum_iterations=maximum)
                self.assertEqual(paired_visits, scalar_visits)
                self.assertEqual(int(actual_rate.view(np.uint32)),
                                 int(expected_rate.view(np.uint32)))
                self.assertEqual(actual_count, expected_count)
                self.assertEqual(actual_count, len(paired_visits))
                self.assertTrue(pairs, 'The available pair provider was not used')
                self.assertEqual(pairs[0], tuple(scalar_visits[:2]))
                if maximum == 0:
                    self.assertEqual(actual_count, 2)
                    self.assertEqual(len(pairs), 1)

    def test_monitor_rounds_inputs_and_retains_double_total_energy(self):
        monitor, reference = WindowConvergence32(3), WindowConvergence(3)
        values = [2**24, 1.00000001, -2.00000001, -3.00000001]
        for index, value in enumerate(values):
            actual = monitor.add(value)
            expected = reference.add(float(np.float32(value)))
            self.assertEqual(actual, expected)
            if index < 2:
                self.assertEqual(actual, np.finfo(np.float64).max)
        # Single float accumulation would lose the second sample's one unit.
        self.assertEqual(monitor.total_energy, 2**24 + 6)
        self.assertEqual(list(monitor.values), [1, -2, -3])

    def test_invalid_control_inputs_fail_early(self):
        with self.assertRaises(ValueError):
            initial_learning_rate32(.1, -1)
        with self.assertRaises(ValueError):
            conjugate_direction32([1, 2], [1], 0)
        with self.assertRaises(ValueError):
            golden_section_search32(lambda x: 0, 2, 1, 0)
        with self.assertRaises(ValueError):
            WindowConvergence32(3).add(np.inf)

    def test_vnl_inner_product_accumulates_float_with_native_contraction(self):
        a = np.array([1e8, 1, -1e8], np.float32)
        b = np.ones(3, np.float32)
        self.assertEqual(_inner_product32(a, b), 0)
        # The native multiply-add has one float rounding, not two.
        a = np.array([1, 1+2**-23], np.float32)
        b = np.array([-1, 1-2**-23], np.float32)
        self.assertEqual(_inner_product32(a, b), np.float32(-2**-46))

    def test_point_distance_has_real_double_arithmetic_before_float_storage(self):
        values = np.array([[1e20, 1e20, 1e20]], np.float32)
        distance = _point_shift32(values, np.zeros_like(values))
        # A float squared norm overflows although the final float distance fits.
        self.assertTrue(np.isfinite(distance))
        self.assertEqual(distance, np.float32(np.sqrt(np.sum(values.astype(np.float64)**2))))

    def test_translation_scale_and_float_step_epsilon(self):
        grid = Grid((1, 1, 1), np.eye(4))
        parameters = np.zeros(3, np.float32)
        scales = parameter_scales32(TranslationModel32(), parameters, grid)
        self.assertEqual(scales.dtype, np.float32)
        np.testing.assert_array_equal(scales, np.ones(3, np.float32))
        self.assertEqual(step_scale32(TranslationModel32(), parameters,
                                     np.full(3, 1e-8, np.float32), grid), 0)

    def test_scales_evaluate_composite_members_before_output_collapse(self):
        matrix=np.diag([.7,1.3,2.1,1.]);matrix[:3,3]=[-96.1234567,44.2345678,-63.3456789]
        grid=Grid((5,6,7),matrix)
        base=AffineTransform32.centered(np.eye(3),[.13,-.23,.09],grid.center)
        rigid=ComposedLinearModel32('Rigid',grid.center,base)
        model=ComposedLinearModel32('Affine',grid.center,rigid.transform([.05,.03,-.02,.012,-.014,.013]))
        p=model.identity_parameters();points=corner_points(grid).astype(np.float32)
        def manual(parameters):
            mapped=points
            for member in model.transform(parameters).components:mapped=member.map_points(mapped)
            return mapped.astype(np.float64)
        original=manual(p);variation=np.float32(.01)
        shifts=[]
        for i in range(len(p)):
            changed=p.copy();changed[i]+=variation
            shifts.append(np.float32(np.sqrt(np.sum((manual(changed)-original)**2,axis=1)).max()))
        shifts=np.array(shifts,np.float32)
        expected=shifts*shifts*np.float32(np.float32(1)/(variation*variation))
        np.testing.assert_array_equal(parameter_scales32(model,p,grid),expected)
        collapsed=SimpleNamespace(transform=model.transform)
        self.assertTrue(np.any(parameter_scales32(collapsed,p,grid)!=expected))

    def test_complete_cg_finds_known_float_minimum_and_rounds_monitor(self):
        grid = Grid((1, 1, 1), np.eye(4))
        result = conjugate_gradient32(QuadraticMetric32(), TranslationModel32(),
                                     np.zeros(3), grid, iterations=70,
                                     convergence_threshold=1e-8, convergence_window=8)
        np.testing.assert_allclose(result.parameters, [.2, -.1, .3], atol=1e-5, rtol=0)
        self.assertEqual(result.parameters.dtype, np.float32)
        self.assertEqual(result.scales.dtype, np.float32)
        self.assertTrue(np.isinf(result.history[0]["convergence"]))
        self.assertTrue(np.isfinite(result.history[7]["convergence"]))
        self.assertGreater(len(result.history), 1)
        self.assertLess(result.history[-1]["value"], result.history[0]["value"]*1e-8)

    def test_no_iterations_preserves_owned_float_parameters(self):
        original = np.array([.1, .2, .3], np.float64)
        result = conjugate_gradient32(None, None, original, None, iterations=0,
                                     scales=np.ones(3), estimate_step_scale=lambda p, g: 1)
        self.assertEqual(result.history, [])
        np.testing.assert_array_equal(result.parameters, original.astype(np.float32))
        self.assertFalse(np.shares_memory(result.parameters, original))

    def test_best_parameters_use_after_step_position_and_before_step_value(self):
        calls = []

        class RisingMetric:
            count = 0

            def evaluate(self, model, parameters, derivative=True):
                if derivative:
                    self.count += 1
                    calls.append(parameters.copy())
                return SimpleNamespace(value=float(self.count), derivative=np.ones(1, np.float32),
                                       valid_points=1)

        result = conjugate_gradient32(RisingMetric(), None, [0], None, iterations=3, monitor=False,
                                     scales=[1], estimate_step_scale=lambda p, g: 1)
        # First value is best, but its saved position is after its update.
        np.testing.assert_array_equal(result.parameters, calls[1])
        self.assertEqual(result.history[0]["value"], 1)
        self.assertGreater(float(result.parameters[0]), 0)


if __name__ == "__main__":
    unittest.main()
