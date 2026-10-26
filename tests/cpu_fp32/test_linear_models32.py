"""Float model contracts and analytic checks, not an upstream bitwise oracle."""

import unittest
import ctypes
import sys
import tempfile
from pathlib import Path
import numpy as np

from fast_ants.pipeline.models32 import AffineTransform32, LinearModel32, ComposedLinearModel32, rotation32, _matmul32, _offset32, _translation32, _trig32
from fast_ants.core.transforms import AffineTransform
from fast_ants.io.transforms import save_affine, load_affine


class LinearModel32Tests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'darwin', 'Installed ANTs oracle uses macOS libSystem')
    def test_float_trig_matches_native_sincos_on_rounding_sensitive_angles(self):
        # Read-only external test oracle; production does not load this symbol.
        class Pair(ctypes.Structure):
            _fields_ = [('sin', ctypes.c_float), ('cos', ctypes.c_float)]
        native = getattr(ctypes.CDLL(None), '__sincosf_stret')
        native.argtypes, native.restype = [ctypes.c_float], Pair
        for angles in ([.156, .1744, .1948], [-.156, -.1744, -.1948], [0, .1, -.1]):
            angles = np.asarray(angles, np.float32)
            expected = []
            for angle in angles:
                value = native(float(angle))
                expected.extend([value.cos, value.sin])
            np.testing.assert_array_equal(np.asarray(_trig32(angles)), np.asarray(expected, np.float32))

    def test_native_float_contraction_preserves_cancellation_residual(self):
        # (1+2^-23)*(1-2^-23)-1 == -2^-46 with one float FMA.
        # Separate product/addition loses this representable residual.
        plus, minus = np.float32(1+2**-23), np.float32(1-2**-23)
        matrix = np.eye(3, dtype=np.float32)
        matrix[0] = [-1, plus, 0]
        point = np.array([1, minus, 0], np.float32)
        product = _matmul32(matrix, point[:, None])
        self.assertEqual(product[0, 0], np.float32(-2**-46))
        transform = AffineTransform32.centered(matrix, [0, 0, 0], [0, 0, 0])
        self.assertEqual(transform.map_points(point)[0], np.float32(-2**-46))

    def test_offset_and_translation_use_native_fma_after_initial_float_sum(self):
        plus, minus = np.float32(1+2**-23), np.float32(1-2**-23)
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 0] = plus
        center = np.array([minus, 0, 0], np.float32)
        translation = np.array([2**-23, 0, 0], np.float32)
        self.assertEqual(_offset32(matrix, translation, center)[0], np.float32(2**-46))
        offset = np.array([-2**-23, 0, 0], np.float32)
        self.assertEqual(_translation32(matrix, offset, center)[0], np.float32(-2**-46))

    def test_state_is_owned_readonly_float32(self):
        center = np.array([10.123456789, -20.123456789, 30.123456789])
        model = LinearModel32('Affine', center)
        p = model.identity_parameters()
        p[-3:] = [.125, -.25, .5]
        transform = model.transform(p)
        np.testing.assert_array_equal(transform.center, center.astype(np.float32))
        np.testing.assert_array_equal(transform.translation, p[-3:])
        for value in (model.center, transform.center, transform.matrix, transform.translation):
            self.assertEqual(value.dtype, np.float32)
            self.assertFalse(value.flags.writeable)
        self.assertEqual(p.dtype, np.float32)
        center[:] = 0
        p[:] = 0
        self.assertNotEqual(model.center[0], 0)
        self.assertEqual(transform.translation[0], .125)

    def test_euler_axes_and_zxy_order(self):
        half_pi = np.float32(np.pi/2)
        np.testing.assert_allclose(rotation32([half_pi, 0, 0]) @ [0, 1, 0], [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(rotation32([0, half_pi, 0]) @ [0, 0, 1], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(rotation32([0, 0, half_pi]) @ [1, 0, 0], [0, 1, 0], atol=1e-7)
        # Y first sends x to -z, X sends -z to y, Z sends y to -x.
        np.testing.assert_allclose(rotation32([half_pi]*3) @ [1, 0, 0], [-1, 0, 0], atol=2e-7)

    def test_map_rounds_world_point_before_transform(self):
        transform = AffineTransform32.centered(np.eye(3), [0, 0, 0], [0, 0, 0])
        mapped = transform.map_points([2**24+1, 0, 0])
        self.assertEqual(mapped.dtype, np.float32)
        np.testing.assert_array_equal(mapped, [2**24, 0, 0])
        # Offset calculation also preserves the float translation+center boundary.
        centered = AffineTransform32.centered(np.eye(3), [1, 0, 0], [2**24, 0, 0])
        self.assertEqual(centered.matrix[0, 3], 0)
        self.assertEqual(centered.translation[0], 1)

    def test_affine_jacobian_rounds_point_and_center_then_subtracts(self):
        model = LinearModel32('Affine', [2**24, 0, 0])
        jacobian = model.jacobian(model.identity_parameters(), [[2**24+1, 2, 3]])
        self.assertEqual(jacobian.dtype, np.float32)
        np.testing.assert_array_equal(jacobian[0, 0, :3], [0, 2, 3])
        np.testing.assert_array_equal(jacobian[0, :, -3:], np.eye(3))

    def test_parameter_jacobians_match_analytic_derivative_directions(self):
        points = np.array([[12, -15, 23], [7, -22, 35], [9, -16, 28.]])
        for kind in ('Rigid', 'Affine'):
            model = LinearModel32(kind, [10, -20, 30])
            p = model.identity_parameters()
            p += np.linspace(.03, .09, len(p), dtype=np.float32)
            actual = model.jacobian(p, points)
            # Finite-difference steps must exceed float coordinate rounding.
            for parameter in range(len(p)):
                delta = np.zeros_like(p); delta[parameter] = .001
                plus = model.transform(p+delta).map_points(points).astype(np.float64)
                minus = model.transform(p-delta).map_points(points).astype(np.float64)
                numerical = (plus-minus)/(2*float(delta[parameter]))
                np.testing.assert_allclose(actual[..., parameter], numerical, atol=.007, rtol=.003)

    def test_composition_order_and_jacobian_chain_rule(self):
        base = AffineTransform.centered([[1.1, .1, 0], [0, .9, .05], [0, 0, 1.02]], [1, -2, 3], [2, 4, 6])
        model = ComposedLinearModel32('Rigid', [2, 4, 6], base)
        local_model = LinearModel32('Rigid', [2, 4, 6])
        p = np.array([.1, -.15, .07, .3, -.4, .2], np.float32)
        points = np.array([[4, 7, 9], [-2, 3, 8.]])
        result = model.transform(p)
        expected = base.map_points(local_model.transform(p).map_points(points))
        np.testing.assert_allclose(result.map_points(points), expected, atol=3e-6, rtol=0)
        local_jacobian = local_model.jacobian(p, points)
        expected_jacobian = np.einsum('ij,njk->nik', base.matrix[:3, :3], local_jacobian)
        np.testing.assert_allclose(model.jacobian(p, points), expected_jacobian, atol=2e-6, rtol=0)
        self.assertEqual(model.base.matrix.dtype, np.float32)

    def test_sequential_view_keeps_intermediate_point_rounding_explicit(self):
        base = AffineTransform32.centered(np.eye(3), [-1, 0, 0], [0, 0, 0])
        model = ComposedLinearModel32('Affine', [0, 0, 0], base)
        p = model.identity_parameters(); p[-3] = 1
        transform = model.transform(p)
        point = np.array([2**24, 0, 0], np.float32)
        # Local +1 rounds away; base -1 then survives. The collapsed matrix
        # instead has net offset zero. Neither result is silently relabeled.
        self.assertEqual(transform.map_points(point)[0], 2**24)
        self.assertEqual(transform.map_points_sequential(point)[0], 2**24-1)
        self.assertEqual(model.evaluation_transform(p).map_points(point)[0], 2**24-1)
        np.testing.assert_array_equal(transform.matrix, np.eye(4, dtype=np.float32))
        self.assertEqual(len(transform.components), 2)
        np.testing.assert_array_equal(transform.components[0].translation, [1, 0, 0])
        self.assertIs(transform.components[1], base)

    def test_components_survive_stage_boundaries_without_growing_per_iteration(self):
        initial = AffineTransform32.centered(np.eye(3), [1.2, -2.3, .4], [4, 5, 6])
        rigid = ComposedLinearModel32('Rigid', [4, 5, 6], initial)
        rp = np.array([.1, -.2, .3, .4, -.5, .6], np.float32)
        rigid_result = rigid.transform(rp)
        affine = ComposedLinearModel32('Affine', [4, 5, 6], rigid_result)
        ap = affine.identity_parameters(); ap[:3] += np.float32(.02)
        result = affine.transform(ap)
        self.assertEqual(len(result.components), 3)
        points = np.array([[5, 8, 2], [3, -1, 9]], np.float32)
        local = LinearModel32('Affine', [4, 5, 6]).transform(ap)
        expected = initial.map_points(rigid_result.components[0].map_points(local.map_points(points)))
        np.testing.assert_array_equal(result.map_points_sequential(points), expected)
        # Each call replaces the active transform rather than appending it.
        ap[-1] += np.float32(.03)
        self.assertEqual(len(affine.transform(ap).components), 3)
        self.assertEqual(len(rigid_result.components), 2)

    def test_sequential_view_and_saved_affine_have_distinct_explicit_contracts(self):
        base = AffineTransform32.centered(np.eye(3), [-1, 0, 0], [0, 0, 0])
        model = ComposedLinearModel32('Affine', [0, 0, 0], base)
        p = model.identity_parameters(); p[-3] = 1
        transform, evaluation = model.transform(p), model.evaluation_transform(p)
        point = np.array([2**24, 0, 0], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'affine.mat'
            with self.assertRaises(TypeError):
                save_affine(evaluation, path)
            save_affine(transform, path, dtype=np.float32)
            stored = load_affine(path)
            np.testing.assert_array_equal(stored.map_points(point), transform.map_points(point))
            self.assertNotEqual(stored.map_points(point)[0], evaluation.map_points(point)[0])

    def test_no_base_and_identity_base_keep_correct_component_counts(self):
        no_base = ComposedLinearModel32('Rigid', [0, 0, 0])
        p = no_base.identity_parameters()
        points = np.array([[2.5, 3, 4]], np.float32)
        np.testing.assert_array_equal(no_base.evaluation_transform(p).map_points(points), points)
        identity = AffineTransform32.centered(np.eye(3), [0, 0, 0], [0, 0, 0])
        with_base = ComposedLinearModel32('Rigid', [0, 0, 0], identity)
        result = with_base.transform(p)
        self.assertEqual(len(result.components), 2)
        np.testing.assert_array_equal(result.map_points_sequential(points), points)

    def test_inverse_and_parameter_recovery(self):
        model = LinearModel32('Affine', [12, -20, 30])
        p = model.identity_parameters()
        p += np.linspace(.001, .012, 12, dtype=np.float32)
        transform = model.transform(p)
        recovered = model.parameters_from_affine(transform)
        np.testing.assert_allclose(recovered, p, atol=3e-6, rtol=0)
        points = np.array([[4, 7, 9], [-2, 3, 8.]])
        np.testing.assert_allclose(transform.inverse().map_points(transform.map_points(points)), points, atol=3e-6, rtol=0)
        self.assertEqual(transform.inverse().matrix.dtype, np.float32)

    def test_invalid_and_float_overflow_inputs_rejected(self):
        with self.assertRaises(ValueError):
            LinearModel32('Unsupported', [0, 0, 0])
        with self.assertRaises(ValueError):
            LinearModel32('Rigid', [1e100, 0, 0])
        model = LinearModel32('Rigid', [0, 0, 0])
        with self.assertRaises(ValueError):
            model.transform([np.nan]*6)
        with self.assertRaises(ValueError):
            model.transform([0]*5)
        with self.assertRaises(ValueError):
            model.jacobian(model.identity_parameters(), [[1e100, 0, 0]])


if __name__ == '__main__':
    unittest.main()
