"""Opt-in GPU component mapping contracts, independent of registration tests.

FAST_ANTS_TEST_METAL=1 python -m unittest discover -s tests -p test_metal_geometry32.py
"""

import os
import unittest

import numpy as np

from fast_ants.pipeline.models32 import AffineTransform32, ComposedLinearModel32


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1',
                     'Metal device tests require explicit opt-in')
class MetalGeometry32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.geometry import map_world_components32
        from fast_ants.backends.metal.runtime import MetalContext, mx
        if not mx.metal.is_available():
            raise unittest.SkipTest('An accessible Apple GPU is required')
        cls.map_points = staticmethod(map_world_components32)
        cls.context_type, cls.mx = MetalContext, mx

    def check_components(self, points, components, device_matrices=False):
        ctx = self.context_type(())
        expected = points.copy()
        for component in components:
            expected = component.map_points(expected)
        matrices = np.stack([component.matrix[:3] for component in components])
        points.setflags(write=False)
        matrices.setflags(write=False)
        original_points, original_matrices = points.copy(), matrices.copy()
        gpu_points = ctx.upload(points, 'test_input')
        supplied_matrices = ctx.upload(matrices, 'test_input') if device_matrices else matrices
        before = ctx.kernels.copy()
        result = self.map_points(ctx, gpu_points, supplied_matrices)
        # A component sequence is one dispatch, not one kernel per component.
        self.assertEqual(ctx.kernels['map_world_components32']-before['map_world_components32'], 1)
        self.assertFalse(any(key.startswith('device_to_host:') for key in ctx.transfers))
        self.assertEqual(result.dtype, self.mx.float32)
        self.assertIsNot(result, gpu_points)
        actual = ctx.download(result, 'test_output')
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(ctx.download(gpu_points, 'test_input_after').view(np.uint32),
                                      original_points.view(np.uint32))
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(matrices, original_matrices)
        return actual

    def test_boundary_coordinates_and_native_fma(self):
        values = np.array([-.5, 0, .5, 63.5], np.float32)
        values = np.concatenate((values, np.nextafter(values, np.float32(-np.inf)),
                                 np.nextafter(values, np.float32(np.inf))))
        points = np.column_stack((values, values[::-1], np.ones_like(values)))
        plus, minus = np.float32(1+2**-23), np.float32(1-2**-23)
        points = np.vstack((points, np.array([[1, minus, 0]], np.float32)))
        matrix = np.eye(3, dtype=np.float32)
        matrix[0] = [-1, plus, 0]
        component = AffineTransform32.centered(matrix, [0, -.25, .5], [0, 0, 0])
        for on_device in (False, True):
            with self.subTest(device_matrices=on_device):
                actual = self.check_components(points.copy(), (component,), on_device)
                self.assertEqual(actual[-1, 0], np.float32(-2**-46))

    def test_three_stage_sequence_uses_application_order(self):
        center = np.array([-96.1234, 44.2345, -63.3456])
        initial = AffineTransform32.centered(np.eye(3), [.13, -.23, .09], center)
        rigid = ComposedLinearModel32('Rigid', center, initial)
        rp = np.array([.05, .03, -.02, .012, -.014, .013], np.float32)
        affine = ComposedLinearModel32('Affine', center, rigid.transform(rp))
        ap = affine.identity_parameters()
        ap[1], ap[3], ap[-3:] = .002, .003, [.001, -.002, .003]
        transform = affine.transform(ap)
        self.assertEqual(len(transform.components), 3)
        points = np.random.default_rng(23).uniform(-120, 120, (37, 3)).astype(np.float32)
        actual = self.check_components(points, transform.components)
        collapsed = transform.map_points(points)
        self.assertTrue(np.any(actual.view(np.uint32) != collapsed.view(np.uint32)))

    def test_large_origin_preserves_each_component_rounding(self):
        base = AffineTransform32.centered(np.eye(3), [-1, 0, 0], [0, 0, 0])
        model = ComposedLinearModel32('Affine', np.zeros(3), base)
        parameters = model.identity_parameters()
        parameters[-3] = 1
        transform = model.transform(parameters)
        points = np.array([[2**24, 0, 0], [2**24+2, 1, -1]], np.float32)
        actual = self.check_components(points, transform.components, device_matrices=True)
        self.assertEqual(actual[0, 0], np.float32(2**24-1))
        self.assertEqual(transform.map_points(points)[0, 0], np.float32(2**24))

    def test_empty_components_and_empty_points_have_explicit_contracts(self):
        ctx = self.context_type(())
        points = np.array([[-0., 0., -.5], [1, 2, 3]], np.float32)
        gpu_points = ctx.upload(points)
        empty_matrices = np.empty((0, 3, 4), np.float32)
        result = self.map_points(ctx, gpu_points, empty_matrices)
        self.assertIsNot(result, gpu_points)
        np.testing.assert_array_equal(ctx.download(result).view(np.uint32), points.view(np.uint32))
        empty = ctx.upload(np.empty((0, 3), np.float32))
        before_kernels, before_transfers = ctx.kernels.copy(), repr(ctx.transfers)
        self.assertIs(self.map_points(ctx, empty, empty_matrices), empty)
        self.assertIs(self.map_points(ctx, empty, np.eye(4, dtype=np.float32)[None, :3]), empty)
        self.assertEqual(ctx.kernels, before_kernels)
        self.assertEqual(repr(ctx.transfers), before_transfers)

    def test_shape_and_dtype_errors_do_not_dispatch(self):
        ctx = self.context_type(())
        points = ctx.upload(np.zeros((2, 3), np.float32))
        matrices = np.eye(4, dtype=np.float32)[None, :3]
        malformed = (
            (np.zeros((2, 3), np.float32), matrices, TypeError),
            (ctx.upload(np.zeros((2, 3), np.int32), dtype=np.int32), matrices, TypeError),
            (ctx.upload(np.zeros((3, 2), np.float32)), matrices, ValueError),
            (points, matrices.astype(np.float64), TypeError),
            (points, matrices[0], ValueError),
            (points, ctx.upload(np.zeros((1, 4, 4), np.float32)), ValueError),
        )
        before = ctx.kernels.copy()
        for p, m, exception in malformed:
            with self.assertRaises(exception):
                self.map_points(ctx, p, m)
        self.assertEqual(ctx.kernels, before)


if __name__ == '__main__':
    unittest.main()
