"""Lazy CC gradients match the existing stored-gradient operator exactly."""
import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from fast_ants.backends.cpu.fp32.metrics.cc_queue import _work32, _work_gradient32, cc_pair_gradient32
from fast_ants.backends.cpu.fp32.kernels import gradient32


class CCFusedGradient32Tests(unittest.TestCase):
    def setUp(self):
        self.threads = get_num_threads()
        set_num_threads(1)

    def tearDown(self):
        set_num_threads(self.threads)

    def check_case(self, fixed, moving, valid, radius, shift, spacing):
        inputs = (fixed, moving, valid, shift, spacing)
        saved = tuple(a.copy() for a in inputs)
        for array in inputs:
            array.setflags(write=False)
        fixed_gradient = gradient32(fixed, shift, spacing)
        moving_gradient = gradient32(moving, shift, spacing)
        expected = _work32(fixed, moving, fixed_gradient, moving_gradient, valid, radius)
        for threads in (1, min(2, self.threads)):
            set_num_threads(threads)
            actual = _work_gradient32(fixed, moving, valid, radius, shift, spacing)
            self.assertEqual(actual[1], expected[1])
            self.assertEqual(np.float32(actual[0]).view(np.uint32),
                             np.float32(expected[0]).view(np.uint32))
            for index in (2, 3, 4):
                self.assertEqual(actual[index].dtype, np.float32)
                np.testing.assert_array_equal(actual[index].view(np.uint32),
                                              expected[index].view(np.uint32))
            public = cc_pair_gradient32(fixed, moving, valid, radius, shift, spacing)
            self.assertEqual(public[0], actual[0])
            self.assertEqual(public[1], actual[1])
            np.testing.assert_array_equal(public[2], actual[2])
            np.testing.assert_array_equal(public[3], actual[3])
        for array, original in zip(inputs, saved):
            np.testing.assert_array_equal(array, original)

    def test_oblique_anisotropic_stencils_masks_and_large_dc(self):
        shape = (5, 7, 9)
        z, y, x = np.indices(shape, dtype=np.float32)
        fixed = np.asarray(.4 + .11 * np.sin(.6 * x) + .07 * y + .03 * z, np.float32)
        moving = np.asarray(.2 + .13 * np.cos(.7 * x + .2) + .05 * y + .04 * z, np.float32)
        # A physical half-spacing stencil with off-diagonal components. It
        # crosses support boundaries on each image face and uses unequal deltas.
        shift = np.array([[.47, -.19, .08], [.16, .53, -.11], [-.07, .09, .49]], np.float32)
        spacing = np.array([.8, 1.3, 2.1], np.float32)
        valid = np.ones(shape, np.bool_)
        valid[:, :, 0] = False
        valid[1:4, 2:5, 3:6] = False
        for dc in (False, True):
            f = (fixed + np.float32(4096)) if dc else fixed.copy()
            m = (moving + np.float32(-8192)) if dc else moving.copy()
            for radius in (0, 1, 2):
                with self.subTest(dc=dc, radius=radius):
                    self.check_case(f, m, valid.copy(), radius, shift.copy(), spacing.copy())

    def test_flat_singleton_strided_and_empty_valid(self):
        rng = np.random.default_rng(20261025)
        for shape in ((1, 1, 7), (1, 5, 9), (4, 5, 9)):
            f = rng.uniform(.1, .9, size=(shape[0], shape[1], shape[2] * 2)).astype(np.float32)
            m = rng.uniform(.1, .9, size=f.shape).astype(np.float32)
            fixed, moving = f[:, :, ::2], m[:, :, ::2]
            shift = np.diag(np.full(3, np.float32(.5)))
            spacing = np.array([.7, 1.1, 2.], np.float32)
            for kind in ('textured', 'flat', 'empty'):
                valid = rng.random(shape) > .2
                if kind == 'flat':
                    fixed = np.full(shape, np.float32(1e6), np.float32)
                    moving = np.full(shape, np.float32(-2e6), np.float32)
                if kind == 'empty':
                    valid.fill(False)
                with self.subTest(shape=shape, kind=kind):
                    self.check_case(fixed, moving, valid, 3, shift.copy(), spacing.copy())


if __name__ == '__main__':
    unittest.main()
