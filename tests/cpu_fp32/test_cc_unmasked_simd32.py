"""Unmasked SIMD statistics match the established masked/general CC path."""
import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from fast_ants.backends.cpu.fp32.metrics.cc_queue import _work_gradient32
from fast_ants.backends.cpu.fp32.metrics.cc_simd import cc_unmasked_gradient32


class CCUnmaskedSIMD32Tests(unittest.TestCase):
    def test_clipped_windows_vector_tails_large_dc_and_degenerate_variance(self):
        rng = np.random.default_rng(7501)
        shift = np.array([[.47, -.19, .08], [.16, .53, -.11], [-.07, .09, .49]], np.float32)
        spacing = np.array([.8, 1.3, 2.1], np.float32)
        previous = get_num_threads()
        try:
            for nx in (1, 3, 4, 5, 8, 9, 13):
                shape = (1, 3, nx) if nx < 4 else (4, 5, nx)
                f = rng.uniform(.2, .8, shape).astype(np.float32)
                m = rng.uniform(.1, .9, shape).astype(np.float32)
                valid = np.ones(shape, np.bool_)
                for kind in ('ordinary', 'dc', 'flat'):
                    fixed, moving = f.copy(), m.copy()
                    if kind == 'dc':
                        fixed += np.float32(4096)
                        moving -= np.float32(8192)
                    elif kind == 'flat':
                        fixed.fill(np.float32(1e6))
                        moving.fill(np.float32(-2e6))
                    fixed.setflags(write=False)
                    moving.setflags(write=False)
                    for radius in (0, 2, 12):
                        expected = _work_gradient32(fixed, moving, valid, radius, shift, spacing)
                        for threads in (1, 2):
                            set_num_threads(threads)
                            actual = cc_unmasked_gradient32(fixed, moving, radius, shift, spacing)
                            with self.subTest(nx=nx, kind=kind, radius=radius, threads=threads):
                                self.assertEqual(actual[1], expected[1])
                                self.assertEqual(np.float32(actual[0]).view(np.uint32),
                                                 np.float32(expected[0]).view(np.uint32))
                                for a, b in zip(actual[2:], expected[2:4]):
                                    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        finally:
            set_num_threads(previous)


if __name__ == '__main__':
    unittest.main()
