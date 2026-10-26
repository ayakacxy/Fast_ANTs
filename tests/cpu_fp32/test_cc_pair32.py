"""Small exact operator comparisons; no registration or upstream execution."""

import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from fast_ants.backends.cpu.fp32.metrics.cc_pair import cc_pair32
from fast_ants.backends.cpu.fp32.kernels import cc32


class CCPair32Tests(unittest.TestCase):
    def setUp(self):
        self.threads = get_num_threads()
        set_num_threads(1)

    def tearDown(self):
        set_num_threads(self.threads)

    def fixtures(self):
        z, y, x = np.indices((3, 4, 5))
        fixed = np.sin(.3*x) + np.cos(.4*y) + .12*z
        moving = np.sin(.3*x+.09) + np.cos(.4*y-.12) + .1*z + .02*x*y
        cases = (
            (fixed, moving),
            (np.ones_like(fixed), np.full_like(moving, 3)),
            (1+fixed*1e-3, 2+moving*1e-3),
            (4096+fixed, -8192+moving),
            (fixed, -moving),
        )
        return [(f.astype(np.float32), m.astype(np.float32)) for f, m in cases]

    def assert_same(self, actual, expected):
        self.assertEqual(actual.dtype, np.dtype('float32'))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_pair_matches_both_directions_and_threads(self):
        rng = np.random.default_rng(20261024)
        shape = (3, 4, 5)
        fixed_gradient = rng.normal(size=(3, *shape)).astype(np.float32)
        moving_gradient = rng.normal(size=(3, *shape)).astype(np.float32)
        full = np.ones(shape, np.bool_)
        holes = full.copy()
        holes[1:, 1:3, 1:4] = False
        sparse = np.zeros(shape, np.bool_)
        sparse[0, 0, 0] = sparse[-1, -1, -1] = True
        empty = np.zeros(shape, np.bool_)
        thread_counts = [1, min(2, self.threads)] if self.threads > 1 else [1]
        for case, (fixed, moving) in enumerate(self.fixtures()):
            for radius in (0, 1, 3):
                for mask_id, valid in enumerate((full, holes, sparse, empty)):
                    set_num_threads(1)
                    value_f, count_f, update_f = cc32(fixed, moving, moving_gradient, valid, radius)
                    value_m, count_m, update_m = cc32(moving, fixed, fixed_gradient, valid, radius)
                    for threads in thread_counts:
                        with self.subTest(case=case, radius=radius, mask=mask_id, threads=threads):
                            set_num_threads(threads)
                            value, count, forward, reverse = cc_pair32(
                                fixed, moving, fixed_gradient, moving_gradient, valid, radius)
                            self.assertEqual(count, count_f)
                            self.assertEqual(count, count_m)
                            self.assert_same(np.asarray(value, np.float32), np.asarray(value_f, np.float32))
                            self.assert_same(np.asarray(value, np.float32), np.asarray(value_m, np.float32))
                            self.assert_same(forward, update_f)
                            self.assert_same(reverse, update_m)

    def test_pair_does_not_modify_or_alias_inputs(self):
        fixed, moving = self.fixtures()[0]
        gradients = np.ones((3, *fixed.shape), np.float32)
        valid = np.ones(fixed.shape, np.bool_)
        inputs = (fixed, moving, gradients, valid)
        originals = tuple(value.copy() for value in inputs)
        _, _, forward, reverse = cc_pair32(fixed, moving, gradients, gradients, valid, 1)
        self.assertFalse(np.shares_memory(forward, reverse))
        for value, original in zip(inputs, originals):
            np.testing.assert_array_equal(value, original)
            self.assertFalse(np.shares_memory(forward, value))
            self.assertFalse(np.shares_memory(reverse, value))


if __name__ == '__main__':
    unittest.main()
