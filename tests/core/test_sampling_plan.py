import tempfile
import unittest
from pathlib import Path

import numpy as np

from fast_ants import Grid
from fast_ants.pipeline.sampling_plan import SamplingPlan, sample_virtual_domain


class SamplingPlanTests(unittest.TestCase):
    def test_dense_points_and_jittered_regular_replay(self):
        grid = Grid((8, 9, 10), np.diag([1.2, 1.5, 2, 1]))
        dense = sample_virtual_domain(grid)
        np.testing.assert_array_equal(dense.points_lps[0], grid.origin)
        np.testing.assert_allclose(dense.points_lps[-1], grid.index_to_physical([9, 8, 7]))
        regular = sample_virtual_domain(grid, 'Regular', .25, 123)
        self.assertEqual(len(regular.points_lps), 180)
        self.assertFalse(np.array_equal(regular.points_lps, dense.points_lps[::4]))
        np.testing.assert_array_equal(regular.points_lps, sample_virtual_domain(grid, 'Regular', .25, 123).points_lps)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'points.npz'
            regular.save(path)
            np.testing.assert_array_equal(SamplingPlan.load(path).points_lps, regular.points_lps)

    def test_random_plan_cardinality_and_seed(self):
        grid = Grid((6, 7, 8), np.eye(4))
        first = sample_virtual_domain(grid, 'Random', .3, 42)
        second = sample_virtual_domain(grid, 'Random', .3, 43)
        self.assertEqual(len(first.points_lps), int(6*7*8*.3))
        self.assertFalse(np.array_equal(first.points_lps, second.points_lps))
        with self.assertRaises(ValueError):
            sample_virtual_domain(grid, 'None', .3)


if __name__=='__main__':
    unittest.main()
