import unittest

import numpy as np

from fast_ants import Grid, DisplacementField
from fast_ants.quality.metrics import field_difference


class QualityTests(unittest.TestCase):
    def test_one_micron_bound_is_on_vector_length(self):
        grid = Grid((3, 4, 5), np.eye(4))
        reference = DisplacementField(np.zeros((3, *grid.shape_zyx)), grid)
        values = reference.data.copy()
        values[:, 1, 1, 1] = [.0008, .0008, 0]
        stats = field_difference(DisplacementField(values, grid), reference, .001)
        self.assertFalse(stats['passed'])
        self.assertAlmostEqual(stats['max_l2_mm'], np.sqrt(2)*.0008, places=15)
        self.assertLess(stats['max_component_abs_mm'], .001)
        values[:, 1, 1, 1] = [.0006, .0006, 0]
        self.assertTrue(field_difference(DisplacementField(values, grid), reference, .001)['passed'])

    def test_mismatched_world_geometry_rejected(self):
        grid = Grid((2, 3, 4), np.eye(4))
        other = np.eye(4); other[0, 3] = .1
        with self.assertRaises(ValueError):
            field_difference(DisplacementField(np.zeros((3, 2, 3, 4)), grid),
                             DisplacementField(np.zeros((3, 2, 3, 4)), Grid(grid.shape_zyx, other)), .001)
