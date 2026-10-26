"""Bidirectional saved-field QA against analytic affine determinants.

No registration, ANTs executable or external image data is used. Fields are
formed directly from physical coordinates and known linear maps rather than
from the registration's transform-to-field or inversion implementation.
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'validation'))
from validation.quality.assess_saved_registration import deformation_regularity
from fast_ants import DisplacementField, Grid


class SavedRegistrationRegularityTests(unittest.TestCase):
    def grid(self):
        angle=.31
        matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
                               [np.sin(angle),np.cos(angle),0],
                               [0,0,1]])@np.diag([.8,1.4,2.])
        matrix[:3,3]=[13,-25,8]
        return Grid((7,6,5),matrix)

    def affine_field(self,grid,linear):
        z,y,x=np.indices(grid.shape_zyx)
        points=grid.index_to_physical(np.stack([x,y,z],axis=-1))
        displacement=(points-grid.center)@(linear-np.eye(3)).T
        return DisplacementField(np.moveaxis(displacement,-1,0),grid)

    def test_affine_pair_reports_both_analytic_determinants(self):
        grid=self.grid()
        linear=np.array([[1.02,.03,0],[0,.97,.02],[.01,0,1.01]])
        inverse=np.linalg.inv(linear)
        report=deformation_regularity(self.affine_field(grid,linear),self.affine_field(grid,inverse))
        self.assertAlmostEqual(report['jacobian_minimum'],np.linalg.det(linear),places=12)
        self.assertAlmostEqual(report['inverse_jacobian_minimum'],1/np.linalg.det(linear),places=12)
        self.assertEqual(report['nonpositive_jacobian_points'],0)
        self.assertEqual(report['nonpositive_inverse_jacobian_points'],0)
        for name in ('forward_field_finite','inverse_field_finite',
                     'forward_jacobian_finite','inverse_jacobian_finite'):
            self.assertIs(report[name],True)
        # Preserve the existing inverse-consistency result and both support
        # directions while adding inverse Jacobian evidence.
        self.assertEqual(set(report['inverse_consistency']),
                         {'forward_after_inverse','inverse_after_forward'})

    def test_inverse_fold_is_reported_when_forward_is_regular(self):
        grid=self.grid()
        forward=self.affine_field(grid,np.eye(3))
        folded=self.affine_field(grid,np.diag([-.5,1.,1.]))
        report=deformation_regularity(forward,folded)
        self.assertEqual(report['jacobian_minimum'],1.)
        self.assertEqual(report['nonpositive_jacobian_points'],0)
        self.assertAlmostEqual(report['inverse_jacobian_minimum'],-.5,places=12)
        self.assertEqual(report['nonpositive_inverse_jacobian_points'],int(np.prod(grid.shape_zyx)))

    def test_zero_determinant_counts_in_each_direction(self):
        # Exact axis-aligned coordinates keep the singular determinant exactly
        # zero, so this checks the <=0 branch without a near-zero tolerance.
        grid=Grid((7,6,5),np.diag([1.,2.,3.,1.]))
        identity=self.affine_field(grid,np.eye(3))
        singular=self.affine_field(grid,np.diag([0.,1.,1.]))
        count=int(np.prod(grid.shape_zyx))
        forward=deformation_regularity(singular,identity)
        reverse=deformation_regularity(identity,singular)
        self.assertEqual(forward['jacobian_minimum'],0.)
        self.assertEqual(forward['nonpositive_jacobian_points'],count)
        self.assertEqual(forward['nonpositive_inverse_jacobian_points'],0)
        self.assertEqual(reverse['inverse_jacobian_minimum'],0.)
        self.assertEqual(reverse['nonpositive_inverse_jacobian_points'],count)
        self.assertEqual(reverse['nonpositive_jacobian_points'],0)


if __name__=='__main__':
    unittest.main()
