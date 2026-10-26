"""Final QA summaries against the independent public field operators."""
import unittest

import numpy as np

from fast_ants.quality.output_kernels import output_quality
from fast_ants.backends.cpu.fp64.fields import inverse_consistency, jacobian_determinant
from fast_ants.core.types import Grid
from fast_ants.core.fields import DisplacementField


class OutputQATests(unittest.TestCase):
    def compare(self,forward,inverse):
        expected=inverse_consistency(forward,inverse)
        jacobian=jacobian_determinant(forward)
        actual=output_quality(forward,inverse)
        self.assertEqual(set(actual),{'nonlinear_inverse_consistency','minimum_nonlinear_jacobian',
                                      'nonpositive_jacobian_points'})
        for direction,report in expected.items():
            summary=actual['nonlinear_inverse_consistency'][direction]
            self.assertEqual(set(summary),set(report))
            for key in ('common_valid_points','total_points','p99_l2_mm','max_l2_mm'):
                self.assertEqual(summary[key],report[key],(direction,key))
            if report['rms_l2_mm'] is None:self.assertIsNone(summary['rms_l2_mm'])
            else:np.testing.assert_allclose(summary['rms_l2_mm'],report['rms_l2_mm'],rtol=3e-14,atol=1e-15)
        self.assertEqual(actual['minimum_nonlinear_jacobian'],float(jacobian.min()))
        self.assertEqual(actual['nonpositive_jacobian_points'],int((jacobian<=0).sum()))
        return actual

    def test_oblique_large_origin_different_grids_and_both_storage_dtypes(self):
        angle=.27
        rotation=np.asarray([[np.cos(angle),-np.sin(angle),0],
                             [np.sin(angle),np.cos(angle),0],[0,0,1]])
        matrix=np.eye(4);matrix[:3,:3]=rotation@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[1e12,-2e12,3e12]
        other=matrix.copy();other[:3,:3]=rotation@np.diag([.8,1.1,1.9])
        other[:3,3]+=[.2,-.13,.09]
        forward_grid,inverse_grid=Grid((4,5,6),matrix),Grid((5,4,7),other)
        for dtype in (np.float32,np.float64):
            with self.subTest(dtype=dtype):
                fields=[]
                for grid,sign in ((forward_grid,1),(inverse_grid,-1)):
                    z,y,x=np.indices(grid.shape_zyx)
                    data=sign*np.asarray([.031*np.sin(.4*x)+.003*y,
                                          .023*np.cos(.3*y)-.002*z,
                                          .017*np.sin(.2*z)+.001*x])
                    fields.append(DisplacementField(data.astype(dtype),grid))
                self.compare(*fields)

    def test_singleton_axis_folding_empty_support_and_nonfinite_rejection(self):
        grid=Grid((1,4,6),np.diag([.7,1.2,2.,1.]))
        for dtype in (np.float32,np.float64):
            with self.subTest(dtype=dtype):
                z,y,x=np.indices(grid.shape_zyx)
                forward=np.zeros((3,*grid.shape_zyx),dtype);forward[0]=-1.5*.7*x
                inverse=np.zeros_like(forward);inverse[1]=.04*y
                report=self.compare(DisplacementField(forward,grid),DisplacementField(inverse,grid))
                self.assertEqual(report['nonpositive_jacobian_points'],x.size)
                outside=DisplacementField(np.full_like(forward,1e6),grid)
                report=self.compare(outside,outside)
                for summary in report['nonlinear_inverse_consistency'].values():
                    self.assertEqual(summary['common_valid_points'],0)
                    self.assertIsNone(summary['p99_l2_mm'])
                finite=DisplacementField(np.zeros_like(forward),grid)
                finite.data[0,0,0,0]=np.nan
                with self.assertRaises(ValueError):output_quality(finite,outside)


if __name__=='__main__':unittest.main()
