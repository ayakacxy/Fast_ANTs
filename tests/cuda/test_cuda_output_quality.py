"""Native CUDA FP64 final QA against the independent CPU summaries."""
import os
import unittest

import numpy as np

from fast_ants.core.fields import DisplacementField
from fast_ants.core.types import Grid
from fast_ants.quality.output_kernels import output_quality as cpu_quality


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA')=='1',
                     'CUDA device tests require explicit opt-in')
class CUDAOutputQualityTests(unittest.TestCase):
    def compare(self,forward,inverse):
        from fast_ants.backends.cuda.ops import CUDAOps
        from fast_ants.backends.cuda.output_quality import output_quality
        ops=CUDAOps()
        actual=output_quality(ops.context,ops.field(forward),ops.field(inverse))
        expected=cpu_quality(forward,inverse)
        self.assertEqual(actual['nonpositive_jacobian_points'],expected['nonpositive_jacobian_points'])
        self.assertEqual(actual['minimum_nonlinear_jacobian'],expected['minimum_nonlinear_jacobian'])
        for direction,report in expected['nonlinear_inverse_consistency'].items():
            summary=actual['nonlinear_inverse_consistency'][direction]
            for key,value in report.items():
                if value is None:self.assertIsNone(summary[key])
                elif key=='rms_l2_mm':np.testing.assert_allclose(summary[key],value,rtol=3e-14,atol=1e-15)
                else:self.assertEqual(summary[key],value,(direction,key))
        return actual

    def test_oblique_large_origin_and_different_support(self):
        angle=.27
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],
                           [np.sin(angle),np.cos(angle),0],[0,0,1]])
        matrix=np.eye(4);matrix[:3,:3]=rotation@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[1e12,-2e12,3e12]
        other=matrix.copy();other[:3,:3]=rotation@np.diag([.8,1.1,1.9])
        other[:3,3]+=[.2,-.13,.09]
        fields=[]
        for grid,sign in ((Grid((4,5,6),matrix),1),(Grid((5,4,7),other),-1)):
            z,y,x=np.indices(grid.shape_zyx)
            data=sign*np.array([.031*np.sin(.4*x)+.003*y,
                               .023*np.cos(.3*y)-.002*z,.017*np.sin(.2*z)+.001*x])
            fields.append(DisplacementField(data.astype(np.float32),grid))
        self.compare(*fields)

    def test_singleton_folding_empty_support_and_nonfinite(self):
        from fast_ants.backends.cuda.ops import CUDAOps
        from fast_ants.backends.cuda.output_quality import output_quality
        grid=Grid((1,4,6),np.diag([.7,1.2,2.,1.]))
        z,y,x=np.indices(grid.shape_zyx)
        forward=np.zeros((3,*grid.shape_zyx),np.float32);forward[0]=-1.5*.7*x
        inverse=np.zeros_like(forward);inverse[1]=.04*y
        report=self.compare(DisplacementField(forward,grid),DisplacementField(inverse,grid))
        self.assertEqual(report['nonpositive_jacobian_points'],x.size)
        outside=DisplacementField(np.full_like(forward,1e6),grid)
        self.compare(outside,outside)
        invalid=DisplacementField(np.zeros_like(forward),grid);invalid.data[0,0,0,0]=np.nan
        ops=CUDAOps()
        with self.assertRaises(ValueError):
            output_quality(ops.context,ops.field(invalid),ops.field(outside))


if __name__=='__main__':unittest.main()
