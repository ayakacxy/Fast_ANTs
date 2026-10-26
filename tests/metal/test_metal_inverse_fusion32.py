"""Targeted equivalence checks for the fused inversion residual/norm pass."""

import os
import unittest

import numpy as np

from fast_ants.core.types import Grid
from fast_ants.core.fields import DisplacementField


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL') == '1', 'Explicit GPU test opt-in')
class MetalInverseFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal import dense
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext
        cls.dense, cls.ops_type, cls.context_type = dense, MetalOps, MetalContext

    def fixture(self):
        matrix=np.eye(4)
        a=.23
        matrix[:3,:3]=np.array([[np.cos(a),-np.sin(a),0],
            [np.sin(a),np.cos(a),0],[0,0,1]])@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[1e7,-2e7,3e7]
        grid=Grid((5,7,9),matrix)
        values=np.random.default_rng(14).normal(0,.15,(3,*grid.shape_zyx)).astype(np.float32)
        return DisplacementField(values,grid)

    def test_residual_and_norms_match_separate_passes(self):
        ctx=self.context_type(('parallel_norms',))
        field=self.fixture()
        outer=ctx.upload(field.data)
        # Include outside support, edge clamping, and different source/output grids.
        inner=ctx.upload(np.random.default_rng(18).uniform(-3,3,(3,4,6,8)).astype(np.float32))
        linear=np.diag([.9,1.1,1.2]);offset=np.array([-.49,.23,-.7])
        inverse=field.grid.lps_to_index[:3,:3];spacing=field.grid.spacing_xyz
        reference=self.dense.compose(ctx,outer,inner,linear,offset,inverse)
        expected,mean,maximum=self.dense.norms(ctx,reference,spacing)
        residual,values=self.dense.compose(ctx,outer,inner,linear,offset,inverse,spacing)
        for actual,wanted in ((residual,reference),(values,expected)):
            np.testing.assert_array_equal(ctx.download(actual),ctx.download(wanted))
        _,got_mean,got_max=self.dense.reduce_norms(ctx,values,mean)
        _,ref_mean,ref_max=self.dense.norms(ctx,reference,spacing,mean)
        self.assertEqual((got_mean,got_max),(ref_mean,ref_max))

    def test_inversion_keeps_boundaries_and_stopping_decisions(self):
        for parallel in ((),('parallel_norms',)):
            baseline=self.ops_type(self.context_type(parallel))
            fused=self.ops_type(self.context_type((*parallel,'fused_inverse_norms')))
            field=self.fixture()
            for boundary,iterations,initial in ((True,4,False),(False,4,True),(True,0,False),
                                                (True,0,'near'),(False,2,'near')):
                with self.subTest(parallel=parallel,boundary=boundary,iterations=iterations):
                    results=[]
                    for ops in (baseline,fused):
                        device=ops.field(field)
                        start_grid=field.grid
                        if initial=='near':
                            matrix=field.grid.index_to_lps.copy()
                            matrix[:3,3]+=[2e-7,-3e-7,1e-7]
                            start_grid=Grid(field.grid.shape_zyx,matrix)
                        start=ops.field(DisplacementField(-field.data,start_grid)) if initial else None
                        results.append(ops.invert(device,start,max_iterations=iterations,
                            mean_tolerance=.001,max_tolerance=.1,enforce_boundary=boundary))
                    ref,actual=results
                    np.testing.assert_array_equal(fused.context.download(actual.field.data),
                                                  baseline.context.download(ref.field.data))
                    self.assertEqual(actual.history,ref.history)
                    self.assertEqual(actual.stop_reason,ref.stop_reason)
                    self.assertEqual(actual.final_residual,ref.final_residual)
            self.assertGreater(fused.context.kernels['compose_norms'],0)
            self.assertEqual(fused.context.kernels['norms'],0)
            self.assertEqual(fused.context.kernels['compose'],0)


if __name__=='__main__':
    unittest.main()
