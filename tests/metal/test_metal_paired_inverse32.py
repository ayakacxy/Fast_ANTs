"""Batched inverse controls preserve independent states and stopping rules."""
import os
import unittest

import numpy as np

from fast_ants.core.types import Grid
from fast_ants.core.fields import DisplacementField


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalPairedInverse32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext
        from fast_ants.backends.metal.fields.inverse import invert_pair
        cls.ops_type,cls.context_type=MetalOps,MetalContext
        cls.pair=staticmethod(invert_pair)

    def fixtures(self):
        angle=.23
        matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[1e7,-2e7,3e7]
        grid=Grid((5,7,9),matrix)
        values=np.random.default_rng(971).normal(0,.22,(3,*grid.shape_zyx)).astype(np.float32)
        return grid,values

    def compare_results(self,ops,actual,expected):
        for got,wanted in zip(actual,expected):
            value=ops.context.download(got.field.data)
            reference=ops.context.download(wanted.field.data)
            self.assertTrue(np.isfinite(value).all())
            np.testing.assert_allclose(value,reference,atol=5e-7,rtol=2e-5)
            self.assertEqual(len(got.history),len(wanted.history))
            self.assertEqual(got.stop_reason,wanted.stop_reason)
            for current,previous in zip(got.history,wanted.history):
                for key in ('mean_scaled_error_before_update','max_scaled_error_before_update'):
                    self.assertAlmostEqual(current[key],previous[key],delta=5e-7)
            if 'evaluated' in got.final_residual:
                self.assertEqual(got.final_residual,{'evaluated':False})
            else:
                for key in got.final_residual:
                    self.assertAlmostEqual(got.final_residual[key],wanted.final_residual[key],delta=5e-7)

    def test_finished_side_stays_finished_and_one_control_read_advances_both(self):
        grid,values=self.fixtures()
        ops=self.ops_type(self.context_type(('parallel_norms','parallel_max')))
        host=(DisplacementField(np.zeros_like(values),grid),DisplacementField(values,grid))
        fields=tuple(ops.field(value) for value in host)
        actual=self.pair(ops,fields,max_iterations=5,_evaluate_final_residual=False)
        self.assertEqual(len(actual[0].history),1)
        self.assertGreater(len(actual[1].history),1)
        reads=ops.context.transfers['device_to_host:control_inverse_pair_norms']['calls']
        self.assertEqual(reads,max(len(value.history) for value in actual))
        self.assertLess(reads,sum(len(value.history) for value in actual))
        expected=tuple(ops.invert(value,max_iterations=5,_evaluate_final_residual=False) for value in fields)
        self.compare_results(ops,actual,expected)
        np.testing.assert_array_equal(ops.context.download(fields[1].data),values)
        for value in actual:
            array=ops.context.download(value.field.data)
            for axis in (1,2,3):
                self.assertTrue(np.all(np.take(array,[0,-1],axis=axis)==0))

    def test_near_geometry_initials_zero_iterations_and_optional_final_diagnostics(self):
        grid,values=self.fixtures()
        matrix=grid.index_to_lps.copy();matrix[:3,3]+=[2e-7,-3e-7,1e-7]
        near=Grid(grid.shape_zyx,matrix)
        for iterations,boundary in ((0,True),(3,False),(3,True)):
            with self.subTest(iterations=iterations,boundary=boundary):
                ops=self.ops_type(self.context_type(('parallel_norms','parallel_max')))
                fields=tuple(ops.field(DisplacementField(v,grid)) for v in (values,values*np.float32(.4)))
                initial=tuple(ops.field(DisplacementField(-v,near)) for v in (values,values*np.float32(.4)))
                saved=tuple(ops.context.download(value.data) for value in initial)
                actual=self.pair(ops,fields,initial,max_iterations=iterations,
                                 enforce_boundary=boundary,_evaluate_final_residual=True)
                expected=tuple(ops.invert(value,start,max_iterations=iterations,enforce_boundary=boundary)
                               for value,start in zip(fields,initial))
                self.compare_results(ops,actual,expected)
                for value,old in zip(initial,saved):
                    np.testing.assert_array_equal(ops.context.download(value.data),old)
                if iterations==0:
                    self.assertFalse(any(value.history for value in actual))
                    self.assertNotIn('device_to_host:control_inverse_pair_norms',ops.context.transfers)
                    self.assertEqual(ops.context.transfers['device_to_host:control_inverse_pair_final_norms']['calls'],1)
        ops=self.ops_type(self.context_type(()))
        fields=tuple(ops.field(DisplacementField(v,grid)) for v in (values,values*np.float32(.4)))
        actual=self.pair(ops,fields,max_iterations=1)
        expected=tuple(ops.invert(value,max_iterations=1,_evaluate_final_residual=False) for value in fields)
        self.compare_results(ops,actual,expected)
        self.assertNotIn('device_to_host:control_inverse_pair_norms',ops.context.transfers)


if __name__=='__main__':unittest.main()
