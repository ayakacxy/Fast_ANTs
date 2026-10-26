"""Asynchronous inverse submission preserves independent stop/update controls."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from fast_ants.core.types import Grid
from fast_ants.core.fields import DisplacementField


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalInverseOverlap32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.fields import overlap as overlap
        from fast_ants.backends.metal import dense
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext
        from fast_ants.backends.metal.fields.inverse import invert_pair
        cls.overlap,cls.dense=overlap,dense
        cls.ops_type,cls.context_type=MetalOps,MetalContext
        cls.reference=staticmethod(invert_pair)

    def fixtures(self):
        angle=.23;matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[1e7,-2e7,3e7]
        grid=Grid((5,7,9),matrix)
        values=np.random.default_rng(971).normal(0,.22,(3,*grid.shape_zyx)).astype(np.float32)
        return grid,values

    def compare(self,ops,actual,expected):
        for got,wanted in zip(actual,expected):
            np.testing.assert_allclose(ops.context.download(got.field.data),
                ops.context.download(wanted.field.data),atol=5e-7,rtol=2e-5)
            self.assertEqual(got.history,wanted.history)
            self.assertEqual(got.stop_reason,wanted.stop_reason)
            self.assertEqual(got.final_residual,wanted.final_residual)

    def test_both_updates_submitted_before_read_and_finished_side_not_advanced(self):
        grid,values=self.fixtures()
        ops=self.ops_type(self.context_type(('parallel_norms','parallel_max')))
        fields=tuple(ops.field(DisplacementField(value,grid)) for value in
                     (np.zeros_like(values),values))
        events=[];submit=self.overlap.mx.async_eval;control=ops.context.control
        def recorded_submit(*arrays):
            events.append(('submit',len(arrays)))
            return submit(*arrays)
        def recorded_control(data,reason,maximum_elements=32):
            if reason=='inverse_pair_norms':events.append(('read',data.size))
            return control(data,reason,maximum_elements)
        with patch.object(self.overlap.mx,'async_eval',side_effect=recorded_submit), \
                patch.object(ops.context,'control',side_effect=recorded_control):
            actual=self.overlap.invert_pair(ops,fields,max_iterations=5)
        rounds=max(len(value.history) for value in actual)
        self.assertEqual(len(actual[0].history),1)
        self.assertGreater(len(actual[1].history),1)
        self.assertEqual(len(events),2*rounds)
        self.assertEqual(events[:2],[('submit',3),('read',4)])
        for queued,read in zip(events[::2],events[1::2]):
            self.assertEqual(queued[0],'submit');self.assertEqual(read[0],'read')
        self.assertTrue(any(event==('submit',2) for event in events))
        self.assertEqual(ops.context.control_guards['inverse_overlap_async_submissions'],rounds)
        self.assertEqual(ops.context.async_submissions['inverse_update'],rounds)
        self.assertEqual(ops.context.kernels['inverse_step_device_max32'],
                         sum(len(value.history) for value in actual))
        expected=self.reference(ops,fields,max_iterations=5)
        self.compare(ops,actual,expected)
        np.testing.assert_array_equal(ops.context.download(fields[1].data),values)
        # Captured/callback observations ask for actual post-update diagnostics.
        near_matrix=grid.index_to_lps.copy();near_matrix[:3,3]+=[2e-7,-3e-7,1e-7]
        near=Grid(grid.shape_zyx,near_matrix)
        initial=tuple(ops.field(DisplacementField(-value,near)) for value in
                      (np.zeros_like(values),values))
        for iterations,boundary in ((0,True),(2,False)):
            options=dict(max_iterations=iterations,enforce_boundary=boundary,_evaluate_final_residual=True)
            self.compare(ops,self.overlap.invert_pair(ops,fields,initial,**options),
                         self.reference(ops,fields,initial,**options))
        serial=self.ops_type(self.context_type(()))
        self.compare(serial,self.overlap.invert_pair(serial,fields,max_iterations=1),
                     self.reference(serial,fields,max_iterations=1))
        self.assertEqual(serial.context.control_guards['inverse_overlap_async_submissions'],0)

    def test_native_stop_max_and_mean_guard_recomputed_update_remain_distinct(self):
        grid=Grid((3,4,5),np.eye(4))
        values=np.zeros((3,*grid.shape_zyx),np.float32);values[0]=np.float32(.1)
        ops=self.ops_type(self.context_type(('parallel_norms','parallel_max')))
        fields=tuple(ops.field(DisplacementField(value,grid)) for value in (values,values.copy()))
        probe=self.reference(ops,fields,max_iterations=1,mean_tolerance=0.,max_tolerance=0.)
        maximum=probe[0].history[0]['max_scaled_error_before_update']
        # A native threshold guard may retain a larger stop control, but that
        # corrected control must not be substituted into the update formula.
        with patch.object(self.dense,'maximum_control',return_value=maximum*2):
            options=dict(max_iterations=2,mean_tolerance=0.,max_tolerance=maximum,
                         enforce_boundary=False,_evaluate_final_residual=True)
            actual=self.overlap.invert_pair(ops,fields,**options)
            expected=self.reference(ops,fields,**options)
        self.compare(ops,actual,expected)
        self.assertEqual(len(actual[0].history),2)
        self.assertEqual(actual[0].history[0]['max_control_error'],maximum*2)
        self.assertEqual(actual[0].history[0]['max_scaled_error_before_update'],maximum)

        original=self.dense._serial_norms
        def changed_maximum(ctx,norms,name='norm_reduce'):
            result=original(ctx,norms,name)
            return ctx.call('test_guard_changed_max',
                'out0[0]=data[0];out0[1]=data[1]*1.25f;',{'data':result},[(2,)],1)
        # Artificially change a guarded maximum to exercise the rare discard/
        # recompute path; real serial and parallel maxima ordinarily agree.
        mean=probe[0].history[0]['mean_scaled_error_before_update']
        with patch.object(self.dense,'_serial_norms',side_effect=changed_maximum):
            options=dict(max_iterations=1,mean_tolerance=mean,max_tolerance=0.,
                         enforce_boundary=False,_evaluate_final_residual=True)
            actual=self.overlap.invert_pair(ops,fields,**options)
            expected=self.reference(ops,fields,**options)
        self.compare(ops,actual,expected)
        self.assertGreater(ops.context.control_guards['inverse_overlap_guard_recomputations'],0)


if __name__=='__main__':unittest.main()
