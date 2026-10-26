"""Deferred paired CC controls preserve updates and public observations."""
import json
import os
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalDeferredControls32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext
        cls.ops_type,cls.context_type=MetalOps,MetalContext

    def fixture(self,flags):
        from fast_ants.core.types import Grid, Image
        angle=.23
        matrix=np.eye(4)
        matrix[:3,:3]=np.asarray([[np.cos(angle),-np.sin(angle),0],
                                 [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[17.,-33.,9.]
        grid=Grid((5,7,9),matrix)
        z,y,x=np.indices(grid.shape_zyx,dtype=np.float32)
        fixed=(.5+.17*np.sin(.6*x)+.11*np.cos(.5*y)+.03*z).astype(np.float32)
        moving=(.4+.15*np.sin(.6*x+.09)+.13*np.cos(.5*y-.12)+.025*z+.004*x*y).astype(np.float32)
        ops=self.ops_type(self.context_type(flags))
        return ops,ops.image(Image(fixed,grid)),ops.image(Image(moving,grid))

    @staticmethod
    def reads(ctx,reason):
        return ctx.transfers.get('device_to_host:control_'+reason,{}).get('calls',0)

    @staticmethod
    def downloads(ctx):
        return {key:dict(value) for key,value in ctx.transfers.items() if key.startswith('device_to_host:')}

    def test_lazy_shared_score_one_read_public_equivalence_empty_mask_and_restore(self):
        from fast_ants.core.results import NoValidSamplesError
        ops,fixed,moving=self.fixture(('parallel_cc',))
        ctx=ops.context
        z,y,x=np.indices(fixed.grid.shape_zyx)
        valid=ctx.upload((x+2*y+3*z)%9!=0,dtype=np.bool_)
        before=self.downloads(ctx)
        forward,reverse=ops.cc_pair(fixed,moving,2,valid,_defer_control=True)
        self.assertIs(forward.pending,reverse.pending)
        self.assertEqual(self.downloads(ctx),before)
        self.assertFalse(ctx.defer_cc_score)
        self.assertEqual(ctx.async_submissions['independent_cc_score'],1)
        self.assertEqual(self.reads(ctx,'cc_final'),0)
        value,count=forward.value,reverse.valid_points
        self.assertEqual(reverse.value,value)
        self.assertEqual(forward.valid_points,count)
        self.assertEqual(self.reads(ctx,'cc_final'),1)
        self.assertEqual(ctx.transfers['device_to_host:control_cc_final']['bytes'],8)
        expected=ops.cc_pair(fixed,moving,2,valid)
        self.assertEqual(self.reads(ctx,'cc_value'),1)
        self.assertEqual(self.reads(ctx,'cc_count'),1)
        for actual,reference in zip((forward,reverse),expected):
            self.assertEqual(actual.value,reference.value)
            self.assertEqual(actual.valid_points,reference.valid_points)
            np.testing.assert_array_equal(ctx.download(actual.update.data),ctx.download(reference.update.data))
        empty=ctx.upload(np.zeros(fixed.grid.shape_zyx,np.bool_),dtype=np.bool_)
        before=self.downloads(ctx)
        invalid=ops.cc_pair(fixed,moving,2,empty,_defer_control=True)
        self.assertEqual(self.downloads(ctx),before)
        self.assertFalse(ctx.defer_cc_score)
        with self.assertRaises(NoValidSamplesError):_ = invalid[0].value
        self.assertFalse(ctx.defer_cc_score)
        # A device construction failure must restore the previous context
        # state as well; it must not become an empty-domain observation.
        ctx.defer_cc_score=True
        try:
            with patch.object(ops,'_cc_pair_arrays',side_effect=RuntimeError('device failure')):
                with self.assertRaisesRegex(RuntimeError,'device failure'):
                    ops.cc_pair(fixed,moving,2,valid,_defer_control=True)
            self.assertTrue(ctx.defer_cc_score)
        finally:ctx.defer_cc_score=False

    def test_syn_step_controls_fields_details_and_capture_public_path(self):
        flags=('paired_cc','parallel_cc','parallel_norms','parallel_max',
               'direct_field_convolution','fused_scaled_compose','batched_inverse')
        baseline,fixed,moving=self.fixture(flags)
        candidate,cfixed,cmoving=self.fixture((*flags,'deferred_syn_controls'))
        self.assertNotIn('cc_window_stats',candidate.context.optimizations)
        kwargs={'radius':2,'gradient_step':.1,'update_variance':.3,'total_variance':.1,
                '_evaluate_final_residual':False}
        old=baseline.syn_step(fixed,moving,baseline.identity(fixed.grid),**kwargs)
        new=candidate.syn_step(cfixed,cmoving,candidate.identity(cfixed.grid),**kwargs)
        self.assertEqual(new.value,old.value)
        self.assertEqual(new.details,old.details)
        json.dumps(new.details)
        for name,field in new.state.fields().items():
            np.testing.assert_array_equal(candidate.context.download(field.data),
                baseline.context.download(old.state.fields()[name].data))
        bctx,cctx=baseline.context,candidate.context
        self.assertEqual(self.reads(bctx,'cc_value')+self.reads(bctx,'cc_count'),2)
        self.assertEqual(self.reads(bctx,'cc_final'),0)
        self.assertEqual(self.reads(cctx,'cc_value')+self.reads(cctx,'cc_count'),0)
        self.assertEqual(self.reads(cctx,'cc_final'),1)
        self.assertEqual(cctx.transfers['device_to_host:control_cc_final']['bytes'],8)
        self.assertEqual(cctx.async_submissions['independent_cc_score'],1)
        self.assertEqual(cctx.report()['async_submissions']['independent_cc_score'],1)
        self.assertFalse(cctx.defer_cc_score)
        kernels=cctx.kernels.copy();submissions=cctx.async_submissions.copy()
        captured=candidate.syn_step(cfixed,cmoving,candidate.identity(cfixed.grid),capture=True,**kwargs)
        self.assertEqual(captured.value,old.value)
        self.assertIn('fixed_scaled_update',captured.intermediates)
        self.assertIn('moving_scaled_update',captured.intermediates)
        self.assertEqual(self.reads(cctx,'cc_final'),1)
        self.assertEqual(self.reads(cctx,'cc_value'),1)
        self.assertEqual(self.reads(cctx,'cc_count'),1)
        self.assertEqual(cctx.async_submissions,submissions)
        self.assertFalse(cctx.defer_cc_score)
        self.assertEqual((cctx.kernels-kernels)['scaled_compose32'],0)
        self.assertEqual((cctx.kernels-kernels)['multiply'],2)
        for name,field in captured.state.fields().items():
            np.testing.assert_array_equal(cctx.download(field.data),
                baseline.context.download(old.state.fields()[name].data))
        for inversion in captured.details['inversions'].values():
            self.assertIn('mean_spacing_scaled_norm',inversion['final_residual'])
        json.dumps(captured.details)


if __name__=='__main__':unittest.main()
