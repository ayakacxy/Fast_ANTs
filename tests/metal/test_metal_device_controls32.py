"""Private lazy GPU MI controls preserve the public metric and cost policy."""
import os
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalDeviceControls32Tests(unittest.TestCase):
    def fixture(self,private=False):
        from fast_ants.core.types import Grid, Image
        from fast_ants.pipeline.models32 import ComposedLinearModel32
        from fast_ants.core.transforms import AffineTransform
        from fast_ants.backends.metal.metrics.mattes.metric import MetalMattesMetric
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext, DEFAULT_OPTIMIZATIONS
        flags=tuple(dict.fromkeys((*DEFAULT_OPTIMIZATIONS,'device_mi_control','batch_mi_search')))
        matrix=np.diag([.8,1.3,2.1,1.]);matrix[:3,3]=[17.,-33.,9.]
        grid=Grid((17,29,37),matrix)
        z,y,x=np.indices(grid.shape_zyx)
        fixed=(.5+.17*np.sin(.6*x)+.11*np.cos(.5*y)+.03*z).astype(np.float32)
        moving=(.4+.15*np.sin(.6*x+.09)+.13*np.cos(.5*y-.12)+.025*z+.004*x*y).astype(np.float32)
        ops=MetalOps(MetalContext(flags))
        metric=MetalMattesMetric(ops,ops.image(Image(fixed,grid)),ops.image(Image(moving,grid)),bins=16)
        metric._device_control_mode=private
        base=AffineTransform.centered(np.asarray([[.97,.11,-.07],[-.12,1.03,.06],
                                                  [.02,-.08,.99]]),[.2,-.1,.15],grid.center)
        model=ComposedLinearModel32('Affine',grid.center,base)
        parameters=model.identity_parameters();parameters[1]=np.float32(.007)
        parameters[-3:]=[.3,-.21,.17]
        return metric,model,parameters

    @staticmethod
    def downloads(ctx):
        return {key:dict(value) for key,value in ctx.transfers.items() if key.startswith('device_to_host:')}

    @staticmethod
    def total_reads(ctx):
        entries=[value for key,value in ctx.transfers.items() if key.startswith('device_to_host:')]
        return sum(entry['calls'] for entry in entries),sum(entry['bytes'] for entry in entries)

    def test_public_pdf_details_and_private_value_derivative_controls(self):
        public,model,parameters=self.fixture()
        private,_,_=self.fixture(True)
        self.assertFalse(public.batch_value_evaluations)
        self.assertTrue(private.batch_value_evaluations)
        before=self.total_reads(private.ctx)
        for derivative in (True,False):
            expected=public.evaluate(model,parameters,derivative)
            actual=private.evaluate(model,parameters,derivative)
            self.assertEqual(actual.valid_points,expected.valid_points)
            np.testing.assert_allclose(actual.value,expected.value,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(actual.derivative,expected.derivative,atol=3e-5,rtol=3e-4)
            for key in ('joint_pdf','fixed_pdf','moving_pdf','pdf_derivative'):
                self.assertIn(key,expected.details)
                self.assertNotIn(key,actual.details)
            self.assertTrue(actual.details['device_finalization'])
            self.assertTrue(actual.details['pdf_details_omitted'])
            self.assertFalse(actual.details['software_expanded_precision'])
            self.assertEqual(actual.details['control_transfer_bytes'],60 if derivative else 12)
        after=self.total_reads(private.ctx)
        self.assertEqual((after[0]-before[0],after[1]-before[1]),(2,72))
        self.assertEqual(private.control_evaluations,2)
        self.assertEqual(private.control_derivative_evaluations,1)
        self.assertEqual(private.control_bytes,72)
        self.assertEqual(private.ctx.transfers['device_to_host:control_mattes_final']['calls'],2)

    def test_deferred_pair_single_read_invalid_cost_and_mask_fallback(self):
        from fast_ants.core.types import Image
        from fast_ants.backends.metal.metrics.mattes.finalize import decode_control
        private,model,parameters=self.fixture(True)
        before=self.downloads(private.ctx)
        packet,size=private._enqueue_device_control(model,parameters,False)
        self.assertEqual(size,0)
        self.assertEqual(packet.shape,(3,))
        self.assertEqual(self.downloads(private.ctx),before)
        self.assertEqual(private.control_evaluations,0)
        # Reading this explicitly constructed packet is separate from the
        # paired API's bounded transfer contract below.
        initial=decode_control(private.ctx.control(packet,'test_lazy_mi',3),0)[0]
        second=parameters.copy();second[-3:]+=np.asarray([.03,-.02,.01],np.float32)
        before=self.total_reads(private.ctx)
        paired=private.evaluate_values_many(model,(parameters,second))
        after=self.total_reads(private.ctx)
        self.assertEqual((after[0]-before[0],after[1]-before[1]),(1,24))
        self.assertEqual(private.ctx.transfers['device_to_host:control_mattes_final_pair']['calls'],1)
        self.assertEqual(private.control_evaluations,2)
        self.assertEqual(private.control_bytes,24)
        before=self.total_reads(private.ctx)
        separate=tuple(private.evaluate(model,p,False).value for p in (parameters,second))
        after=self.total_reads(private.ctx)
        self.assertEqual((after[0]-before[0],after[1]-before[1]),(2,24))
        np.testing.assert_allclose(paired,separate,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(paired[0],initial,atol=1e-6,rtol=1e-6)
        invalid=parameters.copy();invalid[-3:]=np.float32(1e6)
        before=self.total_reads(private.ctx);successful=private.control_evaluations
        costs=private.evaluate_values_many(model,(parameters,invalid))
        after=self.total_reads(private.ctx)
        self.assertEqual((after[0]-before[0],after[1]-before[1]),(1,24))
        np.testing.assert_allclose(costs[0],paired[0],atol=1e-6,rtol=1e-6)
        self.assertEqual(costs[1],np.float32(np.finfo(np.float32).max))
        self.assertEqual(private.control_evaluations,successful+1)
        with patch.object(private,'_enqueue_device_control',side_effect=RuntimeError('device failure')):
            with self.assertRaisesRegex(RuntimeError,'device failure'):
                private.evaluate_values_many(model,(parameters,second))
        private.moving_mask=private.ops.image(Image(np.ones(private.grid.shape_zyx,np.float32),private.grid))
        self.assertFalse(private.batch_value_evaluations)
        before=private.ctx.kernels.copy()
        result=private.evaluate_values_many(model,(parameters,second))
        np.testing.assert_allclose(result,paired,atol=2e-5,rtol=2e-5)
        calls=private.ctx.kernels-before
        self.assertEqual(calls['metric_finalize_prepare32'],0)
        self.assertEqual(calls['metric_finalize_controls32'],0)
        self.assertEqual(calls['mattes_control_pair32'],0)


if __name__=='__main__':unittest.main()
