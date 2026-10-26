"""Fused Mattes controls against the separate GPU and CPU finalizers."""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalMetricFinalizeFused32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import finalize as metric_finalize32, finalize_fused as metric_finalize_fused32
        from fast_ants.backends.metal.runtime import MetalContext
        cls.original,cls.fused=metric_finalize32,metric_finalize_fused32
        cls.context_type=MetalContext

    @staticmethod
    def pack(raw,counts,valid,bad=None):
        tail=np.empty(0,np.uint32) if bad is None else np.asarray(bad,np.uint32)
        return np.r_[raw.ravel().view(np.uint32),counts,np.uint32(valid),tail].astype(np.uint32)

    def test_parameter_extremes_sparse_bins_large_counts_and_deferred_read(self):
        from fast_ants.pipeline.mattes32 import finalize_mattes_float1
        rng=np.random.default_rng(539)
        for bins,parameters,multiplier in ((5,0,1),(16,6,1),(32,12,4096)):
            with self.subTest(bins=bins,parameters=parameters):
                ctx=self.context_type()
                counts=rng.integers(100,200,bins,dtype=np.uint32)*np.uint32(multiplier)
                counts[1]=0
                probabilities=rng.uniform(.05,1,(bins,bins))
                probabilities[:,0]=0;probabilities/=probabilities.sum(axis=1)[:,None]
                histogram=(probabilities*counts[:,None]).astype(np.float32)
                valid=int(counts.astype(np.uint64).sum())
                derivative=(rng.normal(0,.02,(bins,bins,parameters))*valid/(bins*bins)).astype(np.float32)
                derivative[:,0]=0;derivative[1]=0
                raw=np.concatenate((histogram[:,:,None],derivative),axis=2)
                packed=ctx.upload(self.pack(raw,counts,valid),dtype=np.uint32)
                before=ctx.kernels.copy()
                candidate=self.fused.finalize_packed_device(ctx,packed,bins,parameters,.037)
                calls=ctx.kernels-before
                self.assertEqual(sum(calls.values()),1)
                self.assertEqual(calls['metric_finalize_fused32'],1)
                self.assertFalse(any(key.startswith('device_to_host:') for key in ctx.transfers))
                self.assertEqual(candidate.shape,(parameters+3,))
                reference=self.original.finalize_packed_device(ctx,packed,bins,parameters,.037)
                actual=self.original.decode_control(ctx.control(candidate,'fused_test',parameters+3),parameters)
                expected=self.original.decode_control(ctx.control(reference,'original_test',parameters+3),parameters)
                cpu=finalize_mattes_float1(histogram,counts,derivative,valid,np.float32(.037))
                self.assertEqual(actual[2],valid)
                self.assertEqual(actual[2],expected[2])
                np.testing.assert_allclose(actual[0],expected[0],atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(actual[1],expected[1],atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(actual[0],cpu[0],atol=2e-5,rtol=2e-5)
                np.testing.assert_allclose(actual[1],cpu[1],atol=3e-5,rtol=3e-4)
                self.assertEqual(ctx.transfers['device_to_host:control_fused_test']['bytes'],(parameters+3)*4)

    def test_empty_nonfinite_negative_diagnostic_and_normalization_status(self):
        from fast_ants.core.results import NoValidSamplesError
        bins,parameters=5,1
        for kind in ('empty_count','empty_mass','negative','nonfinite_pdf',
                     'nonfinite_derivative','diagnostic','total_overflow'):
            with self.subTest(kind=kind):
                ctx=self.context_type()
                raw=np.ones((bins,bins,parameters+1),np.float32);raw[:,:,1]=np.float32(.025)
                counts=np.full(bins,20,np.uint32);valid=int(counts.sum())
                bad=np.zeros(1,np.uint32)
                if kind=='empty_count':valid=0
                elif kind=='empty_mass':raw[:,:,0]=0
                elif kind=='negative':raw[0,0,0]=-1
                elif kind=='nonfinite_pdf':raw[0,0,0]=np.nan
                elif kind=='nonfinite_derivative':raw[-1,-1,1]=np.inf
                elif kind=='diagnostic':bad[0]=np.uint32(5)
                else:raw[:,:,0]=np.float32(np.finfo(np.float32).max/2)
                packed=ctx.upload(self.pack(raw,counts,valid,bad),dtype=np.uint32)
                candidate=self.fused.finalize_packed_device(ctx,packed,bins,parameters,.1,bad_blocks=1)
                reference=self.original.finalize_packed_device(ctx,packed,bins,parameters,.1,bad_blocks=1)
                actual=ctx.control(candidate,'fused_invalid_test',parameters+3)
                expected=ctx.control(reference,'original_invalid_test',parameters+3)
                np.testing.assert_array_equal(actual,expected)
                error=NoValidSamplesError if kind in ('empty_count','empty_mass') else ValueError
                with self.assertRaises(error):self.original.decode_control(actual,parameters)
        for width in (0.,-1.,np.nan,np.inf):
            with self.subTest(width=width),self.assertRaises(ValueError):
                self.fused.finalize_packed_device(ctx,packed,bins,parameters,width,bad_blocks=1)


if __name__=='__main__':unittest.main()
