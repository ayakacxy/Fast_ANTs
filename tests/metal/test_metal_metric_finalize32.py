"""Bounded GPU Mattes finalization compared with the CPU float1 formulas."""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalMetricFinalize32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import finalize as finalize
        from fast_ants.backends.metal.runtime import MetalContext, mx
        cls.finalize,cls.context_type,cls.mx=finalize,MetalContext,mx

    def test_ordinary_and_large_integer_counts_parameters_and_deferred_read(self):
        from fast_ants.pipeline.mattes32 import finalize_mattes_float1
        rng=np.random.default_rng(322)
        bins=16
        for parameters,multiplier in ((0,1),(6,1),(12,4096)):
            with self.subTest(parameters=parameters,multiplier=multiplier):
                ctx=self.context_type()
                counts=rng.integers(1000,3000,bins,dtype=np.uint32)*np.uint32(multiplier)
                valid=int(counts.astype(np.uint64).sum())
                rows=rng.uniform(.05,1,(bins,bins));rows/=rows.sum(axis=1)[:,None]
                histogram=(rows*counts[:,None]).astype(np.float32)
                derivative=(rng.normal(0,.1,(bins,bins,parameters))*valid/(bins*bins)).astype(np.float32)
                # Empty columns and rows exercise epsilon branches in both
                # implementations without putting values on unstable cutoffs.
                histogram[:,0]=0;derivative[:,0]=0
                counts[1]=0;histogram[1]=0;derivative[1]=0
                valid=int(counts.astype(np.uint64).sum())
                raw=np.concatenate((histogram[:,:,None],derivative),axis=2)
                number=np.r_[counts,np.uint32(valid)].astype(np.uint32)
                width=np.float32(.037)
                expected=finalize_mattes_float1(histogram,counts,derivative,valid,width)
                packet=self.finalize.finalize_device(ctx,ctx.upload(raw),
                    ctx.upload(number,dtype=np.uint32),width)
                self.assertEqual(packet.dtype,self.mx.uint32)
                self.assertEqual(packet.shape,(parameters+3,))
                self.assertFalse(any(key.startswith('device_to_host:') for key in ctx.transfers))
                value,descent,count=self.finalize.read_control(ctx,packet,parameters)
                self.assertEqual(count,valid)
                self.assertEqual(descent.dtype,np.float32)
                np.testing.assert_allclose(value,expected[0],atol=2e-5,rtol=2e-5)
                np.testing.assert_allclose(descent,expected[1],atol=3e-5,rtol=3e-4)
                transfer=ctx.transfers['device_to_host:control_mattes_final']
                self.assertEqual(transfer['calls'],1)
                self.assertEqual(transfer['bytes'],(parameters+3)*4)

    def test_packed_input_exact_integer_count_and_visible_invalid_domains(self):
        from fast_ants.core.results import NoValidSamplesError
        ctx=self.context_type()
        bins,parameters=5,1
        raw=np.ones((bins,bins,parameters+1),np.float32)
        raw[:,:,1]=.025
        valid=2**24+1
        counts=np.zeros(bins+1,np.uint32);counts[2]=valid;counts[-1]=valid
        packed=np.r_[raw.ravel().view(np.uint32),counts,np.uint32(0)].astype(np.uint32)
        packet=self.finalize.finalize_packed_device(ctx,ctx.upload(packed,dtype=np.uint32),
                                                   bins,parameters,.1,bad_blocks=1)
        _,_,count=self.finalize.read_control(ctx,packet,parameters)
        self.assertEqual(count,valid)
        for kind in ('count_zero','mass_zero','negative','nonfinite_pdf','nonfinite_derivative','flag'):
            h=raw.copy();n=counts.copy();bad=np.zeros(1,np.uint32)
            if kind=='count_zero':n[-1]=0
            elif kind=='mass_zero':h[:,:,0]=0
            elif kind=='negative':h[0,0,0]=-1
            elif kind=='nonfinite_pdf':h[0,0,0]=np.nan
            elif kind=='nonfinite_derivative':h[-1,-1,1]=np.inf
            else:bad[0]=1
            p=self.finalize.finalize_device(ctx,ctx.upload(h),ctx.upload(n,dtype=np.uint32),.1,
                                            badflags=ctx.upload(bad,dtype=np.uint32))
            error=NoValidSamplesError if kind in ('count_zero','mass_zero') else ValueError
            with self.subTest(kind=kind),self.assertRaises(error):
                self.finalize.read_control(ctx,p,parameters)
        for width in (0.,-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):
                self.finalize.finalize_device(ctx,ctx.upload(raw),ctx.upload(counts,dtype=np.uint32),width)


if __name__=='__main__':unittest.main()
