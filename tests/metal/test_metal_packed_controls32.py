"""Lossless integer controls and bounded MI diagnostics across one GPU read."""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalPackedControls32Tests(unittest.TestCase):
    def test_counts_above_float_integer_range_indexed_chunks_and_invalid_flags(self):
        from fast_ants.backends.metal.runtime import MetalContext
        from fast_ants.backends.metal.metrics.mattes.histogram import merge_histogram_controls
        ctx=MetalContext()
        chunks,bins,parameters=4,16,6
        values=np.random.default_rng(52).uniform(0,10,(chunks,bins,bins,parameters+1)).astype(np.float32)
        counts=np.full((chunks,bins+1),2**24+1,np.uint32)
        data=ctx.upload(values);numbers=ctx.upload(counts,dtype=np.uint32)
        before=ctx.kernels.copy()
        joint,got,pdf,valid=merge_histogram_controls(ctx,data,numbers,bins,parameters,chunks)
        expected=values.astype(np.float64).sum(axis=0)
        np.testing.assert_allclose(joint,expected[:,:,0],rtol=2e-7,atol=2e-6)
        np.testing.assert_allclose(pdf,expected[:,:,1:],rtol=2e-7,atol=2e-6)
        np.testing.assert_array_equal(got,counts.sum(axis=0,dtype=np.int64)[:-1])
        self.assertEqual(valid,int(counts[:,-1].sum(dtype=np.int64)))
        self.assertEqual(sum((ctx.kernels-before).values()),1)
        self.assertEqual(ctx.transfers['device_to_host:control_mattes_packed']['calls'],1)
        # The allocated fourth chunk must not contribute to indexed samples.
        joint,got,pdf,valid=merge_histogram_controls(ctx,data,numbers,bins,parameters,chunks,
                                                   sample_count=ctx.meta(5000))
        np.testing.assert_allclose(joint,values[:3].astype(np.float64).sum(axis=0)[:,:,0],rtol=2e-7,atol=2e-6)
        np.testing.assert_array_equal(got,np.full(bins,3*(2**24+1),np.int64))
        self.assertEqual(valid,3*(2**24+1))
        flags=np.zeros((chunks,bins,parameters+1),np.uint32);flags[-1,-1,-1]=1
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            merge_histogram_controls(ctx,data,numbers,bins,parameters,chunks,
                                     badflags=ctx.upload(flags,dtype=np.uint32))
        joint,got,pdf,valid=merge_histogram_controls(ctx,data,numbers,bins,parameters,chunks,
                                                   sample_count=ctx.meta(0))
        self.assertEqual(valid,0)
        np.testing.assert_array_equal(joint,0);np.testing.assert_array_equal(got,0)
        np.testing.assert_array_equal(pdf,0)


if __name__=='__main__':unittest.main()
