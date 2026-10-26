"""Device contracts for the FP32 difference-form Gaussian row filter."""
import os
import unittest
import numpy as np
from fast_ants.backends.cpu.fp32.gaussian_sections import coefficients32, section_lines32 as cpu_filter


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit Metal device test opt-in required')
class MetalGaussianSections32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.runtime import MetalContext, mx
        from fast_ants.backends.metal.gaussian_sections import section_lines32
        cls.context_type=MetalContext;cls.mx=mx;cls.filter=staticmethod(section_lines32)
        if not mx.metal.is_available():raise unittest.SkipTest('Accessible Apple GPU required')

    def test_cpu_device_transfer_and_boundaries(self):
        rng=np.random.default_rng(23)
        for length in (1,3,17,513):
            lines=rng.uniform(-.5,1,(7,length)).astype(np.float32)
            for sigma in (.5,1.,3.,8.,16.,64.):
                with self.subTest(length=length,sigma=sigma):
                    ctx=self.context_type(());data=ctx.upload(lines,'test_input');coeff=coefficients32(sigma)
                    actual=self.filter(ctx,data,coeff)
                    self.assertEqual(ctx.kernels['gaussian_sections32'],1)
                    self.assertFalse(any(k.startswith('device_to_host') for k in ctx.transfers))
                    np.testing.assert_allclose(ctx.download(actual),cpu_filter(lines,coeff),rtol=1e-6,atol=1e-6)
                    np.testing.assert_array_equal(ctx.download(data),lines)

    def test_constant_response_and_empty_rows(self):
        ctx=self.context_type(())
        lines=np.repeat(np.array([[-.75],[0],[.125],[1.25]],np.float32),37,axis=1)
        gpu=ctx.upload(lines)
        for sigma in (1.,3.,8.,64.):np.testing.assert_array_equal(ctx.download(self.filter(ctx,gpu,coefficients32(sigma))),lines)
        empty=ctx.upload(np.empty((0,4),np.float32));before=ctx.kernels.copy()
        self.assertIs(self.filter(ctx,empty,coefficients32(1)),empty)
        self.assertEqual(ctx.kernels,before)

    def test_rejects_silent_cast_or_malformed_tables(self):
        ctx=self.context_type(());data=ctx.upload(np.ones((3,4),np.float32));before=ctx.kernels.copy()
        for table in (np.zeros((2,4),np.float64),np.zeros((2,7),np.float32),np.full((2,4),np.nan,np.float32)):
            with self.assertRaises(ValueError):self.filter(ctx,data,table)
        with self.assertRaises(ValueError):self.filter(ctx,np.ones((3,4),np.float32),coefficients32(1))
        self.assertEqual(ctx.kernels,before)


if __name__=='__main__':unittest.main()
