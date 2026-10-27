"""Pixel-major CUDA Gaussian layout against the existing FP32 recurrence."""
import os
import unittest
import numpy as np

from fast_ants.core.types import Grid, Image


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA')=='1','CUDA device tests require explicit opt-in')
class CUDAGaussianSections32Tests(unittest.TestCase):
    def setUp(self):
        from fast_ants.backends.cuda.runtime import CUDAContext
        self.ctx=CUDAContext();self.rng=np.random.default_rng(20261026)

    def same(self,actual,expected):
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

    def test_short_and_partial_warp_rows_preserve_recurrence(self):
        from fast_ants.backends.cpu.fp32.gaussian_sections import coefficients32,section_lines32 as reference
        from fast_ants.backends.cuda.gaussian_sections import section_lines32
        for shape in ((1,1),(3,2),(5,3),(7,4),(37,79),(259,257)):
            values=(4096+self.rng.normal(size=shape)).astype(np.float32)
            values[0]=np.float32(-.75)
            device=self.ctx.upload(values,'test_input')
            for sigma in (.5,3.,64.):
                with self.subTest(shape=shape,sigma=sigma):
                    output=section_lines32(self.ctx,device,coefficients32(sigma))
                    self.same(self.ctx.download(output,'test_output'),reference(values,coefficients32(sigma)))
                    self.assertTrue(output.flags.c_contiguous)
                    self.assertNotEqual(output.data.ptr,device.data.ptr)
            self.same(self.ctx.download(device,'test_input_unchanged'),values)

    def test_native_axes_preserve_order_and_output_ownership(self):
        from fast_ants.backends.cuda.filtering import smooth_image
        from fast_ants.backends.cuda.types import CUDAImage
        from fast_ants.backends.cpu.fp32.filtering import smooth_image32
        for shape in ((5,7,9),(4,8,37),(9,1,7)):
            grid=Grid(shape,np.diag([.7,1.3,2.1,1.]))
            values=(4096+self.rng.normal(size=shape)).astype(np.float32)
            image=Image(values,grid);device=self.ctx.upload(values,'test_input')
            gpu=CUDAImage(device,grid,{'source':'test'})
            for axis in range(3):
                sigma=np.zeros(3);sigma[2-axis]=1.6
                if shape[axis]<4:
                    with self.assertRaises(ValueError):smooth_image(self.ctx,gpu,sigma)
                    continue
                with self.subTest(shape=shape,axis=axis):
                    output=smooth_image(self.ctx,gpu,sigma)
                    expected=smooth_image32(image,sigma).data
                    self.same(self.ctx.download(output.data,'test_output'),expected)
                    self.assertNotEqual(output.data.data.ptr,device.data.ptr)
                    smooth_image(self.ctx,gpu,sigma*2)
                    self.same(self.ctx.download(output.data,'test_retained_output'),expected)
            self.same(self.ctx.download(device,'test_input_unchanged'),values)


if __name__=='__main__':unittest.main()
