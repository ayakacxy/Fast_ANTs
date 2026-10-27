"""Opt-in CUDA image kernels against the existing CPU FP32 definitions.

FAST_ANTS_TEST_CUDA=1 python -m unittest tests.cuda.test_cuda_images32
"""

import os
import unittest

import numpy as np

from fast_ants.core.fields import DisplacementField
from fast_ants.core.transforms import AffineTransform
from fast_ants.core.types import Grid, Image


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA') == '1',
                     'CUDA device tests require explicit opt-in')
class CUDAImages32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.cuda.images import CUDAImageOps, warp
        from fast_ants.backends.cuda.runtime import CUDAContext, cp
        from fast_ants.backends.cuda.types import CUDAImage, CUDAField
        from fast_ants.backends.cpu.fp32 import images
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise unittest.SkipTest('An accessible NVIDIA GPU is required')
        cls.context_type,cls.cp=CUDAContext,cp
        cls.image_type,cls.field_type=CUDAImage,CUDAField
        cls.ops_type,cls.warp,cls.cpu=CUDAImageOps,staticmethod(warp),images

    def setUp(self):
        self.ctx=self.context_type()
        self.ops=self.ops_type(self.ctx)

    def assert_same_bits(self,actual,expected):
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

    def test_device_contracts_upload_zero_and_window(self):
        grid=Grid((3,3,3),np.eye(4))
        image=Image(np.linspace(-3,7,27).reshape(grid.shape_zyx),grid,{'test':'source'})
        gpu=self.ops.image(image)
        self.assertIs(self.ops.image(gpu).data,gpu.data)
        self.assertEqual(gpu.metadata,image.metadata)
        self.assert_same_bits(self.ops.download_image(gpu).data,self.cpu.image32(image).data)
        self.assert_same_bits(self.ctx.download(self.ops.zero(grid).data,'test_output'),
                              self.cpu.zero32(grid).data)
        for values in (image.data,np.full(grid.shape_zyx,5.)):
            source=Image(values,grid)
            actual=self.ops.download_image(self.ops.image(source,normalize=True)).data
            self.assert_same_bits(actual,self.cpu.image32(source,normalize=True).data)
        for values in (image.data,gpu.data.astype(self.cp.float64),gpu.data.transpose(2,1,0)):
            with self.assertRaises(ValueError):self.image_type(values,grid)
        with self.assertRaises(ValueError):self.field_type(gpu.data,grid)
        with self.assertRaises(TypeError):self.ops.pull(image,grid)

    def test_fractional_affine_and_lps_field_match_cpu(self):
        rng=np.random.default_rng(72)
        geometry=np.eye(4);geometry[:3,:3]=np.diag([.7,1.3,2.1]);geometry[:3,3]=[-20.,13.,-8.]
        grid=Grid((7,9,11),geometry)
        image=Image(rng.normal(size=grid.shape_zyx).astype(np.float32),grid)
        values=rng.uniform(-.3,.3,size=(3,*grid.shape_zyx)).astype(np.float32)
        field=DisplacementField(values,grid)
        gpu=self.ops.image(image)
        gpu_field=self.field_type(self.ctx.upload(values,'test_input'),grid)
        linear=AffineTransform.centered([[1.02,.015,-.01],[-.02,.98,.025],[.01,-.02,1.01]],
                                         [-.35,.125,.3],grid.center)
        saved=image.data.copy(),values.copy()
        for nearest in (False,True):
            with self.subTest(nearest=nearest):
                expected=self.cpu.pull_image32(image,grid,field,linear,nearest).data
                before={k:dict(v) for k,v in self.ctx.transfers.items() if k.startswith('device_to_host:')}
                result=self.ops.pull(gpu,grid,gpu_field,linear,nearest)
                self.assertEqual({k:v for k,v in self.ctx.transfers.items() if k.startswith('device_to_host:')},
                                 before)
                self.assert_same_bits(self.ops.download_image(result).data,expected)
        np.testing.assert_array_equal(image.data,saved[0])
        np.testing.assert_array_equal(values,saved[1])

    def test_oblique_anisotropic_large_origin_pull_and_gradient(self):
        rng=np.random.default_rng(73);angle=.23
        direction=np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],[0,0,1]])
        geometry=np.eye(4);geometry[:3,:3]=direction@np.diag([.7,1.3,2.1])
        geometry[:3,3]=[2**28+.125,-2**27+.25,2**26-.375]
        grid=Grid((9,11,13),geometry)
        image=Image(np.float32(1e5)+rng.normal(size=grid.shape_zyx).astype(np.float32),grid)
        target=geometry.copy();target[:3,3]+=[.17,-.31,.23]
        output_grid=Grid(grid.shape_zyx,target)
        gpu=self.ops.image(image)
        result=self.ops.pull(gpu,output_grid)
        self.assert_same_bits(self.ops.download_image(result).data,
                              self.cpu.pull_image32(image,output_grid).data)
        gradient=self.ops.gradient(gpu)
        self.assert_same_bits(self.ctx.download(gradient,'test_output'),self.cpu.gradient_image32(image))

    def test_half_voxel_support_nearest_ties_and_clamped_edges(self):
        from fast_ants.backends.cpu.fp32.dense_kernels import warp32
        positions=np.array([-.5001,-.5,-.25,0,.4999,.5,1.5,3.25,3.4999,3.5],np.float32)
        data=np.array([[[10,20,30,40]]],np.float32)
        field=np.zeros((3,1,1,positions.size),np.float32);field[0,0,0]=positions
        linear=np.zeros((3,3),np.float32);offset=np.zeros(3,np.float32);fl=np.eye(3,dtype=np.float32)
        gpu_data=self.ctx.upload(data,'test_input');gpu_field=self.ctx.upload(field,'test_input')
        for nearest in (False,True):
            expected=warp32(data,field,linear,offset,fl,nearest)
            actual=self.ctx.download(self.warp(self.ctx,gpu_data,gpu_field,linear,offset,fl,nearest),'test_output')
            self.assert_same_bits(actual,expected)
            if nearest:np.testing.assert_array_equal(actual.ravel(),[0,10,10,10,10,20,30,40,40,0])
        # Adding 0.5 to this valid singleton coordinate rounds to one in FP32.
        singleton=self.ctx.upload(np.array([[[7]]],np.float32),'test_input')
        edge=np.zeros((3,1,1,1),np.float32);edge[0,0,0,0]=np.nextafter(np.float32(.5),np.float32(-np.inf))
        actual=self.warp(self.ctx,singleton,self.ctx.upload(edge,'test_input'),linear,offset,fl,True)
        self.assertEqual(self.ctx.download(actual,'test_output').item(),7.)


if __name__ == '__main__':
    unittest.main()
