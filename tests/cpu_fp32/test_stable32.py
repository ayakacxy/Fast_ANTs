import unittest
import numpy as np

from fast_ants import Grid, Image, register32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import local_sample32, local_affine32, local_gradient32
from fast_ants.backends.cpu.fp64.linear import initialization_point
from fast_ants.backends.cpu.fp32.linear import initialization_point32


class Stable32Tests(unittest.TestCase):
    def test_local_interpolation_keeps_small_offsets(self):
        data=np.broadcast_to(np.arange(64,dtype=np.float32),(4,4,64)).copy()
        base=np.array([32,1,1],np.int32);offset=np.array([1e-5,0,0],np.float32)
        residual=local_sample32(data,base,offset,np.float32(32))
        self.assertLess(abs(float(residual)-1e-5),1e-11)
        naive=np.float32(np.float32(32)+np.float32(1e-5))-np.float32(32)
        self.assertGreater(abs(float(naive)-1e-5),1e-7)

    def test_centered_moments_and_native_small_control(self):
        grid=Grid((12,14,16),np.diag([.8,1.4,2.,1.]))
        z,y,x=np.indices(grid.shape_zyx)
        image=Image(np.exp(-((x-7.7)**2+(y-6.4)**2+(z-5.2)**2)/12).astype(np.float32),grid)
        self.assertLess(float(np.linalg.norm(initialization_point32(image,1)-initialization_point(image,1))),1e-6)

    def test_integer_voxel_origins_do_not_quantize_subvoxel_motion(self):
        p=np.array([[63,2,3]],np.float32)
        b,f=local_affine32(p,np.eye(3,dtype=np.int32),np.zeros((3,3),np.float32),
                          np.array([0,0,0],np.int32),np.array([1e-5,0,0],np.float32),np.zeros(3,np.float32))
        self.assertEqual(b.dtype,np.int32);self.assertEqual(f.dtype,np.float32)
        self.assertEqual(int(b[0,0]),63)
        self.assertLess(abs(float(f[0,0])-1e-5),1e-11)

    def test_public_fp32_rejects_retired_emulation_policy(self):
        image=Image(np.ones((4,4,4),np.float32),Grid((4,4,4),np.eye(4)))
        with self.assertRaisesRegex(ValueError,'software-expanded'):
            register32(image,image,linear_precision='paired')
