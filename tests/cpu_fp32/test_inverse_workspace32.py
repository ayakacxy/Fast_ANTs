"""Local-coordinate, reduction and ownership contracts for FP32 inversion."""
import unittest
import numpy as np
from numba import get_num_threads,set_num_threads

from fast_ants.backends.cpu.fp32.inverse import InverseWorkspace32, residual32_into, update32_into
from fast_ants.backends.cpu.fp32.kernels import norm32


class InverseWorkspace32Tests(unittest.TestCase):
    def test_affine_vector_field_local_sampling_norms_and_threshold_guard(self):
        shape=(17,29,37)
        z,y,x=np.indices(shape,dtype=np.float32)
        outer=np.stack((.003*x+.005*y-.002*z,
                        -.002*x+.001*y+.003*z,
                        .001*x-.004*y+.002*z)).astype(np.float32)
        current=np.empty_like(outer)
        current[:]=np.array([.000001,.13,-.09],np.float32)[:,None,None,None]
        # A sample entirely outside support contributes zero outer displacement.
        current[:,0,0,0]=[111,111,111]
        inverse=np.array([[1.1,.13,0],[-.07,.7,0],[0,0,.45]],np.float32)
        spacing=np.array([.9,1.4,2.2],np.float32)
        offset=np.einsum('ij,jzyx->izyx',inverse.astype(np.float64),current.astype(np.float64))
        coordinates=np.stack((x,y,z)).astype(np.float64)+offset
        valid=np.all(coordinates>=-.5,axis=0)&np.all(coordinates<np.array(shape[::-1])[:,None,None,None]-.5,axis=0)
        clipped=np.clip(coordinates,0,np.array(shape[::-1])[:,None,None,None]-1)
        a,b,c=clipped
        sampled=np.stack((.003*a+.005*b-.002*c,-.002*a+.001*b+.003*c,.001*a-.004*b+.002*c))
        expected=current.astype(np.float64)+np.where(valid[None],sampled,0.)
        outer.setflags(write=False);current.setflags(write=False)
        workspace=InverseWorkspace32(shape)
        buffers=tuple(id(v) for v in (workspace.residual,workspace.norms,workspace.partials,workspace.finite,workspace.output))
        old_threads=get_num_threads();previous=None
        try:
            for threads in (1,2):
                set_num_threads(threads)
                residual,norms,mean,maximum=residual32_into(outer,current,inverse,spacing,workspace)
                self.assertIs(residual,workspace.residual);self.assertIs(norms,workspace.norms)
                self.assertEqual(residual.dtype,np.float32)
                np.testing.assert_allclose(residual,expected,atol=1.5e-7,rtol=2e-7)
                np.testing.assert_array_equal(residual[:,0,0,0],current[:,0,0,0])
                expected_norms,serial_mean,serial_max=norm32(residual,spacing)
                np.testing.assert_array_equal(norms,expected_norms)
                np.testing.assert_allclose(mean,serial_mean,rtol=3e-6,atol=0)
                self.assertEqual(maximum,serial_max)
                if previous is not None:
                    np.testing.assert_array_equal(residual,previous[0])
                    self.assertEqual((mean,maximum),previous[1:])
                previous=residual.copy(),mean,maximum
                _,_,guarded_mean,guarded_max=residual32_into(outer,current,inverse,spacing,workspace,
                    store_residual=False,mean_tolerance=float(mean))
                self.assertTrue(workspace.serial_mean_guard_used)
                self.assertEqual((guarded_mean,guarded_max),(serial_mean,serial_max))
        finally:
            set_num_threads(old_threads)
        self.assertEqual(buffers,tuple(id(v) for v in (workspace.residual,workspace.norms,
                         workspace.partials,workspace.finite,workspace.output)))

    def test_update_clipping_boundaries_and_owned_inplace_output(self):
        shape=(17,29,37);rng=np.random.default_rng(307)
        current=rng.normal(0,.01,(3,*shape)).astype(np.float32)
        residual=rng.normal(0,.2,current.shape).astype(np.float32)
        norms,_,maximum=norm32(residual,np.ones(3,np.float32))
        current.setflags(write=False);residual.setflags(write=False)
        workspace=InverseWorkspace32(shape);before=current.copy()
        for boundary in (False,True):
            epsilon=np.float32(.75)
            factor=np.ones(shape,np.float32);limit=epsilon*np.float32(maximum)
            np.divide(limit,norms,out=factor,where=norms>limit)
            expected=current-residual*factor[None]*epsilon
            if boundary:
                expected[:,[0,-1],:,:]=0;expected[:,:,[0,-1],:]=0;expected[:,:,:,[0,-1]]=0
            actual=update32_into(current,residual,norms,maximum,epsilon,boundary,workspace)
            self.assertIs(actual,workspace.output)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(current,before)
        # Only current may alias output; each update reads its own voxel only.
        old=workspace.output.copy();epsilon=np.float32(.5)
        factor=np.ones(shape,np.float32);limit=epsilon*np.float32(maximum)
        np.divide(limit,norms,out=factor,where=norms>limit)
        expected=old-residual*factor[None]*epsilon
        actual=update32_into(workspace.output,residual,norms,maximum,epsilon,False,workspace)
        np.testing.assert_array_equal(actual,expected)
        with self.assertRaises(ValueError):
            update32_into(current,residual,norms,maximum,epsilon,False,workspace,output=residual)
        bad=residual.copy();bad[0,3,3,3]=np.nan
        with self.assertRaises(ValueError):
            update32_into(current,bad,norms,maximum,epsilon,False,workspace)


if __name__=='__main__':
    unittest.main()
