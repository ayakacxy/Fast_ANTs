"""Parallel evaluation must not perturb inversion control reductions."""
import unittest
import numpy as np
from numba import get_num_threads,set_num_threads
from fast_ants.backends.cpu.fp32.norms import norm32, maximum_vector32
from fast_ants.backends.cpu.fp32.kernels import norm32 as reference, maximum_vector32 as maximum_reference


class ParallelNormTests(unittest.TestCase):
    def test_parallel_values_and_scalar_controls_preserve_order(self):
        field=np.random.default_rng(12).normal(size=(3,32,35,33)).astype(np.float32)
        field[:,::5,::7,::3]=np.float32(1e-20)
        spacing=np.array([.7,1.4,2.5],np.float32)
        old_threads=get_num_threads();expected=reference(field,spacing)
        field.setflags(write=False)
        try:
            for threads in (1,2):
                set_num_threads(threads);actual=norm32(field,spacing)
                np.testing.assert_array_equal(actual[0].view(np.uint32),expected[0].view(np.uint32))
                np.testing.assert_array_equal(np.asarray(actual[1:],np.float32).view(np.uint32),
                                              np.asarray(expected[1:],np.float32).view(np.uint32))
        finally:set_num_threads(old_threads)
        self.assertEqual(actual[0].dtype,np.float32)
        self.assertFalse(np.shares_memory(actual[0],field))

    def test_small_and_strided_fields(self):
        for field in (np.zeros((3,4,5,6),np.float32),
                      np.random.default_rng(8).normal(size=(3,64,35,33)).astype(np.float32)[:,::2]):
            spacing=np.array([1,2,3],np.float32)
            expected=reference(field,spacing);actual=norm32(field,spacing)
            np.testing.assert_array_equal(actual[0],expected[0])
            self.assertEqual(actual[1:],expected[1:])

    def test_maximum_vector_preserves_first_tie_zero_underflow_and_strides(self):
        shape=(17,29,37)
        spacing=np.array([1,2,4],np.float32)
        tied=np.zeros((3,*shape),np.float32)
        tied[:,0,0,1]=[1,0,0]
        tied[:,0,0,5]=[-1,0,0]
        tied[:,0,1,0]=[0,2,0]
        tied[:,3,0,2]=[0,0,4]
        zeros=np.zeros_like(tied)
        zeros[:,0,0,0]=[-0.,0.,-0.]
        underflow=np.full_like(tied,np.float32(1e-30))
        underflow[:,0,0,0]=[-1e-30,2e-30,-3e-30]
        underflow[:,-1,-1,-1]=[4e-30,8e-30,16e-30]
        strided=np.random.default_rng(27).normal(size=(3,34,29,37)).astype(np.float32)[:,::2]
        small=tied[:,:2,:3,:7]
        threads=get_num_threads()
        try:
            for field in (tied,zeros,underflow,strided,small):
                expected=maximum_reference(np.ascontiguousarray(field),spacing)
                before=field.copy()
                field.setflags(write=False)
                for workers in (1,2):
                    with self.subTest(shape=field.shape,workers=workers):
                        set_num_threads(workers)
                        actual=maximum_vector32(field,spacing)
                        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
                        self.assertEqual(actual.dtype,np.float32)
                        self.assertFalse(np.shares_memory(actual,field))
                np.testing.assert_array_equal(field.view(np.uint32),before.view(np.uint32))
        finally:
            set_num_threads(threads)


if __name__=='__main__':unittest.main()
