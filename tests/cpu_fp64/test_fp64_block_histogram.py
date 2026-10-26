"""Fixed-state Mattes contracts for deterministic private-block reductions."""

import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from fast_ants.backends.cpu.fp64.metrics import linear_fused as fused


class FP64BlockHistogramTests(unittest.TestCase):
    def fixture(self, count=70003, parameters=12):
        rng=np.random.default_rng(741)
        # Occupancy imbalance represents the large background bin in brain MRI.
        fixed=rng.uniform(0,1,count)
        fixed[:int(.85*count)]=0.
        offsets,ordered=fused.fixed_bin_order(fixed,32,0.,1.)
        values=rng.uniform(-.1,1.1,count)
        if count>=4:values[:4]=[0.,1.,np.nextafter(0.,-np.inf),np.nextafter(1.,np.inf)]
        states=np.ones(count,np.int8);states[7::43]=0
        gradients=rng.normal(size=(count,3));gradients[::5]=0.
        points=rng.normal(size=(count,3))*10
        center=np.array([.2,-.3,.4])
        linear=rng.normal(size=(parameters,3,3))*.1
        translation=rng.normal(size=(parameters,3))
        return offsets,ordered,values,states,gradients,points,center,linear,translation

    def test_values_and_derivatives_preserve_fixed_state_objective(self):
        threads=get_num_threads()
        try:
            for parameters in (6,12):
                data=self.fixture(parameters=parameters)
                value_args=(*data[:4],32,0.,1.)
                derivative_args=(*data,32,0.,1.)
                value_ref=fused._value_histogram_serial(*value_args)
                derivative_ref=fused._derivative_histogram_serial(*derivative_args)
                previous=None
                for workers in (1,2):
                    set_num_threads(workers)
                    value=fused._value_histogram_blocks(*value_args)
                    derivative=fused._derivative_histogram_blocks(*derivative_args)
                    self.assertEqual(value[-1],value_ref[-1])
                    self.assertEqual(derivative[-2:],derivative_ref[-2:])
                    for actual,reference in zip(derivative[:3],derivative_ref[:3]):
                        np.testing.assert_allclose(actual,reference,rtol=2e-12,atol=2e-10)
                    np.testing.assert_allclose(value[0],value_ref[0],rtol=2e-12,atol=2e-10)
                    np.testing.assert_array_equal(value[1],value_ref[1])
                    result=fused._mattes_finish(*(x.copy() for x in derivative[:3]),
                        derivative[3],32,0.,1.)
                    reference=fused._mattes_finish(*(x.copy() for x in derivative_ref[:3]),
                        derivative_ref[3],32,0.,1.)
                    self.assertLess(abs(result[0]-reference[0]),2e-12)
                    np.testing.assert_allclose(result[1],reference[1],rtol=2e-11,atol=2e-12)
                    if previous is not None:
                        for actual,prior in zip(derivative[:3],previous[:3]):
                            np.testing.assert_array_equal(actual,prior)
                    previous=derivative
        finally:
            set_num_threads(threads)

    def test_empty_samples_and_invalid_derivatives_are_not_hidden(self):
        empty=self.fixture(count=0,parameters=6)
        value=fused._value_histogram_blocks(*empty[:4],32,0.,1.)
        derivative=fused._derivative_histogram_blocks(*empty,32,0.,1.)
        self.assertEqual(value[-1],0)
        self.assertEqual(derivative[-2:],(0,False))
        np.testing.assert_array_equal(derivative[2],0.)
        data=list(self.fixture(count=4,parameters=6))
        # Invalid parameter derivatives must still be surfaced for geometrically
        # valid samples even when their intensity lies outside the PDF range.
        data[2][:]=-1.
        data[4][:]=0.
        data[5][:]=1e200
        data[7][:]=1e200
        actual=fused._derivative_histogram_blocks(*data,32,0.,1.)
        reference=fused._derivative_histogram_serial(*data,32,0.,1.)
        self.assertEqual(actual[-2:],reference[-2:])
        self.assertEqual(actual[-2:],(0,True))


if __name__=='__main__':
    unittest.main()
