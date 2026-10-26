"""Gaussian transfer/edge contracts against an analytic convolution oracle."""
import math
import unittest
import numpy as np
from numba import get_num_threads,set_num_threads
from fast_ants.backends.cpu.fp32.gaussian_sections import coefficients32, section_lines32
from fast_ants.backends.cpu.fp32.gaussian_sections import smooth_sections32
from fast_ants.core.types import Image, Grid


def analytic_filter(lines,sigma):
    """Damped-sinusoid impulse response with constant extension, in double."""
    radius=int(math.ceil(20*sigma+64));k=np.arange(radius+1,dtype=np.float64)
    impulse=np.zeros_like(k);gain=0.;center=0.
    for a,b,w,decay in ((1.3530,1.8151,.6681,1.3932),(-.3531,.0902,2.0787,1.3732)):
        e=math.exp(-decay/sigma);c=math.cos(w/sigma);s=math.sin(w/sigma)
        impulse+=np.exp(-decay*k/sigma)*(a*np.cos(w*k/sigma)+b*np.sin(w*k/sigma))
        gain+=(a+e*(b*s-a*c))/(1-2*e*c+e*e);center+=a
    impulse/=2*gain-center
    kernel=np.r_[impulse[:0:-1],impulse]
    return np.stack([np.convolve(np.pad(row.astype(np.float64),radius,mode='edge'),kernel,mode='valid') for row in lines])


class GaussianSections32Tests(unittest.TestCase):
    def test_anisotropic_mm_sigmas_and_axis_order(self):
        matrix=np.diag([-.7,1.2,2.5,1.]);grid=Grid((5,7,9),matrix)
        data=np.random.default_rng(45).uniform(.1,.9,grid.shape_zyx).astype(np.float32)
        expected=data.copy()
        # sigma x=0, y=2, z=.5 voxels. The reference independently applies
        # the analytic row convolution in the upstream z,x,y order.
        for axis,sigma in ((0,.5),(1,2.)):
            moved=np.moveaxis(expected,axis,-1);shape=moved.shape
            result=analytic_filter(moved.reshape(-1,shape[-1]),sigma).astype(np.float32)
            expected=np.moveaxis(result.reshape(shape),-1,axis)
        actual=smooth_sections32(Image(data,grid),[0,2.4,1.25],units='mm')
        np.testing.assert_allclose(actual.data,expected,rtol=0,atol=1e-6)
        self.assertTrue(actual.grid.same_geometry(grid))

    def test_impulse_ramp_and_noise_match_the_gaussian_transfer(self):
        rng=np.random.default_rng(20261023);x=np.linspace(0,1,513,dtype=np.float32)
        impulse=np.zeros_like(x);impulse[0]=1;impulse[256]=1
        lines=np.stack((x,np.float32(.7)+np.float32(.02)*rng.normal(size=x.size).astype(np.float32),
                        np.float32(.5)+np.float32(.5)*np.sin(x*np.float32(14)),impulse))
        for sigma in (.5,1.,2.,3.,5.,8.,16.,32.,64.):
            with self.subTest(sigma=sigma):
                actual=section_lines32(lines,coefficients32(sigma))
                # Normalized image accuracy, including a discontinuous edge
                # impulse; this does not require reproducing upstream ULPs.
                tolerance=1e-6 if sigma<=8 else 2e-5
                np.testing.assert_allclose(actual,analytic_filter(lines,sigma),rtol=0,atol=tolerance)
                self.assertEqual(actual.dtype,np.float32)
                self.assertTrue(np.isfinite(actual).all())

    def test_constants_and_short_line_boundaries(self):
        for length in (1,2,3,4,37):
            lines=np.repeat(np.array([[-.75],[0],[.125],[1.25]],np.float32),length,axis=1)
            for sigma in (.5,3.,8.,64.):
                np.testing.assert_array_equal(section_lines32(lines,coefficients32(sigma)),lines)
        lines=np.array([[0,1,0],[1,1,0]],np.float32)
        np.testing.assert_allclose(section_lines32(lines,coefficients32(3)),analytic_filter(lines,3),rtol=0,atol=2e-7)

    def test_reversal_and_dc_shift_are_consistent(self):
        lines=np.random.default_rng(5).uniform(.2,.7,(7,127)).astype(np.float32)
        coeff=coefficients32(5.)
        actual=section_lines32(lines,coeff)
        np.testing.assert_allclose(section_lines32(lines[:,::-1],coeff)[:,::-1],actual,atol=1e-7,rtol=0)
        np.testing.assert_allclose(section_lines32(lines+np.float32(.25),coeff),actual+.25,atol=2e-7,rtol=0)

    def test_parallel_rows_and_readonly_inputs(self):
        lines=np.random.default_rng(7).random((37,79),dtype=np.float32);saved=lines.copy()
        lines.setflags(write=False);coeff=coefficients32(3.);threads=get_num_threads()
        try:
            set_num_threads(1);one=section_lines32(lines,coeff)
            set_num_threads(2);two=section_lines32(lines,coeff)
        finally:set_num_threads(threads)
        np.testing.assert_array_equal(one,two)
        np.testing.assert_array_equal(lines,saved)
        self.assertFalse(np.shares_memory(one,lines))

    def test_invalid_and_empty_inputs(self):
        for sigma in (0,-1,np.nan,np.inf,65):
            with self.assertRaises(ValueError):coefficients32(sigma)
        for shape in ((0,4),(3,0)):
            empty=np.empty(shape,np.float32)
            self.assertEqual(section_lines32(empty,coefficients32(1)).shape,shape)
        with self.assertRaises(ValueError):section_lines32(np.zeros((1,4)),coefficients32(1))
        with self.assertRaises(ValueError):section_lines32(np.zeros((1,4),np.float32),np.zeros((2,4)))


if __name__=='__main__':unittest.main()
