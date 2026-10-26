"""Float1 bin boundary contracts, with already quantized image samples."""

import unittest
import numpy as np
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import sample_bin_fractions32, histogram_block32


class Float1BinTests(unittest.TestCase):
    def test_rounded_pixel_boundaries_and_ranges(self):
        for low, high in ((0, 1), (-3, 8), (4096, 4128), (-8192, -8160)):
            bins=32
            low,high=np.float32(low),np.float32(high)
            width=np.float32(np.float32(high-low)/np.float32(bins-4))
            origin=np.float32(np.float32(low/width)-np.float32(2))
            edges=np.asarray(low+np.arange(bins-3,dtype=np.float32)*width,np.float32)
            values=np.concatenate((edges,np.nextafter(edges,np.float32(-np.inf)),
                                   np.nextafter(edges,np.float32(np.inf))))
            term=np.subtract(np.divide(values,width,dtype=np.float32),origin,dtype=np.float32)
            expected=np.clip(np.trunc(term),2,bins-3).astype(np.int32)
            indices,fractions=sample_bin_fractions32(values,width,origin,bins)
            np.testing.assert_array_equal(indices,expected)
            np.testing.assert_array_equal(fractions.view(np.uint32),
                np.subtract(term,expected.astype(np.float32),dtype=np.float32).view(np.uint32))

    def test_histogram_with_explicit_bins_matches_uncached_terms(self):
        fixed=np.linspace(-3,8,101,dtype=np.float32)
        moving=fixed[::-1].copy(); bins=32
        low,high=fixed[0],fixed[-1]
        width=np.float32((high-low)/np.float32(bins-4))
        origin=np.float32(low/width-np.float32(2))
        indices,fractions=sample_bin_fractions32(moving,width,origin,bins)
        derivative=np.ones((fixed.size,2),np.float32)
        original=histogram_block32(fixed,moving,derivative,bins,low,high,low,high)
        cached=histogram_block32(fixed,moving,derivative,bins,low,high,low,high,indices,fractions)
        for a,b in zip(original,cached):
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
