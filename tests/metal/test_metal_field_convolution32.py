"""Check direct-axis field smoothing against the existing transpose path."""
import os
import unittest
import numpy as np
from fast_ants.core.types import Grid
from fast_ants.core.fields import DisplacementField


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalFieldConvolutionTests(unittest.TestCase):
    def test_axes_boundaries_and_small_variance_mixing(self):
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext
        baseline=MetalOps(MetalContext(()))
        direct=MetalOps()
        self.assertIn('direct_field_convolution',direct.context.optimizations)
        for shape in ((5,7,9),(1,3,5)):
            grid=Grid(shape,np.eye(4))
            values=np.random.default_rng(26).normal(size=(3,*shape)).astype(np.float32)
            field=DisplacementField(values,grid)
            for variance in (0.,.1,3.):
                with self.subTest(shape=shape,variance=variance):
                    ref=baseline.smooth_field(baseline.field(field),variance)
                    out=direct.smooth_field(direct.field(field),variance)
                    np.testing.assert_array_equal(direct.context.download(out.data),
                                                  baseline.context.download(ref.data))
        self.assertEqual(direct.context.kernels['convolve'],0)
        self.assertEqual(direct.context.kernels['mix_boundary'],0)
        self.assertGreater(direct.context.kernels['convolve_field_axis_mix'],0)


if __name__=='__main__':
    unittest.main()
