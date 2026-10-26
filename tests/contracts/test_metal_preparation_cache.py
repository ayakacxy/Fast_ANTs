"""Preparation reuse is local to an input/geometry/filter and stays bounded."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from fast_ants.backends.metal.preparation import _PreparationCache


class PreparationCacheTests(unittest.TestCase):
    def test_identity_geometry_filter_zero_sigma_and_capacity(self):
        data=np.zeros((2,3,4),np.float32)
        image=SimpleNamespace(data=data,grid=object())
        other=SimpleNamespace(data=data,grid=object())
        ops=SimpleNamespace(smooth_image=Mock(side_effect=lambda image,sigma,units:
            SimpleNamespace(data=image.data.copy(),grid=image.grid)))
        cache=_PreparationCache(2*data.nbytes)
        first=cache.smooth(ops,image,1.,'vox')
        self.assertIs(cache.smooth(ops,image,1.,'vox'),first)
        self.assertIsNot(cache.smooth(ops,other,1.,'vox'),first)
        self.assertEqual(cache.bytes,2*data.nbytes)
        self.assertIsNot(cache.smooth(ops,image,1.,'mm'),first)
        self.assertIsNot(cache.smooth(ops,image,2.,'vox'),first)
        zero=cache.smooth(ops,image,0.,'vox')
        self.assertIsNot(cache.smooth(ops,image,0.,'vox'),zero)
        self.assertEqual(cache.bytes,cache.maximum_bytes)
        self.assertEqual(cache.hits,1)
        self.assertEqual(ops.smooth_image.call_count,6)
        anisotropic=_PreparationCache(data.nbytes)
        result=anisotropic.smooth(ops,image,(.5,1.,2.),'vox')
        self.assertIs(anisotropic.smooth(ops,image,np.asarray([.5,1.,2.]),'vox'),result)
        self.assertEqual(anisotropic.hits,1)


if __name__=='__main__':unittest.main()
