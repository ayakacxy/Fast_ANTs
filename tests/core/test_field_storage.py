"""Public field storage validation and ownership with large parallel scans."""
import unittest
import numpy as np
from fast_ants import Grid
from fast_ants.core.fields import DisplacementField


class FieldStorageTests(unittest.TestCase):
    def test_finite_scan_rejects_every_nonfinite_value_across_layouts(self):
        grid=Grid((19,33,65),np.eye(4))
        for dtype in (np.float32,np.float64):
            values=np.zeros((3,*grid.shape_zyx),dtype=dtype)
            values.ravel()[1]=-0.
            values.ravel()[2]=np.nextafter(dtype(0),dtype(1))
            values.ravel()[3]=np.finfo(dtype).max
            values.setflags(write=False)
            self.assertEqual(DisplacementField(values,grid).data.dtype,dtype)
            for bad in (np.nan,np.inf,-np.inf):
                for offset in (0,16383,16384,values.size-1):
                    probe=values.copy();probe.ravel()[offset]=bad
                    for view in (probe,np.asfortranarray(probe),probe[:,:,:,::-1]):
                        with self.assertRaisesRegex(ValueError,'finite'):
                            DisplacementField(view,grid)
