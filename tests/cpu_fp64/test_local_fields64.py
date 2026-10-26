"""Analytic and historical-reference checks for local FP64 inversion."""
import unittest
import numpy as np
from fast_ants import Grid, DisplacementField
from fast_ants.backends.cpu.fp64.fields import invert_field, _InversionWorkspace


class LocalInverseTests(unittest.TestCase):
    def test_origin_independence_and_packed_constant_inverse(self):
        matrix=np.eye(4)
        angle=.23
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.8,1.4,2.])
        values=np.empty((3,5,7,9),np.float64)
        values[:]=np.array([.0001,-.002,.01])[:,None,None,None]
        outputs=[]
        for origin in ([13,-25,8],[1e13,-2e13,3e13]):
            matrix[:3,3]=origin
            field=DisplacementField(values.copy(),Grid(values.shape[1:],matrix.copy()))
            for packed in (False,True):
                actual=invert_field(field,enforce_boundary=False,max_iterations=100,
                    mean_tolerance=1e-12,max_tolerance=1e-12,
                    _workspace=_InversionWorkspace(field.grid.shape_zyx,packed=packed))
                np.testing.assert_allclose(actual.field.data,-values,atol=2e-12,rtol=0)
                outputs.append(actual.field.data)
        for output in outputs[1:]:
            np.testing.assert_allclose(output,outputs[0],atol=1e-15,rtol=0)

    def test_smooth_field_agrees_with_world_path_and_preserves_input(self):
        shape=(17,29,37)
        matrix=np.diag([.8,1.4,2.,1.]);matrix[:3,3]=[13,-25,8]
        grid=Grid(shape,matrix)
        z,y,x=np.indices(shape)
        values=np.stack([.12*np.sin(y*.21),.04*np.cos(z*.17),.02*np.sin(x*.15)])
        field=DisplacementField(values,grid)
        before=field.data.copy()
        initial=DisplacementField(-.5*values,grid)
        expected=invert_field(field,initial,arithmetic='itk',max_iterations=8)
        actual=invert_field(field,initial,max_iterations=8,
                            _workspace=_InversionWorkspace(shape,packed=True))
        np.testing.assert_allclose(actual.field.data,expected.field.data,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(field.data,before)
        np.testing.assert_array_equal(initial.data,-.5*values)
        self.assertTrue(np.isfinite(actual.field.data).all())
        self.assertEqual(np.count_nonzero(actual.field.data[:,:,0,:]),0)
        self.assertLess(abs(actual.final_residual['mean_spacing_scaled_norm']-
                            expected.final_residual['mean_spacing_scaled_norm']),1e-12)
        with self.assertRaises(ValueError):
            invert_field(field,arithmetic='fp32')


if __name__=='__main__':
    unittest.main()
