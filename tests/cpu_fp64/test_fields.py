import unittest
import numpy as np

from fast_ants import Grid, DisplacementField, AffineTransform, transform_to_field
from fast_ants.backends.cpu.fp64.fields import compose_fields, invert_field, inverse_consistency, jacobian_determinant, resample_field, scale_update, zero_field


class FieldTests(unittest.TestCase):
    def grid(self):
        angle = .31
        matrix = np.eye(4)
        matrix[:3,:3] = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.8,1.4,2.])
        matrix[:3,3] = [13,-25,8]
        return Grid((15,17,19),matrix)

    def test_noncommuting_composition_and_long_chain(self):
        grid = self.grid()
        a = AffineTransform.centered(np.diag([1.01,.99,1]), [.04,-.06,0], grid.center)
        b = AffineTransform.centered(np.array([[1,.015,0],[0,1,0],[0,0,1]]), [-.03,.02,0], grid.center)
        fa, fb = transform_to_field(grid,a), transform_to_field(grid,b)
        c, valid = compose_fields(fa,fb,return_valid=True)
        z,y,x = np.indices(grid.shape_zyx)
        points = grid.index_to_physical(np.stack([x,y,z],axis=-1))
        truth = a.map_points(b.map_points(points))-points
        np.testing.assert_allclose(c.data[:,1:-1,1:-1,1:-1],np.moveaxis(truth,-1,0)[:,1:-1,1:-1,1:-1],atol=1e-13)
        self.assertGreater(np.max(np.abs(c.data-compose_fields(fb,fa).data)),1e-5)
        field = zero_field(grid)
        expected = points.copy()
        for i in range(8):
            field = compose_fields(fa,field)
            expected = a.map_points(expected)
        np.testing.assert_allclose(field.data[:,3:-3,3:-3,3:-3],np.moveaxis(expected-points,-1,0)[:,3:-3,3:-3,3:-3],atol=1e-12)

    def test_spacing_scaled_step_and_constant_resample(self):
        grid = self.grid()
        data = np.zeros((3,*grid.shape_zyx)); data[0]=.8; data[1]=1.4
        field = DisplacementField(data,grid)
        scaled, info = scale_update(field,.1)
        self.assertAlmostEqual(info['maximum_spacing_scaled_norm'],np.sqrt(2))
        self.assertAlmostEqual(np.linalg.norm(scaled.data[:,7,8,9]/grid.spacing_xyz),.1)
        from fast_ants.pipeline.pyramid import shrink_grid
        smaller = resample_field(field,shrink_grid(grid,2))
        np.testing.assert_allclose(smaller.data[:,1:-1,1:-1,1:-1],np.broadcast_to(np.array([.8,1.4,0])[:,None,None,None],smaller.data[:,1:-1,1:-1,1:-1].shape),atol=1e-14)

    def test_zero_constant_inverse_boundaries_and_stop_logic(self):
        grid = self.grid()
        zero = invert_field(zero_field(grid))
        self.assertEqual(len(zero.history),1)
        self.assertEqual(np.max(np.abs(zero.field.data)),0)
        data = np.zeros((3,*grid.shape_zyx)); data[0]=.02
        field = DisplacementField(data,grid)
        result = invert_field(field,enforce_boundary=False,max_iterations=100,max_tolerance=1e-10,mean_tolerance=1e-10)
        np.testing.assert_allclose(result.field.data[0],-.02,atol=1e-10)
        boundary = invert_field(field,max_iterations=2)
        self.assertEqual(np.max(np.abs(boundary.field.data[:,:,0,:])),0)
        # Mean threshold can stop while the maximum is still above its threshold.
        data*=0; data[0,7,8,9]=.3
        result = invert_field(DisplacementField(data,grid),max_tolerance=0,mean_tolerance=.001)
        self.assertEqual(len(result.history),1)
        self.assertEqual(result.stop_reason,'mean_error_threshold')
        self.assertGreater(result.history[0]['max_scaled_error_before_update'],0)

    def test_smooth_shear_inverse_and_jacobian(self):
        matrix = np.diag([1.,1.3,2.,1.]); matrix[:3,3]=[-9,-10,-11]
        grid = Grid((13,15,17),matrix)
        z,y,x = np.indices(grid.shape_zyx)
        data = np.zeros((3,*grid.shape_zyx)); data[0]=.1*np.sin(y*.3)
        field = DisplacementField(data,grid)
        result = invert_field(field,enforce_boundary=False,max_iterations=120,max_tolerance=1e-13,mean_tolerance=1e-13)
        np.testing.assert_allclose(result.field.data,-data,atol=2e-11,rtol=0)
        np.testing.assert_allclose(jacobian_determinant(field),1,atol=1e-14)
        for check in inverse_consistency(field,result.field).values():
            self.assertLess(check['max_l2_mm'],2e-11)

    def test_affine_jacobian_in_physical_axes_and_invalid_support(self):
        grid = self.grid()
        transform = AffineTransform.centered(np.array([[1.02,.03,0],[0,.97,.02],[.01,0,1.01]]),[0,0,0],grid.center)
        field = transform_to_field(grid,transform)
        np.testing.assert_allclose(jacobian_determinant(field),np.linalg.det(transform.matrix[:3,:3]),atol=2e-14)
        data = np.zeros_like(field.data); data[0]=500
        inverse = DisplacementField(data,grid)
        report = inverse_consistency(field,inverse)
        self.assertEqual(report['forward_after_inverse']['common_valid_points'],0)
        self.assertIsNone(report['forward_after_inverse']['max_l2_mm'])
