"""Focused contracts for exact-grid field samples in FP64 image pulls."""

import unittest

import numpy as np

from fast_ants import AffineTransform, DisplacementField, Grid, Image, apply_image
from fast_ants.backends.cpu.fp64 import pull as pull64
from fast_ants.backends.cpu.fp64.sampling import _pull_parameters
from fast_ants.core.transforms import CompositeTransform


class DirectFieldPull64Tests(unittest.TestCase):
    def reference(self, image, grid, transform, nearest=False, default=-7):
        """Run the existing generic physical/interpolated-field path explicitly."""
        output=np.empty(grid.shape_zyx,image.data.dtype if nearest else np.float64)
        finite=np.empty(grid.shape_zyx[:2],np.bool_)
        data=image.data if nearest else np.asarray(image.data,np.float64)
        pull64._pull_serial(data,output,grid.index_to_lps,
            image.grid.lps_to_index[:3,:3],image.grid.origin,
            *_pull_parameters(transform),default,nearest,finite,False)
        self.assertTrue(finite.all())
        return output

    def test_direct_samples_match_analytic_ramp_with_oblique_geometry(self):
        angle=.23
        matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[17,-33,9]
        # Exercise both the small serial and production-size parallel dispatch.
        for shape in ((5,7,9),(17,29,37)):
            grid=Grid(shape,matrix)
            z,y,x=np.indices(shape,dtype=np.float64)
            image=Image(1+2*x+3*y+5*z,grid)
            field=DisplacementField(np.stack((.12*np.sin(y),.07*np.cos(x),-.08*np.sin(z))),grid)
            affine=AffineTransform.centered(np.eye(3),[.13,-.07,.03],grid.center)
            points=np.stack((x,y,z),axis=-1)@matrix[:3,:3].T+matrix[:3,3]
            for transform in (field,CompositeTransform((field,affine))):
                with self.subTest(shape=shape,affine=transform is not field):
                    self.assertTrue(pull64._field_matches_output_grid(grid,_pull_parameters(transform)))
                    mapped=points+np.moveaxis(field.data,0,-1)
                    if transform is not field:
                        mapped=mapped+affine.translation
                    indices=(mapped-grid.origin)@grid.lps_to_index[:3,:3].T
                    valid=np.all(indices>=-.5,axis=-1)&np.all(indices<np.asarray(grid.shape_xyz)-.5,axis=-1)
                    clamped=np.clip(indices,0,np.asarray(grid.shape_xyz)-1)
                    expected=np.where(valid,1+2*clamped[...,0]+3*clamped[...,1]+5*clamped[...,2],-7)
                    actual=apply_image(image,grid,transform,default_value=-7).data
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
                    np.testing.assert_allclose(actual,self.reference(image,grid,transform),rtol=0,atol=2e-12)

    def test_near_grid_and_affine_first_keep_generic_sampling_and_edges(self):
        grid=Grid((3,4,7),np.eye(4))
        z,y,x=np.indices(grid.shape_zyx,dtype=np.float64)
        image=Image((1+2*x+3*y+5*z).astype(np.uint16),grid)
        values=np.stack((.12*x,.03*y,-.04*z))
        matrix=np.eye(4);matrix[0,3]=5e-7
        nearby=Grid(grid.shape_zyx,matrix)
        self.assertTrue(grid.same_geometry(nearby))
        for field_grid in (nearby,grid):
            field=DisplacementField(values,field_grid)
            transform=field if field_grid is nearby else CompositeTransform((
                AffineTransform.centered(np.eye(3),[.25,0,0],[0,0,0]),field))
            self.assertFalse(pull64._field_matches_output_grid(grid,_pull_parameters(transform)))
            for nearest in (False,True):
                interpolation='NearestNeighbor' if nearest else 'Linear'
                actual=apply_image(image,grid,transform,interpolation,default_value=0).data
                np.testing.assert_array_equal(actual,self.reference(image,grid,transform,nearest,0))

        # Exact-grid direct samples retain inclusive lower/exclusive upper
        # half-voxel support and nearest-neighbour tie handling in the image.
        for shift in (-.5,np.nextafter(-.5,-np.inf),.5):
            values=np.zeros((3,*grid.shape_zyx));values[0]=shift
            field=DisplacementField(values,grid)
            self.assertTrue(pull64._field_matches_output_grid(grid,_pull_parameters(field)))
            for nearest in (False,True):
                actual=apply_image(image,grid,field,'NearestNeighbor' if nearest else 'Linear',default_value=0).data
                np.testing.assert_array_equal(actual,self.reference(image,grid,field,nearest,0))


if __name__=='__main__':
    unittest.main()
