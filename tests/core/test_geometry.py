import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fast_ants import AffineTransform, CompositeTransform, Grid, Image, load_image, save_image


class GeometryTests(unittest.TestCase):
    def test_explicit_itk_scaled_integer_io_preserves_its_fp32_boundary(self):
        raw = np.array([0, 1, 31, 1023, 2177, 32760], dtype=np.int16).reshape(1, 2, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"scaled-integer.nii.gz"
            image = nib.Nifti1Image(raw, np.eye(4))
            image.header.set_slope_inter(1.835897445678711, .125)
            nib.save(image, path)
            header = nib.load(path)
            nifti = load_image(path)
            itk = load_image(path, scaling_policy="itk")
            expected = (raw.astype(np.float32).astype(np.float64)*header.dataobj.slope+
                        header.dataobj.inter).astype(np.float32).astype(np.float64)
            np.testing.assert_array_equal(itk.data, expected.transpose(2, 1, 0))
            np.testing.assert_array_equal(nifti.data, header.get_fdata().transpose(2, 1, 0))
            self.assertEqual(itk.data.dtype, np.float64)
            self.assertGreater(np.max(abs(itk.data-nifti.data)), 1e-5)
            self.assertEqual(itk.metadata["scaling_policy"], "itk")
            with self.assertRaises(ValueError):
                load_image(path, scaling_policy="unknown")

    def test_itk_scanner_geometry_uses_pixdim_and_normalized_sform(self):
        angle=.21
        affine=np.eye(4)
        affine[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([1.2,1.4,1.6])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'scanner.nii.gz'
            image=nib.Nifti1Image(np.zeros((5,6,7)),affine)
            image.set_sform(affine,1); image.set_qform(affine,1)
            nib.save(image,path)
            actual=load_image(path,geometry_policy='itk_scanner')
            header=nib.load(path)
            expected=header.affine.copy()
            expected[:3,:3]/=np.sqrt(np.sum(expected[:3,:3]**2,axis=0))
            expected[:3,:3]*=header.header.get_zooms()
            np.testing.assert_allclose(actual.grid.affine_ras,expected,atol=1e-15)
            np.testing.assert_allclose(actual.grid.spacing_xyz,header.header.get_zooms(),atol=1e-15)
            self.assertEqual(actual.metadata['geometry_policy'],'itk_scanner')
            image.set_sform(affine,2); nib.save(image,path)
            with self.assertRaises(ValueError):
                load_image(path,geometry_policy='itk_scanner')

    def setUp(self):
        angle = .37
        self.matrix = np.array([[np.cos(angle)*1.2, -np.sin(angle)*2, 0, 125],
                                [np.sin(angle)*1.2, np.cos(angle)*2, 0, -250],
                                [0, 0, -3.5, 1000], [0, 0, 0, 1]])
        self.grid = Grid((7, 9, 11), self.matrix)

    def test_oblique_anisotropic_roundtrip_and_center(self):
        index = np.array([[0, 0, 0], [10, 8, 6], [3.25, 2.5, 1.125]])
        physical = index @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        np.testing.assert_allclose(self.grid.index_to_physical(index), physical, atol=1e-12)
        np.testing.assert_allclose(self.grid.physical_to_index(physical), index, atol=1e-12)
        np.testing.assert_allclose(self.grid.center, self.grid.index_to_physical([5, 4, 3]))
        np.testing.assert_allclose(self.grid.spacing_xyz, [1.2, 2, 3.5])

    def test_gradient_covector_transform(self):
        physical_gradient = np.array([.3, -.7, 1.2])
        index_gradient = self.matrix[:3, :3].T @ physical_gradient
        np.testing.assert_allclose(self.grid.gradient_to_physical(index_gradient), physical_gradient, atol=1e-14)

    def test_nifti_axis_values_and_ras_lps_roundtrip(self):
        data = np.arange(7*9*11, dtype=np.float64).reshape(7, 9, 11)
        image = Image(data, self.grid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.nii.gz"
            save_image(image, path)
            raw = nib.load(path)
            self.assertEqual(raw.shape, (11, 9, 7))
            self.assertEqual(raw.dataobj[3, 2, 1], data[1, 2, 3])
            np.testing.assert_allclose(raw.affine[:2], -self.matrix[:2], atol=1e-5)
            recovered = load_image(path)
            np.testing.assert_array_equal(recovered.data, data)
            self.assertTrue(self.grid.same_geometry(recovered.grid, atol=1e-5))

    def test_sform_priority_and_spatial_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit.nii"
            affine = np.diag([.001, .002, .003, 1])
            image = nib.Nifti1Image(np.zeros((3, 4, 5)), affine)
            image.set_sform(affine, 2)
            image.set_qform(np.diag([.004, .004, .004, 1]), 1)
            image.header.set_xyzt_units("meter")
            nib.save(image, path)
            recovered = load_image(path)
            np.testing.assert_allclose(recovered.grid.spacing_xyz, [1, 2, 3], rtol=1e-7)
            self.assertEqual(recovered.metadata["affine_source"], "sform")
            self.assertFalse(recovered.metadata["qform_sform_agree"])

    def test_invalid_and_sheared_grid_fail(self):
        for shape, matrix in [((0, 2, 3), np.eye(4)), ((2, 3, 4), np.zeros((4, 4))),
                              ((2, 3, 4), np.array([[1, .1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))]:
            with self.assertRaises(ValueError):
                Grid(shape, matrix)
        with self.assertRaises(ValueError):
            Image(np.full(self.grid.shape_zyx, np.nan), self.grid)

    def test_centered_affine_and_noncommuting_composite(self):
        center = np.array([10, -20, 30])
        linear = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotation = AffineTransform.centered(linear, [0, 0, 0], center)
        shift = AffineTransform.centered(np.eye(3), [1, 2, 3], [0, 0, 0])
        points = np.array([[12, -25, 33], [9, -18, 27]])
        expected = (points - center) @ linear.T + center + [1, 2, 3]
        composite = CompositeTransform((rotation, shift))
        np.testing.assert_allclose(composite.map_points(points), expected)
        np.testing.assert_allclose(composite.inverse().map_points(expected), points)
        self.assertFalse(np.allclose(CompositeTransform((shift, rotation)).map_points(points), expected))

    def test_three_real_mri_geometry_and_axes(self):
        paths = sorted(Path("validation/data/public").glob("*.nii.gz"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            image = load_image(path, dtype=np.float32)
            original = nib.load(path)
            np.testing.assert_array_equal(image.data[17, 23, 31], original.dataobj[31, 23, 17])
            point = image.grid.index_to_physical([31, 23, 17])
            expected = nib.affines.apply_affine(original.affine, [31, 23, 17]) * [-1, -1, 1]
            np.testing.assert_allclose(point, expected, atol=1e-12)
            self.assertTrue(image.data.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
