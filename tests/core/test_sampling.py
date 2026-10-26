import tempfile
import unittest
from pathlib import Path

import numpy as np

from fast_ants import AffineTransform, DisplacementField, Grid, Image, apply_image, load_affine, load_field, sample_image, save_affine, save_field


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid((5, 6, 7), np.eye(4))
        z, y, x = np.indices(self.grid.shape_zyx, dtype=np.float64)
        self.image = Image(1 + 2*x + 3*y + 5*z, self.grid)

    def test_linear_ramp_and_half_voxel_support(self):
        points = np.array([[1.25, 2.5, 3.25], [-.5, 1, 2], [-.50001, 1, 2],
                           [6.49999, 2, 3], [6.5, 2, 3], [6, 5, 4]])
        values, valid = sample_image(self.image, points, default_value=-7)
        np.testing.assert_array_equal(valid, [True, True, False, True, False, True])
        np.testing.assert_allclose(values, [27.25, 14, -7, 34, -7, 48], atol=1e-12)

    def test_nearest_ties_labels_and_outside(self):
        labels = Image(self.image.data.astype(np.uint16), self.grid)
        values, valid = sample_image(labels, [[.5, 0, 0], [-.5, 0, 0], [-.5001, 0, 0]], "NearestNeighbor")
        np.testing.assert_array_equal(values, [3, 1, 0])
        self.assertEqual(values.dtype, np.uint16)
        with self.assertRaises(ValueError):
            sample_image(labels, [[0, 0, 0]], "NearestNeighbor", default_value=-1)

    def test_identity_translation_and_singleton_dimension(self):
        np.testing.assert_array_equal(apply_image(self.image, self.grid).data, self.image.data)
        transform = AffineTransform.centered(np.eye(3), [1, 0, 0], [0, 0, 0])
        shifted = apply_image(self.image, self.grid, transform)
        np.testing.assert_array_equal(shifted.data[:, :, :-1], self.image.data[:, :, 1:])
        np.testing.assert_array_equal(shifted.data[:, :, -1], 0)
        line = Image(np.arange(6, dtype=np.float64).reshape(1, 1, 6), Grid((1, 1, 6), np.eye(4)))
        np.testing.assert_allclose(sample_image(line, [[2.25, .25, -.25]])[0], [2.25])

    def test_field_components_interpolation_and_outside_identity(self):
        z, y, x = np.indices(self.grid.shape_zyx, dtype=np.float64)
        field = DisplacementField(np.stack([.1*x, .2*y, -.3*z]), self.grid)
        points = np.array([[1.25, 2.5, 3.25], [-1, 1, 1]])
        np.testing.assert_allclose(field.map_points(points), [[1.375, 3, 2.275], [-1, 1, 1]], atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.nii.gz"
            save_field(field, path)
            recovered = load_field(path)
            np.testing.assert_array_equal(recovered.data, field.data)
            self.assertTrue(recovered.grid.same_geometry(field.grid))

    def test_affine_file_roundtrip_and_original_mat_subset(self):
        transform = AffineTransform.centered([[1.1, .2, 0], [0, .9, -.1], [0, 0, 1]], [2, -3, 4], [10, -20, 30])
        with tempfile.TemporaryDirectory() as directory:
            for extension in (".mat", ".tfm"):
                path = Path(directory) / ("affine" + extension)
                save_affine(transform, path)
                recovered = load_affine(path)
                np.testing.assert_allclose(recovered.matrix, transform.matrix, atol=1e-14)
                np.testing.assert_allclose(recovered.center, transform.center)
            bad = Path(directory) / "bad.mat"
            bad.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                load_affine(bad)
        for floating in (0, 1):
            path = Path(f"validation/baseline/p00/ixi002_to_ixi012/float{floating}/reg_0GenericAffine.mat")
            recovered = load_affine(path)
            probes = np.array([[0, 0, 0], [10, -20, 30]])
            np.testing.assert_allclose(recovered.inverse().map_points(recovered.map_points(probes)), probes, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
