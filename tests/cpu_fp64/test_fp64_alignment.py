import json
from pathlib import Path
import unittest

import numpy as np

from fast_ants import AffineTransform, load_image
from fast_ants.core.linalg import inverse3, svd3
from fast_ants.pipeline.models64 import ComposedLinearModel
from fast_ants.backends.cpu.fp64.metrics.mattes import MattesMetric
from fast_ants.backends.cpu.fp64.preprocessing import prepare_level, preprocess_image


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT = Path(__file__).resolve().parents[2]


class FP64AlignmentTests(unittest.TestCase):
    def test_inverse_matches_independent_native_geometry_fixtures(self):
        fixture = json.loads((FIXTURES / "fp64_geometry.json").read_text())
        for case in fixture["geometries"]:
            with self.subTest(case=case["name"]):
                matrix = np.asarray(case["matrix"])
                before = matrix.copy()
                matrix.setflags(write=False)
                np.testing.assert_array_equal(inverse3(matrix), case["inverse"])
                np.testing.assert_array_equal(matrix, before)

    def test_svd_reconstruction_and_inverse_residual(self):
        rng = np.random.default_rng(825)
        for _ in range(24):
            rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            # Include reflected axes, anisotropy, and repeated singular values.
            scales = rng.uniform(.1, 4.0, 3)
            scales[1] = scales[0]
            scales[2] *= -1
            matrix = rotation@np.diag(scales)
            u, s, v = svd3(matrix)
            np.testing.assert_allclose(u@np.diag(s)@v.T, matrix, rtol=0, atol=8e-14)
            np.testing.assert_allclose(u.T@u, np.eye(3), rtol=0, atol=8e-15)
            np.testing.assert_allclose(matrix@inverse3(matrix), np.eye(3), rtol=0, atol=8e-14)
        with self.assertRaises(ValueError):
            inverse3(np.zeros((3, 3)))

    def test_unsmoothed_mattes_at_frozen_official_transform(self):
        # The former LU/no-FMA coordinates changed fixed bin assignments and
        # missed this independent official metric by approximately 2.3e-8.
        fixture = json.loads((FIXTURES / "fp64_mattes_fixed_state.json").read_text())
        fixed, moving = [preprocess_image(load_image(PROJECT / fixture[side], geometry_policy="itk_scanner"))
                         for side in ("fixed", "moving")]
        level = prepare_level(fixed, moving, 1, 0, "vox")
        initial = AffineTransform(fixture["matrix"], fixture["center"])
        model = ComposedLinearModel("Rigid", initial.center, initial)
        metric = MattesMetric(level.fixed, level.moving, level.virtual_grid)
        evaluation = metric.evaluate(model, model.identity_parameters(), False)
        # Native log prints 13 significant decimal digits.
        self.assertLess(abs(evaluation.value-fixture["native_metric"]), 5e-13)


if __name__ == "__main__":
    unittest.main()
