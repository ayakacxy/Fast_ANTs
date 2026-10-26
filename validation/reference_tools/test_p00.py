"""P00 numerical-fixture and provenance checks; standard-library unittest runner."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from validation.common.p00_common import ROOT, PROJECT, block_mean, sample_field_lps, sha256
from validation.reference_tools.prepare_p00 import phantom
from validation.reference_tools.run_p00_baseline import command_for


class P00Checks(unittest.TestCase):
    def test_three_mri_and_publisher_checksums(self):
        manifest = json.loads((ROOT / "sources/downloaded-files.json").read_text())
        images = [f for f in manifest["files"] if f["name"].endswith(".nii.gz")]
        self.assertEqual(len(images), 3)
        for entry in manifest["files"]:
            path = PROJECT / entry["path"]
            algorithm, expected = entry["publisher_checksum"].split(":")
            self.assertEqual(hashlib.new(algorithm, path.read_bytes()).hexdigest(), expected)
            self.assertEqual(sha256(path), entry["sha256"])

    def test_block_mean_preserves_physical_center(self):
        index = np.indices((8, 8, 8), dtype=np.float64)
        values = index[0] + 2 * index[1] + 3 * index[2]
        affine = np.array([[0, -2, 0, 15], [1.3, 0, 0, -25], [0, 0, 3, 42], [0, 0, 0, 1]])
        result, geometry = block_mean(values, affine, 4)
        np.testing.assert_allclose(result, [[[9, 21], [17, 29]], [[13, 25], [21, 33]]])
        np.testing.assert_allclose(nib.affines.apply_affine(geometry, [1, 0, 1]),
                                   nib.affines.apply_affine(affine, [5.5, 1.5, 5.5]))

    def test_analytic_maps_and_exact_shear_inverse(self):
        cases = json.loads((ROOT / "cases.json").read_text())["cases"]
        for case in cases:
            if case["kind"] != "analytic":
                continue
            fixed = np.array(case["fixed_landmarks_lps_mm"]) * [-1, -1, 1]
            moving = np.array(case["moving_landmarks_lps_mm"]) * [-1, -1, 1]
            truth = case["truth"]
            if truth["matrix"] is not None:
                recovered = nib.affines.apply_affine(np.linalg.inv(truth["matrix"]), moving)
            else:
                recovered = moving.copy()
                recovered[:, 0] -= 1.5 * np.sin(moving[:, 1] / 14)
                # A shear along x depending only on y has an exact unit determinant.
                jacobian = np.eye(3)
                jacobian[0, 1] = 1.5 / 14 * np.cos(.37)
                self.assertEqual(np.linalg.det(jacobian), 1)
            np.testing.assert_allclose(recovered, fixed, atol=1e-12)
            np.testing.assert_allclose(phantom(recovered), phantom(fixed), atol=1e-12)

    def test_lps_field_interpolation_uses_image_geometry(self):
        affine = np.array([[0, -2, 0, 9], [1.5, 0, 0, -8], [0, 0, 2.5, 13], [0, 0, 0, 1]])
        index = np.moveaxis(np.indices((6, 7, 8), dtype=np.float64), 0, -1)
        ras = nib.affines.apply_affine(affine, index)
        lps = ras * [-1, -1, 1]
        field = .02 * lps + [1, -2, 3]
        probes = nib.affines.apply_affine(affine, [[1.25, 2.5, 3.75], [2.5, 1.75, 4.25]]) * [-1, -1, 1]
        with tempfile.TemporaryDirectory(prefix="fast-ants-p00-") as temporary:
            path = Path(temporary) / "field.nii.gz"
            nib.save(nib.Nifti1Image(field[:, :, :, None, :], affine), path)
            np.testing.assert_allclose(sample_field_lps(path, probes), .02 * probes + [1, -2, 3], atol=1e-12)

    def test_command_has_explicit_precision_and_dense_sampling(self):
        config = json.loads((ROOT / "configs/p00_dense.json").read_text())
        case = json.loads((ROOT / "cases.json").read_text())["cases"][0]
        command = command_for(config, case, 0, Path("/tmp/p00"), {"antsRegistration": "antsRegistration"})
        self.assertEqual(command[command.index("--float") + 1], "0")
        metrics = [command[i + 1] for i, value in enumerate(command) if value == "--metric"]
        self.assertEqual(len(metrics), 3)
        self.assertTrue(all(metric.endswith(",None,1,0]") for metric in metrics))
        self.assertNotIn("ixi002_to_ixi013", config["default_cases"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
