"""Prepare exactly three public MRI subjects and small analytic numerical fixtures."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import json
from pathlib import Path

import nibabel as nib
import numpy as np

from validation.common.p00_common import ROOT, PROJECT, block_mean, image_info, sha256, write_json


def save(path, values, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(values.astype(np.float32), affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return str(path.relative_to(PROJECT))


def read_landmarks(path):
    markup = json.loads(path.read_text())["markups"][0]
    points = np.array([p["position"] for p in markup["controlPoints"]])
    coordinate_system = markup["coordinateSystem"]
    if coordinate_system == "RAS":
        points *= [-1, -1, 1]
    elif coordinate_system != "LPS":
        raise ValueError(f"Unsupported landmark coordinates: {coordinate_system}")
    if points.shape != (4, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid four-point annotation: {path}")
    return points.tolist()


def phantom(points):
    out = np.zeros(points.shape[:-1])
    for center, width, amplitude in [((-4, 1, 0), (9, 12, 10), 0.8),
                                     ((9, -6, 6), (3, 5, 4), 0.65),
                                     ((-8, 9, -7), (4, 3, 3), 0.6),
                                     ((3, 7, 10), (3, 3, 3), 0.35)]:
        out += amplitude * np.exp(-0.5 * np.sum(((points - center) / width) ** 2, axis=-1))
    return out


def synthetic_cases():
    shape = (40, 36, 32)
    angle = np.deg2rad(17)
    direction = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag([1.2, 1.4, 1.6])
    affine[:3, 3] = -affine[:3, :3] @ ((np.array(shape) - 1) / 2)
    indices = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1)
    points = nib.affines.apply_affine(affine, indices)
    probes = np.array([[-5, 0, 0], [6, -4, 4], [-7, 6, -5], [2, 5, 8]])
    fixed_path = save(ROOT / "data/analytic/fixed.nii.gz", phantom(points), affine)
    cases = []
    for kind in ("identity", "rigid", "affine", "smooth"):
        transform = np.eye(4)
        if kind in ("rigid", "affine"):
            theta = np.deg2rad(4)
            transform[:3, :3] = [[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            transform[:3, 3] = [1.2, -0.8, 0.6]
        if kind == "affine":
            transform[:3, :3] = transform[:3, :3] @ np.array([[1.02, .015, 0],
                                                            [0, .98, .01], [0, 0, 1.01]])
        inverse_points = nib.affines.apply_affine(np.linalg.inv(transform), points)
        targets = nib.affines.apply_affine(transform, probes)
        if kind == "smooth":
            # Analytic shear: exact inverse and determinant 1; no resampling library.
            inverse_points[..., 0] -= 1.5 * np.sin(points[..., 1] / 14)
            targets[:, 0] += 1.5 * np.sin(probes[:, 1] / 14)
        moving_path = save(ROOT / f"data/analytic/{kind}.nii.gz", phantom(inverse_points), affine)
        cases.append({"id": f"synthetic_{kind}", "split": "development", "kind": "analytic",
                      "fixed": fixed_path, "moving": moving_path,
                      "fixed_landmarks_lps_mm": (probes * [-1, -1, 1]).tolist(),
                      "moving_landmarks_lps_mm": (targets * [-1, -1, 1]).tolist(),
                      "truth": {"map_direction": "fixed_to_moving", "coordinate_system": "RAS",
                                "matrix": transform.tolist() if kind != "smooth" else None,
                                "smooth_map": "(x+1.5*sin(y/14),y,z)" if kind == "smooth" else None,
                                "generation": "analytic continuous phantom; no interpolation"}})
    # Tiny operator cases are kept in one archive, not extra MRI subjects.
    fixtures = ROOT / "data/analytic/operator-fixtures.npz"
    np.savez(fixtures, constant=np.ones((5, 5, 5), np.float32),
             low_contrast=np.linspace(.7499, .7501, 125, dtype=np.float32).reshape(5, 5, 5),
             empty_mask=np.zeros((5, 5, 5), bool),
             boundary_indices=np.array([[-.001, 0, 0], [0, 0, 0], [3.5, 3.5, 3.5], [4, 4, 4]]),
             nonfinite=np.array([0, np.nan, np.inf], np.float32),
             large_origin_small_update=np.array([1000, 1e-5], np.float32))
    return cases


def main():
    downloaded = json.loads((ROOT / "sources/downloaded-files.json").read_text())["files"]
    images, landmarks, inventory, preparation = {}, {}, [], []
    for entry in downloaded:
        path = PROJECT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Input changed: {path}")
        subject = entry["subject"]
        if path.name.endswith("_annot.json"):
            landmarks[subject] = read_landmarks(path)
            continue
        inventory.append(image_info(path))
        original = nib.load(path)
        values, affine = block_mean(original.get_fdata(dtype=np.float32), original.affine, 4)
        # Fixed deterministic scale for numerical comparisons, not histogram matching.
        scale = float(values.max())
        if scale <= 0:
            raise ValueError(f"Empty MRI: {path}")
        images[subject] = save(ROOT / "data/prepared" / path.name, values / scale, affine)
        preparation.append({"subject": subject, "operation": "4x4x4 block mean, crop high edge remainder",
                            "intensity_divisor": scale, "image": image_info(PROJECT / images[subject])})
    if len(images) != 3:
        raise RuntimeError("P00 source data is deliberately limited to exactly 3 MRI subjects")
    subjects = sorted(images)
    cases = synthetic_cases()
    for moving in subjects[1:]:
        cases.append({"id": "ixi002_to_" + moving.split("-")[0].lower(),
                      "kind": "real_mri", "split": "reserved" if moving == subjects[2] else "development",
                      "fixed": images[subjects[0]], "moving": images[moving],
                      "fixed_landmarks_lps_mm": landmarks[subjects[0]],
                      "moving_landmarks_lps_mm": landmarks[moving],
                      "note": "Four non-expert head landmarks; exploratory numerical check, not a clinical quality target"})
    inputs = {p for case in cases for p in (case["fixed"], case["moving"])}
    write_json(ROOT / "cases.json", {"schema_version": 1, "raw_mri_subject_count": 3,
               "reserved_note": "IXI013 reserved; IXI002 is shared fixed image, not independent cohort validation",
               "cases": cases, "inputs": [image_info(PROJECT / p) for p in sorted(inputs)]})
    write_json(ROOT / "sources/input-inventory.json", {"raw_mri": inventory, "preparation": preparation,
               "operator_fixture_sha256": sha256(ROOT / "data/analytic/operator-fixtures.npz")})
    print(f"Prepared {len(images)} MRI subjects, {len(cases)} cases and tiny operator fixtures", flush=True)


if __name__ == "__main__":
    main()
