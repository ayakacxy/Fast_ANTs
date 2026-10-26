"""Small independent P00 reference helpers, not production registration operators."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION
from validation.common.runtime import sha256, write_json


import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = _TOOL_VALIDATION
PROJECT = ROOT.parent






def image_info(path):
    image = nib.load(path)
    values = image.get_fdata(dtype=np.float32)
    if len(image.shape) != 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite 3D scalar image: {path}")
    if abs(np.linalg.det(image.affine[:3, :3])) < 1e-10:
        raise ValueError(f"Singular image geometry: {path}")
    return {"path": str(Path(path).relative_to(PROJECT)), "sha256": sha256(path),
            "shape": list(image.shape), "dtype": str(image.get_data_dtype()),
            "affine_ras_mm": image.affine.tolist(),
            "spacing_mm": [float(v) for v in image.header.get_zooms()],
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"]),
            "minimum": float(values.min()), "maximum": float(values.max())}


def block_mean(data, affine, factor=4):
    """P00 data preparation: block average, preserve each block's physical center."""
    shape = tuple(s // factor for s in data.shape)
    cut = data[tuple(slice(0, n * factor) for n in shape)]
    small = cut.reshape(shape[0], factor, shape[1], factor, shape[2], factor)
    small = small.mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32)
    index_map = np.eye(4)
    index_map[:3, :3] *= factor
    index_map[:3, 3] = (factor - 1) / 2
    return small, affine @ index_map


def sample_field_lps(path, points_lps):
    """Independent eight-corner interpolation at a handful of validation points."""
    image = nib.load(path)
    field = image.get_fdata()
    if field.ndim == 5 and field.shape[3] == 1:
        field = field[:, :, :, 0, :]
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Unexpected displacement layout: {field.shape}")
    ras = np.asarray(points_lps, dtype=np.float64) * np.array([-1, -1, 1])
    index = nib.affines.apply_affine(np.linalg.inv(image.affine), ras)
    lower = np.floor(index).astype(int)
    fraction = index - lower
    if np.any(lower < 0) or np.any(lower + 1 >= np.array(field.shape[:3])):
        raise ValueError("Validation points outside field interpolation domain")
    out = np.zeros((len(index), 3))
    for i in range(8):
        offset = np.array([(i >> axis) & 1 for axis in range(3)])
        location = lower + offset
        weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
        out += field[location[:, 0], location[:, 1], location[:, 2]] * weight[:, None]
    return out


def difference(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "p99_abs": float(np.percentile(np.abs(delta), 99))}
