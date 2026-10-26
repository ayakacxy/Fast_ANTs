"""ANTs affine MAT-v4/ITK-text subset and ANTs LPS-vector NIfTI fields."""

from pathlib import Path
import struct

import nibabel as nib
import numpy as np

from fast_ants.core.types import Grid
from fast_ants.io.images import _MM_SCALE
from fast_ants.io.nifti import _save_nifti_storage
from fast_ants.core.fields import DisplacementField
from fast_ants.core.transforms import AffineTransform
from fast_ants.core.transforms import CompositeTransform


def _mat_vectors(data):
    position, records = 0, {}
    while position < len(data):
        if len(data) - position < 20:
            raise ValueError("Truncated MAT-v4 header")
        selected = None
        for endian in ("<", ">"):
            kind, rows, cols, imaginary, length = struct.unpack_from(endian + "5i", data, position)
            machine, precision, matrix_kind = kind // 1000, (kind // 10) % 10, kind % 10
            if (machine == (0 if endian == "<" else 1) and precision in (0, 1) and
                    matrix_kind == 0 and 1 <= rows <= 10000 and 1 <= cols <= 10000 and
                    imaginary == 0 and 1 <= length <= 1024):
                selected = endian, rows, cols, length, precision
                break
        if selected is None:
            raise ValueError("Unsupported MAT file; only real IEEE MAT-v4 numeric vectors are supported")
        endian, rows, cols, length, precision = selected
        position += 20
        name_end = position + length
        if name_end > len(data) or data[name_end - 1] != 0:
            raise ValueError("Invalid MAT-v4 variable name")
        name = data[position:name_end - 1].decode("ascii")
        dtype = np.dtype(endian + ("f8" if precision == 0 else "f4"))
        end = name_end + rows * cols * dtype.itemsize
        if end > len(data) or rows * cols > 12:
            raise ValueError("Truncated or unsupported transform parameter vector")
        if name in records:
            raise ValueError("Multiple transform blocks are not supported")
        records[name] = np.frombuffer(data[name_end:end], dtype=dtype).astype(np.float64)
        position = end
    return records


def load_affine(path):
    data = Path(path).read_bytes()
    if data.startswith(b"#Insight"):
        records = {}
        for line in data.decode("utf-8").splitlines():
            if line.startswith(("Transform:", "Parameters:", "FixedParameters:")):
                key, value = line.split(":", 1)
                if key in records:
                    raise ValueError("Multiple transform blocks are not supported")
                records[key] = value.strip()
        name = records.get("Transform", "")
        parameters = np.fromstring(records.get("Parameters", ""), sep=" ")
        center = np.fromstring(records.get("FixedParameters", ""), sep=" ")
    else:
        records = _mat_vectors(data)
        names = [name for name in records if name != "fixed"]
        if len(names) != 1 or set(records) != {names[0], "fixed"}:
            raise ValueError("Expected one affine and one fixed-parameter record")
        name = names[0]
        parameters, center = records[name], records["fixed"]
    supported = {f"{kind}_{dtype}_3_3" for kind in ("AffineTransform", "MatrixOffsetTransformBase")
                 for dtype in ("float", "double")}
    if name not in supported or parameters.shape != (12,) or center.shape != (3,):
        raise ValueError(f"Unsupported linear transform: {name}; expected a 3D affine with 12+3 parameters")
    return AffineTransform.centered(parameters[:9].reshape(3, 3), parameters[9:], center)


def save_affine(transform, path, dtype=np.float64):
    if not isinstance(transform, AffineTransform):
        raise TypeError("save_affine requires an AffineTransform")
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("Affine output dtype must be float32 or float64")
    name = "AffineTransform_" + ("float" if dtype == np.float32 else "double") + "_3_3"
    parameters = np.r_[transform.matrix[:3, :3].ravel(), transform.translation]
    if Path(path).suffix == ".tfm":
        text = ("#Insight Transform File V1.0\n#Transform 0\nTransform: " + name +
                "\nParameters: " + " ".join(format(float(x), ".17g") for x in parameters.astype(dtype,copy=False)) +
                "\nFixedParameters: " + " ".join(format(float(x), ".17g") for x in transform.center.astype(dtype,copy=False)) + "\n")
        Path(path).write_text(text)
        return
    if Path(path).suffix != ".mat":
        raise ValueError("Affine output format must be .mat or .tfm")
    output = bytearray()
    for variable, values in ((name, parameters), ("fixed", transform.center)):
        encoded = variable.encode("ascii") + b"\0"
        output.extend(struct.pack("<5i", 10 if dtype == np.float32 else 0, len(values), 1, 0, len(encoded)))
        output.extend(encoded)
        output.extend(np.asarray(values, dtype="<f4" if dtype == np.float32 else "<f8").tobytes())
    Path(path).write_bytes(output)


def load_field(path, dtype=np.float64):
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    if values.ndim == 5 and values.shape[3:] == (1, 3):
        values = values[:, :, :, 0, :]
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Expected ANTs vector field [x,y,z,1,3] or [x,y,z,3]")
    unit = image.header.get_xyzt_units()[0]
    if unit not in _MM_SCALE:
        raise ValueError("Unsupported field units")
    scale = _MM_SCALE[unit]
    matrix = np.array(image.affine, dtype=np.float64)
    matrix[:3] *= scale
    # This API deliberately reads ANTs' LPS components, not arbitrary RAS vectors.
    field = np.asarray(values.transpose(3, 2, 1, 0) * scale, dtype=dtype)
    return DisplacementField(field, Grid.from_ras(values.shape[:3], matrix))


def save_field(field, path, dtype=None):
    dtype = field.data.dtype if dtype is None else np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("Field output dtype must be float32 or float64")
    storage = field.data.astype(dtype,copy=False)
    values = storage.transpose(3, 2, 1, 0)[:, :, :, None, :]
    image = nib.Nifti1Image(values, field.grid.affine_ras, dtype=dtype)
    image.header.set_intent("vector")
    image.header.set_xyzt_units("mm")
    image.set_qform(field.grid.affine_ras, 1)
    image.set_sform(field.grid.affine_ras, 1)
    _save_nifti_storage(image, path, storage)


def load_ants_transform_list(paths):
    """Image-pull convention: -t warp -t affine maps x to affine(warp(x)).

    This is the map used to sample a moving image on the reference grid. It
    must not be confused with an anatomical moving-to-fixed point push map.
    """
    transforms = []
    for item in paths:
        if isinstance(item, tuple):
            path, invert = item
        else:
            path, invert = item, False
        transform = load_field(path) if str(path).endswith((".nii", ".nii.gz")) else load_affine(path)
        if invert:
            transform = transform.inverse()
        transforms.append(transform)
    return CompositeTransform(tuple(transforms))
