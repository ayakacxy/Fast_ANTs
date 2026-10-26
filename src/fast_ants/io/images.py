"""NIfTI scalar I/O. sform > qform > fallback follows NiBabel's best affine."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np

from fast_ants.core.types import Grid
from fast_ants.core.types import Image
from fast_ants.io.nifti import _save_nifti_storage

_MM_SCALE = {"mm": 1.0, "meter": 1000.0, "micron": 0.001, "unknown": 1.0}


def load_image(path, dtype=np.float64, *, geometry_policy="nifti", scaling_policy="nifti"):
    """Load a scalar image with explicit geometry and voxel-scaling policies.

    ``itk`` scaling reproduces NiftiImageIO's component-type conversion before
    conversion to the requested dtype. In particular, scaled integer inputs
    pass through FP32 even when antsRegistration uses ``--float 0``. The
    default ``nifti`` policy retains NiBabel's ordinary scaling semantics.
    """
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError("Only three-dimensional scalar NIfTI images are supported")
    unit = image.header.get_xyzt_units()[0]
    if unit not in _MM_SCALE:
        raise ValueError(f"Unsupported NIfTI spatial unit: {unit}")
    affine = np.array(image.affine, dtype=np.float64)
    if geometry_policy not in ("nifti", "itk_scanner"):
        raise ValueError("Geometry policy must be nifti or itk_scanner")
    if geometry_policy == "itk_scanner":
        # Bounded ITK-compatible subset: a scanner sform is authoritative.
        # ITK normalizes columns then uses header pixdim for voxel spacing.
        if int(image.header["sform_code"]) != 1:
            raise ValueError("itk_scanner geometry requires a coded scanner-anatomical sform")
        linear = affine[:3, :3]
        norms = np.sqrt(np.sum(linear*linear, axis=0))
        direction = linear/norms
        if not np.allclose(direction.T@direction, np.eye(3), rtol=0, atol=1e-5):
            raise ValueError("itk_scanner geometry requires orthogonal sform direction columns")
        affine[:3, :3] = direction*np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    affine[:3] *= _MM_SCALE[unit]
    if scaling_policy not in ("nifti", "itk"):
        raise ValueError("Scaling policy must be nifti or itk")
    if scaling_policy == "itk":
        raw = np.asarray(image.dataobj.get_unscaled())
        slope, intercept = float(image.dataobj.slope), float(image.dataobj.inter)
        epsilon = np.finfo(np.float64).eps
        rescale = abs(slope) > epsilon and (abs(slope-1.0) > epsilon or abs(intercept) > epsilon)
        component_dtype = np.dtype(np.float32) if rescale and raw.dtype.kind in "biu" else raw.dtype
        values = raw.astype(component_dtype)
        if rescale:
            # ITK first casts integers to its FP32 IO buffer, computes the
            # scale/offset in double, then stores back in that buffer's type.
            values = (values.astype(np.float64)*slope+intercept).astype(component_dtype)
    else:
        values = np.asarray(image.dataobj)
    if dtype is not None:
        values = values.astype(dtype)
    sform, qform = int(image.header["sform_code"]), int(image.header["qform_code"])
    metadata = {"source": str(Path(path)), "affine_source": "sform" if sform else "qform" if qform else "fallback",
                "sform_code": sform, "qform_code": qform, "source_spatial_unit": unit,
                "unknown_unit_assumed_mm": unit == "unknown", "geometry_policy":geometry_policy,
                "scaling_policy":scaling_policy}
    if sform and qform:
        metadata["qform_sform_agree"] = bool(np.allclose(image.get_sform(), image.get_qform(), rtol=0, atol=1e-5))
    return Image(values.transpose(2, 1, 0), Grid.from_ras(image.shape, affine), metadata)


def load_images(paths, dtype=np.float64, *, geometry_policy="nifti", scaling_policy="nifti"):
    """Read independent images with at most two workers, retaining input order.

    Every item uses ``load_image`` with the same policies and no shared cache.
    Empty and single-item inputs avoid creating worker threads. Exceptions
    propagate from the corresponding ordered result.
    """
    paths=tuple(paths)
    if not paths:
        return ()
    def read(path):
        return load_image(path,dtype,geometry_policy=geometry_policy,scaling_policy=scaling_policy)
    if len(paths)==1:
        return (read(paths[0]),)
    with ThreadPoolExecutor(max_workers=2) as readers:
        return tuple(readers.map(read,paths))


def save_image(image, path):
    values = image.data.transpose(2, 1, 0)
    nifti = nib.Nifti1Image(values, image.grid.affine_ras, dtype=values.dtype)
    nifti.set_qform(image.grid.affine_ras, 1)
    nifti.set_sform(image.grid.affine_ras, 1)
    nifti.header.set_xyzt_units("mm")
    _save_nifti_storage(nifti, path, image.data)
