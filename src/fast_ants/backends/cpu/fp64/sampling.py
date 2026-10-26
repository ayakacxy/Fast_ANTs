from fast_ants.core.fields import DisplacementField
"""CPU FP64 reference sampling and pull resampling with explicit physical maps."""

from dataclasses import dataclass
import numpy as np

from fast_ants.backends.cpu.common.geometry import grid_points
from fast_ants.backends.cpu.common.geometry import physical_to_indices
from fast_ants.backends.cpu.common.reductions import all_finite
from fast_ants.backends.cpu.fp64.sampling_kernels import add_vectors
from fast_ants.backends.cpu.fp64.sampling_kernels import sample_linear
from fast_ants.backends.cpu.fp64.sampling_kernels import sample_nearest
from fast_ants.backends.cpu.fp64.sampling_kernels import sample_vector
from fast_ants.core.types import Grid
from fast_ants.core.types import Image
from fast_ants.core.types import points_array
from fast_ants.core.transforms import AffineTransform
from fast_ants.core.transforms import CompositeTransform


def sample_image(image, points_lps, interpolation="Linear", default_value=0):
    points = points_array(points_lps)
    flat = np.ascontiguousarray(points.reshape(-1, 3))
    indices = physical_to_indices(flat, image.grid.lps_to_index[:3, :3], image.grid.origin)
    if not np.isfinite(default_value):
        raise ValueError("Outside value must be finite")
    if interpolation == "Linear":
        values, valid = sample_linear(np.asarray(image.data, dtype=np.float64), indices, float(default_value))
    elif interpolation == "NearestNeighbor":
        if image.data.dtype.kind in "iu":
            limits = np.iinfo(image.data.dtype)
            if default_value != int(default_value) or not limits.min <= default_value <= limits.max:
                raise ValueError("Outside value is not representable by label dtype")
        values, valid = sample_nearest(image.data, indices, default_value)
    else:
        raise ValueError("Supported interpolation: Linear or NearestNeighbor")
    return values.reshape(points.shape[:-1]), valid.reshape(points.shape[:-1])




def _pull_parameters(transform):
    """Pack exact built-in maps; overridden/custom map_points retain fallback."""
    field, affine = None, None
    if type(transform) is AffineTransform:
        affine, mode = transform, 1
    elif type(transform) is DisplacementField:
        field, mode = transform, 2
    elif type(transform) is CompositeTransform and len(transform.transforms) == 2:
        first, second = transform.transforms
        if type(first) is DisplacementField and type(second) is AffineTransform:
            field, affine, mode = first, second, 3
        elif type(first) is AffineTransform and type(second) is DisplacementField:
            affine, field, mode = first, second, 4
        else:
            return None
    else:
        return None
    field_data = np.zeros((3, 1, 1, 1)) if field is None else np.asarray(field.data, dtype=np.float64)
    field_inverse = np.eye(3) if field is None else field.grid.lps_to_index[:3, :3]
    field_origin = np.zeros(3) if field is None else field.grid.origin
    linear = np.eye(3) if affine is None else affine.matrix[:3, :3]
    center = np.zeros(3) if affine is None else affine.center
    translation = np.zeros(3) if affine is None else affine.translation
    return field_data, field_inverse, field_origin, linear, center, translation, mode


def apply_image(moving, reference, fixed_to_moving=None, interpolation="Linear", default_value=0, chunk_slices=8):
    """For each reference point x, sample moving at fixed_to_moving(x)."""
    grid = reference.grid if isinstance(reference, Image) else reference
    if not isinstance(grid, Grid):
        raise TypeError("Reference must be an Image or Grid")
    if not isinstance(chunk_slices, int) or chunk_slices <= 0:
        raise ValueError("chunk_slices must be a positive integer")
    transform = AffineTransform.identity() if fixed_to_moving is None else fixed_to_moving
    parameters = _pull_parameters(transform)
    if parameters is not None:
        from fast_ants.backends.cpu.fp64.pull import pull_image64
        if not np.isfinite(default_value):
            raise ValueError("Outside value must be finite")
        if interpolation == "Linear":
            data, default = np.asarray(moving.data, dtype=np.float64), float(default_value)
        elif interpolation == "NearestNeighbor":
            if moving.data.dtype.kind in "iu":
                limits = np.iinfo(moving.data.dtype)
                if default_value != int(default_value) or not limits.min <= default_value <= limits.max:
                    raise ValueError("Outside value is not representable by label dtype")
            data, default = moving.data, default_value
        else:
            raise ValueError("Supported interpolation: Linear or NearestNeighbor")
        output = pull_image64(data, grid, moving.grid.lps_to_index[:3, :3],
                              moving.grid.origin, parameters, default,
                              interpolation == "NearestNeighbor")
        return Image(output, grid)
    dtype = moving.data.dtype if interpolation == "NearestNeighbor" else np.float64
    output = np.empty(grid.shape_zyx, dtype=dtype)
    for start in range(0, grid.shape_zyx[0], chunk_slices):
        stop = min(start + chunk_slices, grid.shape_zyx[0])
        points = grid_points(grid.index_to_lps, grid.shape_zyx, start, stop)
        mapped = transform.map_points(points)
        values, valid = sample_image(moving, mapped, interpolation, default_value)
        output[start:stop] = values.reshape(stop - start, grid.shape_zyx[1], grid.shape_zyx[2])
    return Image(output, grid)


def transform_to_field(grid, transform):
    parameters = _pull_parameters(transform)
    if parameters is not None:
        from fast_ants.backends.cpu.fp64.pull import rasterize_transform64
        return DisplacementField(rasterize_transform64(grid, parameters), grid)
    output = np.empty((3, *grid.shape_zyx), dtype=np.float64)
    for z in range(grid.shape_zyx[0]):
        points = grid_points(grid.index_to_lps, grid.shape_zyx, z, z + 1)
        values = transform.map_points(points) - points
        output[:, z] = values.T.reshape(3, grid.shape_zyx[1], grid.shape_zyx[2])
    return DisplacementField(output, grid)
