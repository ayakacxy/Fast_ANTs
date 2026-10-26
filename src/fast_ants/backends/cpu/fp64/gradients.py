"""ITK-style index central differences and physical-point central differences."""

import numpy as np

from fast_ants.backends.cpu.fp64.filter_kernels import central_gradient
from fast_ants.backends.cpu.fp64.filter_kernels import physical_point_gradient
from fast_ants.core.types import points_array
from fast_ants.backends.cpu.fp64.sampling import sample_image


def gradient_image(image):
    """Index +/-1 stencil, zero local component at its index boundary, LPS/mm."""
    return central_gradient(np.asarray(image.data, dtype=np.float64), image.grid.lps_to_index[:3, :3])


def gradient_at_points(image, points_lps):
    """Match CentralDifferenceImageFunction::Evaluate(Point), not its index overload.

    Physical axes are offset by half the corresponding spacing. If either
    neighbour lies outside the half-voxel image support, that component is 0.
    """
    points = points_array(points_lps)
    output = physical_point_gradient(np.asarray(image.data, dtype=np.float64),
                                     np.ascontiguousarray(points.reshape(-1, 3)), image.grid.origin,
                                     image.grid.lps_to_index[:3, :3], image.grid.spacing_xyz)
    return output.reshape(points.shape)
