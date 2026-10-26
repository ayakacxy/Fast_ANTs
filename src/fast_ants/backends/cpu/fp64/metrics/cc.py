from fast_ants.core.results import CCEvaluation
"""Neighbourhood CC on a common physical grid, with ITK's local update."""

from dataclasses import dataclass
import numpy as np

from fast_ants.backends.cpu.fp64.metrics.cc_kernels import local_cc
from fast_ants.backends.cpu.fp64.metrics.cc_pair import local_cc_pair
from fast_ants.backends.cpu.common.geometry import grid_points
from fast_ants.backends.cpu.fp64.gradients import gradient_at_points
from fast_ants.pipeline.masks import sample_mask
from fast_ants.core.results import NoValidSamplesError
from fast_ants.core.fields import DisplacementField




def neighborhood_cc_pair(fixed, moving, radius=2, *, fixed_mask=None, moving_mask=None):
    """Two directed source-order evaluations, with a shared unmasked traversal."""
    if fixed_mask is not None or moving_mask is not None:
        return (neighborhood_cc(fixed, moving, radius, fixed_mask=fixed_mask, moving_mask=moving_mask),
                neighborhood_cc(moving, fixed, radius, fixed_mask=moving_mask, moving_mask=fixed_mask))
    if not fixed.grid.same_geometry(moving.grid):
        raise ValueError('CC inputs must be resampled onto the same physical grid')
    radii = np.asarray(radius)
    if radii.ndim == 0:
        radii = np.repeat(radii, 3)
    if radii.shape != (3,) or not np.isfinite(radii).all() or np.any(radii < 0) or np.any(radii != np.floor(radii)):
        raise ValueError('CC radius must be a nonnegative integer or x,y,z triple')
    fs, rs, forward, reverse = local_cc_pair(
        np.asarray(fixed.data, dtype=np.float64), np.asarray(moving.data, dtype=np.float64),
        radii.astype(np.int64), fixed.grid.index_to_lps, fixed.grid.origin,
        fixed.grid.lps_to_index[:3, :3], fixed.grid.spacing_xyz)
    count = fixed.data.size
    return (CCEvaluation(float(fs/count), DisplacementField(forward, fixed.grid), count),
            CCEvaluation(float(rs/count), DisplacementField(reverse, fixed.grid), count))


def neighborhood_cc(fixed, moving, radius=2, *, valid=None, fixed_mask=None,
                    moving_mask=None, return_statistics=False):
    """Compute a dense CC value and centre-local *descent* field for moving.

    Inputs must already occupy the same grid (e.g. SyN's two midpoint images).
    The field has no division by the number of valid centres. Clipped windows
    ignore invalid neighbours; an invalid centre never contributes. A degenerate
    window contributes score 1 and zero update, following ITK, including radius 0.
    """
    if not fixed.grid.same_geometry(moving.grid):
        raise ValueError("CC inputs must be resampled onto the same physical grid")
    raw_radius = np.asarray(radius)
    if raw_radius.ndim == 0:
        raw_radius = np.repeat(raw_radius, 3)
    if raw_radius.shape != (3,) or not np.isfinite(raw_radius).all() or np.any(raw_radius < 0) or np.any(raw_radius != np.floor(raw_radius)):
        raise ValueError("CC radius must be a nonnegative integer or x,y,z triple")
    radii = raw_radius.astype(np.int64)
    if valid is None:
        valid_array = np.ones(fixed.grid.shape_zyx, dtype=np.bool_)
    else:
        supplied = np.asarray(valid)
        if supplied.shape != fixed.grid.shape_zyx or supplied.dtype != np.bool_:
            raise ValueError("CC validity must be a Boolean array on the common grid")
        valid_array = supplied.copy()
    gradient = np.empty((3, *fixed.grid.shape_zyx), dtype=np.float64)
    for z in range(fixed.grid.shape_zyx[0]):
        points = grid_points(fixed.grid.index_to_lps, fixed.grid.shape_zyx, z, z+1)
        gradient[:, z] = gradient_at_points(moving, points).T.reshape(3, *fixed.grid.shape_zyx[1:])
        for mask in (fixed_mask, moving_mask):
            if mask is not None:
                valid_array[z] &= sample_mask(mask, points).reshape(fixed.grid.shape_zyx[1:])
    value_sum, count, update, stats = local_cc(np.asarray(fixed.data, dtype=np.float64),
                                               np.asarray(moving.data, dtype=np.float64),
                                               gradient, np.ascontiguousarray(valid_array),
                                               radii, bool(return_statistics))
    if count == 0:
        raise NoValidSamplesError("No valid CC centre exists in the supplied common domain")
    names = ("count", "fixed_mean", "moving_mean", "fixed_variance", "moving_variance", "covariance", "score")
    statistics = dict(zip(names, stats)) if return_statistics else None
    return CCEvaluation(float(value_sum/count), DisplacementField(update, fixed.grid), int(count), statistics)
