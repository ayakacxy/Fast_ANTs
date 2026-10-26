from fast_ants.core.results import NoValidSamplesError
from fast_ants.core.results import MetricEvaluation
"""Metric control and explicit ITK descent-direction convention."""

from dataclasses import dataclass
import numpy as np

from fast_ants.backends.cpu.common.geometry import grid_points
from fast_ants.backends.cpu.fp64.metrics.linear_fused import evaluate_linear_mattes
from fast_ants.backends.cpu.fp64.metrics.linear_fused import fixed_bin_order
from fast_ants.backends.cpu.fp64.metrics.mattes_kernels import mattes_samples
from fast_ants.backends.cpu.fp64.metrics.mattes_kernels import mean_squares_samples
from fast_ants.backends.cpu.common.reductions import minmax
from fast_ants.backends.cpu.fp64.gradients import gradient_at_points
from fast_ants.pipeline.models64 import ComposedLinearModel
from fast_ants.pipeline.models64 import LinearModel
from fast_ants.pipeline.masks import sample_mask
from fast_ants.backends.cpu.fp64.sampling import sample_image






def evaluate_mattes_samples(fixed, moving, intensity_derivatives=None, bins=32, fixed_range=None, moving_range=None):
    fixed, moving = np.ascontiguousarray(fixed, dtype=np.float64), np.ascontiguousarray(moving, dtype=np.float64)
    if fixed.ndim != 1 or moving.shape != fixed.shape or fixed.size == 0:
        raise ValueError("Mattes input samples must be nonempty matching 1D arrays")
    if not np.isfinite(fixed).all() or not np.isfinite(moving).all() or bins < 5 or int(bins) != bins:
        raise ValueError("Invalid Mattes sample values or histogram bin count")
    fixed_range = minmax(fixed) if fixed_range is None else fixed_range
    moving_range = minmax(moving) if moving_range is None else moving_range
    if (not np.isfinite(fixed_range).all() or not np.isfinite(moving_range).all() or
            fixed_range[1] <= fixed_range[0] or moving_range[1] <= moving_range[0]):
        raise ValueError("Mattes intensity ranges must be finite and nonconstant")
    derivatives = np.zeros((len(fixed), 0)) if intensity_derivatives is None else np.asarray(intensity_derivatives, dtype=np.float64)
    if derivatives.ndim != 2 or derivatives.shape[0] != len(fixed) or not np.isfinite(derivatives).all():
        raise ValueError("Intensity derivatives must have shape [samples, parameters]")
    value, descent, joint, fixed_pdf, moving_pdf, pdf_derivative, valid = mattes_samples(
        fixed, moving, derivatives, int(bins), *fixed_range, *moving_range)
    if valid == 0:
        raise NoValidSamplesError("Mattes metric has no valid moving samples")
    return MetricEvaluation(float(value), descent, valid, {"joint_pdf": joint, "fixed_pdf": fixed_pdf,
                            "moving_pdf": moving_pdf, "pdf_derivative": pdf_derivative})


def _image_range(image, mask):
    if mask is None:
        return minmax(np.asarray(image.data, dtype=np.float64).ravel())
    points = grid_points(image.grid.index_to_lps, image.grid.shape_zyx, 0, image.grid.shape_zyx[0])
    selected = sample_mask(mask, points)
    if not selected.any():
        raise NoValidSamplesError("Mask contains no image pixel centres")
    return minmax(np.ascontiguousarray(image.data.ravel()[selected], dtype=np.float64))


class MattesMetric:
    """Mattes evaluation with a fixed-sample cache for built-in linear models.

    Replacing public arrays, ranges, masks, or moving geometry safely returns
    to the ordinary evaluation path. Fixed-value array contents are treated as
    immutable for this metric instance's cache lifetime; in-place edits of
    those contents are not detected by a full-array scan on each evaluation.
    """
    def __init__(self, fixed, moving, virtual_grid=None, bins=32, points_lps=None, fixed_mask=None, moving_mask=None):
        self.fixed, self.moving, self.bins = fixed, moving, bins
        self.virtual_grid = fixed.grid if virtual_grid is None else virtual_grid
        self.moving_mask = moving_mask
        self.points = (grid_points(self.virtual_grid.index_to_lps, self.virtual_grid.shape_zyx,
                                   0, self.virtual_grid.shape_zyx[0]) if points_lps is None else
                       np.ascontiguousarray(points_lps, dtype=np.float64))
        values, valid = sample_image(fixed, self.points)
        if fixed_mask is not None:
            valid &= sample_mask(fixed_mask, self.points)
        self.points = np.ascontiguousarray(self.points[valid])
        self.fixed_values = np.ascontiguousarray(values[valid])
        self.fixed_range, self.moving_range = _image_range(fixed, fixed_mask), _image_range(moving, moving_mask)
        if points_lps is not None and len(self.points):
            # ITK sparse initialization uses nearest fixed pixel intensities at
            # the actual fixed-space sample points to estimate this range.
            nearest, inside = sample_image(fixed, self.points, interpolation="NearestNeighbor")
            self.fixed_range = minmax(np.ascontiguousarray(nearest[inside], dtype=np.float64))
        if not len(self.fixed_values):
            raise NoValidSamplesError("No fixed samples in metric domain")
        self._fused_fixed_order = None
        self._fused_cache_state = None
        # Only the established built-in linear models use this path. Keep
        # masks/custom models and invalid-range error handling on the existing
        # implementation rather than silently changing their public contract.
        if (type(self) is MattesMetric and moving_mask is None and
                isinstance(bins, (int, float, np.integer, np.floating)) and np.isfinite(bins) and
                bins >= 5 and int(bins) == bins and
                np.isfinite(self.fixed_values).all() and
                np.isfinite(self.fixed_range).all() and np.isfinite(self.moving_range).all() and
                self.fixed_range[1] > self.fixed_range[0] and self.moving_range[1] > self.moving_range[0]):
            self._fused_fixed_order = fixed_bin_order(self.fixed_values, int(bins), *self.fixed_range)
            self._fused_cache_state = (
                self.points, (self.points.shape, self.points.dtype, self.points.strides),
                self.fixed_values, (self.fixed_values.shape, self.fixed_values.dtype, self.fixed_values.strides),
                int(bins), tuple(self.fixed_range), self.moving, self.moving.grid)
        self._fused_origin = moving.grid.origin
        self._fused_inverse_linear = moving.grid.lps_to_index[:3, :3]
        self._fused_spacing = moving.grid.spacing_xyz

    def _fused_cache_matches(self):
        """Check constant-size metadata before entering unchecked JIT indexing."""
        state = self._fused_cache_state
        if state is None or self.moving_mask is not None:
            return False
        if (self.points is not state[0] or self.fixed_values is not state[2] or
                self.moving is not state[6] or self.moving.grid is not state[7]):
            return False
        if ((self.points.shape, self.points.dtype, self.points.strides) != state[1] or
                (self.fixed_values.shape, self.fixed_values.dtype, self.fixed_values.strides) != state[3]):
            return False
        # Only four scalar range endpoints are inspected. Content scans of the
        # cached point/fixed-value arrays remain outside the per-evaluation path.
        numeric = (int, float, np.integer, np.floating)
        try:
            if not isinstance(self.bins, numeric) or not np.isfinite(self.bins) or self.bins != state[4]:
                return False
            if len(self.fixed_range) != 2 or len(self.moving_range) != 2:
                return False
            fmin, fmax = self.fixed_range
            mmin, mmax = self.moving_range
            if not all(isinstance(value, numeric) for value in (fmin, fmax, mmin, mmax)):
                return False
            if fmin != state[5][0] or fmax != state[5][1]:
                return False
            if not np.isfinite(mmin) or not np.isfinite(mmax) or mmax <= mmin:
                return False
        except (AttributeError, IndexError, TypeError, ValueError):
            return False
        return True

    def evaluate(self, model, parameters, derivative=True):
        transform = model.transform(parameters)
        if (self._fused_fixed_order is not None and type(model) in (LinearModel, ComposedLinearModel) and
                self._fused_cache_matches()):
            linear_derivatives, translation_derivatives = model.derivatives(parameters) if derivative else (None, None)
            result, geometric_count = evaluate_linear_mattes(
                np.asarray(self.moving.data, dtype=np.float64), self.points, transform,
                self._fused_origin, self._fused_inverse_linear, self._fused_spacing,
                self.fixed_values, *self._fused_fixed_order, int(self.bins), self.moving_range,
                model.center if derivative else None, linear_derivatives, translation_derivatives)
            if geometric_count == 0:
                raise NoValidSamplesError("All samples map outside moving image")
            value, descent, joint, fixed_pdf, moving_pdf, pdf_derivative, valid = result
            if valid == 0:
                raise NoValidSamplesError("Mattes metric has no valid moving samples")
            return MetricEvaluation(float(value), descent, valid, {
                "joint_pdf": joint, "fixed_pdf": fixed_pdf, "moving_pdf": moving_pdf,
                "pdf_derivative": pdf_derivative})
        mapped = transform.map_points(self.points)
        moving_values, valid = sample_image(self.moving, mapped)
        if self.moving_mask is not None:
            valid &= sample_mask(self.moving_mask, mapped)
        if not valid.any():
            raise NoValidSamplesError("All samples map outside moving image")
        fixed, moving = self.fixed_values[valid], moving_values[valid]
        intensity_derivatives = None
        if derivative:
            gradient = gradient_at_points(self.moving, mapped[valid])
            intensity_derivatives = model.intensity_derivatives(parameters, self.points[valid], gradient)
        return evaluate_mattes_samples(fixed, moving, intensity_derivatives, self.bins,
                                       self.fixed_range, self.moving_range)


class MeanSquaresMetric(MattesMetric):
    """Debug metric with the same sampling and physical-transform conventions."""

    def evaluate(self, model, parameters, derivative=True):
        mapped = model.transform(parameters).map_points(self.points)
        moving, valid = sample_image(self.moving, mapped)
        if self.moving_mask is not None:
            valid &= sample_mask(self.moving_mask, mapped)
        if not valid.any():
            raise NoValidSamplesError("All MSE samples lie outside the moving image")
        derivatives = np.zeros((int(valid.sum()), 0))
        if derivative:
            derivatives = model.intensity_derivatives(parameters, self.points[valid],
                                                      gradient_at_points(self.moving, mapped[valid]))
        value, descent = mean_squares_samples(self.fixed_values[valid], moving[valid], derivatives)
        return MetricEvaluation(float(value), descent, int(valid.sum()), {})
