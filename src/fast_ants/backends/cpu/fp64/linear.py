from fast_ants.core.config import LinearStage
"""Independent multi-level rigid/affine registration using Mattes MI."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from fast_ants.core.types import RegistrationResult
from fast_ants.pipeline.models64 import ComposedLinearModel
from fast_ants.backends.cpu.fp64.metrics.mattes import MattesMetric
from fast_ants.backends.cpu.fp64.metrics.mattes import MeanSquaresMetric
from fast_ants.pipeline.optimizer64 import conjugate_gradient
from fast_ants.backends.cpu.fp64.preprocessing import prepare_level
from fast_ants.backends.cpu.fp64.preprocessing import preprocess_image
from fast_ants.backends.cpu.fp64.sampling import apply_image
from fast_ants.pipeline.sampling_plan import sample_virtual_domain
from fast_ants.core.transforms import AffineTransform


@njit(cache=True, fastmath={"contract"})
def _mass_physical(data, index_to_lps):
    # ImageMomentsCalculator accumulates physical first moments directly.
    # Converting the averaged index afterwards changes their FP64 rounding.
    mass, moments = 0.0, np.zeros(3)
    for z in range(data.shape[0]):
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                value = data[z, y, x]
                mass += value
                for i in range(3):
                    point = index_to_lps[i, 3]
                    point += index_to_lps[i, 0]*x
                    point += index_to_lps[i, 1]*y
                    point += index_to_lps[i, 2]*z
                    moments[i] += point*value
    return mass, moments


def initialization_point(image, feature=1):
    if feature == 0:
        return image.grid.center
    if feature == 2:
        return image.grid.origin
    if feature != 1:
        raise ValueError("Initialization feature must be geometric center (0), intensity moments (1), or origin (2)")
    mass, moments = _mass_physical(np.asarray(image.data, dtype=np.float64), image.grid.index_to_lps)
    if abs(mass) <= np.finfo(float).eps:
        raise ValueError("Intensity-moment initialization requires nonzero total mass")
    return moments/mass




def register_linear(fixed, moving, stages=None, initialization_feature=1, seed=20261022,
                    fixed_mask=None, moving_mask=None, output_directory=None):
    stages = (LinearStage("Rigid"), LinearStage("Affine")) if stages is None else tuple(stages)
    if not stages:
        raise ValueError("At least one linear stage is required")
    center, moving_center = initialization_point(fixed, initialization_feature), initialization_point(moving, initialization_feature)
    normalized_fixed, normalized_moving = preprocess_image(fixed), preprocess_image(moving)
    current = AffineTransform.centered(np.eye(3), moving_center-center, center)
    history = []
    out = None if output_directory is None else Path(output_directory)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for stage_index, stage in enumerate(stages):
        model = ComposedLinearModel(stage.kind, center, current)
        parameters = model.identity_parameters()
        for level, (iterations, shrink, sigma) in enumerate(zip(stage.iterations, stage.shrink_factors, stage.smoothing_sigmas)):
            images = prepare_level(normalized_fixed, normalized_moving, shrink, sigma, stage.smoothing_units)
            plan = sample_virtual_domain(images.virtual_grid, stage.sampling_strategy, stage.sampling_fraction, seed+stage_index*100+level)
            points = None if stage.sampling_strategy == "None" else plan.points_lps
            if out is not None and points is not None:
                plan.save(out/f"samples-stage{stage_index}-level{level}.npz")
            metric_type = MattesMetric if stage.metric == "MI" else MeanSquaresMetric
            metric = metric_type(images.fixed, images.moving, images.virtual_grid, stage.bins, points,
                                 fixed_mask, moving_mask)
            result = conjugate_gradient(metric, model, parameters, images.virtual_grid, iterations,
                                         stage.gradient_step, stage.convergence_threshold, stage.convergence_window)
            parameters = result.parameters
            current = model.transform(parameters)
            final = metric.evaluate(model, parameters, False)
            history.append({"stage": stage_index, "kind": stage.kind, "level": level,
                            "shrink_factor": shrink, "sigma": sigma, "sigma_units": stage.smoothing_units,
                            "sampling": stage.sampling_strategy, "sample_count": len(plan.points_lps),
                            "stop_reason": result.stop_reason, "scales": result.scales.tolist(),
                            "final_value_reevaluated": final.value, "iterations": result.history})
    warped = apply_image(moving, fixed, current)
    return RegistrationResult(fixed.grid, moving.grid, current, current.inverse(), warped, history)
