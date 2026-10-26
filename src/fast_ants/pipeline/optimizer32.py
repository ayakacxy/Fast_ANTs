"""Float1 linear optimizer and physical-shift scales for additive models.

Adapted control rules from ITK 5.4.7, Copyright NumFOCUS, Apache-2.0:
ConjugateGradientLineSearchOptimizerv4, GradientDescentLineSearchOptimizerv4,
GradientDescentOptimizerv4 and WindowConvergenceMonitoringFunction.

Additional type definitions audited against official ITK v5.4.1 (the installed
oracle's ITK version): ObjectToObjectOptimizerBase, OptimizerParameterScalesEstimator,
GradientDescentOptimizerv4, itkPoint and VNL vnl_c_vector. Source URLs are under
https://github.com/InsightSoftwareConsortium/ITK/tree/v5.4.1/Modules .
Parameters, directions, scales, inner products and optimizer convergence values
are float. Point-distance arithmetic and window spline data are double. Native
float FMA preserves the contraction boundary of the CPU reference build; there
is no extended-precision emulation. End-to-end agreement still depends on the
caller supplying float1-compatible transform models, metrics and sampling.

Window monitoring is mixed precision in ITK: float energy inputs, double total
energy and spline data, float spline weights. It is not double-single emulation.
The helpers below operate only on small host-side control vectors/scalars; they
make no end-to-end float1 accuracy or performance claim.
"""

import numpy as np
from numba import njit

from fast_ants.core.results import NoValidSamplesError
from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.core.results import OptimizationResult
from fast_ants.pipeline.optimizer64 import corner_points
from fast_ants.pipeline.optimizer64 import _maximum_shift
from fast_ants.core.arithmetic32 import fma32


FLOAT1_EPSILON = np.float32(np.finfo(np.float32).eps)
FLOAT1_MAX = np.float32(np.finfo(np.float32).max)
# The C++ constructor assigns m_Phi before calculating m_Resphi. Rounding
# 2 - 1.618034 just once gives a different float, so preserve both assignments.
_PHI = np.float32(1.618034)
_RESPHI = np.float32(np.float32(2) - _PHI)


def _finite_scalar32(value, name):
    with np.errstate(over="ignore", invalid="ignore"):
        scalar = np.float32(value)
    if np.ndim(scalar) or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite float32 scalar")
    return scalar


def _vector32(values, name):
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite one-dimensional float32 vector")
    return vector


def polak_ribiere_gamma32(numerator, denominator):
    """Apply float-template epsilon and restart rules to an existing reduction.

    VNL's ``inner_product<float>`` returns float from a sequential float sum.
    Both operands of the subsequent division therefore remain float.
    """
    numerator = _finite_scalar32(numerator, "gamma numerator")
    denominator = _finite_scalar32(denominator, "gamma denominator")
    gamma = np.float32(0)
    if denominator > FLOAT1_EPSILON:
        with np.errstate(over="ignore"):
            gamma = np.float32(numerator / denominator)
    if gamma < np.float32(0) or gamma > np.float32(5):
        gamma = np.float32(0)
    return gamma


def conjugate_direction32(gradient, previous_direction, gamma):
    """Form a new owned float direction, with multiplication preceding addition."""
    gradient = _vector32(gradient, "gradient")
    previous = _vector32(previous_direction, "previous direction")
    if gradient.shape != previous.shape:
        raise ValueError("Gradient and previous direction must have the same shape")
    gamma = _finite_scalar32(gamma, "gamma")
    return np.add(gradient, np.multiply(previous, gamma, dtype=np.float32), dtype=np.float32)


def apply_parameter_step32(parameters, direction, learning_rate):
    """Return owned parameters after the float gradient scaling and update.

    This supports additive parameter updates only. Rigid/affine model matrices,
    their Jacobians and transform-specific updates are not implemented here.
    """
    parameters = _vector32(parameters, "parameters")
    direction = _vector32(direction, "direction")
    if parameters.shape != direction.shape:
        raise ValueError("Parameters and direction must have the same shape")
    rate = _finite_scalar32(learning_rate, "learning rate")
    return np.add(parameters, np.multiply(direction, rate, dtype=np.float32), dtype=np.float32)


def initial_learning_rate32(maximum_physical_step, estimated_step_scale):
    """Consume an audited scale estimate, preserving ITK's float epsilon branch."""
    maximum = _finite_scalar32(maximum_physical_step, "maximum physical step")
    scale = _finite_scalar32(estimated_step_scale, "estimated step scale")
    if maximum <= np.float32(0) or scale < np.float32(0):
        raise ValueError("Physical step must be positive and estimated scale nonnegative")
    if scale <= FLOAT1_EPSILON:
        return np.float32(1)
    return np.float32(maximum / scale)


@njit(cache=True, fastmath=False)
def _inner_product32(first, second):
    value = np.float32(0)
    for i in range(first.size):
        value = fma32(first[i], second[i], value)
    return value


@njit(cache=True, fastmath=False)
def _golden_point32(left, middle, right, resphi):
    if right-middle > middle-left:
        return fma32(resphi, np.float32(right-middle), middle)
    return fma32(-resphi, np.float32(middle-left), middle)


def _mapped_corners32(model, parameters, points):
    # Scale estimation evaluates the active CompositeTransform, whose float
    # point state rounds at each member. Output collapsing is an IO operation.
    evaluate = getattr(model, "evaluation_transform", model.transform)
    mapped = np.asarray(evaluate(parameters).map_points(points), dtype=np.float32)
    if mapped.shape != points.shape or not np.isfinite(mapped).all():
        raise ValueError("Scale estimation requires finite three-dimensional mapped corners")
    return mapped


def _point_shift32(new, old):
    # itk::Point<float>::RealType is double. Cast each point *before*
    # subtraction, take its double Euclidean distance, then store float scales.
    distance = _maximum_shift(new.astype(np.float64)-old.astype(np.float64))
    return np.float32(distance)


def parameter_scales32(model, parameters, grid, variation=.01):
    """Physical-shift scales for global rigid/affine additive parameters.

    The supplied model owns float1 transform arithmetic. Casting a double-model
    result to float here cannot substitute for a float matrix/offset evaluation.
    """
    parameters = _vector32(parameters, "parameters")
    variation = _finite_scalar32(variation, "parameter variation")
    if variation <= np.float32(0):
        raise ValueError("Parameter variation must be positive")
    points = np.asarray(corner_points(grid), dtype=np.float32)
    original = _mapped_corners32(model, parameters, points)
    shifts = np.empty(parameters.size, np.float32)
    for index in range(parameters.size):
        changed = parameters.copy()
        changed[index] = np.float32(changed[index]+variation)
        shifts[index] = _point_shift32(_mapped_corners32(model, changed, points), original)
    positive = shifts[shifts > FLOAT1_EPSILON]
    if not positive.size:
        return np.ones(parameters.size, np.float32)
    shifts = np.where(shifts > FLOAT1_EPSILON, shifts, positive.min()).astype(np.float32)
    squared_variation = np.float32(variation*variation)
    normalization = np.float32(np.float32(1)/squared_variation)
    return np.multiply(np.multiply(shifts, shifts, dtype=np.float32), normalization, dtype=np.float32)


def step_scale32(model, parameters, direction, grid, variation=.01):
    """Float scale estimator for a global-transform step, including float epsilon."""
    parameters = _vector32(parameters, "parameters")
    direction = _vector32(direction, "direction")
    if parameters.shape != direction.shape or not parameters.size:
        raise ValueError("Scale estimation requires nonempty matching parameter/direction vectors")
    maximum = np.max(np.abs(direction))
    if maximum <= FLOAT1_EPSILON:
        return np.float32(0)
    variation = _finite_scalar32(variation, "parameter variation")
    if variation <= np.float32(0):
        raise ValueError("Parameter variation must be positive")
    factor = np.float32(variation/maximum)
    points = np.asarray(corner_points(grid), dtype=np.float32)
    original = _mapped_corners32(model, parameters, points)
    changed = apply_parameter_step32(parameters, direction, factor)
    distance = _point_shift32(_mapped_corners32(model, changed, points), original)
    return np.float32(distance/factor)


def golden_section_search32(function, a, b, c, epsilon=.2, maximum_iterations=20):
    """Return ``(float32 rate, evaluation count)`` for a float1 bracket.

    Defaults follow the ANTs registration helper's epsilon 0.2; callers supply
    the helper's bracket ``(0, learning_rate, 2 * learning_rate)``. The callback
    receives an ``np.float32`` rate and must return a finite value or FLOAT1_MAX
    for an invalid sample domain. Exceptions propagate. No transform is updated
    inside this helper: callers must evaluate each rate from the same base.
    """
    left = _finite_scalar32(a, "left bracket")
    middle = _finite_scalar32(b, "middle bracket")
    right = _finite_scalar32(c, "right bracket")
    epsilon = _finite_scalar32(epsilon, "line-search epsilon")
    if not left <= middle <= right or epsilon <= np.float32(0):
        raise ValueError("Line search requires an ordered bracket and positive epsilon")
    if int(maximum_iterations) != maximum_iterations or maximum_iterations < 0:
        raise ValueError("Maximum line-search iterations must be a nonnegative integer")
    middle_value = FLOAT1_MAX
    depth = evaluations = 0
    while True:
        # The upstream recursion checks '>' before incrementing its counter.
        if depth > maximum_iterations:
            return np.float32(np.float32(right + left) / np.float32(2)), evaluations
        depth += 1
        longer_right = np.float32(right - middle) > np.float32(middle - left)
        point = np.float32(_golden_point32(left, middle, right, _RESPHI))
        tolerance = np.float32(epsilon * np.float32(np.abs(middle) + np.abs(point)))
        if np.abs(np.float32(right - left)) < tolerance:
            return np.float32(np.float32(right + left) / np.float32(2)), evaluations
        pair = getattr(function, 'evaluate_pair', None)
        if middle_value == FLOAT1_MAX and pair is not None:
            # These two costs depend on the same immutable base/direction,
            # not on one another. Device metrics can submit them together.
            first, second = pair(point, middle)
            value = _finite_scalar32(first, "line-search metric")
            middle_value = _finite_scalar32(second, "line-search metric")
            evaluations += 2
        else:
            value = _finite_scalar32(function(point), "line-search metric")
            evaluations += 1
        if middle_value == FLOAT1_MAX and pair is None:
            middle_value = _finite_scalar32(function(middle), "line-search metric")
            evaluations += 1
        if value < middle_value:
            if longer_right:
                left, middle = middle, point
            else:
                middle, right = point, middle
            middle_value = value
        elif longer_right:
            right = point
        elif value == FLOAT1_MAX:
            # ITK keeps the lower bracket if both metrics have no valid points.
            middle, right, middle_value = point, middle, value
        else:
            left = point


class WindowConvergence32(WindowConvergence):
    """Float energy input to ITK's mixed-precision convergence monitor.

    ``add`` returns the monitor's *double* convergence value, including double
    max until the window fills. ``conjugate_gradient32`` casts that result to
    float optimizer storage before comparing its float threshold. Zero energy
    inherits the existing project safeguard; ITK's source has no explicit zero
    denominator guard, so this case is not a claim of upstream equivalence.
    """

    def add(self, value):
        return super().add(float(_finite_scalar32(value, "energy")))


def conjugate_gradient32(metric, model, initial_parameters, grid, iterations=100, gradient_step=.1,
                         convergence_threshold=1e-6, convergence_window=10, monitor=True, *,
                         scales=None, estimate_step_scale=None, progress=None):
    """ANTs float1 CG controls for a caller-supplied additive transform model.

    The positional interface and result match ``optimizers.conjugate_gradient``.
    ``metric.evaluate(model, parameters, derivative)`` supplies value, derivative
    and valid_points, all consumed at float1 assignment boundaries. ``model``
    must evaluate a float1 transform from float32 parameters. Optional ``scales``
    and ``estimate_step_scale(parameters, scaled_gradient)`` permit an externally
    audited scale estimator; otherwise the physical corner estimators above run.
    No displacement-field/local-support optimizer is implied by this interface.
    ``progress`` optionally observes completed history entries; it does not
    request metric evaluations or change optimizer controls.
    """
    if int(iterations) != iterations or iterations < 0:
        raise ValueError("Iteration budget must be a nonnegative integer")
    gradient_step = _finite_scalar32(gradient_step, "physical step")
    if gradient_step <= np.float32(0):
        raise ValueError("Physical step must be positive")
    threshold = _finite_scalar32(convergence_threshold, "convergence threshold")
    parameters = _vector32(initial_parameters, "parameters").copy()
    if not parameters.size:
        raise ValueError("Optimizer requires nonempty parameters")
    scales = (parameter_scales32(model, parameters, grid) if scales is None
              else _vector32(scales, "parameter scales").copy())
    if scales.shape != parameters.shape or np.any(scales <= FLOAT1_EPSILON):
        raise ValueError("Parameter scales must match parameters and exceed float epsilon")
    factor = np.divide(np.float32(1), scales, dtype=np.float32)
    last_gradient = np.zeros_like(parameters)
    conjugate = np.zeros_like(parameters)
    learning_rate = gradient_step
    convergence = WindowConvergence32(convergence_window)
    best_value, best_parameters = FLOAT1_MAX, parameters.copy()
    history, reason = [], "maximum_iterations"
    for iteration in range(int(iterations)):
        evaluation = metric.evaluate(model, parameters, True)
        value = _finite_scalar32(evaluation.value, "metric")
        derivative = _vector32(evaluation.derivative, "metric derivative")
        if derivative.shape != parameters.shape:
            raise ValueError("Metric derivative must match parameter shape")
        gradient = np.multiply(derivative, factor, dtype=np.float32)
        convergence_value = np.float32(np.inf)
        if monitor:
            # NumericTraits<float>::RealType makes the monitor double, while
            # GradientDescentOptimizerv4<float>::m_ConvergenceValue is float.
            with np.errstate(over="ignore"):
                convergence_value = np.float32(convergence.add(value))
        entry = {"iteration": iteration+1, "value": float(value),
                 "valid_points": evaluation.valid_points, "convergence": float(convergence_value)}
        if monitor and convergence_value <= threshold:
            reason = "window_converged"
            history.append(entry)
            if progress is not None:progress(entry)
            break
        if iteration == 0:
            scale = (step_scale32(model, parameters, gradient, grid) if estimate_step_scale is None
                     else estimate_step_scale(parameters, gradient))
            learning_rate = initial_learning_rate32(gradient_step, scale)
        denominator = np.float32(_inner_product32(last_gradient, last_gradient))
        numerator = np.float32(0)
        if denominator > FLOAT1_EPSILON:
            difference = np.subtract(gradient, last_gradient, dtype=np.float32)
            numerator = np.float32(_inner_product32(difference, gradient))
        gamma = polak_ribiere_gamma32(numerator, denominator)
        last_gradient = gradient.copy()
        conjugate = conjugate_direction32(gradient, conjugate, gamma)
        base = parameters.copy()

        def cost(rate):
            try:
                trial = apply_parameter_step32(base, conjugate, rate)
                raw_value = metric.evaluate(model, trial, False).value
                with np.errstate(over="ignore", invalid="ignore"):
                    trial_value = np.float32(raw_value)
                return trial_value if np.isfinite(trial_value) else FLOAT1_MAX
            except (NoValidSamplesError, ValueError):
                return FLOAT1_MAX

        if getattr(metric, 'batch_value_evaluations', False):
            def pair_cost(first, second):
                trials=[apply_parameter_step32(base, conjugate, rate) for rate in (first, second)]
                return metric.evaluate_values_many(model, trials)
            cost.evaluate_pair=pair_cost

        learning_rate, evaluations = golden_section_search32(
            cost, np.float32(0), learning_rate, np.float32(np.float32(2)*learning_rate))
        parameters = apply_parameter_step32(base, conjugate, learning_rate)
        # ITK compares the metric from before the update but retains the
        # parameters after that update for ReturnBestParametersAndValue.
        if value < best_value:
            best_value, best_parameters = value, parameters.copy()
        entry.update({"learning_rate": float(learning_rate), "gamma": float(gamma),
                      "line_search_evaluations": evaluations,
                      "parameters_after_step": parameters.tolist()})
        history.append(entry)
        if progress is not None:progress(entry)
    return OptimizationResult(best_parameters, history, reason, scales)
