from fast_ants.core.results import OptimizationResult
from fast_ants.pipeline.convergence import WindowConvergence
"""Physical-shift scales, ITK-style conjugate-gradient golden-section search.

Adapted algorithm/control rules from ITK 5.4.7, Copyright NumFOCUS,
Apache-2.0. This modified Python implementation uses explicit small matrices,
Numba metrics and a two-control-point specialization of window convergence.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np
from numba import njit

from fast_ants.core.results import NoValidSamplesError




@njit(cache=True, fastmath={"contract"})
def _corner_points(index_to_lps, shape):
    points = np.empty((8, 3))
    for n in range(8):
        for i in range(3):
            value = index_to_lps[i, 3]
            for j in range(3):
                value += index_to_lps[i, j]*((shape[j]-1)*((n >> j) & 1))
            points[n, i] = value
    return points


def corner_points(grid):
    # The scale estimator uses the discrete-index overload (origin first).
    return _corner_points(grid.index_to_lps, grid.shape_xyz)


@njit(cache=True, fastmath={"contract"})
def _maximum_shift(displacements):
    maximum = 0.0
    for n in range(displacements.shape[0]):
        squared = 0.0
        for j in range(3):
            squared += displacements[n, j]*displacements[n, j]
        maximum = max(maximum, np.sqrt(squared))
    return maximum


@njit(cache=True, fastmath={"contract"})
def _inner_product(a, b):
    total = 0.0
    for i in range(a.size):
        total += a[i]*b[i]
    return total


@njit(cache=True, fastmath={"contract"})
def _golden_point(left, middle, right, resphi):
    if right-middle > middle-left:
        return middle+resphi*(right-middle)
    return middle-resphi*(middle-left)


def parameter_scales(model, parameters, grid, variation=.01):
    points = corner_points(grid)
    original = model.transform(parameters).map_points(points)
    shifts = np.empty(len(parameters))
    for p in range(len(parameters)):
        delta = np.zeros(len(parameters)); delta[p] = variation
        displacement = model.transform(parameters+delta).map_points(points)-original
        shifts[p] = _maximum_shift(displacement)
    positive = shifts[shifts > np.finfo(float).eps]
    if not len(positive):
        return np.ones(len(parameters))
    shifts = np.where(shifts > np.finfo(float).eps, shifts, positive.min())
    return shifts*shifts*(1.0/(variation*variation))


def step_scale(model, parameters, direction, grid, variation=.01):
    maximum = float(np.max(np.abs(direction)))
    if maximum <= np.finfo(float).eps:
        return 0.0
    factor = variation/maximum
    points = corner_points(grid)
    old = model.transform(parameters).map_points(points)
    new = model.transform(parameters+direction*factor).map_points(points)
    return float(_maximum_shift(new-old)/factor)


def golden_section_search(function, a, b, c, epsilon=.2, maximum_iterations=20):
    resphi = 2-1.618034
    sentinel = np.finfo(float).max
    evaluations = 0

    def search(left, middle, right, middle_value=sentinel, depth=0):
        nonlocal evaluations
        if depth > maximum_iterations:
            return (right+left)/2
        point = _golden_point(left, middle, right, resphi)
        if abs(right-left) < epsilon*(abs(middle)+abs(point)):
            return (right+left)/2
        value = function(point); evaluations += 1
        if middle_value == sentinel:
            middle_value = function(middle); evaluations += 1
        if value < middle_value:
            if right-middle > middle-left:
                return search(middle, point, right, value, depth+1)
            return search(left, point, middle, value, depth+1)
        if right-middle > middle-left:
            return search(left, middle, point, middle_value, depth+1)
        if value == sentinel:
            return search(left, point, middle, value, depth+1)
        return search(point, middle, right, middle_value, depth+1)

    return search(float(a), float(b), float(c)), evaluations




def conjugate_gradient(metric, model, initial_parameters, grid, iterations=100, gradient_step=.1,
                       convergence_threshold=1e-6, convergence_window=10, monitor=True):
    if int(iterations) != iterations or iterations < 0 or gradient_step <= 0 or not np.isfinite(gradient_step):
        raise ValueError("Invalid iteration budget or physical step size")
    parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    scales = parameter_scales(model, parameters, grid)
    last_gradient, conjugate = np.zeros_like(parameters), np.zeros_like(parameters)
    learning_rate = float(gradient_step)
    convergence = WindowConvergence(convergence_window)
    best_value, best_parameters = np.finfo(float).max, parameters.copy()
    history, reason = [], "maximum_iterations"
    for iteration in range(int(iterations)):
        evaluation = metric.evaluate(model, parameters, True)
        value, gradient = evaluation.value, evaluation.derivative*(1.0/scales)
        convergence_value = convergence.add(value)
        entry = {"iteration": iteration+1, "value": float(value), "valid_points": evaluation.valid_points,
                 "convergence": float(convergence_value)}
        if monitor and convergence_value <= convergence_threshold:
            reason = "window_converged"
            history.append(entry)
            break
        if iteration == 0:
            scale = step_scale(model, parameters, gradient, grid)
            learning_rate = 1.0 if scale <= np.finfo(float).eps else gradient_step/scale
        denominator = _inner_product(last_gradient, last_gradient)
        gamma = float(_inner_product(gradient-last_gradient, gradient)/denominator) if denominator > np.finfo(float).eps else 0.0
        if gamma < 0 or gamma > 5:
            gamma = 0.0
        last_gradient = gradient.copy()
        conjugate = gradient+gamma*conjugate
        base = parameters.copy()

        def cost(rate):
            try:
                result = metric.evaluate(model, base+rate*conjugate, False).value
                return float(result) if np.isfinite(result) else np.finfo(float).max
            except (NoValidSamplesError, ValueError):
                return np.finfo(float).max

        learning_rate, evaluations = golden_section_search(cost, 0, learning_rate, 2*learning_rate)
        parameters = base+learning_rate*conjugate
        # Match the upstream ReturnBestParametersAndValue bookkeeping: the
        # comparison value precedes the step, and the saved position follows it.
        if value < best_value:
            best_value, best_parameters = float(value), parameters.copy()
        entry.update({"learning_rate": learning_rate, "gamma": gamma, "line_search_evaluations": evaluations,
                      "parameters_after_step": parameters.tolist()})
        history.append(entry)
    return OptimizationResult(best_parameters, history, reason, scales)
