"""Shared metric, optimizer, inversion and registration result contracts."""
from dataclasses import dataclass
import numpy as np
from fast_ants.core.types import RegistrationResult
from fast_ants.core.fields import DisplacementField
from fast_ants.core.state import SyNState
@dataclass
class OptimizationResult:
    parameters: np.ndarray
    history: list
    stop_reason: str
    scales: np.ndarray

@dataclass
class SyNRegistrationResult(RegistrationResult):
    linear_transform: object = None
    forward_field: object = None
    inverse_field: object = None
    state: object = None
    configuration: dict | None = None
    quality: dict | None = None

class NoValidSamplesError(ValueError):
    pass


@dataclass
class MetricEvaluation:
    value: float
    derivative: np.ndarray  # ITK updates parameters by adding this descent direction.
    valid_points: int
    details: dict

    @property
    def gradient(self):
        return -self.derivative

@dataclass
class CCEvaluation:
    value: float
    update: DisplacementField
    valid_points: int
    statistics: dict | None = None

@dataclass
class InversionResult:
    field: DisplacementField
    history: list
    stop_reason: str
    final_residual: dict

@dataclass
class SyNStepResult:
    state: SyNState
    value: float
    details: dict
    intermediates: dict

