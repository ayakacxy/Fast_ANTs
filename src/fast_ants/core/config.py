"""Shared stage configuration and the original ANTs float-parsed parameters."""
from dataclasses import dataclass
import numpy as np
from fast_ants.pipeline.convergence import WindowConvergence
@dataclass
class LinearStage:
    kind: str
    iterations: tuple = (30, 15)
    shrink_factors: tuple = (2, 1)
    smoothing_sigmas: tuple = (1, 0)
    smoothing_units: str = "vox"
    gradient_step: float = .1
    metric: str = "MI"
    bins: int = 32
    convergence_threshold: float = 1e-7
    convergence_window: int = 8
    sampling_strategy: str = "None"
    sampling_fraction: float = 1.0

    def __post_init__(self):
        if self.kind not in ("Rigid", "Affine") or self.metric not in ("MI", "MeanSquares"):
            raise ValueError("Unsupported linear stage transform or metric")
        if not len(self.iterations) or len(self.iterations) != len(self.shrink_factors) or len(self.iterations) != len(self.smoothing_sigmas):
            raise ValueError("Iteration, shrink and smoothing schedules must have equal nonzero length")
        if not np.isfinite(self.gradient_step) or self.gradient_step<=0:
            raise ValueError("Linear gradient step must be finite and positive")
        # antsRegistrationTemplateHeader parses learningRate as float even
        # for --float 0; retain its effective parameter, not Python's .1 double.
        self.gradient_step=float(np.float32(self.gradient_step))

@dataclass
class SyNStage:
    iterations: tuple = (8,4)
    shrink_factors: tuple = (2,1)
    smoothing_sigmas: tuple = (1,0)
    smoothing_units: str = "vox"
    gradient_step: float = .1
    radius: int = 2
    update_variance: float = 3.
    total_variance: float = 0.
    convergence_threshold: float = 1e-7
    convergence_window: int = 8

    def __post_init__(self):
        if not len(self.iterations) or len(self.iterations)!=len(self.shrink_factors) or len(self.iterations)!=len(self.smoothing_sigmas):
            raise ValueError("SyN schedules must have equal nonzero lengths")
        if any(not isinstance(i,int) or i<0 for i in self.iterations):
            raise ValueError("SyN iterations must be nonnegative integers")
        if not np.isfinite([self.gradient_step,self.update_variance,self.total_variance,self.convergence_threshold]).all() or min(self.gradient_step,self.update_variance,self.total_variance,self.convergence_threshold)<0:
            raise ValueError("SyN numerical settings must be finite and nonnegative")
        if not isinstance(self.radius,int) or self.radius<0:
            raise ValueError("SyN CC radius must be a nonnegative integer")
        if self.smoothing_units not in ("vox","mm"):
            raise ValueError("SyN image smoothing units must be vox or mm")
        WindowConvergence(self.convergence_window)
        self.gradient_step=float(np.float32(self.gradient_step))
        self.update_variance=float(np.float32(self.update_variance))
        self.total_variance=float(np.float32(self.total_variance))

