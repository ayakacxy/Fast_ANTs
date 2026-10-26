"""Euler3D (default ITK Z-X-Y order) and centered affine parameter models."""

from dataclasses import dataclass
import numpy as np
from numba import njit

from fast_ants.core.transforms import AffineTransform


def rotation_and_derivatives(angles):
    x, y, z = angles
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    dx = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]])
    dy = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]])
    dz = np.array([[-sz, -cz, 0], [cz, -sz, 0], [0, 0, 0]])
    return rz@rx@ry, np.stack([rz@dx@ry, rz@rx@dy, dz@rx@ry])


@njit(cache=True, fastmath=False)
def intensity_parameter_derivatives(points, center, gradients, linear_derivatives, translation_derivatives):
    count = linear_derivatives.shape[0]
    output = np.empty((points.shape[0], count), np.float64)
    for n in range(points.shape[0]):
        for parameter in range(count):
            total = 0.0
            for c in range(3):
                spatial = translation_derivatives[parameter, c]
                for j in range(3):
                    spatial += linear_derivatives[parameter, c, j]*(points[n, j]-center[j])
                total += gradients[n, c]*spatial
            output[n, parameter] = total
    return output


@dataclass
class LinearModel:
    kind: str
    center: np.ndarray

    def __post_init__(self):
        if self.kind not in ("Rigid", "Affine"):
            raise ValueError("Linear model must be Rigid or Affine")
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.center.shape != (3,) or not np.isfinite(self.center).all():
            raise ValueError("Linear model center must be a finite LPS point")

    @property
    def parameter_count(self):
        return 6 if self.kind == "Rigid" else 12

    def identity_parameters(self):
        return np.zeros(6) if self.kind == "Rigid" else np.r_[np.eye(3).ravel(), np.zeros(3)]

    def transform(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self.parameter_count,) or not np.isfinite(parameters).all():
            raise ValueError("Invalid linear transform parameters")
        linear = rotation_and_derivatives(parameters[:3])[0] if self.kind == "Rigid" else parameters[:9].reshape(3, 3)
        return AffineTransform.centered(linear, parameters[-3:], self.center)

    def parameters_from_affine(self, transform):
        if self.kind != "Affine":
            raise ValueError("Only an affine model accepts a general existing affine")
        translation = transform.matrix[:3, 3] - self.center + transform.matrix[:3, :3]@self.center
        return np.r_[transform.matrix[:3, :3].ravel(), translation]

    def derivatives(self, parameters):
        count = self.parameter_count
        linear = np.zeros((count, 3, 3))
        translation = np.zeros((count, 3))
        if self.kind == "Rigid":
            linear[:3] = rotation_and_derivatives(parameters[:3])[1]
        else:
            for i in range(9):
                linear[i, i//3, i%3] = 1
        translation[-3:] = np.eye(3)
        return linear, translation

    def intensity_derivatives(self, parameters, points, moving_gradients):
        linear, translation = self.derivatives(parameters)
        return intensity_parameter_derivatives(points, self.center, moving_gradients, linear, translation)


@dataclass
class ComposedLinearModel(LinearModel):
    """Optimize an incremental map before a frozen moving-initial transform.

    With ANTs initialize-transforms-per-stage=0, the new stage is initialized
    to identity and the previous linear map remains in the composite.
    """
    base: AffineTransform | None = None

    def transform(self, parameters):
        local = super().transform(parameters)
        if self.base is None:
            return local
        return AffineTransform(self.base.matrix@local.matrix, self.center)

    def derivatives(self, parameters):
        linear, translation = super().derivatives(parameters)
        if self.base is not None:
            linear = np.stack([self.base.matrix[:3, :3]@d for d in linear])
            translation = translation@self.base.matrix[:3, :3].T
        return linear, translation
