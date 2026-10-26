"""Directed physical transforms; a composite lists maps in application order."""

from dataclasses import dataclass

import numpy as np

from fast_ants.backends.cpu.common.geometry import affine_points
from fast_ants.core.types import points_array


@dataclass(frozen=True, eq=False)
class AffineTransform:
    matrix: np.ndarray
    center: np.ndarray | None = None

    def __post_init__(self):
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12)):
            raise ValueError("Expected a finite affine 4x4 matrix")
        if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
            raise ValueError("Affine transform is singular")
        center = np.zeros(3) if self.center is None else np.array(self.center, dtype=np.float64, copy=True)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Transform center must contain three finite LPS coordinates")
        matrix.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "center", center)

    @classmethod
    def identity(cls):
        return cls(np.eye(4))

    @classmethod
    def centered(cls, linear, translation, center):
        linear = np.asarray(linear, dtype=np.float64)
        translation, center = np.asarray(translation, dtype=np.float64), np.asarray(center, dtype=np.float64)
        if linear.shape != (3, 3) or translation.shape != (3,) or center.shape != (3,):
            raise ValueError("Expected 3x3 linear matrix and two three-component vectors")
        matrix = np.eye(4)
        matrix[:3, :3] = linear
        matrix[:3, 3] = center + translation - linear @ center
        return cls(matrix, center)

    @property
    def translation(self):
        return self.matrix[:3, 3] - self.center + self.matrix[:3, :3] @ self.center

    def map_points(self, points_lps):
        points = points_array(points_lps)
        flat = np.ascontiguousarray(points.reshape(-1, 3))
        return affine_points(flat, self.matrix[:3, :3], self.center, self.translation).reshape(points.shape)

    def inverse(self):
        return AffineTransform(np.linalg.inv(self.matrix), self.center)


@dataclass(frozen=True)
class CompositeTransform:
    transforms: tuple

    def __post_init__(self):
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if any(not callable(getattr(t, "map_points", None)) for t in self.transforms):
            raise TypeError("Composite transforms must implement map_points")

    def map_points(self, points_lps):
        out = points_array(points_lps)
        for transform in self.transforms:
            out = transform.map_points(out)
        return out

    def inverse(self):
        return CompositeTransform(tuple(t.inverse() for t in reversed(self.transforms)))
