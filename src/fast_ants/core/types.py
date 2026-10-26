"""Image geometry: z,y,x arrays; x,y,z index vectors; LPS physical millimetres."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fast_ants.core.linalg import inverse3

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def points_array(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite (..., 3) physical or index coordinates")
    return points


class Transform(Protocol):
    """An explicitly directed map between LPS physical points."""

    def map_points(self, points_lps): ...
    def inverse(self): ...


@dataclass(frozen=True, eq=False)
class Grid:
    shape_zyx: tuple[int, int, int]
    index_to_lps: np.ndarray
    _lps_to_index: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        shape = tuple(self.shape_zyx)
        if len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
            raise ValueError("Grid shape must contain three positive integer z,y,x sizes")
        matrix = np.array(self.index_to_lps, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Grid geometry must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError("Grid geometry must be affine")
        axes = matrix[:3, :3]
        spacing = np.linalg.norm(axes, axis=0)
        if np.any(spacing <= 0) or abs(np.linalg.det(axes)) < 1e-12:
            raise ValueError("Singular image geometry")
        direction = axes / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
            raise ValueError("Sheared image grids are not supported; an orthogonal direction matrix is required")
        # ITK uses a 3x3 SVD inverse. A 4x4 LU inverse is mathematically
        # equivalent, but its different rounding can move fixed intensities
        # across Mattes' discontinuous histogram-bin boundaries.
        inverse = np.eye(4)
        inverse[:3, :3] = inverse3(axes)
        inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
        matrix.setflags(write=False)
        inverse.setflags(write=False)
        object.__setattr__(self, "shape_zyx", shape)
        object.__setattr__(self, "index_to_lps", matrix)
        object.__setattr__(self, "_lps_to_index", inverse)

    @classmethod
    def from_ras(cls, shape_xyz, affine_ras_mm):
        return cls(tuple(shape_xyz)[::-1], RAS_TO_LPS @ np.asarray(affine_ras_mm))

    @property
    def shape_xyz(self):
        return self.shape_zyx[::-1]

    @property
    def spacing_xyz(self):
        return np.linalg.norm(self.index_to_lps[:3, :3], axis=0)

    @property
    def direction(self):
        return self.index_to_lps[:3, :3] / self.spacing_xyz

    @property
    def origin(self):
        return self.index_to_lps[:3, 3].copy()

    @property
    def affine_ras(self):
        return RAS_TO_LPS @ self.index_to_lps

    @property
    def lps_to_index(self):
        return self._lps_to_index

    @property
    def center(self):
        return self.index_to_physical((np.asarray(self.shape_xyz) - 1) / 2)

    def index_to_physical(self, index_xyz):
        points = points_array(index_xyz)
        return points @ self.index_to_lps[:3, :3].T + self.index_to_lps[:3, 3]

    def physical_to_index(self, points_lps):
        # Subtract origin before rotation: preserves small offsets at large origins.
        points = points_array(points_lps)
        return (points - self.origin) @ self._lps_to_index[:3, :3].T

    def gradient_to_physical(self, gradient_index_xyz):
        gradient = points_array(gradient_index_xyz)
        return gradient @ self._lps_to_index[:3, :3]

    def same_geometry(self, other, atol=1e-6):
        return (self.shape_zyx == other.shape_zyx and
                np.allclose(self.index_to_lps, other.index_to_lps, rtol=0, atol=atol))


@dataclass(eq=False)
class Image:
    data: np.ndarray
    grid: Grid
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        data = np.asarray(self.data)
        if data.shape != self.grid.shape_zyx:
            raise ValueError(f"Image data {data.shape} does not match grid {self.grid.shape_zyx}")
        if data.dtype.kind not in "biuf" or not np.isfinite(data).all():
            raise ValueError("Image values must be finite real scalars")
        # Normalize non-native endian arrays before handing them to Numba.
        dtype = data.dtype.newbyteorder("=")
        self.data = np.ascontiguousarray(data, dtype=dtype)
        self.metadata = dict(self.metadata)


@dataclass
class RegistrationResult:
    fixed_grid: Grid
    moving_grid: Grid
    fixed_to_moving: Transform
    moving_to_fixed: Transform | None = None
    warped: Image | None = None
    history: list[dict] = field(default_factory=list)
