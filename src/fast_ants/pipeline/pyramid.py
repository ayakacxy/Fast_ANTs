"""Centered virtual grids and pyramid metadata, independent of execution."""
from dataclasses import dataclass
import numpy as np
from fast_ants.core.types import Grid
from fast_ants.core.types import Image

def shrink_grid(grid, factors):
    factors = np.full(3, factors) if np.ndim(factors) == 0 else np.asarray(factors)
    if factors.shape != (3,) or np.any(factors != np.floor(factors)) or np.any(factors < 1):
        raise ValueError("Shrink factors must be one or three positive integers")
    factors = factors.astype(int)
    shape_xyz = np.maximum(np.asarray(grid.shape_xyz)//factors, 1)
    matrix = grid.index_to_lps.copy()
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(factors)
    matrix[:3, 3] = grid.center - matrix[:3, :3] @ ((shape_xyz-1)/2)
    return Grid(tuple(int(n) for n in shape_xyz[::-1]), matrix)


@dataclass
class PyramidLevel:
    fixed: Image
    moving: Image
    virtual_grid: Grid
    shrink_factor: int | tuple
    sigma: float
    sigma_units: str

