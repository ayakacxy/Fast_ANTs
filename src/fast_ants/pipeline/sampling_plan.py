"""Explicit dense/regular/random virtual samples, persisted for reproducibility."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from fast_ants.backends.cpu.common.geometry import grid_points


@dataclass
class SamplingPlan:
    points_lps: np.ndarray
    strategy: str
    fraction: float
    seed: int

    def save(self, path):
        points = np.ascontiguousarray(self.points_lps, dtype=np.float64)
        metadata = {"strategy": self.strategy, "fraction": self.fraction, "seed": self.seed,
                    "coordinate_system": "LPS_mm", "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                    "rng": "numpy PCG64; stored points are authoritative, not an ITK RNG bitwise match"}
        np.savez_compressed(path, points_lps=points, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            points = np.ascontiguousarray(archive["points_lps"], dtype=np.float64)
            metadata = json.loads(str(archive["metadata"]))
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Invalid saved sample points")
        if hashlib.sha256(points.tobytes()).hexdigest() != metadata["points_sha256"]:
            raise ValueError("Saved sample point checksum mismatch")
        return cls(points, metadata["strategy"], metadata["fraction"], metadata["seed"])


def sample_virtual_domain(grid, strategy="None", fraction=1.0, seed=20261022):
    if strategy not in ("None", "Regular", "Random") or not 0 < fraction <= 1:
        raise ValueError("Sampling needs None/Regular/Random and fraction in (0,1]")
    points = grid_points(grid.index_to_lps, grid.shape_zyx, 0, grid.shape_zyx[0])
    if strategy == "None":
        if fraction != 1:
            raise ValueError("Dense sampling uses fraction=1")
        return SamplingPlan(points, strategy, fraction, seed)
    rng = np.random.default_rng(seed)
    if strategy == "Regular":
        points = points[::int(np.ceil(1/fraction))].copy()
    else:
        count = int(len(points)*fraction)
        if count == 0:
            raise ValueError("Sampling fraction selects zero samples")
        points = points[rng.integers(0, len(points), size=count)].copy()
    # Match the strategy's perturbation geometry: independent physical-axis
    # offsets, one third of the corresponding virtual spacing, including Regular.
    points += rng.normal(size=points.shape)*(grid.spacing_xyz/3)
    return SamplingPlan(np.ascontiguousarray(points), strategy, fraction, seed)
