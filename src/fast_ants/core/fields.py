"""Public LPS-mm displacement data and explicitly invoked CPU sampling."""
from dataclasses import dataclass
import numpy as np
from fast_ants.core.types import Grid
from fast_ants.core.types import points_array
@dataclass(eq=False)
class DisplacementField:
    data: np.ndarray
    grid: Grid
    inverse_field: "DisplacementField | None" = None

    def __post_init__(self):
        from fast_ants.backends.cpu.common.reductions import all_finite
        values = np.asarray(self.data)
        if values.shape != (3, *self.grid.shape_zyx) or values.dtype not in (np.float32, np.float64):
            raise ValueError("Displacement data must be float32/float64 [3,z,y,x], in LPS mm")
        values=np.ascontiguousarray(values)
        if not all_finite(values):
            raise ValueError("Displacement field must be finite")
        self.data=values

    def sample(self, points_lps):
        from fast_ants.backends.cpu.common.geometry import physical_to_indices; from fast_ants.backends.cpu.fp64.sampling_kernels import sample_vector
        points = points_array(points_lps)
        flat = np.ascontiguousarray(points.reshape(-1, 3))
        index = physical_to_indices(flat, self.grid.lps_to_index[:3, :3], self.grid.origin)
        values, valid = sample_vector(np.asarray(self.data, dtype=np.float64), index)
        return values.reshape(points.shape), valid.reshape(points.shape[:-1])

    def map_points(self, points_lps):
        from fast_ants.backends.cpu.fp64.sampling_kernels import add_vectors
        points = points_array(points_lps)
        values, valid = self.sample(points)
        return add_vectors(np.ascontiguousarray(points.reshape(-1, 3)), values.reshape(-1, 3)).reshape(points.shape)

    def inverse(self):
        if self.inverse_field is None:
            raise ValueError("Inverse field is not available; supply or compute it explicitly")
        return self.inverse_field

