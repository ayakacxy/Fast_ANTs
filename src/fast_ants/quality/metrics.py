"""Physical-unit numerical comparisons on an explicitly common grid."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _field_error_lengths(actual, reference):
    _, nz, ny, nx = actual.shape
    norms = np.empty(nz*ny*nx, dtype=np.float64)
    reference_norms = np.empty(norms.shape, dtype=np.float64)
    squared_total, maximum_component = 0.0, 0.0
    n = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                squared, reference_squared = 0.0, 0.0
                for c in range(3):
                    delta = np.float64(actual[c, z, y, x])-np.float64(reference[c, z, y, x])
                    squared += delta*delta
                    component = np.float64(reference[c,z,y,x])
                    reference_squared += component*component
                    maximum_component = max(maximum_component, abs(delta))
                norms[n] = np.sqrt(squared)
                reference_norms[n] = np.sqrt(reference_squared)
                squared_total += squared
                n += 1
    return norms, np.sqrt(squared_total/n), maximum_component, reference_norms


def field_difference(actual, reference, tolerance_mm=None, *, relative_scale_floor_mm=.001):
    if not actual.grid.same_geometry(reference.grid):
        raise ValueError("Displacement comparisons require the same declared physical grid")
    if not np.isfinite(actual.data).all() or not np.isfinite(reference.data).all():
        raise ValueError("Non-finite displacement cannot pass a numerical comparison")
    if tolerance_mm is not None and (not np.isfinite(tolerance_mm) or tolerance_mm <= 0):
        raise ValueError("Displacement tolerance must be finite and positive")
    if not np.isfinite(relative_scale_floor_mm) or relative_scale_floor_mm<=0:
        raise ValueError("Relative comparison needs a finite positive scale floor")
    norms, rms, component, reference_norms = _field_error_lengths(actual.data, reference.data)
    maximum = float(norms.max())
    relative = norms/np.maximum(reference_norms,relative_scale_floor_mm)
    result = {"point_count": int(norms.size), "max_l2_mm": maximum, "rms_l2_mm": float(rms),
              "p95_l2_mm":float(np.percentile(norms,95)),
              "p99_l2_mm": float(np.percentile(norms, 99)), "max_component_abs_mm": float(component),
              "relative_scale_floor_mm":float(relative_scale_floor_mm),
              "relative_max":float(relative.max()),"relative_p95":float(np.percentile(relative,95)),
              "relative_p99":float(np.percentile(relative,99)),"relative_rms":float(np.sqrt(np.mean(relative*relative)))}
    if tolerance_mm is not None:
        result.update({"tolerance_mm": float(tolerance_mm), "passed": maximum <= tolerance_mm})
    return result
