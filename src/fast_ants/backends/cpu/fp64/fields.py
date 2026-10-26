from fast_ants.core.results import InversionResult
"""Physical-mm field composition, ITK-style inversion and geometric QA."""

from dataclasses import dataclass
import numpy as np

from fast_ants.backends.cpu.fp64.field_kernels import compose_arrays
from fast_ants.backends.cpu.fp64.field_kernels import multiply_field
from fast_ants.backends.cpu.fp64.field_kernels import scaled_residual_norms
from fast_ants.backends.cpu.fp64.field_kernels import inverse_step
from fast_ants.backends.cpu.fp64.field_kernels import jacobian_determinants
from fast_ants.backends.cpu.fp64.field_kernels import _inversion_residual
from fast_ants.backends.cpu.fp64.field_kernels import _inversion_update
from fast_ants.backends.cpu.fp64.field_kernels import _inversion_residual_into
from fast_ants.backends.cpu.fp64.field_kernels import _inversion_update_into
from fast_ants.backends.cpu.fp64.field_kernels import compose_arrays_into
from fast_ants.backends.cpu.fp64.field_kernels import multiply_field_inplace
from fast_ants.backends.cpu.fp64.field_kernels import pack_field_into
from fast_ants.backends.cpu.fp64.local_fields import inversion_residual_into as _local_inversion_residual_into
from numba import get_num_threads
from fast_ants.backends.cpu.common.reductions import maximum_norm
from fast_ants.core.fields import DisplacementField


def zero_field(grid):
    return DisplacementField(np.zeros((3, *grid.shape_zyx), dtype=np.float64), grid)


def compose_fields(outer, inner, *, return_valid=False, arithmetic="itk", _output=None, _valid=None):
    """Return T_outer o T_inner on inner.grid: inner(x)+outer(x+inner(x)).

    Outer samples outside its support are zero; the optional mask identifies
    those samples. Default FP64 arithmetic preserves ITK's world-coordinate
    additions/subtraction. ``local`` avoids cancellation for precision audits.
    """
    if arithmetic not in ("itk", "local"):
        raise ValueError("Composition arithmetic must be itk or local")
    if _output is None:
        values, valid = compose_arrays(np.asarray(outer.data, dtype=np.float64),
                                      np.asarray(inner.data, dtype=np.float64),
                                      inner.grid.index_to_lps[:3, :3], inner.grid.origin, outer.grid.origin,
                                      np.ascontiguousarray(outer.grid.lps_to_index[:3, :3]), arithmetic=="itk")
    else:
        values, valid = compose_arrays_into(np.asarray(outer.data, dtype=np.float64),
                                      np.asarray(inner.data, dtype=np.float64),
                                      inner.grid.index_to_lps[:3, :3], inner.grid.origin, outer.grid.origin,
                                      np.ascontiguousarray(outer.grid.lps_to_index[:3, :3]), arithmetic=="itk",_output,_valid,inner.data.size>=49152 and get_num_threads()>1)
    field = DisplacementField(values, inner.grid)
    return (field, valid) if return_valid else field


def resample_field(field, reference_grid):
    """Resample LPS-mm components; a change of spacing does not scale vectors."""
    return compose_fields(field, zero_field(reference_grid), arithmetic="local")


def scale_update(field, learning_rate, *, _inplace=False):
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("Learning rate must be finite and nonnegative")
    maximum = float(maximum_norm(field.data, field.grid.spacing_xyz))
    factor = float(learning_rate/maximum if maximum > 0 else learning_rate)
    return DisplacementField(multiply_field_inplace(field.data,factor,field.data.size>=49152 and get_num_threads()>1) if _inplace else multiply_field(field.data, factor), field.grid), {"maximum_spacing_scaled_norm":maximum, "scale":factor}




class _InversionWorkspace:
    """Private scratch; every value is overwritten before it is consumed."""
    def __init__(self,shape,packed=False):
        self.residual=np.empty((3,*shape),np.float64)
        self.norms=np.empty(shape,np.float64)
        self.finite=np.empty(shape[:2],np.bool_)
        self.partials=np.empty((shape[0]*shape[1],2),np.float64)
        self.packed=np.empty((*shape,3),np.float64) if packed else None


def invert_field(field, initial=None, *, max_iterations=20, mean_tolerance=.001,
                 max_tolerance=.1, enforce_boundary=True, arithmetic="local", _workspace=None, _output=None):
    """Match InvertDisplacementFieldImageFilter, including its AND continuation.

    Error norms use LPS components divided by the corresponding image spacing,
    as in ITK, rather than Euclidean millimetres. History norms are measured
    *before* each update; final_residual is an explicit additional evaluation.
    The default uses local same-grid coordinates and a pairwise norm sum.
    ``arithmetic='itk'`` retains the historical world-rounding/serial-sum path
    for diagnostics; both paths use FP64 and the same inverse update rules.
    """
    if arithmetic not in ("local","itk"):
        raise ValueError("Inverse arithmetic must be local or itk")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("Inverse iteration limit must be a nonnegative integer")
    if not np.isfinite([mean_tolerance, max_tolerance]).all() or min(mean_tolerance, max_tolerance) < 0:
        raise ValueError("Inverse error tolerances must be finite and nonnegative")
    if initial is not None and not field.grid.same_geometry(initial.grid):
        raise ValueError("Initial inverse must share the forward field grid")
    # The first residual reads the caller's initial field. Its first update
    # writes a fresh owned array; later updates reuse that array in place.
    # A zero-iteration request still returns an independent validated copy.
    if initial is None:
        inverse = np.zeros((3,*field.grid.shape_zyx),np.float64)
        owned = inverse
    elif max_iterations == 0:
        inverse = np.asarray(initial.data,dtype=np.float64).copy()
        owned = inverse
    else:
        inverse = np.asarray(initial.data,dtype=np.float64)
        owned = np.empty(inverse.shape,np.float64) if _output is None else _output
    # The private inverse loop owns every updated array. It can validate values
    # at the write site instead of wrapping/scanning a temporary field at each
    # iteration. Public composition and caller-owned arrays remain unchanged.
    outer = np.asarray(field.data, dtype=np.float64)
    linear, origin = field.grid.index_to_lps[:3, :3], field.grid.origin
    inverse_linear = np.ascontiguousarray(field.grid.lps_to_index[:3, :3])
    spacing = field.grid.spacing_xyz
    # Reuse owned buffers. Updating the inverse in place is safe only after
    # the complete residual and its global norm reduction are finished.
    workspace=_InversionWorkspace(field.grid.shape_zyx) if _workspace is None else _workspace
    residual,norms,finite=workspace.residual,workspace.norms,workspace.finite
    packed=workspace.packed is not None
    if packed:
        outer=pack_field_into(outer,workspace.packed)
    parallel = inverse.size >= 49152 and get_num_threads() > 1
    def evaluate_residual(current,store_residual):
        if arithmetic=="local":
            return _local_inversion_residual_into(outer,current,inverse_linear,spacing,
                residual,norms,finite,workspace.partials,store_residual,parallel,packed)
        return _inversion_residual_into(outer,current,linear,origin,inverse_linear,spacing,
            residual,norms,finite,store_residual,parallel,packed)
    history, reason = [], 'max_iterations'
    for i in range(max_iterations):
        _, _, mean, maximum, flags = evaluate_residual(inverse,True)
        if not flags.all():
            raise ValueError("Displacement field must be finite")
        epsilon = .75 if i == 0 else .5
        _, flags = _inversion_update_into(inverse, residual, norms, maximum,
            epsilon, bool(enforce_boundary), owned, finite, parallel)
        inverse = owned
        if not flags.all():
            raise ValueError("Displacement field must be finite")
        history.append({'iteration':i+1, 'mean_scaled_error_before_update':float(mean),
                        'max_scaled_error_before_update':float(maximum), 'epsilon':epsilon})
        if maximum <= max_tolerance or mean <= mean_tolerance:
            reason = 'maximum_error_threshold' if maximum <= max_tolerance else 'mean_error_threshold'
            break
    result = DisplacementField(inverse, field.grid)
    # The explicit final evaluation needs norms, not a three-component volume.
    _, _, mean, maximum, flags = evaluate_residual(inverse,False)
    if not flags.all():
        raise ValueError("Displacement field must be finite")
    return InversionResult(result, history, reason,
                           {'mean_spacing_scaled_norm':float(mean), 'max_spacing_scaled_norm':float(maximum)})


def jacobian_determinant(field):
    """det(I+du/dx): central index differences, one-sided at boundary.

    A singleton axis has zero displacement derivative. This declared QA stencil
    is separate from registration gradients and ANTs' optional geometric stencil.
    """
    return jacobian_determinants(np.asarray(field.data, dtype=np.float64), np.ascontiguousarray(field.grid.lps_to_index[:3, :3]))


def inverse_consistency(forward, inverse):
    """Report both compositions on their mapped common support, in mm."""
    report = {}
    for name, outer, inner in (('forward_after_inverse',forward,inverse),
                                ('inverse_after_forward',inverse,forward)):
        residual, valid = compose_fields(outer, inner, return_valid=True)
        vectors = residual.data[:, valid].astype(np.float64)
        norms = np.sqrt(np.sum(vectors*vectors, axis=0))
        report[name] = {'common_valid_points':int(valid.sum()), 'total_points':int(valid.size),
                        'max_l2_mm':float(norms.max()) if norms.size else None,
                        'rms_l2_mm':float(np.sqrt(np.mean(norms*norms))) if norms.size else None,
                        'p99_l2_mm':float(np.percentile(norms,99)) if norms.size else None}
    return report
