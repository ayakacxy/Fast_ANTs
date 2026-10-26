"""Strict CPU field operators; inversion follows ITK, NumFOCUS/Apache-2.0.

Independent rows execute in parallel; scalar reductions keep source order.
"""


import math
import numpy as np
from numba import get_num_threads, njit, prange
from fast_ants.backends.cpu.fp64.sampling_kernels import inside
from fast_ants.backends.cpu.fp64.sampling_kernels import linear_value
from fast_ants.core.arithmetic64 import fma64


@njit(cache=True, fastmath=False, inline="always")
def _component(data, c, x0, x1, y0, y1, z0, z1, dx, dy, dz):
    a0, b0 = data[c,z0,y0,x0], data[c,z0,y1,x0]
    c0, d0 = data[c,z1,y0,x0], data[c,z1,y1,x0]
    a = fma64(data[c,z0,y0,x1]-a0, dx, a0)
    b = fma64(data[c,z0,y1,x1]-b0, dx, b0)
    cc = fma64(data[c,z1,y0,x1]-c0, dx, c0)
    d = fma64(data[c,z1,y1,x1]-d0, dx, d0)
    lower, upper = fma64(b-a, dy, a), fma64(d-cc, dy, cc)
    return fma64(upper-lower, dz, lower)


@njit(cache=True, fastmath=False, inline="always")
def _vector(data, x, y, z):
    _, nz, ny, nx = data.shape
    x, y, z = min(max(x,0.),nx-1), min(max(y,0.),ny-1), min(max(z,0.),nz-1)
    x0,y0,z0 = int(math.floor(x)),int(math.floor(y)),int(math.floor(z))
    x1,y1,z1 = min(x0+1,nx-1),min(y0+1,ny-1),min(z0+1,nz-1)
    dx,dy,dz = x-x0,y-y0,z-z0
    return (_component(data,0,x0,x1,y0,y1,z0,z1,dx,dy,dz),
            _component(data,1,x0,x1,y0,y1,z0,z1,dx,dy,dz),
            _component(data,2,x0,x1,y0,y1,z0,z1,dx,dy,dz))


@njit(cache=True, fastmath=False)
def _compose_arrays_row(outer, inner, inner_linear, inner_origin, outer_origin, outer_inverse, world_rounding, z, y, output, valid):
    _, nz, ny, nx = inner.shape
    _, oz, oy, ox = outer.shape
    for x in range(nx):
        p0 = inner_origin[0] + inner_linear[0,0]*x + inner_linear[0,1]*y + inner_linear[0,2]*z
        u0 = inner[0,z,y,x]
        p1 = inner_origin[1] + inner_linear[1,0]*x + inner_linear[1,1]*y + inner_linear[1,2]*z
        u1 = inner[1,z,y,x]
        p2 = inner_origin[2] + inner_linear[2,0]*x + inner_linear[2,1]*y + inner_linear[2,2]*z
        u2 = inner[2,z,y,x]
        if world_rounding:
            l0 = p0 + u0 - outer_origin[0]
            l1 = p1 + u1 - outer_origin[1]
            l2 = p2 + u2 - outer_origin[2]
        else:
            l0 = inner_origin[0]-outer_origin[0]+inner_linear[0,0]*x+inner_linear[0,1]*y+inner_linear[0,2]*z+u0
            l1 = inner_origin[1]-outer_origin[1]+inner_linear[1,0]*x+inner_linear[1,1]*y+inner_linear[1,2]*z+u1
            l2 = inner_origin[2]-outer_origin[2]+inner_linear[2,0]*x+inner_linear[2,1]*y+inner_linear[2,2]*z+u2
        i0 = 0.0 + outer_inverse[0,0]*l0 + outer_inverse[0,1]*l1 + outer_inverse[0,2]*l2
        i1 = 0.0 + outer_inverse[1,0]*l0 + outer_inverse[1,1]*l1 + outer_inverse[1,2]*l2
        i2 = 0.0 + outer_inverse[2,0]*l0 + outer_inverse[2,1]*l1 + outer_inverse[2,2]*l2
        good = inside(i0,i1,i2,ox,oy,oz)
        valid[z,y,x] = good
        d0,d1,d2 = _vector(outer,i0,i1,i2) if good else (0.,0.,0.)
        if world_rounding:
            output[0,z,y,x] = p0 + u0 + d0 - p0
            output[1,z,y,x] = p1 + u1 + d1 - p1
            output[2,z,y,x] = p2 + u2 + d2 - p2
        else:
            output[0,z,y,x] = u0 + d0
            output[1,z,y,x] = u1 + d1
            output[2,z,y,x] = u2 + d2


@njit(cache=True, fastmath=False)
def _inverse_step_row(inverse, residual, norms, maximum, epsilon, enforce_boundary, z, y, output):
    _, nz, ny, nx = inverse.shape
    for x in range(nx):
        factor = 1.0
        norm = norms[z, y, x]
        if norm > epsilon * maximum:
            factor = epsilon * maximum / norm
        boundary = x == 0 or y == 0 or z == 0 or (x == nx - 1) or (y == ny - 1) or (z == nz - 1)
        for c in range(3):
            value = inverse[c, z, y, x] - residual[c, z, y, x] * factor * epsilon
            output[c, z, y, x] = 0.0 if enforce_boundary and boundary else value


@njit(cache=True, fastmath=False)
def _jacobian_determinants_row(field, inverse_linear, z, y, output):
    _, nz, ny, nx = field.shape
    sizes = (nx, ny, nz)
    for x in range(nx):
        d_index = np.zeros((3, 3), dtype=np.float64)
        for axis in range(3):
            index = (x, y, z)[axis]
            lo, hi = (max(index - 1, 0), min(index + 1, sizes[axis] - 1))
            if lo == hi:
                continue
            for c in range(3):
                if axis == 0:
                    delta = field[c, z, y, hi] - field[c, z, y, lo]
                elif axis == 1:
                    delta = field[c, z, hi, x] - field[c, z, lo, x]
                else:
                    delta = field[c, hi, y, x] - field[c, lo, y, x]
                d_index[c, axis] = delta / (hi - lo)
        j = np.empty((3, 3), dtype=np.float64)
        for c in range(3):
            for d in range(3):
                value = 0.0
                for k in range(3):
                    value += d_index[c, k] * inverse_linear[k, d]
                j[c, d] = value
        for c in range(3):
            j[c, c] += 1.0
        output[z, y, x] = j[0, 0] * (j[1, 1] * j[2, 2] - j[1, 2] * j[2, 1]) - j[0, 1] * (j[1, 0] * j[2, 2] - j[1, 2] * j[2, 0]) + j[0, 2] * (j[1, 0] * j[2, 1] - j[1, 1] * j[2, 0])



@njit(cache=True, fastmath=False, parallel=True)
def _compose_arrays_work(outer, inner, inner_linear, inner_origin, outer_origin, outer_inverse, world_rounding, use_parallel):
    _, nz, ny, nx = inner.shape
    output = np.empty(inner.shape, dtype=np.float64)
    valid = np.empty((nz, ny, nx), dtype=np.bool_)
    if not use_parallel:
        for row in range(nz*ny):
            _compose_arrays_row(outer, inner, inner_linear, inner_origin, outer_origin, outer_inverse, world_rounding, row//ny, row%ny, output, valid)
    else:
        for task in prange(nz*ny):
            row = np.int64(task)
            _compose_arrays_row(outer, inner, inner_linear, inner_origin, outer_origin, outer_inverse, world_rounding, row//ny, row%ny, output, valid)
    return output, valid


@njit(cache=True, fastmath=False, parallel=True)
def _multiply_field_work(values, factor, use_parallel):
    output = np.empty(values.shape, dtype=np.float64)
    src, dst = values.ravel(), output.ravel()
    if not use_parallel:
        for i in range(values.size):
            dst[i] = src[i]*factor
    else:
        for i in prange(values.size):
            dst[i] = src[i]*factor
    return output


@njit(cache=True, fastmath=False)
def _norm_row(residual, spacing, z, y, norms):
    for x in range(residual.shape[3]):
        value = 0.0
        for c in range(3):
            component = residual[c, z, y, x]/spacing[c]
            value += component*component
        norms[z, y, x] = math.sqrt(value)


@njit(cache=True, fastmath=False, parallel=True)
def _scaled_residual_norms_work(residual, spacing, use_parallel):
    _, nz, ny, nx = residual.shape
    norms = np.empty((nz, ny, nx), dtype=np.float64)
    if not use_parallel:
        for row in range(nz*ny):
            _norm_row(residual, spacing, row//ny, row%ny, norms)
    else:
        for task in prange(nz*ny):
            row = np.int64(task)
            _norm_row(residual, spacing, row//ny, row%ny, norms)
    # This loop intentionally stays serial, including its original max semantics.
    values = norms.ravel()
    total, maximum = 0.0, 0.0
    for i in range(values.size):
        total += values[i]
        maximum = max(maximum, values[i])
    return norms, total/(nz*ny*nx), maximum


@njit(cache=True, fastmath=False, parallel=True)
def _inverse_step_work(inverse, residual, norms, maximum, epsilon, enforce_boundary, use_parallel):
    _, nz, ny, nx = inverse.shape
    output = np.empty(inverse.shape, dtype=np.float64)
    if not use_parallel:
        for row in range(nz*ny):
            _inverse_step_row(inverse, residual, norms, maximum, epsilon, enforce_boundary, row//ny, row%ny, output)
    else:
        for task in prange(nz*ny):
            row = np.int64(task)
            _inverse_step_row(inverse, residual, norms, maximum, epsilon, enforce_boundary, row//ny, row%ny, output)
    return output


@njit(cache=True, fastmath=False, parallel=True)
def _jacobian_determinants_work(field, inverse_linear, use_parallel):
    _, nz, ny, nx = field.shape
    output = np.empty((nz, ny, nx), dtype=np.float64)
    if not use_parallel:
        for row in range(nz*ny):
            _jacobian_determinants_row(field, inverse_linear, row//ny, row%ny, output)
    else:
        for task in prange(nz*ny):
            row = np.int64(task)
            _jacobian_determinants_row(field, inverse_linear, row//ny, row%ny, output)
    return output


# Keep runtime thread-pool queries outside compiled code: Numba marks functions
# referencing this dynamic runtime address uncacheable on the supported runtime.
def compose_arrays(outer, inner, inner_linear, inner_origin, outer_origin, outer_inverse, world_rounding):
    parallel = inner.size >= 49152 and get_num_threads() > 1
    return _compose_arrays_work(outer, inner, inner_linear, inner_origin, outer_origin, outer_inverse, world_rounding, parallel)


def multiply_field(values, factor):
    return _multiply_field_work(values, factor, values.size >= 49152 and get_num_threads() > 1)


def scaled_residual_norms(residual, spacing):
    return _scaled_residual_norms_work(residual, spacing, residual.size >= 49152 and get_num_threads() > 1)


def inverse_step(inverse, residual, norms, maximum, epsilon, enforce_boundary):
    parallel = inverse.size >= 49152 and get_num_threads() > 1
    return _inverse_step_work(inverse, residual, norms, maximum, epsilon, enforce_boundary, parallel)


def jacobian_determinants(field, inverse_linear):
    return _jacobian_determinants_work(field, inverse_linear, field.size >= 49152 and get_num_threads() > 1)


@njit(cache=True, fastmath=False)
def _inversion_residual_row(outer, inner, linear, origin, inverse_linear,
                            spacing, residual, norms, store_residual, z, y):
    """Compose in world coordinates and immediately consume each component.

    Unlike public composition, inversion does not need a support mask. The
    expressions match _compose_arrays_row(world_rounding=True) and _norm_row.
    """
    _, oz, oy, ox = outer.shape
    finite = True
    for x in range(inner.shape[3]):
        p0 = origin[0]+linear[0,0]*x+linear[0,1]*y+linear[0,2]*z
        u0 = inner[0,z,y,x]
        l0 = p0+u0-origin[0]
        p1 = origin[1]+linear[1,0]*x+linear[1,1]*y+linear[1,2]*z
        u1 = inner[1,z,y,x]
        l1 = p1+u1-origin[1]
        p2 = origin[2]+linear[2,0]*x+linear[2,1]*y+linear[2,2]*z
        u2 = inner[2,z,y,x]
        l2 = p2+u2-origin[2]
        i0 = 0.0 + inverse_linear[0,0]*l0 + inverse_linear[0,1]*l1 + inverse_linear[0,2]*l2
        i1 = 0.0 + inverse_linear[1,0]*l0 + inverse_linear[1,1]*l1 + inverse_linear[1,2]*l2
        i2 = 0.0 + inverse_linear[2,0]*l0 + inverse_linear[2,1]*l1 + inverse_linear[2,2]*l2
        good = inside(i0,i1,i2,ox,oy,oz)
        d0,d1,d2 = _vector(outer,i0,i1,i2) if good else (0.,0.,0.)
        v0 = p0+u0+d0-p0
        v1 = p1+u1+d1-p1
        v2 = p2+u2+d2-p2
        if store_residual:
            residual[0,z,y,x] = v0
            residual[1,z,y,x] = v1
            residual[2,z,y,x] = v2
        if not (math.isfinite(v0) and math.isfinite(v1) and math.isfinite(v2)):
            finite = False
        q0 = v0/spacing[0]
        q1 = v1/spacing[1]
        q2 = v2/spacing[2]
        norms[z,y,x] = math.sqrt(0.0 + q0*q0 + q1*q1 + q2*q2)
    return finite


@njit(cache=True, fastmath=False, inline="always")
def _component_packed(data, c, x0, x1, y0, y1, z0, z1, dx, dy, dz):
    a0, b0 = data[z0,y0,x0,c], data[z0,y1,x0,c]
    c0, d0 = data[z1,y0,x0,c], data[z1,y1,x0,c]
    a = fma64(data[z0,y0,x1,c]-a0, dx, a0)
    b = fma64(data[z0,y1,x1,c]-b0, dx, b0)
    cc = fma64(data[z1,y0,x1,c]-c0, dx, c0)
    d = fma64(data[z1,y1,x1,c]-d0, dx, d0)
    lower, upper = fma64(b-a, dy, a), fma64(d-cc, dy, cc)
    return fma64(upper-lower, dz, lower)


@njit(cache=True, fastmath=False, inline="always")
def _vector_packed(data, x, y, z):
    nz, ny, nx, _ = data.shape
    x, y, z = min(max(x,0.),nx-1), min(max(y,0.),ny-1), min(max(z,0.),nz-1)
    x0,y0,z0 = int(math.floor(x)),int(math.floor(y)),int(math.floor(z))
    x1,y1,z1 = min(x0+1,nx-1),min(y0+1,ny-1),min(z0+1,nz-1)
    dx,dy,dz = x-x0,y-y0,z-z0
    return (_component_packed(data,0,x0,x1,y0,y1,z0,z1,dx,dy,dz),
            _component_packed(data,1,x0,x1,y0,y1,z0,z1,dx,dy,dz),
            _component_packed(data,2,x0,x1,y0,y1,z0,z1,dx,dy,dz))


@njit(cache=True, fastmath=False)
def _inversion_residual_packed_row(outer, inner, linear, origin, inverse_linear,
                            spacing, residual, norms, store_residual, z, y):
    """Compose in world coordinates and immediately consume each component.

    Unlike public composition, inversion does not need a support mask. The
    expressions match _compose_arrays_row(world_rounding=True) and _norm_row.
    """
    oz, oy, ox, _ = outer.shape
    finite = True
    for x in range(inner.shape[3]):
        p0 = origin[0]+linear[0,0]*x+linear[0,1]*y+linear[0,2]*z
        u0 = inner[0,z,y,x]
        l0 = p0+u0-origin[0]
        p1 = origin[1]+linear[1,0]*x+linear[1,1]*y+linear[1,2]*z
        u1 = inner[1,z,y,x]
        l1 = p1+u1-origin[1]
        p2 = origin[2]+linear[2,0]*x+linear[2,1]*y+linear[2,2]*z
        u2 = inner[2,z,y,x]
        l2 = p2+u2-origin[2]
        i0 = 0.0 + inverse_linear[0,0]*l0 + inverse_linear[0,1]*l1 + inverse_linear[0,2]*l2
        i1 = 0.0 + inverse_linear[1,0]*l0 + inverse_linear[1,1]*l1 + inverse_linear[1,2]*l2
        i2 = 0.0 + inverse_linear[2,0]*l0 + inverse_linear[2,1]*l1 + inverse_linear[2,2]*l2
        good = inside(i0,i1,i2,ox,oy,oz)
        d0,d1,d2 = _vector_packed(outer,i0,i1,i2) if good else (0.,0.,0.)
        v0 = p0+u0+d0-p0
        v1 = p1+u1+d1-p1
        v2 = p2+u2+d2-p2
        if store_residual:
            residual[0,z,y,x] = v0
            residual[1,z,y,x] = v1
            residual[2,z,y,x] = v2
        if not (math.isfinite(v0) and math.isfinite(v1) and math.isfinite(v2)):
            finite = False
        q0 = v0/spacing[0]
        q1 = v1/spacing[1]
        q2 = v2/spacing[2]
        norms[z,y,x] = math.sqrt(0.0 + q0*q0 + q1*q1 + q2*q2)
    return finite


@njit(cache=True, fastmath=False, parallel=True)
def _inversion_residual_into(outer, inner, linear, origin, inverse_linear,
                             spacing, residual, norms, finite, store_residual, use_parallel, packed=False):
    _, nz, ny, nx = inner.shape
    if packed:
        for task in prange(nz*ny):
            row=np.int64(task)
            finite[row//ny,row%ny]=_inversion_residual_packed_row(outer,inner,linear,origin,inverse_linear,spacing,residual,norms,store_residual,row//ny,row%ny)
    elif not use_parallel:
        for row in range(nz * ny):
            finite[row // ny, row % ny] = _inversion_residual_row(
                outer, inner, linear, origin, inverse_linear, spacing,
                residual, norms, store_residual, row // ny, row % ny)
    else:
        for task in prange(nz * ny):
            row = np.int64(task)
            finite[row // ny, row % ny] = _inversion_residual_row(
                outer, inner, linear, origin, inverse_linear, spacing,
                residual, norms, store_residual, row // ny, row % ny)
    # Keep the original ravelled z/y/x total and max order. Norm overflow is
    # permitted just as before; only nonfinite displacement values are errors.
    values = norms.ravel()
    total, maximum = 0.0, 0.0
    for i in range(values.size):
        total += values[i]
        maximum = max(maximum, values[i])
    return residual, norms, total / (nz * ny * nx), maximum, finite


@njit(cache=True, fastmath=False)
def _inversion_residual_work(outer, inner, linear, origin, inverse_linear,
                             spacing, store_residual, use_parallel):
    _,nz,ny,nx=inner.shape
    residual=np.empty(inner.shape if store_residual else (3,0,0,0),np.float64)
    norms=np.empty((nz,ny,nx),np.float64)
    finite=np.empty((nz,ny),np.bool_)
    return _inversion_residual_into(outer,inner,linear,origin,inverse_linear,
        spacing,residual,norms,finite,store_residual,use_parallel)


@njit(cache=True, fastmath=False)
def _inversion_update_row(inverse, residual, norms, maximum, epsilon,
                          enforce_boundary, z, y, output):
    _, nz, ny, nx = inverse.shape
    finite = True
    for x in range(nx):
        factor = 1.0
        norm = norms[z, y, x]
        if norm > epsilon * maximum:
            factor = epsilon * maximum / norm
        boundary = x == 0 or y == 0 or z == 0 or (x == nx - 1) or (y == ny - 1) or (z == nz - 1)
        for c in range(3):
            value = inverse[c, z, y, x] - residual[c, z, y, x] * factor * epsilon
            value = 0.0 if enforce_boundary and boundary else value
            output[c, z, y, x] = value
            if not math.isfinite(value):
                finite = False
    return finite


@njit(cache=True, fastmath=False, parallel=True)
def _inversion_update_into(inverse, residual, norms, maximum, epsilon,
                           enforce_boundary, output, finite, use_parallel):
    _, nz, ny, nx = inverse.shape
    if not use_parallel:
        for row in range(nz * ny):
            finite[row // ny, row % ny] = _inversion_update_row(
                inverse, residual, norms, maximum, epsilon, enforce_boundary,
                row // ny, row % ny, output)
    else:
        for task in prange(nz * ny):
            row = np.int64(task)
            finite[row // ny, row % ny] = _inversion_update_row(
                inverse, residual, norms, maximum, epsilon, enforce_boundary,
                row // ny, row % ny, output)
    return output, finite


@njit(cache=True, fastmath=False)
def _inversion_update_work(inverse, residual, norms, maximum, epsilon,
                           enforce_boundary, use_parallel):
    output=np.empty(inverse.shape,np.float64)
    finite=np.empty(inverse.shape[1:3],np.bool_)
    return _inversion_update_into(inverse,residual,norms,maximum,epsilon,
        enforce_boundary,output,finite,use_parallel)


def _inversion_residual(outer, inner, linear, origin, inverse_linear, spacing, store_residual=True):
    """Private inverse-only path; public compose_fields retains its contract."""
    residual, norms, mean, maximum, finite = _inversion_residual_work(
        outer, inner, linear, origin, inverse_linear, spacing, bool(store_residual),
        inner.size >= 49152 and get_num_threads() > 1)
    if not finite.all():
        raise ValueError("Displacement field must be finite")
    return residual, norms, mean, maximum


def _inversion_update(inverse, residual, norms, maximum, epsilon, enforce_boundary):
    output, finite = _inversion_update_work(
        inverse, residual, norms, maximum, epsilon, enforce_boundary,
        inverse.size >= 49152 and get_num_threads() > 1)
    if not finite.all():
        raise ValueError("Displacement field must be finite")
    return output


@njit(cache=True,fastmath=False,parallel=True)
def compose_arrays_into(outer,inner,inner_linear,inner_origin,outer_origin,outer_inverse,
                        world_rounding,output,valid,use_parallel):
    _,nz,ny,nx=inner.shape
    if use_parallel:
        for task in prange(nz*ny):
            row=np.int64(task)
            _compose_arrays_row(outer,inner,inner_linear,inner_origin,outer_origin,outer_inverse,world_rounding,row//ny,row%ny,output,valid)
    else:
        for row in range(nz*ny):
            _compose_arrays_row(outer,inner,inner_linear,inner_origin,outer_origin,outer_inverse,world_rounding,row//ny,row%ny,output,valid)
    return output,valid


@njit(cache=True,fastmath=False,parallel=True)
def multiply_field_inplace(values,factor,use_parallel):
    flat=values.ravel()
    if use_parallel:
        for i in prange(flat.size):flat[i]=flat[i]*factor
    else:
        for i in range(flat.size):flat[i]=flat[i]*factor
    return values


@njit(cache=True,fastmath=False,parallel=True)
def pack_field_into(values,packed):
    _,nz,ny,nx=values.shape
    for task in prange(nz*ny):
        row=np.int64(task)
        z,y=row//ny,row%ny
        for x in range(nx):
            for c in range(3):packed[z,y,x,c]=values[c,z,y,x]
    return packed
