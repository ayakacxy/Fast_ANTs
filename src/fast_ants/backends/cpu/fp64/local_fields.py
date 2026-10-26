"""Same-grid FP64 inverse residuals without a world-coordinate round trip.

The displacement and spacing conventions match the standard inverse filter.
Row-wise positive norm sums are combined in a fixed pairwise tree; they do not
replay the original serial sum. There is no fast-math or precision reduction.
"""

import math
import numpy as np
from numba import njit, prange

from fast_ants.backends.cpu.fp64.field_kernels import _vector
from fast_ants.backends.cpu.fp64.field_kernels import _vector_packed
from fast_ants.backends.cpu.fp64.sampling_kernels import inside


@njit(cache=True, fastmath=False)
def _residual_row(outer, inner, inverse_linear, spacing, residual, norms,
                  store_residual, packed, z, y):
    _,nz,ny,nx=inner.shape
    finite=True
    total,maximum=0.0,0.0
    for x in range(nx):
        u0,u1,u2=inner[0,z,y,x],inner[1,z,y,x],inner[2,z,y,x]
        # Both fields have the same grid. Cancel L^-1 L analytically and keep
        # small physical displacements separate from any large world origin.
        i0=x+(inverse_linear[0,0]*u0+inverse_linear[0,1]*u1+inverse_linear[0,2]*u2)
        i1=y+(inverse_linear[1,0]*u0+inverse_linear[1,1]*u1+inverse_linear[1,2]*u2)
        i2=z+(inverse_linear[2,0]*u0+inverse_linear[2,1]*u1+inverse_linear[2,2]*u2)
        d0,d1,d2=0.0,0.0,0.0
        if inside(i0,i1,i2,nx,ny,nz):
            if packed:
                d0,d1,d2=_vector_packed(outer,i0,i1,i2)
            else:
                d0,d1,d2=_vector(outer,i0,i1,i2)
        v0,v1,v2=u0+d0,u1+d1,u2+d2
        if store_residual:
            residual[0,z,y,x]=v0
            residual[1,z,y,x]=v1
            residual[2,z,y,x]=v2
        if not (math.isfinite(v0) and math.isfinite(v1) and math.isfinite(v2)):
            finite=False
        q0,q1,q2=v0/spacing[0],v1/spacing[1],v2/spacing[2]
        value=math.sqrt(q0*q0+q1*q1+q2*q2)
        norms[z,y,x]=value
        total+=value
        maximum=max(maximum,value)
    return finite,total,maximum


@njit(cache=True, fastmath=False, parallel=True)
def inversion_residual_into(outer,inner,inverse_linear,spacing,residual,norms,
                            finite,partials,store_residual,use_parallel,packed=False):
    """Write residual/norm scratch, collecting reductions without a volume scan."""
    _,nz,ny,nx=inner.shape
    count=nz*ny
    if use_parallel:
        for task in prange(count):
            row=np.int64(task);z,y=row//ny,row%ny
            finite[z,y],partials[row,0],partials[row,1]=_residual_row(
                outer,inner,inverse_linear,spacing,residual,norms,store_residual,packed,z,y)
    else:
        for row in range(count):
            z,y=row//ny,row%ny
            finite[z,y],partials[row,0],partials[row,1]=_residual_row(
                outer,inner,inverse_linear,spacing,residual,norms,store_residual,packed,z,y)
    # O(rows) scratch traffic instead of another O(voxels) norm scan. All terms
    # are nonnegative; the balanced tree bounds accumulation depth. Retain the
    # old behavior of reporting +inf when finite displacements overflow norms.
    stride=1
    while stride<count:
        for row in range(0,count,2*stride):
            other=row+stride
            if other<count:
                partials[row,0]+=partials[other,0]
                partials[row,1]=max(partials[row,1],partials[other,1])
        stride*=2
    return residual,norms,partials[0,0]/(nz*ny*nx),partials[0,1],finite
