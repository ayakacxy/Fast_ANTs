"""Parallel FP32 voxel norms and deterministic maximum-vector selection.

Only independent norm evaluations execute in parallel. The mean's Kahan sum
and maximum retain the original flattened z/y/x traversal and FP32 state.
Maximum-vector selection parallelizes independent rows and retains the
earliest flattened index for tied squared norms, including all-zero norms.
"""
import numpy as np
from numba import njit,prange,get_num_threads
from fast_ants.backends.cpu.fp32.kernels import norm32 as serial_norm32
from fast_ants.backends.cpu.fp32.kernels import maximum_vector32 as serial_maximum_vector32
from fast_ants.backends.cpu.fp32.kernels import add32


@njit(cache=True,fastmath=False,parallel=True)
def _norm_values32(field,spacing):
    _,nz,ny,nx=field.shape
    norms=np.empty((nz,ny,nx),np.float32)
    for task in prange(nz):
        z=np.int64(task)
        for y in range(ny):
            for x in range(nx):
                squared=np.float32(0)
                for c in range(3):
                    component=field[c,z,y,x]/spacing[c]
                    squared+=component*component
                norms[z,y,x]=np.sqrt(squared)
    return norms


@njit(cache=True,fastmath=False)
def _control32(norms):
    total=correction=maximum=np.float32(0)
    for value in norms.ravel():
        maximum=max(maximum,value)
        total,correction=add32(total,correction,value)
    return total/np.float32(norms.size),maximum


def norm32(field,spacing):
    if field.size//3<16384 or get_num_threads()==1:
        return serial_norm32(field,spacing)
    norms=_norm_values32(field,spacing)
    mean,maximum=_control32(norms)
    return norms,mean,maximum


@njit(cache=True,fastmath=False,parallel=True)
def _maximum_vector32(field,spacing):
    _,nz,ny,nx=field.shape
    scores=np.empty(nz*ny,np.float32)
    indices=np.empty(nz*ny,np.int64)
    for task in prange(nz*ny):
        row=np.int64(task)
        z,y=row//ny,row%ny
        maximum=np.float32(0)
        selected=row*nx
        for x in range(nx):
            squared=np.float32(0)
            for c in range(3):
                component=field[c,z,y,x]/spacing[c]
                squared+=component*component
            if squared>maximum:
                maximum=squared
                selected=row*nx+x
        scores[row]=maximum
        indices[row]=selected
    # This merge consumes only one score per row. The raw vector at index
    # zero remains selected if every squared norm is zero or underflows.
    maximum=np.float32(0)
    selected=0
    for row in range(scores.size):
        score,index=scores[row],indices[row]
        if score>maximum or (score==maximum and index<selected):
            maximum=score
            selected=index
    z,y,x=selected//(ny*nx),(selected//nx)%ny,selected%nx
    output=np.empty(3,np.float32)
    for c in range(3):
        output[c]=field[c,z,y,x]
    return output


def maximum_vector32(field,spacing):
    """Return the earliest FP32 spacing-scaled squared-norm maximum vector."""
    if field.size//3<16384 or get_num_threads()==1:
        return serial_maximum_vector32(np.ascontiguousarray(field),spacing)
    return _maximum_vector32(field,spacing)
