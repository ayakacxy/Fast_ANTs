"""Reusable ordinary-FP32 inverse residual, norm and update kernels.

The two fields occupy the same grid. Integer voxel origins plus local FP32
offsets avoid world-coordinate cancellation and preserve small fractions.
Norms are accumulated with row-wise Kahan sums and a fixed pairwise merge.
No image-sized FP64 temporary or software-expanded floating point is used.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import ONE
from fast_ants.backends.cpu.fp32.kernels import HALF
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.norms import _control32


class InverseWorkspace32:
    """Private scratch; returned buffers are invalidated when reused.

    ``output`` can be updated in place within one inversion. A caller reusing
    this workspace for another inversion must provide a separately owned
    output if the previous result is still live or is the new forward field.
    """
    def __init__(self, shape):
        shape=tuple(shape)
        if len(shape)!=3 or any(not isinstance(n,(int,np.integer)) or n<=0 for n in shape):
            raise ValueError('Inverse workspace needs three positive grid dimensions')
        self.shape=shape
        self.residual=np.empty((3,*shape),np.float32)
        self.norms=np.empty(shape,np.float32)
        self.partials=np.empty((shape[0]*shape[1],2),np.float32)
        self.finite=np.empty(shape[:2],np.bool_)
        self.output=np.empty((3,*shape),np.float32)
        self.serial_mean_guard_used=False


@njit(cache=True,fastmath=False,inline='always')
def _axis32(base, offset, size):
    if not np.isfinite(offset):
        return 0,0,ZERO,False
    whole=int(np.floor(offset))
    fraction=offset-np.float32(whole)
    index=base+whole
    valid=not (index < -1 or (index == -1 and fraction < HALF) or
               index >= size or (index == size-1 and fraction >= HALF))
    if index<0:
        index,fraction=0,ZERO
    elif index>=size-1:
        index,fraction=size-1,ZERO
    return index,min(index+1,size-1),fraction,valid


@njit(cache=True,fastmath=False,inline='always')
def _sample_component32(data,c,x0,x1,y0,y1,z0,z1,dx,dy,dz):
    a=data[c,z0,y0,x0]+(data[c,z0,y0,x1]-data[c,z0,y0,x0])*dx
    b=data[c,z0,y1,x0]+(data[c,z0,y1,x1]-data[c,z0,y1,x0])*dx
    low=data[c,z1,y0,x0]+(data[c,z1,y0,x1]-data[c,z1,y0,x0])*dx
    high=data[c,z1,y1,x0]+(data[c,z1,y1,x1]-data[c,z1,y1,x0])*dx
    lo,hi=a+(b-a)*dy,low+(high-low)*dy
    return lo+(hi-lo)*dz


@njit(cache=True,fastmath=False)
def _residual_row32(outer,current,inverse,spacing,residual,norms,store_residual,z,y):
    _,nz,ny,nx=current.shape
    total,correction,maximum=ZERO,ZERO,ZERO
    finite=True
    for x in range(nx):
        u0,u1,u2=current[0,z,y,x],current[1,z,y,x],current[2,z,y,x]
        dx=inverse[0,0]*u0+inverse[0,1]*u1+inverse[0,2]*u2
        dy=inverse[1,0]*u0+inverse[1,1]*u1+inverse[1,2]*u2
        dz=inverse[2,0]*u0+inverse[2,1]*u1+inverse[2,2]*u2
        x0,x1,fx,vx=_axis32(x,dx,nx)
        y0,y1,fy,vy=_axis32(y,dy,ny)
        z0,z1,fz,vz=_axis32(z,dz,nz)
        valid=vx and vy and vz
        squared=ZERO
        for c in range(3):
            value=current[c,z,y,x]
            if valid:
                value+=_sample_component32(outer,c,x0,x1,y0,y1,z0,z1,fx,fy,fz)
            if store_residual:
                residual[c,z,y,x]=value
            if not np.isfinite(value):
                finite=False
            component=value/spacing[c]
            squared+=component*component
        norm=np.sqrt(squared)
        norms[z,y,x]=norm
        maximum=max(maximum,norm)
        # Preserve +inf for overflowing positive norms rather than injecting a
        # spurious NaN through inf-inf in the compensation state.
        if np.isfinite(norm) and np.isfinite(total):
            total,correction=add32(total,correction,norm)
        else:
            total+=norm
            correction=ZERO
    return finite,total,maximum


@njit(cache=True,fastmath=False,parallel=True)
def _residual_into32(outer,current,inverse,spacing,residual,norms,partials,
                     finite,store_residual,use_parallel):
    _,nz,ny,nx=current.shape
    count=nz*ny
    if use_parallel:
        for task in prange(count):
            row=np.int64(task);z,y=row//ny,row%ny
            finite[z,y],partials[row,0],partials[row,1]=_residual_row32(
                outer,current,inverse,spacing,residual,norms,store_residual,z,y)
    else:
        for row in range(count):
            z,y=row//ny,row%ny
            finite[z,y],partials[row,0],partials[row,1]=_residual_row32(
                outer,current,inverse,spacing,residual,norms,store_residual,z,y)
    stride=1
    while stride<count:
        for row in range(0,count,2*stride):
            other=row+stride
            if other<count:
                partials[row,0]+=partials[other,0]
                partials[row,1]=max(partials[row,1],partials[other,1])
        stride*=2
    return partials[0,0]/np.float32(nz*ny*nx),partials[0,1]


@njit(cache=True,fastmath=False)
def _update_row32(current,residual,norms,maximum,epsilon,boundary,z,y,output):
    _,nz,ny,nx=current.shape
    limit=epsilon*maximum
    finite=True
    for x in range(nx):
        factor=ONE
        if norms[z,y,x]>limit:
            factor=limit/norms[z,y,x]
        edge=x==0 or y==0 or z==0 or x==nx-1 or y==ny-1 or z==nz-1
        for c in range(3):
            value=current[c,z,y,x]-residual[c,z,y,x]*factor*epsilon
            value=ZERO if boundary and edge else value
            output[c,z,y,x]=value
            if not np.isfinite(value):
                finite=False
    return finite


@njit(cache=True,fastmath=False,parallel=True)
def _update_into32(current,residual,norms,maximum,epsilon,boundary,output,finite,use_parallel):
    nz,ny=current.shape[1:3]
    if use_parallel:
        for task in prange(nz*ny):
            row=np.int64(task);z,y=row//ny,row%ny
            finite[z,y]=_update_row32(current,residual,norms,maximum,epsilon,boundary,z,y,output)
    else:
        for row in range(nz*ny):
            z,y=row//ny,row%ny
            finite[z,y]=_update_row32(current,residual,norms,maximum,epsilon,boundary,z,y,output)


def _field32(values,shape):
    if values.shape!=(3,*shape) or values.dtype!=np.float32 or not values.flags.c_contiguous:
        raise ValueError('Inverse fields must be contiguous FP32 arrays on the workspace grid')


def residual32_into(outer,current,inverse_linear,spacing,workspace,*,
                    store_residual=True,mean_tolerance=None):
    """Return borrowed residual/norm buffers and FP32 mean/maximum controls.

    The mean threshold is a native host control, not a quantized FP32 limit.
    Close to it, repeat the established serial compensated sum of the same
    norms. The caller retains the existing maximum-vector threshold guard.
    """
    _field32(outer,workspace.shape);_field32(current,workspace.shape)
    if (inverse_linear.shape!=(3,3) or spacing.shape!=(3,) or
            inverse_linear.dtype!=np.float32 or spacing.dtype!=np.float32):
        raise ValueError('Inverse geometry must be explicitly supplied as FP32')
    workspace.serial_mean_guard_used=False
    mean,maximum=_residual_into32(outer,current,inverse_linear,spacing,
        workspace.residual,workspace.norms,workspace.partials,workspace.finite,
        bool(store_residual),workspace.norms.size>=16384 and get_num_threads()>1)
    if not workspace.finite.all():
        raise ValueError('Displacement field must be finite')
    if mean_tolerance is not None and np.isfinite(mean):
        scale=max(float(mean),float(mean_tolerance),float(np.finfo(np.float32).tiny))
        if abs(float(mean)-float(mean_tolerance))<=32*np.finfo(np.float32).eps*scale:
            mean,maximum=_control32(workspace.norms)
            workspace.serial_mean_guard_used=True
    return workspace.residual,workspace.norms,mean,maximum


def update32_into(current,residual,norms,maximum,epsilon,enforce_boundary,
                  workspace,*,output=None):
    """Write the clipped inverse update; output may alias current only.

    The caller must also keep this output separate from its forward field.
    All current samples are local, so updating an owned current buffer in
    place is safe after the complete residual pass has finished.
    """
    output=workspace.output if output is None else output
    for array in (current,residual,output):
        _field32(array,workspace.shape)
    if norms.shape!=workspace.shape or norms.dtype!=np.float32 or not norms.flags.c_contiguous:
        raise ValueError('Inverse norms must be contiguous FP32 on the workspace grid')
    if np.shares_memory(output,residual) or np.shares_memory(output,norms):
        raise ValueError('Inverse update output cannot overwrite residual or norm scratch')
    _update_into32(current,residual,norms,np.float32(maximum),np.float32(epsilon),
        bool(enforce_boundary),output,workspace.finite,
        norms.size>=16384 and get_num_threads()>1)
    if not workspace.finite.all():
        raise ValueError('Displacement field must be finite')
    return output
