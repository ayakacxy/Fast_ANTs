"""Bounded-memory final field QA, with the public FP64 geometry conventions.

FP32 field storage is read directly and each scalar is promoted before any
arithmetic. Only one scalar norm buffer is needed per composition direction;
Jacobian determinants are reduced as they are calculated, without a volume.
"""

import math

import numpy as np
from numba import get_num_threads, njit, prange

from fast_ants.core.arithmetic64 import fma64
from fast_ants.backends.cpu.fp64.sampling_kernels import inside


_BLOCK=4096


@njit(cache=True,fastmath=False,inline='always')
def _component64(data,c,x0,x1,y0,y1,z0,z1,dx,dy,dz):
    # Cast before subtracting: subtracting two FP32 loads first would change
    # the public compose_fields(...astype(float64)) interpolation semantics.
    a0,b0=np.float64(data[c,z0,y0,x0]),np.float64(data[c,z0,y1,x0])
    c0,d0=np.float64(data[c,z1,y0,x0]),np.float64(data[c,z1,y1,x0])
    a=fma64(np.float64(data[c,z0,y0,x1])-a0,dx,a0)
    b=fma64(np.float64(data[c,z0,y1,x1])-b0,dx,b0)
    cc=fma64(np.float64(data[c,z1,y0,x1])-c0,dx,c0)
    d=fma64(np.float64(data[c,z1,y1,x1])-d0,dx,d0)
    lower,upper=fma64(b-a,dy,a),fma64(d-cc,dy,cc)
    return fma64(upper-lower,dz,lower)


@njit(cache=True,fastmath=False,inline='always')
def _vector64(data,x,y,z):
    _,nz,ny,nx=data.shape
    x,y,z=min(max(x,0.),nx-1),min(max(y,0.),ny-1),min(max(z,0.),nz-1)
    x0,y0,z0=int(math.floor(x)),int(math.floor(y)),int(math.floor(z))
    x1,y1,z1=min(x0+1,nx-1),min(y0+1,ny-1),min(z0+1,nz-1)
    dx,dy,dz=x-x0,y-y0,z-z0
    return (_component64(data,0,x0,x1,y0,y1,z0,z1,dx,dy,dz),
            _component64(data,1,x0,x1,y0,y1,z0,z1,dx,dy,dz),
            _component64(data,2,x0,x1,y0,y1,z0,z1,dx,dy,dz))


@njit(cache=True,fastmath=False)
def _norm_block(outer,inner,linear,origin,outer_origin,inverse,block,norms,counts,statistics):
    _,nz,ny,nx=inner.shape
    _,oz,oy,ox=outer.shape
    begin=block*_BLOCK;end=min(begin+_BLOCK,nz*ny*nx)
    count=0;maximum=0.;sumsq=0.;finite=True
    for n in range(begin,end):
        x=n%nx;y=(n//nx)%ny;z=n//(nx*ny)
        p0=origin[0]+linear[0,0]*x+linear[0,1]*y+linear[0,2]*z
        p1=origin[1]+linear[1,0]*x+linear[1,1]*y+linear[1,2]*z
        p2=origin[2]+linear[2,0]*x+linear[2,1]*y+linear[2,2]*z
        u0=np.float64(inner[0,z,y,x]);u1=np.float64(inner[1,z,y,x]);u2=np.float64(inner[2,z,y,x])
        l0=p0+u0-outer_origin[0];l1=p1+u1-outer_origin[1];l2=p2+u2-outer_origin[2]
        i0=0.0+inverse[0,0]*l0+inverse[0,1]*l1+inverse[0,2]*l2
        i1=0.0+inverse[1,0]*l0+inverse[1,1]*l1+inverse[1,2]*l2
        i2=0.0+inverse[2,0]*l0+inverse[2,1]*l1+inverse[2,2]*l2
        good=inside(i0,i1,i2,ox,oy,oz)
        d0,d1,d2=_vector64(outer,i0,i1,i2) if good else (0.,0.,0.)
        # Preserve ITK world-coordinate additions/subtraction, even for large
        # origins. Local u+d would change the declared reference QA rounding.
        v0=p0+u0+d0-p0;v1=p1+u1+d1-p1;v2=p2+u2+d2-p2
        if not (math.isfinite(v0) and math.isfinite(v1) and math.isfinite(v2)):
            finite=False
        if good:
            norm=math.sqrt(0.0+v0*v0+v1*v1+v2*v2)
            norms[begin+count]=norm;count+=1
            maximum=max(maximum,norm);sumsq+=norm*norm
            if not math.isfinite(norm):finite=False
    counts[block]=count
    statistics[block,0]=maximum;statistics[block,1]=sumsq
    statistics[block,2]=1. if finite and math.isfinite(sumsq) else 0.


@njit(cache=True,fastmath=False,parallel=True)
def _norm_work(outer,inner,linear,origin,outer_origin,inverse,norms,counts,statistics,parallel):
    if parallel:
        for task in prange(counts.size):
            _norm_block(outer,inner,linear,origin,outer_origin,inverse,np.int64(task),norms,counts,statistics)
    else:
        for block in range(counts.size):
            _norm_block(outer,inner,linear,origin,outer_origin,inverse,block,norms,counts,statistics)


@njit(cache=True,fastmath=False)
def _compact_norms(norms,counts,statistics):
    count=0;maximum=0.;sumsq=0.;finite=True
    for block in range(counts.size):
        begin=block*_BLOCK
        for offset in range(counts[block]):
            norms[count+offset]=norms[begin+offset]
        count+=counts[block]
        maximum=max(maximum,statistics[block,0]);sumsq+=statistics[block,1]
        if statistics[block,2]==0.:finite=False
    return count,maximum,sumsq,finite and math.isfinite(sumsq)


@njit(cache=True,fastmath=False,inline='always')
def _difference(data,c,x,y,z,axis):
    _,nz,ny,nx=data.shape
    if axis==0:
        lo=max(x-1,0);hi=min(x+1,nx-1)
        if lo==hi:return 0.
        delta=np.float64(data[c,z,y,hi])-np.float64(data[c,z,y,lo])
    elif axis==1:
        lo=max(y-1,0);hi=min(y+1,ny-1)
        if lo==hi:return 0.
        delta=np.float64(data[c,z,hi,x])-np.float64(data[c,z,lo,x])
    else:
        lo=max(z-1,0);hi=min(z+1,nz-1)
        if lo==hi:return 0.
        delta=np.float64(data[c,hi,y,x])-np.float64(data[c,lo,y,x])
    return delta/(hi-lo)


@njit(cache=True,fastmath=False,inline='always')
def _world_derivative(a,b,c,inverse,d):
    return 0.0+a*inverse[0,d]+b*inverse[1,d]+c*inverse[2,d]


@njit(cache=True,fastmath=False,inline='always')
def _determinant(data,inverse,x,y,z):
    a=_difference(data,0,x,y,z,0);b=_difference(data,0,x,y,z,1);c=_difference(data,0,x,y,z,2)
    j00=_world_derivative(a,b,c,inverse,0)+1.;j01=_world_derivative(a,b,c,inverse,1)
    j02=_world_derivative(a,b,c,inverse,2)
    a=_difference(data,1,x,y,z,0);b=_difference(data,1,x,y,z,1);c=_difference(data,1,x,y,z,2)
    j10=_world_derivative(a,b,c,inverse,0);j11=_world_derivative(a,b,c,inverse,1)+1.
    j12=_world_derivative(a,b,c,inverse,2)
    a=_difference(data,2,x,y,z,0);b=_difference(data,2,x,y,z,1);c=_difference(data,2,x,y,z,2)
    j20=_world_derivative(a,b,c,inverse,0);j21=_world_derivative(a,b,c,inverse,1)
    j22=_world_derivative(a,b,c,inverse,2)+1.
    return j00*(j11*j22-j12*j21)-j01*(j10*j22-j12*j20)+j02*(j10*j21-j11*j20)


@njit(cache=True,fastmath=False)
def _jacobian_block(data,inverse,block,statistics,counts):
    _,nz,ny,nx=data.shape
    begin=block*_BLOCK;end=min(begin+_BLOCK,nz*ny*nx)
    minimum=math.inf;nonpositive=0;finite=True
    for n in range(begin,end):
        x=n%nx;y=(n//nx)%ny;z=n//(nx*ny)
        value=_determinant(data,inverse,x,y,z)
        minimum=min(minimum,value);nonpositive+=int(value<=0.)
        if not math.isfinite(value):finite=False
    statistics[block,0]=minimum;statistics[block,1]=1. if finite else 0.
    counts[block]=nonpositive


@njit(cache=True,fastmath=False,parallel=True)
def _jacobian_work(data,inverse,statistics,counts,parallel):
    if parallel:
        for task in prange(counts.size):
            _jacobian_block(data,inverse,np.int64(task),statistics,counts)
    else:
        for block in range(counts.size):_jacobian_block(data,inverse,block,statistics,counts)


def _data(field):
    values=np.asarray(field.data)
    if values.dtype not in (np.float32,np.float64) or values.shape!=(3,*field.grid.shape_zyx):
        raise ValueError('Output QA needs float32/float64 displacement fields [3,z,y,x]')
    return values


def _direction(outer,inner,outer_data,inner_data):
    size=int(np.prod(inner.grid.shape_zyx));blocks=(size+_BLOCK-1)//_BLOCK
    norms=np.empty(size,np.float64)
    counts=np.empty(blocks,np.int64);statistics=np.empty((blocks,3),np.float64)
    _norm_work(outer_data,inner_data,inner.grid.index_to_lps[:3,:3],inner.grid.origin,
               outer.grid.origin,np.ascontiguousarray(outer.grid.lps_to_index[:3,:3]),
               norms,counts,statistics,size>=16384 and get_num_threads()>1)
    count,maximum,sumsq,finite=_compact_norms(norms,counts,statistics)
    if not finite:raise ValueError('Output QA composition and norms must be finite')
    # Prefix compaction is in place and percentile may partition this owned
    # scratch. No advanced indexing, residual vector, validity mask, or second
    # N-element percentile copy is needed when support is only partial.
    return {'common_valid_points':int(count),'total_points':size,
            'max_l2_mm':float(maximum) if count else None,
            'rms_l2_mm':float(math.sqrt(sumsq/count)) if count else None,
            'p99_l2_mm':float(np.percentile(norms[:count],99,overwrite_input=True)) if count else None}


def output_quality(forward,inverse):
    """Return final nonlinear composition and forward-Jacobian QA summaries.

    Formulas, half-voxel support, interpolation FMAs and default linear 99th
    percentile match ``fields.inverse_consistency``/``jacobian_determinant``.
    RMS uses independent FP64 block sums, permitting reduction-rounding noise.
    Nonfinite inputs/results are explicitly rejected rather than hidden by a
    support mask or an extremum reduction.
    """
    forward_data,inverse_data=_data(forward),_data(inverse)
    consistency={
        'forward_after_inverse':_direction(forward,inverse,forward_data,inverse_data),
        'inverse_after_forward':_direction(inverse,forward,inverse_data,forward_data)}
    size=int(np.prod(forward.grid.shape_zyx));blocks=(size+_BLOCK-1)//_BLOCK
    statistics=np.empty((blocks,2),np.float64);counts=np.empty(blocks,np.int64)
    _jacobian_work(forward_data,np.ascontiguousarray(forward.grid.lps_to_index[:3,:3]),
                   statistics,counts,size>=16384 and get_num_threads()>1)
    if not statistics[:,1].all():raise ValueError('Output QA Jacobian determinants must be finite')
    return {'nonlinear_inverse_consistency':consistency,
            'minimum_nonlinear_jacobian':float(statistics[:,0].min()),
            'nonpositive_jacobian_points':int(counts.sum())}
