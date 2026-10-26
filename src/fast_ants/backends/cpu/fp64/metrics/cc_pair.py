"""Two source-order CC updates from one plane queue and lazy gradients.

The two directed formulas remain distinct. Sharing their identical raw window
sums avoids repeating the neighbourhood traversal; physical gradients are only
needed for centres whose two variances permit an update.
"""

import math
import numpy as np
from numba import get_num_threads, njit, prange
from fast_ants.backends.cpu.fp64.metrics.cc_kernels import _fill_plane
from fast_ants.backends.cpu.fp64.sampling_kernels import inside
from fast_ants.backends.cpu.fp64.sampling_kernels import linear_value
from fast_ants.core.arithmetic64 import fma64


@njit(cache=True, fastmath=False)
def _score_weight(fixed, moving, sums, reverse):
    count = sums[5]
    if reverse:
        sf2, sm2, sf, sm = sums[1], sums[0], sums[3], sums[2]
    else:
        sf2, sm2, sf, sm = sums[0], sums[1], sums[2], sums[3]
    mf, mm = sf/count, sm/count
    ff = sf2-mf*sf-mf*sf+count*mf*mf
    vv = sm2-mm*sm-mm*sm+count*mm*mm
    fm = sums[4]-mm*sf-mf*sm+count*mm*mf
    product = ff*vv
    epsilon = np.finfo(np.float64).eps
    score = 1.0
    if abs(product) > epsilon:
        score = fm*fm/product
    active = ff > epsilon and vv > epsilon
    weight = 0.0
    if active:
        weight = 2.0*fm/product*(fixed-mf-fm/vv*(moving-mm))
    return score, weight, active


@njit(cache=True, fastmath=False)
def _point_gradient(data, point, origin, inverse, spacing, negative, positive, gradient):
    nz, ny, nx = data.shape
    for c in range(3):
        gradient[c] = 0.0
        offset = .5*spacing[c]
        for i in range(3):
            minus, plus = 0.0, 0.0
            for j in range(3):
                coordinate = point[j]-origin[j]
                minus += inverse[i, j]*(coordinate-(offset if c == j else 0.0))
                plus += inverse[i, j]*(coordinate+(offset if c == j else 0.0))
            negative[i], positive[i] = minus, plus
        delta = (point[c]+offset)-(point[c]-offset)
        if (inside(negative[0], negative[1], negative[2], nx, ny, nz) and
                inside(positive[0], positive[1], positive[2], nx, ny, nz) and delta > 10*np.finfo(np.float64).eps):
            gradient[c] = (linear_value(data, positive[0], positive[1], positive[2])-
                           linear_value(data, negative[0], negative[1], negative[2]))/delta


@njit(cache=True,fastmath=False,inline="always")
def _pair_value(fixed,moving,x,y,z):
    nz,ny,nx=fixed.shape
    x,y,z=min(max(x,0.),nx-1),min(max(y,0.),ny-1),min(max(z,0.),nz-1)
    x0,y0,z0=int(math.floor(x)),int(math.floor(y)),int(math.floor(z))
    x1,y1,z1=min(x0+1,nx-1),min(y0+1,ny-1),min(z0+1,nz-1)
    dx,dy,dz=x-x0,y-y0,z-z0
    fa0,fb0=fixed[z0,y0,x0],fixed[z0,y1,x0]
    fc0,fd0=fixed[z1,y0,x0],fixed[z1,y1,x0]
    fa=fma64(fixed[z0,y0,x1]-fa0,dx,fa0)
    fb=fma64(fixed[z0,y1,x1]-fb0,dx,fb0)
    fc=fma64(fixed[z1,y0,x1]-fc0,dx,fc0)
    fd=fma64(fixed[z1,y1,x1]-fd0,dx,fd0)
    flo,fhi=fma64(fb-fa,dy,fa),fma64(fd-fc,dy,fc)
    fvalue=fma64(fhi-flo,dz,flo)
    ma0,mb0=moving[z0,y0,x0],moving[z0,y1,x0]
    mc0,md0=moving[z1,y0,x0],moving[z1,y1,x0]
    ma=fma64(moving[z0,y0,x1]-ma0,dx,ma0)
    mb=fma64(moving[z0,y1,x1]-mb0,dx,mb0)
    mc=fma64(moving[z1,y0,x1]-mc0,dx,mc0)
    md=fma64(moving[z1,y1,x1]-md0,dx,md0)
    mlo,mhi=fma64(mb-ma,dy,ma),fma64(md-mc,dy,mc)
    mvalue=fma64(mhi-mlo,dz,mlo)
    return fvalue,mvalue

@njit(cache=True, fastmath=False)
def _row(fixed, moving, valid, radii, geometry, origin, inverse, spacing,
         z, y, forward, reverse, scores):
    nz,ny,nx = fixed.shape
    rx,ry,rz = radii
    # Keep each x plane's zz/yy addition sequence, but stream adjacent x
    # values. No separable box sums or reassociation of window statistics.
    planes = np.zeros((6,nx),np.float64)
    for zz in range(max(0,z-rz),min(nz,z+rz+1)):
        for yy in range(max(0,y-ry),min(ny,y+ry+1)):
            for x in range(nx):
                f,m = fixed[zz,yy,x],moving[zz,yy,x]
                planes[0,x] += f*f
                planes[1,x] += m*m
                planes[2,x] += f
                planes[3,x] += m
                planes[4,x] += f*m
                planes[5,x] += 1.0
    totals = np.empty(6,np.float64)
    point,negative,positive,gradient = (np.empty(3,np.float64),np.empty(3,np.float64),
                                       np.empty(3,np.float64),np.empty(3,np.float64))
    for x in range(nx):
        for j in range(6):
            totals[j] = 0.0
        for xx in range(x-rx,x+rx+1):
            for j in range(6):
                totals[j] += planes[j,xx] if 0 <= xx < nx else 0.0
        fs,fw,fa = _score_weight(fixed[z,y,x],moving[z,y,x],totals,False)
        rs,rw,ra = _score_weight(moving[z,y,x],fixed[z,y,x],totals,True)
        scores[0,z,y,x],scores[1,z,y,x]=fs,rs
        if fa or ra:
            for i in range(3):
                point[i] = geometry[i,3]+geometry[i,0]*x+geometry[i,1]*y+geometry[i,2]*z
            # Both images share the physical gradient stencil coordinates.
            # Preserve the existing CC rounding contract for these neighbours.
            for c in range(3):
                offset=.5*spacing[c]
                for i in range(3):
                    minus,plus=0.0,0.0
                    for j in range(3):
                        coordinate=point[j]-origin[j]
                        minus += inverse[i,j]*(coordinate-(offset if c==j else 0.0))
                        plus += inverse[i,j]*(coordinate+(offset if c==j else 0.0))
                    negative[i],positive[i]=minus,plus
                delta=(point[c]+offset)-(point[c]-offset)
                good=(inside(negative[0],negative[1],negative[2],nx,ny,nz) and
                      inside(positive[0],positive[1],positive[2],nx,ny,nz) and
                      delta > 10*np.finfo(np.float64).eps)
                if good:
                    pf,pm=_pair_value(fixed,moving,positive[0],positive[1],positive[2])
                    nf,nm=_pair_value(fixed,moving,negative[0],negative[1],negative[2])
                    gf,gm=(pf-nf)/delta,(pm-nm)/delta
                else:
                    gf,gm=0.0,0.0
                if fa:
                    forward[c,z,y,x]=fw*gm
                if ra:
                    reverse[c,z,y,x]=rw*gf


@njit(cache=True, fastmath=False)
def _finish(scores):
    f, r = scores[0].ravel(), scores[1].ravel()
    fs, rs = 0.0, 0.0
    for n in range(f.size):
        fs -= f[n]
        rs -= r[n]
    return fs, rs


@njit(cache=True, fastmath=False, parallel=True)
def _work(fixed, moving, radii, geometry, origin, inverse, spacing, use_parallel):
    nz, ny, nx = fixed.shape
    valid = np.empty((0,0,0), np.bool_)  # unmasked pair; no volume-sized all-ones mask
    forward = np.zeros((3, nz, ny, nx), np.float64)
    reverse = np.zeros((3, nz, ny, nx), np.float64)
    scores = np.empty((2, nz, ny, nx), np.float64)
    if use_parallel:
        for task in prange(nz*ny):
            row = np.int64(task)
            _row(fixed, moving, valid, radii, geometry, origin, inverse, spacing,
                 row//ny, row%ny, forward, reverse, scores)
    else:
        for row in range(nz*ny):
            _row(fixed, moving, valid, radii, geometry, origin, inverse, spacing,
                 row//ny, row%ny, forward, reverse, scores)
    fs, rs = _finish(scores)
    return fs, rs, forward, reverse


def local_cc_pair(fixed, moving, radii, geometry, origin, inverse, spacing):
    return _work(fixed, moving, radii, geometry, origin, inverse, spacing,
                 fixed.size >= 16384 and get_num_threads() > 1)
