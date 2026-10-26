"""Ordinary FP32 Gaussian candidate using two parallel second-order sections.

Derived from ITK recursive Gaussian coefficients (NumFOCUS, Apache-2.0).

The existing fourth-order transfer function is a sum of two damped sinusoids.
Keeping the conjugate pole pairs separate reduces feedback cancellation.
Only the bounded coefficient table is constructed in native CPU double.
The states are driven by neighboring-pixel differences and the output is
input plus a symmetric correction; constant signals are preserved exactly.
All per-image data, residuals, products and recurrence states stay float32.
The ordinary-FP32 image smoothing wrappers use these sections by default.
"""
import math
from functools import lru_cache
import numpy as np
from numba import njit, prange
from fast_ants.core.arithmetic32 import fma32
from fast_ants.pipeline.gaussian import recursive_gaussian_coefficients


@lru_cache(maxsize=64)
def coefficients32(sigma):
    if not np.isfinite(sigma) or not 0 < sigma <= 64:
        raise ValueError("Section Gaussian sigma must lie in (0,64] voxels")
    numerator,denominator,*_=recursive_gaussian_coefficients(sigma)
    normalization=(1.3530-.3531)/numerator[0]
    rows=[]
    for a,b,w,decay in ((1.3530,1.8151,.6681,1.3932),(-.3531,.0902,2.0787,1.3732)):
        e=math.exp(-decay/sigma);c=math.cos(w/sigma);s=math.sin(w/sigma)
        d1,d2=-2*e*c,e*e
        n0,n1=a/normalization,e*(b*s-a*c)/normalization
        m1,m2=n1-d1*n0,-d2*n0
        reverse_gain=(m1+m2)/(1+d1+d2)
        rows.append([n0,n1,d1,d2,m1,m2,reverse_gain])
    # Verify the algebra before quantizing the small table, independently of
    # the recurrence implementation and without a polynomial root solver.
    first,second=rows
    reconstructed=(np.convolve(first[:2],[1,*second[2:4]])+
                   np.convolve(second[:2],[1,*first[2:4]]))
    if (not np.allclose(reconstructed,numerator,rtol=5e-13,atol=5e-15) or
        not np.allclose(np.convolve([1,*first[2:4]],[1,*second[2:4]])[1:],denominator,rtol=5e-13,atol=5e-15)):
        raise ArithmeticError("Section factorization did not reconstruct the transfer function")
    # For each causal section H(q), G=H(1). Its residual H-G has a
    # difference-input realization A*(x_i-x_{i-1})+B*(x_{i-1}-x_{i-2}).
    # The anti-causal residual is the same H-G applied in reverse. The two
    # symmetric sections have total DC gain one, so output=x+sum(residuals).
    table=[]
    for n0,n1,d1,d2,_,_,_ in rows:
        gain=(n0+n1)/(1+d1+d2)
        table.append([d1,d2,n0-gain,d2*gain])
    table=np.asarray(table,np.float32)
    if not np.isfinite(table).all() or np.any(np.float32(1)+table[:,0]+table[:,1]<=0):
        raise ArithmeticError("Section coefficients are not stable in float32")
    table.setflags(write=False)
    return table


@njit(cache=True,fastmath=False,parallel=True)
def _section_lines32(lines,coefficients):
    output=np.empty(lines.shape,np.float32);length=lines.shape[1]
    for line in prange(lines.shape[0]):
        forward=np.empty((2,length),np.float32)
        for section in range(2):
            previous=previous2=old_delta=np.float32(0)
            old=lines[line,0]
            d1,d2,a,b=coefficients[section]
            for i in range(length):
                current=lines[line,i];delta=current-old
                value=a*delta
                value=fma32(b,old_delta,value)
                value=fma32(-d1,previous,value)
                value=fma32(-d2,previous2,value)
                forward[section,i]=value
                previous2,previous=previous,value
                old_delta,old=delta,current
        reverse1=reverse2=reverse12=reverse22=old_delta=np.float32(0)
        old=lines[line,length-1]
        for i in range(length-1,-1,-1):
            current=lines[line,i];delta=current-old
            value1=coefficients[0,2]*delta
            value1=fma32(coefficients[0,3],old_delta,value1)
            value1=fma32(-coefficients[0,0],reverse1,value1)
            value1=fma32(-coefficients[0,1],reverse12,value1)
            value2=coefficients[1,2]*delta
            value2=fma32(coefficients[1,3],old_delta,value2)
            value2=fma32(-coefficients[1,0],reverse2,value2)
            value2=fma32(-coefficients[1,1],reverse22,value2)
            reverse12,reverse1=reverse1,value1;reverse22,reverse2=reverse2,value2
            old_delta,old=delta,current
            correction=np.float32(forward[0,i]+value1)+np.float32(forward[1,i]+value2)
            output[line,i]=current+correction
    return output


def section_lines32(lines,coefficients):
    """Apply the two sections to FP32 rows; CPU voxel arithmetic stays FP32."""
    if not isinstance(lines,np.ndarray) or lines.dtype!=np.float32 or lines.ndim!=2:
        raise ValueError('Section filter requires a two-dimensional float32 array')
    if not isinstance(coefficients,np.ndarray) or coefficients.dtype!=np.float32 or coefficients.shape!=(2,4):
        raise ValueError('Section filter requires a (2,4) float32 coefficient table')
    if not lines.size:return lines.copy()
    return _section_lines32(np.ascontiguousarray(lines),coefficients)


def smooth_sections32(image,sigma,units='vox'):
    from fast_ants.core.types import Image
    sigmas=np.full(3,float(sigma)) if np.ndim(sigma)==0 else np.asarray(sigma,np.float64)
    if sigmas.shape!=(3,) or not np.isfinite(sigmas).all() or np.any(sigmas<0):raise ValueError('Invalid sigmas')
    if units=='mm':sigmas=sigmas/image.grid.spacing_xyz
    elif units!='vox':raise ValueError('Invalid units')
    if image.data.dtype!=np.float32:raise ValueError('Ordinary float32 image required')
    values=image.data.copy()
    for dimension in (2,0,1):
        if sigmas[dimension]==0:continue
        axis=2-dimension
        if values.shape[axis]<4:raise ValueError('Four pixels required along filtered axes')
        moved=np.moveaxis(values,axis,-1);shape=moved.shape
        lines=np.ascontiguousarray(moved.reshape(-1,shape[-1]))
        filtered=section_lines32(lines,coefficients32(float(sigmas[dimension])))
        values=np.ascontiguousarray(np.moveaxis(filtered.reshape(shape),-1,axis))
    return Image(values,image.grid,image.metadata)
