"""Explicit FP32 dense kernels, strict math, local coordinates and stable CC.

Gaussian recurrence and inverse clipping follow ITK, NumFOCUS/Apache-2.0.
No FP64 value is used in these dense loops; geometry coefficients are supplied
as FP32 after small CPU FP64 matrix setup. See validation dtype audits.
"""

import numpy as np
from numba import njit,prange
from fast_ants.pipeline.float1_policy import FLOAT1_CC_EPS

ZERO=np.float32(0)
ONE=np.float32(1)
HALF=np.float32(.5)
TWO=np.float32(2)
# Match the float1 CC metric's score/update degeneracy branches.
CC_EPS=FLOAT1_CC_EPS


@njit(cache=True,fastmath=False)
def add32(total,correction,value):
    corrected=value-correction
    updated=total+corrected
    return updated,(updated-total)-corrected


@njit(cache=True,fastmath=False)
def inside32(x,y,z,nx,ny,nz):
    return (-HALF<=x<np.float32(nx)-HALF and -HALF<=y<np.float32(ny)-HALF and -HALF<=z<np.float32(nz)-HALF)


@njit(cache=True,fastmath=False)
def linear32(data,x,y,z):
    nz,ny,nx=data.shape
    x=min(max(x,ZERO),np.float32(nx-1)); y=min(max(y,ZERO),np.float32(ny-1)); z=min(max(z,ZERO),np.float32(nz-1))
    x0,y0,z0=int(np.floor(x)),int(np.floor(y)),int(np.floor(z))
    x1,y1,z1=min(x0+1,nx-1),min(y0+1,ny-1),min(z0+1,nz-1)
    dx,dy,dz=x-np.float32(x0),y-np.float32(y0),z-np.float32(z0)
    a=data[z0,y0,x0]+(data[z0,y0,x1]-data[z0,y0,x0])*dx
    b=data[z0,y1,x0]+(data[z0,y1,x1]-data[z0,y1,x0])*dx
    c=data[z1,y0,x0]+(data[z1,y0,x1]-data[z1,y0,x0])*dx
    d=data[z1,y1,x0]+(data[z1,y1,x1]-data[z1,y1,x0])*dx
    lo,hi=a+(b-a)*dy,c+(d-c)*dy
    return lo+(hi-lo)*dz


@njit(cache=True,fastmath=False)
def linear_shift32(data,x,y,z,anchor):
    """Interpolate deviations from a shared anchor before taking a gradient."""
    nz,ny,nx=data.shape
    x=min(max(x,ZERO),np.float32(nx-1));y=min(max(y,ZERO),np.float32(ny-1));z=min(max(z,ZERO),np.float32(nz-1))
    x0,y0,z0=int(np.floor(x)),int(np.floor(y)),int(np.floor(z))
    x1,y1,z1=min(x0+1,nx-1),min(y0+1,ny-1),min(z0+1,nz-1)
    dx,dy,dz=x-np.float32(x0),y-np.float32(y0),z-np.float32(z0)
    a=(data[z0,y0,x0]-anchor)+(data[z0,y0,x1]-data[z0,y0,x0])*dx
    b=(data[z0,y1,x0]-anchor)+(data[z0,y1,x1]-data[z0,y1,x0])*dx
    c=(data[z1,y0,x0]-anchor)+(data[z1,y0,x1]-data[z1,y0,x0])*dx
    d=(data[z1,y1,x0]-anchor)+(data[z1,y1,x1]-data[z1,y1,x0])*dx
    lo,hi=a+(b-a)*dy,c+(d-c)*dy
    return lo+(hi-lo)*dz


@njit(cache=True,fastmath=False)
def warp32(data,field,index_linear,index_offset,field_to_index,nearest=False):
    _,nz,ny,nx=field.shape
    output=np.zeros((nz,ny,nx),dtype=np.float32)
    iz,iy,ix=data.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                index=np.empty(3,np.float32)
                for c in range(3):
                    value=index_offset[c]+index_linear[c,0]*np.float32(x)+index_linear[c,1]*np.float32(y)+index_linear[c,2]*np.float32(z)
                    for j in range(3):value+=field_to_index[c,j]*field[j,z,y,x]
                    index[c]=value
                if inside32(index[0],index[1],index[2],ix,iy,iz):
                    if nearest:
                        output[z,y,x]=data[int(np.floor(index[2]+HALF)),int(np.floor(index[1]+HALF)),int(np.floor(index[0]+HALF))]
                    else:
                        output[z,y,x]=linear32(data,index[0],index[1],index[2])
    return output


@njit(cache=True,fastmath=False)
def gradient32(data,half_spacing_to_index,spacing):
    nz,ny,nx=data.shape
    output=np.zeros((3,nz,ny,nx),dtype=np.float32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(3):
                    dx,dy,dz=half_spacing_to_index[0,c],half_spacing_to_index[1,c],half_spacing_to_index[2,c]
                    px,py,pz=np.float32(x)+dx,np.float32(y)+dy,np.float32(z)+dz
                    mx,my,mz=np.float32(x)-dx,np.float32(y)-dy,np.float32(z)-dz
                    if inside32(px,py,pz,nx,ny,nz) and inside32(mx,my,mz,nx,ny,nz):
                        anchor=data[z,y,x]
                        output[c,z,y,x]=(linear_shift32(data,px,py,pz,anchor)-linear_shift32(data,mx,my,mz,anchor))/spacing[c]
    return output


@njit(cache=True,fastmath=False,parallel=True)
def cc32(fixed,moving,gradient,valid,radius):
    nz,ny,nx=fixed.shape
    update=np.zeros((3,nz,ny,nx),np.float32)
    scores=np.zeros(fixed.shape,np.float32);selected=np.zeros(fixed.shape,np.bool_)
    for task in prange(nz):
        z=np.int64(task)
        for y in range(ny):
            for x in range(nx):
                if not valid[z,y,x]:continue
                f0,m0=fixed[z,y,x],moving[z,y,x]
                sf,sm,cf,cm=ZERO,ZERO,ZERO,ZERO
                count=0
                for zz in range(max(0,z-radius),min(nz,z+radius+1)):
                    for yy in range(max(0,y-radius),min(ny,y+radius+1)):
                        for xx in range(max(0,x-radius),min(nx,x+radius+1)):
                            if valid[zz,yy,xx]:
                                sf,cf=add32(sf,cf,fixed[zz,yy,xx]-f0)
                                sm,cm=add32(sm,cm,moving[zz,yy,xx]-m0)
                                count+=1
                if count==0:continue
                mean_f,mean_m=sf/np.float32(count),sm/np.float32(count)
                ff,mm,fm,cff,cmm,cfm=ZERO,ZERO,ZERO,ZERO,ZERO,ZERO
                for zz in range(max(0,z-radius),min(nz,z+radius+1)):
                    for yy in range(max(0,y-radius),min(ny,y+radius+1)):
                        for xx in range(max(0,x-radius),min(nx,x+radius+1)):
                            if valid[zz,yy,xx]:
                                f=(fixed[zz,yy,xx]-f0)-mean_f
                                m=(moving[zz,yy,xx]-m0)-mean_m
                                ff,cff=add32(ff,cff,f*f)
                                mm,cmm=add32(mm,cmm,m*m)
                                fm,cfm=add32(fm,cfm,f*m)
                product=ff*mm
                score=ONE
                if abs(product)>CC_EPS:score=fm*fm/product
                if ff>CC_EPS and mm>CC_EPS:
                    weight=TWO*fm/product*(-mean_f+fm/mm*mean_m)
                    for c in range(3):update[c,z,y,x]=weight*gradient[c,z,y,x]
                scores[z,y,x]=-score;selected[z,y,x]=True
    total,total_correction=ZERO,ZERO;centres=0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if selected[z,y,x]:
                    total,total_correction=add32(total,total_correction,scores[z,y,x]);centres+=1
    value=total/np.float32(centres) if centres else ZERO
    return value,centres,update


@njit(cache=True,fastmath=False)
def recursive32(lines,numerator,denominator,anticausal,causal_boundary,anti_boundary):
    output=np.empty(lines.shape,np.float32)
    for line_number in range(lines.shape[0]):
        values=lines[line_number]
        length=values.size
        anchor=values[0]
        forward,backward=np.empty(length,np.float32),np.empty(length,np.float32)
        for i in range(length):
            feed,feedback,feed_c,feedback_c=ZERO,ZERO,ZERO,ZERO
            for k in range(4):
                feed,feed_c=add32(feed,feed_c,numerator[k]*(values[max(0,i-k)]-anchor))
                v=denominator[k]*forward[i-k-1] if i-k-1>=0 else causal_boundary[k]*(values[0]-anchor)
                feedback,feedback_c=add32(feedback,feedback_c,v)
            forward[i]=feed-feedback
        for i in range(length-1,-1,-1):
            feed,feedback,feed_c,feedback_c=ZERO,ZERO,ZERO,ZERO
            for k in range(4):
                feed,feed_c=add32(feed,feed_c,anticausal[k]*(values[min(length-1,i+k+1)]-anchor))
                v=denominator[k]*backward[i+k+1] if i+k+1<length else anti_boundary[k]*(values[length-1]-anchor)
                feedback,feedback_c=add32(feedback,feedback_c,v)
            backward[i]=feed-feedback
        for i in range(length):output[line_number,i]=forward[i]+backward[i]+anchor
    return output


@njit(cache=True,fastmath=False)
def convolve32(lines,kernel):
    output=np.empty(lines.shape,np.float32)
    radius=kernel.size//2
    for line in range(lines.shape[0]):
        for i in range(lines.shape[1]):
            total,correction=ZERO,ZERO
            for k in range(kernel.size):
                value=kernel[k]*lines[line,min(max(i+k-radius,0),lines.shape[1]-1)]
                total,correction=add32(total,correction,value)
            output[line,i]=total
    return output


@njit(cache=True,fastmath=False)
def mix_boundary32(smoothed,original,weight):
    output=np.empty(original.shape,np.float32)
    _,nz,ny,nx=original.shape
    for c in range(3):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    boundary=(x==0 or y==0 or z==0 or x==nx-1 or y==ny-1 or z==nz-1)
                    output[c,z,y,x]=ZERO if boundary else smoothed[c,z,y,x]*weight+original[c,z,y,x]*(ONE-weight)
    return output


@njit(cache=True,fastmath=False)
def norm32(field,spacing):
    _,nz,ny,nx=field.shape
    norms=np.empty((nz,ny,nx),np.float32)
    total,correction,maximum=ZERO,ZERO,ZERO
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                squared=ZERO
                for c in range(3):
                    component=field[c,z,y,x]/spacing[c]
                    squared+=component*component
                norm=np.sqrt(squared)
                norms[z,y,x]=norm
                maximum=max(maximum,norm)
                total,correction=add32(total,correction,norm)
    return norms,total/np.float32(nz*ny*nx),maximum


@njit(cache=True,fastmath=False)
def multiply32(field,factor):
    output=np.empty(field.shape,np.float32)
    src,dst=field.ravel(),output.ravel()
    for i in range(src.size):dst[i]=src[i]*factor
    return output


@njit(cache=True,fastmath=False,parallel=True)
def multiply_into32(field,factor,output):
    """Private scratch-output form; callers retain the unscaled source."""
    src,dst=field.ravel(),output.ravel()
    for i in prange(src.size):dst[i]=src[i]*factor
    return output


@njit(cache=True,fastmath=False,inline='always')
def _compose_index32(c,x,y,z,d0,d1,d2,index_linear,index_offset,outer_inverse):
    # Preserve the original left-to-right FP32 coordinate arithmetic without
    # allocating a three-element array for every output voxel.
    value=index_offset[c]+index_linear[c,0]*np.float32(x)+index_linear[c,1]*np.float32(y)+index_linear[c,2]*np.float32(z)
    value+=outer_inverse[c,0]*d0
    value+=outer_inverse[c,1]*d1
    value+=outer_inverse[c,2]*d2
    return value


@njit(cache=True,fastmath=False,parallel=True)
def compose_into32(outer,inner,index_linear,index_offset,outer_inverse,output):
    """Compose three field components using one interpolation geometry."""
    _,nz,ny,nx=inner.shape
    _,oz,oy,ox=outer.shape
    for task in prange(nz*ny):
        row=np.int64(task)
        z,y=row//ny,row%ny
        for x in range(nx):
            d0,d1,d2=inner[0,z,y,x],inner[1,z,y,x],inner[2,z,y,x]
            px=_compose_index32(0,x,y,z,d0,d1,d2,index_linear,index_offset,outer_inverse)
            py=_compose_index32(1,x,y,z,d0,d1,d2,index_linear,index_offset,outer_inverse)
            pz=_compose_index32(2,x,y,z,d0,d1,d2,index_linear,index_offset,outer_inverse)
            if inside32(px,py,pz,ox,oy,oz):
                # Bounds, floors and fractional weights are shared by the
                # vector's three components; interpolate in the original
                # x-then-y-then-z order for each component.
                px=min(max(px,ZERO),np.float32(ox-1))
                py=min(max(py,ZERO),np.float32(oy-1))
                pz=min(max(pz,ZERO),np.float32(oz-1))
                x0,y0,z0=int(np.floor(px)),int(np.floor(py)),int(np.floor(pz))
                x1,y1,z1=min(x0+1,ox-1),min(y0+1,oy-1),min(z0+1,oz-1)
                dx,dy,dz=px-np.float32(x0),py-np.float32(y0),pz-np.float32(z0)
                for c in range(3):
                    a=outer[c,z0,y0,x0]+(outer[c,z0,y0,x1]-outer[c,z0,y0,x0])*dx
                    b=outer[c,z0,y1,x0]+(outer[c,z0,y1,x1]-outer[c,z0,y1,x0])*dx
                    low=outer[c,z1,y0,x0]+(outer[c,z1,y0,x1]-outer[c,z1,y0,x0])*dx
                    high=outer[c,z1,y1,x0]+(outer[c,z1,y1,x1]-outer[c,z1,y1,x0])*dx
                    lo,hi=a+(b-a)*dy,low+(high-low)*dy
                    value=lo+(hi-lo)*dz
                    output[c,z,y,x]=inner[c,z,y,x]+value
            else:
                for c in range(3):output[c,z,y,x]=inner[c,z,y,x]+ZERO
    return output


@njit(cache=True,fastmath=False)
def compose32(outer,inner,index_linear,index_offset,outer_inverse):
    output=np.empty(inner.shape,np.float32)
    return compose_into32(outer,inner,index_linear,index_offset,outer_inverse,output)


@njit(cache=True,fastmath=False)
def inverse_step32(inverse,residual,norms,maximum,epsilon,enforce_boundary):
    _,nz,ny,nx=inverse.shape
    output=np.empty(inverse.shape,np.float32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                factor=ONE
                if norms[z,y,x]>epsilon*maximum:factor=epsilon*maximum/norms[z,y,x]
                boundary=(x==0 or y==0 or z==0 or x==nx-1 or y==ny-1 or z==nz-1)
                for c in range(3):
                    value=inverse[c,z,y,x]-residual[c,z,y,x]*factor*epsilon
                    output[c,z,y,x]=ZERO if enforce_boundary and boundary else value
    return output


@njit(cache=True,fastmath=False)
def rescale32(data):
    flat=data.ravel()
    minimum,maximum=flat[0],flat[0]
    for i in range(1,flat.size):minimum=min(minimum,flat[i]);maximum=max(maximum,flat[i])
    output=np.empty(data.shape,np.float32)
    dst=output.ravel()
    span=maximum-minimum
    for i in range(flat.size):dst[i]=(flat[i]-minimum)/span if span>ZERO else ZERO
    return output


@njit(cache=True,fastmath=False)
def range32(data):
    values=data.ravel();minimum,maximum=values[0],values[0]
    for i in range(1,values.size):minimum=min(minimum,values[i]);maximum=max(maximum,values[i])
    return minimum,maximum


@njit(cache=True,fastmath=False)
def window32(data,low,high,scale):
    output=np.empty(data.shape,np.float32)
    src,dst=data.ravel(),output.ravel()
    for i in range(src.size):
        value=src[i]
        if scale==ZERO:dst[i]=ZERO
        elif value<low:dst[i]=ZERO
        elif value>high:dst[i]=ONE
        else:dst[i]=(value-low)*scale
    return output


@njit(cache=True,fastmath=False)
def maximum_vector32(field,spacing):
    values=field.reshape(3,-1);maximum=ZERO;selected=0
    for i in range(values.shape[1]):
        squared=ZERO
        for c in range(3):
            v=values[c,i]/spacing[c];squared+=v*v
        if squared>maximum:maximum=squared;selected=i
    return values[:,selected].copy()
