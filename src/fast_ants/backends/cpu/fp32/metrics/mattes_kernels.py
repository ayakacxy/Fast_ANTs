"""Compensated FP32 Mattes histograms; ITK algorithm, NumFOCUS/Apache-2.0."""

import numpy as np
from numba import njit,prange
from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import ONE
from fast_ants.backends.cpu.fp32.kernels import TWO
from fast_ants.backends.cpu.fp32.kernels import HALF
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.kernels import inside32
from fast_ants.backends.cpu.fp32.kernels import linear32
from fast_ants.backends.cpu.fp32.kernels import linear_shift32
from fast_ants.pipeline.float1_policy import FLOAT1_PDF_EPS
from fast_ants.core.arithmetic32 import fma32

THREE=np.float32(3)
FOUR=np.float32(4)
SIX=np.float32(6)
ONE_HALF=np.float32(1.5)
SENTINEL=np.float32(np.finfo(np.float32).max)


@njit(cache=True,fastmath=False)
def cubic32(argument):
    a=abs(argument)
    if a<ONE:return fma32(a*a,fma32(THREE,a,-SIX),FOUR)/SIX
    if a<TWO:return (TWO-a)*(TWO-a)*(TWO-a)/SIX
    return ZERO


@njit(cache=True,fastmath=False)
def cubic_derivative32(argument):
    a=abs(argument);sign=ONE if argument>=ZERO else -ONE
    if a<ONE:return sign*a*fma32(ONE_HALF,a,-TWO)
    if a<TWO:return -HALF*sign*(TWO-a)*(TWO-a)
    return ZERO


@njit(cache=True,fastmath=False)
def mattes32(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax):
    fw,mw=(fmax-fmin)/np.float32(bins-4),(mmax-mmin)/np.float32(bins-4)
    joint=np.zeros((bins,bins),np.float32);correction=np.zeros((bins,bins),np.float32)
    fixed_pdf=np.zeros(bins,np.float32)
    np_=d_intensity.shape[1]
    derivative=np.zeros((bins,bins,np_),np.float32);dcorrection=np.zeros((bins,bins,np_),np.float32)
    valid=0
    for n in range(fixed.size):
        if moving[n]<mmin or moving[n]>mmax:continue
        fi=min(max(int(fixed[n]/fw-(fmin/fw-TWO)),2),bins-3)
        term=moving[n]/mw-(mmin/mw-TWO)
        mi=min(max(int(term),2),bins-3)
        fixed_pdf[fi]+=ONE;valid+=1
        for j in range(mi-1,mi+3):
            argument=np.float32(j)-term
            joint[fi,j],correction[fi,j]=add32(joint[fi,j],correction[fi,j],cubic32(argument))
            weight=cubic_derivative32(argument)
            for p in range(np_):
                derivative[fi,j,p],dcorrection[fi,j,p]=add32(derivative[fi,j,p],dcorrection[fi,j,p],d_intensity[n,p]*weight)
    total,tc=ZERO,ZERO
    for i in range(bins):
        for j in range(bins):total,tc=add32(total,tc,joint[i,j])
    moving_pdf=np.zeros(bins,np.float32)
    if not valid or total<FLOAT1_PDF_EPS:return SENTINEL,np.zeros(np_,np.float32),joint,fixed_pdf,moving_pdf,derivative,valid
    factor=-ONE/(mw*np.float32(valid))
    for i in range(bins):
        fixed_pdf[i]/=total
        for j in range(bins):
            joint[i,j]/=total
            for p in range(np_):derivative[i,j,p]*=factor
    for j in range(bins):
        subtotal,sc=ZERO,ZERO
        for i in range(bins):subtotal,sc=add32(subtotal,sc,joint[i,j])
        moving_pdf[j]=subtotal
    information,ic=ZERO,ZERO
    descent=np.zeros(np_,np.float32);dc=np.zeros(np_,np.float32)
    for i in range(bins):
        if fixed_pdf[i]<=FLOAT1_PDF_EPS:continue
        log_fixed=np.log(fixed_pdf[i])
        for j in range(bins):
            if joint[i,j]>FLOAT1_PDF_EPS and moving_pdf[j]>FLOAT1_PDF_EPS:
                ratio=np.log(joint[i,j]/moving_pdf[j])
                information,ic=add32(information,ic,joint[i,j]*(ratio-log_fixed))
                for p in range(np_):descent[p],dc[p]=add32(descent[p],dc[p],derivative[i,j,p]*ratio)
    return -information,descent,joint,fixed_pdf,moving_pdf,derivative,valid


@njit(cache=True,fastmath=False)
def index_points32(shape_zyx):
    nz,ny,nx=shape_zyx
    points=np.empty((nz*ny*nx,3),np.float32);n=0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                points[n,0]=np.float32(x);points[n,1]=np.float32(y);points[n,2]=np.float32(z);n+=1
    return points


@njit(cache=True,fastmath=False)
def affine_indices32(points,linear,offset):
    output=np.empty(points.shape,np.float32)
    for n in range(points.shape[0]):
        for c in range(3):
            value=offset[c]
            for j in range(3):value+=linear[c,j]*points[n,j]
            output[n,c]=value
    return output


@njit(cache=True,fastmath=False)
def point_samples32(data,indices,nearest=False):
    values=np.zeros(indices.shape[0],np.float32);valid=np.zeros(indices.shape[0],np.bool_)
    nz,ny,nx=data.shape
    for n in range(indices.shape[0]):
        x,y,z=indices[n,0],indices[n,1],indices[n,2]
        if inside32(x,y,z,nx,ny,nz):
            valid[n]=True
            if nearest:values[n]=data[int(np.floor(z+HALF)),int(np.floor(y+HALF)),int(np.floor(x+HALF))]
            else:values[n]=linear32(data,x,y,z)
    return values,valid


@njit(cache=True,fastmath=False)
def point_gradient32(data,indices,half_shift,spacing):
    output=np.zeros(indices.shape,np.float32)
    nz,ny,nx=data.shape
    for n in range(indices.shape[0]):
        x,y,z=indices[n,0],indices[n,1],indices[n,2]
        for c in range(3):
            dx,dy,dz=half_shift[0,c],half_shift[1,c],half_shift[2,c]
            if inside32(x-dx,y-dy,z-dz,nx,ny,nz) and inside32(x+dx,y+dy,z+dz,nx,ny,nz):
                anchor=linear32(data,x,y,z)
                output[n,c]=(linear_shift32(data,x+dx,y+dy,z+dz,anchor)-linear_shift32(data,x-dx,y-dy,z-dz,anchor))/spacing[c]
    return output


@njit(cache=True,fastmath=False,parallel=True)
def intensity_derivatives32(local_points,gradients,linear_derivatives,translation_derivatives):
    count=linear_derivatives.shape[0]
    output=np.empty((local_points.shape[0],count),np.float32)
    for n in prange(local_points.shape[0]):
        for p in range(count):
            total=ZERO
            for c in range(3):
                spatial=translation_derivatives[p,c]
                for j in range(3):spatial=fma32(linear_derivatives[p,c,j],local_points[n,j],spatial)
                total=fma32(gradients[n,c],spatial,total)
            output[n,p]=total
    return output


@njit(cache=True,fastmath=False)
def moments32(data):
    totals=np.zeros(4,np.float32);corrections=np.zeros(4,np.float32)
    for z in range(data.shape[0]):
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                value=data[z,y,x]
                totals[0],corrections[0]=add32(totals[0],corrections[0],value)
                totals[1],corrections[1]=add32(totals[1],corrections[1],value*(np.float32(x)-np.float32(data.shape[2]-1)*HALF))
                totals[2],corrections[2]=add32(totals[2],corrections[2],value*(np.float32(y)-np.float32(data.shape[1]-1)*HALF))
                totals[3],corrections[3]=add32(totals[3],corrections[3],value*(np.float32(z)-np.float32(data.shape[0]-1)*HALF))
    return totals


@njit(cache=True,fastmath=False,parallel=True)
def centered_affine_indices32(points,linear,offset,center):
    output=np.empty(points.shape,np.float32)
    for n in prange(points.shape[0]):
        for c in range(3):
            value=offset[c]
            for j in range(3):value+=linear[c,j]*(points[n,j]-center[j])
            output[n,c]=value
    return output


@njit(cache=True,fastmath=False)
def histogram_block32(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax,bin_indices=None,bin_fractions=None):
    """Ordinary FP32 values, Kahan sums and integer counts; no FP emulation."""
    fw=(fmax-fmin)/np.float32(bins-4);mw=(mmax-mmin)/np.float32(bins-4)
    joint=np.zeros((bins,bins),np.float32);jc=np.zeros(joint.shape,np.float32)
    counts=np.zeros(bins,np.int64);P=d_intensity.shape[1]
    derivative=np.zeros((bins,bins,P),np.float32);dc=np.zeros(derivative.shape,np.float32)
    valid=0
    for n in range(fixed.size):
        if moving[n]<mmin or moving[n]>mmax:continue
        fi=min(max(int(fixed[n]/fw-(fmin/fw-TWO)),2),bins-3)
        term=moving[n]/mw-(mmin/mw-TWO);mi=min(max(int(term),2),bins-3)
        if bin_indices is not None:mi=bin_indices[n]
        counts[fi]+=1;valid+=1
        for j in range(mi-1,mi+3):
            argument=np.float32(j)-term if bin_fractions is None else np.float32(j-mi)-bin_fractions[n]
            joint[fi,j],jc[fi,j]=add32(joint[fi,j],jc[fi,j],cubic32(argument))
            weight=cubic_derivative32(argument)
            for p in range(P):
                derivative[fi,j,p],dc[fi,j,p]=add32(derivative[fi,j,p],dc[fi,j,p],d_intensity[n,p]*weight)
    return joint,counts,derivative,valid,mw


@njit(cache=True,fastmath=False)
def histograms32_serial(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax,bin_indices=None,bin_fractions=None):
    """Fixed partitions and merge order shared with the Metal implementation."""
    block=max(2048,(fixed.size+127)//128);P=d_intensity.shape[1]
    joint=np.zeros((bins,bins),np.float32);jc=np.zeros(joint.shape,np.float32)
    derivatives=np.zeros((bins,bins,P),np.float32);dc=np.zeros(derivatives.shape,np.float32)
    counts=np.zeros(bins,np.int64);valid=0;mw=ONE
    for start in range(0,fixed.size,block):
        end=min(start+block,fixed.size)
        bi=None if bin_indices is None else bin_indices[start:end]
        bf=None if bin_fractions is None else bin_fractions[start:end]
        j,c,d,n,mw=histogram_block32(fixed[start:end],moving[start:end],d_intensity[start:end],bins,fmin,fmax,mmin,mmax,bi,bf)
        valid+=n
        for i in range(bins):
            counts[i]+=c[i]
            for k in range(bins):
                joint[i,k],jc[i,k]=add32(joint[i,k],jc[i,k],j[i,k])
                for p in range(P):derivatives[i,k,p],dc[i,k,p]=add32(derivatives[i,k,p],dc[i,k,p],d[i,k,p])
    return joint,counts,derivatives,valid,mw


@njit(cache=True,fastmath=False,parallel=True)
def local_affine32(points,integer_linear,residual_linear,integer_offset,residual_offset,origin):
    """Integer voxel location + ordinary FP32 displacement, not FP emulation."""
    bases=np.empty(points.shape,np.int32);offsets=np.empty(points.shape,np.float32)
    for n in prange(points.shape[0]):
        for c in range(3):
            base=integer_offset[c];value=residual_offset[c]
            for j in range(3):
                q=points[n,j]-origin[j];integer=int(np.floor(q));fraction=q-np.float32(integer)
                base+=integer_linear[c,j]*integer
                value=fma32(residual_linear[c,j],np.float32(integer),value)
                value=fma32(np.float32(integer_linear[c,j])+residual_linear[c,j],fraction,value)
            bases[n,c]=base;offsets[n,c]=value
    return bases,offsets


@njit(cache=True,fastmath=False,inline='always')
def _local_affine_integer_point32(points,n,integer_linear,residual_linear,integer_offset,
                                  residual_offset,origin,base,offset):
    """Dense-plan points and the floor-selected origin are exact integers."""
    q0=int(points[n,0]-origin[0]);q1=int(points[n,1]-origin[1]);q2=int(points[n,2]-origin[2])
    for c in range(3):
        b=integer_offset[c]
        b+=integer_linear[c,0]*q0
        b+=integer_linear[c,1]*q1
        b+=integer_linear[c,2]*q2
        v=fma32(residual_linear[c,0],np.float32(q0),residual_offset[c])
        v=fma32(residual_linear[c,1],np.float32(q1),v)
        v=fma32(residual_linear[c,2],np.float32(q2),v)
        base[c]=b;offset[c]=v


@njit(cache=True,fastmath=False,parallel=True)
def local_affine_integer32(points,integer_linear,residual_linear,integer_offset,residual_offset,origin):
    """Specialize dense integer samples; sparse points retain local_affine32."""
    bases=np.empty(points.shape,np.int32);offsets=np.empty(points.shape,np.float32)
    for n in prange(points.shape[0]):
        _local_affine_integer_point32(points,n,integer_linear,residual_linear,integer_offset,
                                      residual_offset,origin,bases[n],offsets[n])
    return bases,offsets


@njit(cache=True,fastmath=False)
def local_inside32(base,offset,shape):
    for c in range(3):
        whole=int(np.floor(offset[c]));f=offset[c]-np.float32(whole);b=base[c]+whole
        if b < -1 or b==-1 and f<HALF or b>=shape[2-c] or b==shape[2-c]-1 and f>=HALF:return False
    return True


@njit(cache=True,fastmath=False,inline='always')
def _local_axis32(base,offset,size):
    whole=int(np.floor(offset));f=offset-np.float32(whole);b=base+whole
    valid=not (b < -1 or b==-1 and f<HALF or b>=size or b==size-1 and f>=HALF)
    if b<0:b=0;f=ZERO
    if b>=size-1:b=size-1;f=ZERO
    return b,min(b+1,size-1),f,valid


@njit(cache=True,fastmath=False,inline='always')
def _local_at32(data,bx,by,bz,x,y,z,anchor):
    x0,x1,dx,_=_local_axis32(bx,x,data.shape[2])
    y0,y1,dy,_=_local_axis32(by,y,data.shape[1])
    z0,z1,dz,_=_local_axis32(bz,z,data.shape[0])
    a=fma32(data[z0,y0,x1]-data[z0,y0,x0],dx,data[z0,y0,x0]-anchor)
    b=fma32(data[z0,y1,x1]-data[z0,y1,x0],dx,data[z0,y1,x0]-anchor)
    c=fma32(data[z1,y0,x1]-data[z1,y0,x0],dx,data[z1,y0,x0]-anchor)
    d=fma32(data[z1,y1,x1]-data[z1,y1,x0],dx,data[z1,y1,x0]-anchor)
    lo=fma32(b-a,dy,a);hi=fma32(d-c,dy,c)
    return fma32(hi-lo,dz,lo)


@njit(cache=True,fastmath=False,inline='always')
def _interpolate_axes32(data,x0,x1,y0,y1,z0,z1,dx,dy,dz,anchor):
    """Interpolate already checked axes without repeating floors and clipping."""
    a=fma32(data[z0,y0,x1]-data[z0,y0,x0],dx,data[z0,y0,x0]-anchor)
    b=fma32(data[z0,y1,x1]-data[z0,y1,x0],dx,data[z0,y1,x0]-anchor)
    c=fma32(data[z1,y0,x1]-data[z1,y0,x0],dx,data[z1,y0,x0]-anchor)
    d=fma32(data[z1,y1,x1]-data[z1,y1,x0],dx,data[z1,y1,x0]-anchor)
    lo=fma32(b-a,dy,a);hi=fma32(d-c,dy,c)
    return fma32(hi-lo,dz,lo)


@njit(cache=True,fastmath=False,inline='always')
def _sample_inside32(data,base,offset):
    x0,x1,dx,vx=_local_axis32(base[0],offset[0],data.shape[2])
    if not vx:return ZERO,False
    y0,y1,dy,vy=_local_axis32(base[1],offset[1],data.shape[1])
    if not vy:return ZERO,False
    z0,z1,dz,vz=_local_axis32(base[2],offset[2],data.shape[0])
    if not vz:return ZERO,False
    return _interpolate_axes32(data,x0,x1,y0,y1,z0,z1,dx,dy,dz,ZERO),True


@njit(cache=True,fastmath=False,inline='always')
def local_sample32(data,base,offset,anchor,nearest=False):
    if nearest:
        x=base[0]+int(np.floor(offset[0]+HALF));y=base[1]+int(np.floor(offset[1]+HALF));z=base[2]+int(np.floor(offset[2]+HALF))
        return data[z,y,x]-anchor
    return _local_at32(data,base[0],base[1],base[2],offset[0],offset[1],offset[2],anchor)


@njit(cache=True,fastmath=False,parallel=True)
def local_samples32(data,bases,offsets,nearest=False):
    output=np.zeros(bases.shape[0],np.float32);valid=np.zeros(bases.shape[0],np.bool_)
    for n in prange(bases.shape[0]):
        if local_inside32(bases[n],offsets[n],data.shape):
            valid[n]=True;output[n]=local_sample32(data,bases[n],offsets[n],ZERO,nearest)
    return output,valid


@njit(cache=True,fastmath=False,parallel=True)
def local_gradient32(data,bases,offsets,shift,spacing):
    output=np.zeros(offsets.shape,np.float32)
    for n in prange(bases.shape[0]):
        bx,by,bz=bases[n];x,y,z=offsets[n]
        anchor=_local_at32(data,bx,by,bz,x,y,z,ZERO)
        for c in range(3):
            px=x+shift[0,c];py=y+shift[1,c];pz=z+shift[2,c]
            mx=x-shift[0,c];my=y-shift[1,c];mz=z-shift[2,c]
            _,_,_,v1=_local_axis32(bx,px,data.shape[2]);_,_,_,v2=_local_axis32(by,py,data.shape[1]);_,_,_,v3=_local_axis32(bz,pz,data.shape[0])
            _,_,_,v4=_local_axis32(bx,mx,data.shape[2]);_,_,_,v5=_local_axis32(by,my,data.shape[1]);_,_,_,v6=_local_axis32(bz,mz,data.shape[0])
            if v1 and v2 and v3 and v4 and v5 and v6:
                output[n,c]=(_local_at32(data,bx,by,bz,px,py,pz,anchor)-_local_at32(data,bx,by,bz,mx,my,mz,anchor))/spacing[c]
    return output


@njit(cache=True,fastmath=False,parallel=True)
def local_bin_fractions32(data,bases,offsets,values,minimum,inverse_width,anchors,biases,bins):
    """Interpolate deviations around each bin's binary intensity origin.

    Bin integer and unit-cell fraction are geometric histogram coordinates.
    Every image, displacement and intermediate arithmetic value is one FP32.
    """
    indices=np.empty(values.size,np.int32);fractions=np.empty(values.size,np.float32)
    for n in prange(values.size):
        k=min(max(int((values[n]-minimum)*inverse_width),0),bins-5)
        fraction=ZERO
        for attempt in range(3):
            residual=local_sample32(data,bases[n],offsets[n],anchors[k])
            fraction=(residual-biases[k])*inverse_width
            if fraction<ZERO and k>0:k-=1;continue
            if fraction>=ONE and k<bins-5:k+=1;continue
            break
        indices[n]=k+2;fractions[n]=fraction
    return indices,fractions


@njit(cache=True,fastmath=False)
def sample_bin_fractions32(values,width,normalized_min,bins):
    """Float1 Parzen terms from already rounded moving-image samples.

    Match the two division/subtraction assignment boundaries in ITK rather
    than sampling the image a second time around a different intensity anchor.
    """
    indices=np.empty(values.size,np.int32)
    fractions=np.empty(values.size,np.float32)
    for n in range(values.size):
        term=values[n]/width-normalized_min
        index=min(max(int(term),2),bins-3)
        indices[n]=index
        fractions[n]=term-np.float32(index)
    return indices,fractions


@njit(cache=True,fastmath=False)
def fixed_point_histograms32(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax,bin_indices,bin_fractions,joint_scale,derivative_scale):
    """Native integer histogram accumulation, with explicit overflow bounds.

    This is a reduction algorithm, not software floating-point arithmetic.
    Per-sample interpolation, spline weights and products remain ordinary FP32.
    """
    fw=(fmax-fmin)/np.float32(bins-4)
    joint=np.zeros((bins,bins),np.int64);counts=np.zeros(bins,np.int64)
    P=d_intensity.shape[1];derivatives=np.zeros((bins,bins,P),np.int64);valid=0
    for n in range(fixed.size):
        if moving[n]<mmin or moving[n]>mmax:continue
        fi=min(max(int((fixed[n]-fmin)/fw+TWO),2),bins-3);mi=bin_indices[n]
        counts[fi]+=1;valid+=1
        for j in range(mi-1,mi+3):
            argument=np.float32(j-mi)-bin_fractions[n]
            joint[fi,j]+=np.int64(np.rint(cubic32(argument)*joint_scale))
            weight=cubic_derivative32(argument)
            for p in range(P):derivatives[fi,j,p]+=np.int64(np.rint((d_intensity[n,p]*weight)*derivative_scale))
    return joint,counts,derivatives,valid


@njit(cache=True,fastmath=False,parallel=True)
def histograms32_parallel(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax,bin_indices=None,bin_fractions=None):
    block=max(2048,(fixed.size+127)//128);P=d_intensity.shape[1];chunks=(fixed.size+block-1)//block
    partial_joint=np.empty((chunks,bins,bins),np.float32)
    partial_derivatives=np.empty((chunks,bins,bins,P),np.float32)
    partial_counts=np.empty((chunks,bins),np.int64);partial_valid=np.empty(chunks,np.int64)
    for task in prange(chunks):
        chunk=np.int64(task);start=chunk*block;end=min(start+block,fixed.size)
        bi=None if bin_indices is None else bin_indices[start:end]
        bf=None if bin_fractions is None else bin_fractions[start:end]
        j,c,d,n,width=histogram_block32(fixed[start:end],moving[start:end],d_intensity[start:end],bins,fmin,fmax,mmin,mmax,bi,bf)
        partial_valid[chunk]=n
        for i in range(bins):
            partial_counts[chunk,i]=c[i]
            for k in range(bins):
                partial_joint[chunk,i,k]=j[i,k]
                for p in range(P):partial_derivatives[chunk,i,k,p]=d[i,k,p]
    joint=np.zeros((bins,bins),np.float32);jc=np.zeros(joint.shape,np.float32)
    derivatives=np.zeros((bins,bins,P),np.float32);dc=np.zeros(derivatives.shape,np.float32)
    counts=np.zeros(bins,np.int64);valid=0
    for chunk in range(chunks):
        valid+=partial_valid[chunk]
        for i in range(bins):
            counts[i]+=partial_counts[chunk,i]
            for k in range(bins):
                joint[i,k],jc[i,k]=add32(joint[i,k],jc[i,k],partial_joint[chunk,i,k])
                for p in range(P):derivatives[i,k,p],dc[i,k,p]=add32(derivatives[i,k,p],dc[i,k,p],partial_derivatives[chunk,i,k,p])
    width=(mmax-mmin)/np.float32(bins-4)
    return joint,counts,derivatives,valid,width


def histograms32(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax,bin_indices=None,bin_fractions=None):
    function=histograms32_parallel if fixed.size>=16384 else histograms32_serial
    return function(fixed,moving,d_intensity,bins,fmin,fmax,mmin,mmax,bin_indices,bin_fractions)
