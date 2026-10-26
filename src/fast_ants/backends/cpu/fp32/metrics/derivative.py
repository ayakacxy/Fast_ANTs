"""Streaming FP32 local sampling, image gradients and Mattes derivatives.

Each fixed sample block owns bounded histogram scratch. No mapped-coordinate,
gradient or parameter-derivative array grows with the sample count. Per-sample
float1 formulas are retained; private Kahan sums are merged deterministically.
Returned arrays borrow the workspace and are overwritten by its next use.
"""
import numpy as np
from numba import get_num_threads,njit,prange,literal_unroll

from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import ONE
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import _local_axis32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import _local_affine_integer_point32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import _sample_inside32,_interpolate_axes32
from fast_ants.core.arithmetic32 import fma32
from fast_ants.backends.cpu.fp32.metrics.parzen import parzen_add4,derivative_add4
from fast_ants.backends.cpu.fp32.metrics.coordinates import map_integer4,integer_plan_kind

_SAFE_BOUND=np.float32(1e8)


class DerivativeWorkspace32:
    """Reusable block PDFs and reductions, independent of voxel-sized arrays."""
    def __init__(self,count,bins,parameters):
        if count<0 or bins<5 or not 1<=parameters<=12:
            raise ValueError('Derivative workspace needs count>=0, bins>=5 and 1..12 parameters')
        self.count,self.bins,self.parameters=int(count),int(bins),int(parameters)
        self.block=max(4096,(self.count+127)//128)
        self.chunks=(self.count+self.block-1)//self.block
        joint_shape=(self.chunks,self.bins,self.bins)
        derivative_shape=(*joint_shape,self.parameters)
        self.partial_joint=np.empty(joint_shape,np.float32)
        self.partial_joint_correction=np.empty(joint_shape,np.float32)
        self.partial_derivative=np.empty(derivative_shape,np.float32)
        self.partial_derivative_correction=np.empty(derivative_shape,np.float32)
        self.partial_counts=np.empty((self.chunks,self.bins),np.int64)
        self.sizes=np.empty((self.chunks,2),np.int64)
        self.bad=np.empty(self.chunks,np.bool_)
        self.joint=np.empty((self.bins,self.bins),np.float32)
        self.joint_correction=np.empty_like(self.joint)
        self.derivative=np.empty((self.bins,self.bins,self.parameters),np.float32)
        self.derivative_correction=np.empty_like(self.derivative)
        self.counts=np.empty(self.bins,np.int64)
        self.empty_zero_support=np.empty((0,0,0),np.bool_)


@njit(cache=True,fastmath=False,inline='always')
def _map_general(points,n,integer_linear,residual_linear,integer_offset,
                  residual_offset,origin,base,offset):
    for c in range(3):
        b=integer_offset[c];value=residual_offset[c]
        for j in range(3):
            q=points[n,j]-origin[j]
            whole=int(np.floor(q));fraction=q-np.float32(whole)
            b+=integer_linear[c,j]*whole
            value=fma32(residual_linear[c,j],np.float32(whole),value)
            value=fma32(np.float32(integer_linear[c,j])+residual_linear[c,j],fraction,value)
        base[c]=b;offset[c]=value


@njit(cache=True,fastmath=False,inline='always')
def _gradient(data,base,offset,pixel,shift,spacing,gradient):
    bx,by,bz=base;xx,yy,zz=offset
    for c in range(3):
        px,py,pz=xx+shift[0,c],yy+shift[1,c],zz+shift[2,c]
        mx,my,mz=xx-shift[0,c],yy-shift[1,c],zz-shift[2,c]
        xp0,xp1,fxp,v1=_local_axis32(bx,px,data.shape[2])
        yp0,yp1,fyp,v2=_local_axis32(by,py,data.shape[1])
        zp0,zp1,fzp,v3=_local_axis32(bz,pz,data.shape[0])
        xm0,xm1,fxm,v4=_local_axis32(bx,mx,data.shape[2])
        ym0,ym1,fym,v5=_local_axis32(by,my,data.shape[1])
        zm0,zm1,fzm,v6=_local_axis32(bz,mz,data.shape[0])
        gradient[c]=ZERO
        if v1 and v2 and v3 and v4 and v5 and v6:
            gradient[c]=(_interpolate_axes32(data,xp0,xp1,yp0,yp1,zp0,zp1,fxp,fyp,fzp,pixel)-
                         _interpolate_axes32(data,xm0,xm1,ym0,ym1,zm0,zm1,fxm,fym,fzm,pixel))/spacing[c]


@njit(cache=True,fastmath=False)
def _safe_coefficients(linear_derivatives,translation_derivatives):
    for p in range(linear_derivatives.shape[0]):
        for c in range(3):
            if not np.isfinite(translation_derivatives[p,c]) or abs(translation_derivatives[p,c])>_SAFE_BOUND:
                return False
            for j in range(3):
                if not np.isfinite(linear_derivatives[p,c,j]) or abs(linear_derivatives[p,c,j])>_SAFE_BOUND:
                    return False
    return True


@njit(cache=True,fastmath=False)
def _jacobian_plan32(linear_derivatives,translation_derivatives):
    """Recognize exact coordinate bases once per evaluation, in any order.

    Rows are (kind, component, coordinate): kinds 1/2 are unit linear and
    translation bases, kind 3 is one linear column, kind 4 is a translation
    vector, and kind 0 is generic.
    The plan is only a structural hint; per-sample finite/bounded guards in
    the chunk preserve the full expression for unsafe arithmetic.
    """
    plan=np.zeros((linear_derivatives.shape[0],3),np.int32)
    for p in range(linear_derivatives.shape[0]):
        linear_count=translation_count=0
        columns=0
        linear_unit=translation_unit=True
        linear_component=linear_coordinate=translation_component=0
        for c in range(3):
            value=translation_derivatives[p,c]
            if value!=ZERO:
                translation_count+=1;translation_component=c
                if value!=ONE:translation_unit=False
            for j in range(3):
                value=linear_derivatives[p,c,j]
                if value!=ZERO:
                    linear_count+=1;linear_component=c;linear_coordinate=j
                    columns|=1<<j
                    if value!=ONE:linear_unit=False
        if linear_count==1 and linear_unit and translation_count==0:
            plan[p,0]=1;plan[p,1]=linear_component;plan[p,2]=linear_coordinate
        elif linear_count==0 and translation_count==1 and translation_unit:
            plan[p,0]=2;plan[p,1]=translation_component
        elif translation_count==0 and columns in (1,2,4):
            plan[p,0]=3;plan[p,2]=0 if columns==1 else (1 if columns==2 else 2)
        elif linear_count==0:
            plan[p,0]=4
    return plan


@njit(cache=True,fastmath=False)
def _chunk(data,points,fixed_bins,integer_linear,residual_linear,integer_offset,
           residual_offset,origin,bins,minimum,maximum,width,normalized_min,
           shift,spacing,local_linear,local_offset,local_center,linear_derivatives,
           translation_derivatives,jacobian_plan,points_are_integer,safe_coefficients,start,end,
           joint,joint_correction,counts,derivative,derivative_correction,zero_support):
    joint.fill(ZERO);joint_correction.fill(ZERO);counts.fill(0)
    derivative.fill(ZERO);derivative_correction.fill(ZERO)
    base=np.empty(3,np.int32);offset=np.empty(3,np.float32)
    batch_bases=np.empty((3,4),np.int32);batch_offsets=np.empty((3,4),np.float32)
    point_vector_end=end-(end-start)%4
    gradient=np.empty(3,np.float32);local=np.empty(3,np.float32)
    intensity=np.empty(linear_derivatives.shape[0],np.float32)
    geometric=valid=0;bad=False
    for n in range(start,end):
        if points_are_integer:
            if points_are_integer==2 and n<point_vector_end:
                lane=(n-start)%4
                if lane==0:
                    map_integer4(points,n,integer_linear,residual_linear,integer_offset,
                                 residual_offset,origin,batch_bases,batch_offsets)
                for c in range(3):base[c]=batch_bases[c,lane];offset[c]=batch_offsets[c,lane]
            else:
                _local_affine_integer_point32(points,n,integer_linear,residual_linear,
                    integer_offset,residual_offset,origin,base,offset)
        else:
            _map_general(points,n,integer_linear,residual_linear,integer_offset,
                residual_offset,origin,base,offset)
        pixel,inside=_sample_inside32(data,base,offset)
        if not inside:continue
        geometric+=1
        # Match the existing FP32 path: out-of-range intensity samples never
        # contribute a derivative PDF and need no gradient/Jacobian work.
        if pixel<minimum or pixel>maximum:continue
        known_zero=False
        if zero_support.size:
            xx=min(max(base[0]+int(np.floor(offset[0])),0),data.shape[2]-1)
            yy=min(max(base[1]+int(np.floor(offset[1])),0),data.shape[1]-1)
            zz=min(max(base[2]+int(np.floor(offset[2])),0),data.shape[0]-1)
            known_zero=zero_support[zz,yy,xx]
        if known_zero:
            gradient.fill(ZERO)
        else:
            _gradient(data,base,offset,pixel,shift,spacing,gradient)
        bounded_local=True
        for c in range(3):
            value=local_offset[c]
            # This local physical coordinate uses the half-voxel centre, not
            # the integer floor centre used by the sampling decomposition.
            for j in range(3):value+=local_linear[c,j]*(points[n,j]-local_center[j])
            local[c]=value
            if not np.isfinite(value) or abs(value)>_SAFE_BOUND:bounded_local=False
        zero_derivative=(gradient[0]==ZERO and gradient[1]==ZERO and gradient[2]==ZERO
                         and safe_coefficients and bounded_local)
        if not zero_derivative:
            # A nonselected infinite gradient still contributes 0*inf in the
            # full Jacobian expression. Do not hide it with a basis shortcut.
            planned=(safe_coefficients and bounded_local and
                     np.isfinite(gradient[0]) and np.isfinite(gradient[1]) and
                     np.isfinite(gradient[2]))
            for p in range(linear_derivatives.shape[0]):
                kind=jacobian_plan[p,0]
                if planned and kind==1:
                    total=gradient[jacobian_plan[p,1]]*local[jacobian_plan[p,2]]
                elif planned and kind==2:
                    total=gradient[jacobian_plan[p,1]]
                elif planned and kind==3:
                    column=jacobian_plan[p,2]
                    coordinate=local[column]
                    total=ZERO
                    for c in range(3):
                        spatial=fma32(linear_derivatives[p,c,column],coordinate,ZERO)
                        total=fma32(gradient[c],spatial,total)
                elif planned and kind==4:
                    total=ZERO
                    for c in range(3):
                        total=fma32(gradient[c],translation_derivatives[p,c],total)
                else:
                    total=ZERO
                    for c in range(3):
                        spatial=translation_derivatives[p,c]
                        for j in range(3):spatial=fma32(linear_derivatives[p,c,j],local[j],spatial)
                        total=fma32(gradient[c],spatial,total)
                intensity[p]=total
                if not np.isfinite(total):bad=True
        term=pixel/width-normalized_min
        moving_bin=min(max(int(term),2),bins-3);fraction=term-np.float32(moving_bin)
        fixed_bin=fixed_bins[n];counts[fixed_bin]+=1;valid+=1
        weights=parzen_add4(joint,joint_correction,fixed_bin,moving_bin,fraction)
        if not zero_derivative:
            parameters=linear_derivatives.shape[0]
            parameter_vector_end=(parameters//4)*4
            for slot in literal_unroll((0,1,2,3)):
                j=moving_bin-1+slot
                weight=weights[slot]
                for p in range(0,parameter_vector_end,4):
                    derivative_add4(derivative,derivative_correction,intensity,fixed_bin,j,p,weight)
                for p in range(parameter_vector_end,parameters):
                    derivative[fixed_bin,j,p],derivative_correction[fixed_bin,j,p]=add32(
                        derivative[fixed_bin,j,p],derivative_correction[fixed_bin,j,p],intensity[p]*weight)
    return valid,geometric,bad


@njit(cache=True,fastmath=False,parallel=True)
def _histograms(data,points,fixed_bins,integer_linear,residual_linear,integer_offset,
                residual_offset,origin,bins,minimum,maximum,width,normalized_min,
                shift,spacing,local_linear,local_offset,local_center,linear_derivatives,
                translation_derivatives,points_are_integer,parallel,block,partial_joint,
                partial_joint_correction,partial_counts,partial_derivative,
                partial_derivative_correction,sizes,bad,joint,joint_correction,counts,
                derivative,derivative_correction,zero_support):
    safe=_safe_coefficients(linear_derivatives,translation_derivatives)
    jacobian_plan=_jacobian_plan32(linear_derivatives,translation_derivatives)
    chunks=partial_joint.shape[0]
    if parallel:
        for task in prange(chunks):
            chunk=np.int64(task)
            sizes[chunk,0],sizes[chunk,1],bad[chunk]=_chunk(data,points,fixed_bins,
                integer_linear,residual_linear,integer_offset,residual_offset,origin,
                bins,minimum,maximum,width,normalized_min,shift,spacing,local_linear,
                local_offset,local_center,linear_derivatives,translation_derivatives,
                jacobian_plan,points_are_integer,safe,chunk*block,min((chunk+1)*block,points.shape[0]),
                partial_joint[chunk],partial_joint_correction[chunk],partial_counts[chunk],
                partial_derivative[chunk],partial_derivative_correction[chunk],zero_support)
    else:
        for chunk in range(chunks):
            sizes[chunk,0],sizes[chunk,1],bad[chunk]=_chunk(data,points,fixed_bins,
                integer_linear,residual_linear,integer_offset,residual_offset,origin,
                bins,minimum,maximum,width,normalized_min,shift,spacing,local_linear,
                local_offset,local_center,linear_derivatives,translation_derivatives,
                jacobian_plan,points_are_integer,safe,chunk*block,min((chunk+1)*block,points.shape[0]),
                partial_joint[chunk],partial_joint_correction[chunk],partial_counts[chunk],
                partial_derivative[chunk],partial_derivative_correction[chunk],zero_support)
    joint.fill(ZERO);joint_correction.fill(ZERO);counts.fill(0)
    derivative.fill(ZERO);derivative_correction.fill(ZERO)
    valid=geometric=0;bad_derivative=False
    for chunk in range(chunks):
        valid+=sizes[chunk,0];geometric+=sizes[chunk,1];bad_derivative|=bad[chunk]
        for i in range(bins):
            counts[i]+=partial_counts[chunk,i]
            for j in range(bins):
                joint[i,j],joint_correction[i,j]=add32(joint[i,j],joint_correction[i,j],partial_joint[chunk,i,j])
                for p in range(linear_derivatives.shape[0]):
                    derivative[i,j,p],derivative_correction[i,j,p]=add32(
                        derivative[i,j,p],derivative_correction[i,j,p],partial_derivative[chunk,i,j,p])
    return joint,counts,derivative,valid,geometric,bad_derivative


def derivative_histogram32(data,points,fixed_bins,mapping,bins,moving_range,width,normalized_min,
                           gradient_geometry,local_geometry,linear_derivatives,
                           translation_derivatives,*,points_are_integer=False,workspace=None,zero_support=None):
    """Consume validated FP32 samples; return borrowed PDFs and sample flags.

    ``gradient_geometry`` is (LPS half-spacing shift, spacing), while
    ``local_geometry`` is the exact centred-affine (linear, offset, half-centre)
    triple already used by the unfused FP32 parameter-Jacobian implementation.
    """
    parameters=linear_derivatives.shape[0]
    workspace=(DerivativeWorkspace32(len(points),bins,parameters) if workspace is None else workspace)
    if (workspace.count,workspace.bins,workspace.parameters)!=(len(points),bins,parameters):
        raise ValueError('Derivative workspace does not match samples, bins and parameters')
    if zero_support is None:
        zero_support=workspace.empty_zero_support
    elif (zero_support.shape!=data.shape or zero_support.dtype!=np.bool_
          or not zero_support.flags.c_contiguous or not np.isfinite(gradient_geometry[0]).all()
          or np.any(np.abs(gradient_geometry[0])>1)):
        raise ValueError('Zero support needs a matching Boolean volume and shifts within one voxel')
    return _histograms(data,points,fixed_bins,*mapping,bins,np.float32(moving_range[0]),
        np.float32(moving_range[1]),np.float32(width),np.float32(normalized_min),
        *gradient_geometry,*local_geometry,linear_derivatives,translation_derivatives,
        integer_plan_kind(points,mapping,points_are_integer),len(points)>=16384 and get_num_threads()>1,workspace.block,
        workspace.partial_joint,workspace.partial_joint_correction,workspace.partial_counts,
        workspace.partial_derivative,workspace.partial_derivative_correction,workspace.sizes,
        workspace.bad,workspace.joint,workspace.joint_correction,workspace.counts,
        workspace.derivative,workspace.derivative_correction,zero_support)
