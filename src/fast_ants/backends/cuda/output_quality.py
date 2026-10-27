"""Final field QA in native FP64, without downloading dense QA scratch.

Fitting storage remains FP32. Each field load is promoted before arithmetic;
geometry, interpolation FMAs and half-voxel support follow the CPU reference.
Only bounded summaries and two exact percentile order statistics leave CUDA.
"""

from functools import lru_cache
import math

import cupy as cp
import numpy as np

from fast_ants.backends.cuda.runtime import OPTIONS


SOURCE=r'''
__device__ double qa_component(const float* data,unsigned int c,
        int nx,int ny,int nz,int x0,int x1,int y0,int y1,int z0,int z1,
        double dx,double dy,double dz) {
    const float* p=data+c*(unsigned int)(nx*ny*nz);
    double a0=p[(z0*ny+y0)*nx+x0],b0=p[(z0*ny+y1)*nx+x0];
    double c0=p[(z1*ny+y0)*nx+x0],d0=p[(z1*ny+y1)*nx+x0];
    double a=fma((double)p[(z0*ny+y0)*nx+x1]-a0,dx,a0);
    double b=fma((double)p[(z0*ny+y1)*nx+x1]-b0,dx,b0);
    double cc=fma((double)p[(z1*ny+y0)*nx+x1]-c0,dx,c0);
    double d=fma((double)p[(z1*ny+y1)*nx+x1]-d0,dx,d0);
    double lower=fma(b-a,dy,a),upper=fma(d-cc,dy,cc);
    return fma(upper-lower,dz,lower);
}

extern "C" __global__ void qa_norms(const float* outer,const float* inner,
        const double* g,double* norms,unsigned int size) {
    unsigned int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=size)return;
    int ox=(int)g[24],oy=(int)g[25],oz=(int)g[26];
    int nx=(int)g[27],ny=(int)g[28];
    int x=n%nx,y=(n/nx)%ny,z=n/(nx*ny);
    double p0=g[9]+g[0]*x+g[1]*y+g[2]*z;
    double p1=g[10]+g[3]*x+g[4]*y+g[5]*z;
    double p2=g[11]+g[6]*x+g[7]*y+g[8]*z;
    double u0=inner[n],u1=inner[size+n],u2=inner[2*size+n];
    double l0=p0+u0-g[12],l1=p1+u1-g[13],l2=p2+u2-g[14];
    double i0=0.0+g[15]*l0+g[16]*l1+g[17]*l2;
    double i1=0.0+g[18]*l0+g[19]*l1+g[20]*l2;
    double i2=0.0+g[21]*l0+g[22]*l1+g[23]*l2;
    bool good=(-.5<=i0 && i0<ox-.5 && -.5<=i1 && i1<oy-.5 && -.5<=i2 && i2<oz-.5);
    double d0=0.,d1=0.,d2=0.;
    if(good) {
        double xx=fmin(fmax(i0,0.),ox-1),yy=fmin(fmax(i1,0.),oy-1),zz=fmin(fmax(i2,0.),oz-1);
        int x0=(int)floor(xx),y0=(int)floor(yy),z0=(int)floor(zz);
        int x1=min(x0+1,ox-1),y1=min(y0+1,oy-1),z1=min(z0+1,oz-1);
        double dx=xx-x0,dy=yy-y0,dz=zz-z0;
        d0=qa_component(outer,0,ox,oy,oz,x0,x1,y0,y1,z0,z1,dx,dy,dz);
        d1=qa_component(outer,1,ox,oy,oz,x0,x1,y0,y1,z0,z1,dx,dy,dz);
        d2=qa_component(outer,2,ox,oy,oz,x0,x1,y0,y1,z0,z1,dx,dy,dz);
    }
    double v0=p0+u0+d0-p0,v1=p1+u1+d1-p1,v2=p2+u2+d2-p2;
    norms[n]=(!isfinite(v0) || !isfinite(v1) || !isfinite(v2)) ? nan("") :
        (good ? sqrt(0.0+v0*v0+v1*v1+v2*v2) : -1.);
}

__device__ double qa_difference(const float* data,int c,int nx,int ny,int nz,
        int x,int y,int z,int axis) {
    int value=axis==0 ? x : (axis==1 ? y : z);
    int length=axis==0 ? nx : (axis==1 ? ny : nz);
    int lo=max(value-1,0),hi=min(value+1,length-1);
    if(lo==hi)return 0.;
    unsigned int base=c*(unsigned int)(nx*ny*nz)+(z*ny+y)*nx+x;
    int stride=axis==0 ? 1 : (axis==1 ? nx : nx*ny);
    return ((double)data[base+(hi-value)*stride]-(double)data[base+(lo-value)*stride])/(hi-lo);
}

extern "C" __global__ void qa_jacobian(const float* data,const double* inverse,
        double* values,unsigned int size,int nx,int ny,int nz) {
    unsigned int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=size)return;
    int x=n%nx,y=(n/nx)%ny,z=n/(nx*ny);
    double j[9];
    for(int c=0;c<3;c++) {
        double a=qa_difference(data,c,nx,ny,nz,x,y,z,0);
        double b=qa_difference(data,c,nx,ny,nz,x,y,z,1);
        double d=qa_difference(data,c,nx,ny,nz,x,y,z,2);
        for(int axis=0;axis<3;axis++)
            j[3*c+axis]=0.0+a*inverse[axis]+b*inverse[3+axis]+d*inverse[6+axis];
        j[3*c+c]+=1.;
    }
    values[n]=j[0]*(j[4]*j[8]-j[5]*j[7])-j[1]*(j[3]*j[8]-j[5]*j[6])+
              j[2]*(j[3]*j[7]-j[4]*j[6]);
}
'''


@lru_cache(2)
def _kernel(name):
    return cp.RawKernel(SOURCE,name,options=OPTIONS)


def _geometry(ctx,values):
    values=np.ascontiguousarray(values,dtype=np.float64)
    result=cp.asarray(values,blocking=True)
    ctx._record('host_to_device','output_qa_fp64_geometry',values)
    return result


def _control(ctx,values):
    if values.size>8:raise ValueError('Output QA summaries must be bounded')
    result=cp.asnumpy(values,stream=ctx.stream,blocking=True)
    ctx._record('device_to_host','control_output_qa_fp64',result)
    return result


def _direction(ctx,outer,inner):
    nz,ny,nx=inner.grid.shape_zyx;oz,oy,ox=outer.grid.shape_zyx
    size=nx*ny*nz
    geometry=_geometry(ctx,np.r_[inner.grid.index_to_lps[:3,:3].ravel(),inner.grid.origin,
        outer.grid.origin,outer.grid.lps_to_index[:3,:3].ravel(),ox,oy,oz,nx,ny,nz])
    norms=cp.empty(size,cp.float64)
    _kernel('qa_norms')(((size+255)//256,),(256,),
        (outer.data,inner.data,geometry,norms,np.uint32(size)),stream=ctx.stream)
    ctx.kernels['output_qa_norms64']+=1
    valid=norms>=0.
    packet=_control(ctx,cp.stack((cp.count_nonzero(valid),cp.max(norms),
        cp.sum(cp.where(valid,norms*norms,0.),dtype=cp.float64),cp.all(cp.isfinite(norms)))))
    count=int(packet[0])
    if not packet[3] or not math.isfinite(packet[2]):
        raise ValueError('Output QA composition and norms must be finite')
    percentile=None
    if count:
        # Invalid support uses a negative sentinel, so these are exactly the
        # two order statistics of the valid norms. Partition owned scratch.
        index=(count-1)*.99;lower=int(math.floor(index));weight=index-lower
        first=size-count+lower;last=min(first+1,size-1)
        norms.partition((first,last))
        a,b=_control(ctx,norms[first:last+1])[[0,-1]]
        delta=b-a
        percentile=float(b-delta*(1.-weight) if weight>=.5 else a+delta*weight)
    return {'common_valid_points':count,'total_points':size,
        'max_l2_mm':float(packet[1]) if count else None,
        'rms_l2_mm':float(math.sqrt(packet[2]/count)) if count else None,
        'p99_l2_mm':percentile}


def output_quality(ctx,forward,inverse):
    """Match the CPU FP64 final summaries using explicit CUDA QA scratch.

    Native double storage is private to output QA. It never enters fitting
    operators or widens the runtime's FP32/integer storage contract.
    """
    for field in (forward,inverse):
        ctx.require(field.data)
        if field.data.dtype!=cp.float32 or field.data.shape!=(3,*field.grid.shape_zyx):
            raise ValueError('CUDA output QA needs FP32 displacement fields [3,z,y,x]')
    with ctx.device,ctx.stream:
        finite=_control(ctx,cp.stack((cp.all(cp.isfinite(forward.data)),cp.all(cp.isfinite(inverse.data)))))
        if not finite.all():raise ValueError('Output QA displacement fields must be finite')
        consistency={'forward_after_inverse':_direction(ctx,forward,inverse),
                     'inverse_after_forward':_direction(ctx,inverse,forward)}
        nz,ny,nx=forward.grid.shape_zyx;size=nx*ny*nz
        geometry=_geometry(ctx,forward.grid.lps_to_index[:3,:3])
        values=cp.empty(size,cp.float64)
        _kernel('qa_jacobian')(((size+255)//256,),(256,),
            (forward.data,geometry,values,np.uint32(size),np.int32(nx),np.int32(ny),np.int32(nz)),stream=ctx.stream)
        ctx.kernels['output_qa_jacobian64']+=1
        packet=_control(ctx,cp.stack((cp.min(values),cp.count_nonzero(values<=0.),cp.all(cp.isfinite(values)))))
    if not packet[2]:raise ValueError('Output QA Jacobian determinants must be finite')
    return {'nonlinear_inverse_consistency':consistency,
            'minimum_nonlinear_jacobian':float(packet[0]),
            'nonpositive_jacobian_points':int(packet[1])}
