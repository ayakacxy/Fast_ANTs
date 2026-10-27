"""Device displacement composition, norm controls, and inverse updates."""
import numpy as np
from fast_ants.backends.cuda import reductions


def multiply(ctx,values,factor):
    return ctx.call('multiply','out0[i]=data[i]*factor[0];',
        {'data':values,'factor':ctx.geometry([factor])},[values.shape],values.size)


def compose(ctx,outer,inner,linear,offset,inverse):
    _,nz,ny,nx=inner.shape;_,oz,oy,ox=outer.shape
    return ctx.call('compose','''int3 n=make_int3(meta[0],meta[1],meta[2]);
      int3 src=make_int3(meta[3],meta[4],meta[5]),p=xyz(i,n);
      unsigned int N=meta[0]*meta[1]*meta[2],O=meta[3]*meta[4]*meta[5];float q[3];
      for(int c=0;c<3;c++){
        float v=offset[c]+linear[c*3]*float(p.x)+linear[c*3+1]*float(p.y)+linear[c*3+2]*float(p.z);
        for(int j=0;j<3;j++)v+=inverse[c*3+j]*inner[j*N+i];q[c]=v;}
      float3 point=make_float3(q[0],q[1],q[2]);bool valid=inside(point,src);
      for(int c=0;c<3;c++)out0[c*N+i]=inner[c*N+i]+(valid?sample1(outer+c*O,point,src):0.0f);''',
      {'outer':outer,'inner':inner,'linear':ctx.geometry(linear),'offset':ctx.geometry(offset),
       'inverse':ctx.geometry(inverse),'meta':ctx.meta(nx,ny,nz,ox,oy,oz)},[inner.shape],nx*ny*nz)


def residual_arrays(ctx,outer,current,inverse,spacing):
    """Queue the exact residual and voxel norm arrays without a host read."""
    _,nz,ny,nx=current.shape
    residual,values=ctx.call('inverse_residual_norms','''
      int3 n=make_int3(meta[0],meta[1],meta[2]),p=xyz(i,n);unsigned int N=meta[0]*meta[1]*meta[2];
      float u0=current[i],u1=current[N+i],u2=current[2*N+i];
      float dx=inverse[0]*u0+inverse[1]*u1+inverse[2]*u2;
      float dy=inverse[3]*u0+inverse[4]*u1+inverse[5]*u2;
      float dz=inverse[6]*u0+inverse[7]*u1+inverse[8]*u2;
      LocalAxis x=local_axis(p.x,dx,n.x),y=local_axis(p.y,dy,n.y),z=local_axis(p.z,dz,n.z);
      bool valid=x.valid&&y.valid&&z.valid;float squared=0.0f;
      for(int c=0;c<3;c++){
        float value=current[c*N+i];
        if(valid){
          const float* data=outer+c*N;
          float a=data[idx(x.lo,y.lo,z.lo,n)]+(data[idx(x.hi,y.lo,z.lo,n)]-data[idx(x.lo,y.lo,z.lo,n)])*x.fraction;
          float b=data[idx(x.lo,y.hi,z.lo,n)]+(data[idx(x.hi,y.hi,z.lo,n)]-data[idx(x.lo,y.hi,z.lo,n)])*x.fraction;
          float low=data[idx(x.lo,y.lo,z.hi,n)]+(data[idx(x.hi,y.lo,z.hi,n)]-data[idx(x.lo,y.lo,z.hi,n)])*x.fraction;
          float high=data[idx(x.lo,y.hi,z.hi,n)]+(data[idx(x.hi,y.hi,z.hi,n)]-data[idx(x.lo,y.hi,z.hi,n)])*x.fraction;
          float lo=a+(b-a)*y.fraction,hi=low+(high-low)*y.fraction;
          value+=lo+(hi-lo)*z.fraction;
        }
        out0[c*N+i]=value;float v=value/spacing[c];squared+=v*v;
      }
      out1[i]=sqrtf(squared);''',
      {'outer':outer,'current':current,'inverse':ctx.geometry(inverse),'spacing':ctx.geometry(spacing),
       'meta':ctx.meta(nx,ny,nz)},[current.shape,current.shape[1:]],nx*ny*nz)
    return residual,values


def residual_controls(ctx,outer,current,inverse,spacing):
    """Queue the exact residual, row Kahan reduction and device controls."""
    residual,values=residual_arrays(ctx,outer,current,inverse,spacing)
    return residual,values,reductions.row_norms(ctx,values)


def guarded_norms(ctx,values,mean,maximum,mean_tolerance=None):
    """Apply the existing finite and near-threshold scalar controls."""
    if not np.isfinite([mean,maximum]).all():raise ValueError('Inverse residual norms must be finite')
    if mean_tolerance is not None:
        scale=max(float(mean),float(mean_tolerance),np.finfo(np.float32).tiny)
        if abs(float(mean)-mean_tolerance)<=32*np.finfo(np.float32).eps*scale:
            mean,maximum=ctx.control(reductions.serial_norms(ctx,values),'inverse_norms_guard',2)
    return float(mean),float(maximum)


def residual_norms(ctx,outer,current,inverse,spacing,mean_tolerance=None):
    """Evaluate the CPU FP32 inverse's integer-origin local interpolant."""
    residual,values,controls=residual_controls(ctx,outer,current,inverse,spacing)
    mean,maximum=ctx.control(controls,'inverse_norms',2)
    mean,maximum=guarded_norms(ctx,values,mean,maximum,mean_tolerance)
    return residual,values,mean,maximum


def maximum_control(ctx,field,spacing):
    candidate=reductions.maximum_vector(ctx,field,spacing)
    values=ctx.control(candidate,'maximum_vector',3).astype(np.float64)/np.asarray(spacing,np.float64)
    maximum=float(np.sqrt(values@values))
    if not np.isfinite(maximum):raise ValueError('Displacement norms must be finite')
    return maximum


def inverse_step(ctx,current,residual,norms,maximum,epsilon,enforce_boundary):
    _,nz,ny,nx=current.shape
    return ctx.call('inverse_step','''unsigned int N=meta[0]*meta[1]*meta[2];
      int3 n=make_int3(meta[0],meta[1],meta[2]),p=xyz(i,n);float maximum=params[0],eps=params[1];
      float factor=norms[i]>eps*maximum?eps*maximum/norms[i]:1.0f;
      for(int c=0;c<3;c++){float v=current[c*N+i]-residual[c*N+i]*factor*eps;
        out0[c*N+i]=meta[3]&&edge(p,n)?0.0f:v;}''',
      {'current':current,'residual':residual,'norms':norms,'params':ctx.geometry([maximum,epsilon]),
       'meta':ctx.meta(nx,ny,nz,int(enforce_boundary))},[current.shape],nx*ny*nz)


def inverse_step_device(ctx,current,residual,norms,controls,index,epsilon,enforce_boundary):
    """Queue an immutable update using the already reduced device maximum."""
    _,nz,ny,nx=current.shape
    return ctx.call('inverse_step_device_max','''unsigned int N=meta[0]*meta[1]*meta[2];
      int3 n=make_int3(meta[0],meta[1],meta[2]),p=xyz(i,n);
      float maximum=controls[2*meta[4]+1],eps=params[0];
      float factor=norms[i]>eps*maximum?eps*maximum/norms[i]:1.0f;
      for(int c=0;c<3;c++){float v=current[c*N+i]-residual[c*N+i]*factor*eps;
        out0[c*N+i]=meta[3]&&edge(p,n)?0.0f:v;}''',
      {'current':current,'residual':residual,'norms':norms,'controls':controls,
       'params':ctx.geometry([epsilon]),'meta':ctx.meta(nx,ny,nz,int(enforce_boundary),index)},
      [current.shape],nx*ny*nz)
