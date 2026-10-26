"""Device displacement composition, norm controls, and inverse updates."""
import numpy as np
from fast_ants.backends.metal import reductions
from fast_ants.backends.metal import serial


def multiply(ctx,values,factor):
    return ctx.call('multiply', 'uint i=thread_position_in_grid.x;out0[i]=data[i]*factor[0];',
        {'data':values,'factor':ctx.upload([factor])},[values.shape],values.size)


_COMPOSE_COORDINATES = '''uint i=thread_position_in_grid.x;int3 n=int3(meta[0],meta[1],meta[2]);
      int3 src=int3(meta[3],meta[4],meta[5]),p=xyz(i,n);uint N=meta[0]*meta[1]*meta[2],O=meta[3]*meta[4]*meta[5];float3 q;
      for(int c=0;c<3;c++){
        float v=offset[c]+linear[c*3]*float(p.x)+linear[c*3+1]*float(p.y)+linear[c*3+2]*float(p.z);
        for(int j=0;j<3;j++)v+=inverse[c*3+j]*inner[j*N+i];q[c]=v;}
      bool valid=inside(q,src);'''


def compose(ctx,outer,inner,linear,offset,inverse,spacing=None):
    """Compose fields, optionally producing residual norms in the same pass.

    Inversion needs both arrays. Keeping each FP32 residual in a register
    avoids reading its three stored components again in a separate kernel.
    Interpolation, component order and subsequent norm reduction are unchanged.
    """
    _,nz,ny,nx=inner.shape;_,oz,oy,ox=outer.shape
    inputs={'outer':outer,'inner':inner,'linear':ctx.upload(linear),'offset':ctx.upload(offset),
            'inverse':ctx.upload(inverse),'meta':ctx.meta(nx,ny,nz,ox,oy,oz)}
    if spacing is None:
        return ctx.call('compose', _COMPOSE_COORDINATES + '''
          for(int c=0;c<3;c++)out0[c*N+i]=inner[c*N+i]+(valid?sample1(outer+c*O,q,src):0.0f);''',
          inputs,[inner.shape],nx*ny*nz)
    inputs['spacing']=ctx.upload(spacing)
    return ctx.call('compose_norms', _COMPOSE_COORDINATES + '''
      float sum=0.0f;
      for(int c=0;c<3;c++){
        float residual=inner[c*N+i]+(valid?sample1(outer+c*O,q,src):0.0f);
        out0[c*N+i]=residual;
        float v=residual/spacing[c];sum+=v*v;
      }
      out1[i]=metal::sqrt(sum);''',inputs,[inner.shape,inner.shape[1:]],nx*ny*nz)


def _serial_norms(ctx,values,name='norm_reduce'):
    if values.size>serial.SLICE:return serial.norms(ctx,values)
    return ctx.call(name, '''float2 sum=float2(0);float maximum=0;
      for(uint i=0;i<meta[0];i++){sum=compensated(sum,data[i]);maximum=metal::max(maximum,data[i]);}
      out0[0]=sum.x/float(meta[0]);out0[1]=maximum;''',{'data':values,'meta':ctx.meta(values.size)},[(2,)],1)


def norms(ctx,field,spacing,mean_tolerance=None):
    n=field.size//3
    values=ctx.call('norms', '''uint i=thread_position_in_grid.x,N=meta[0];float sum=0;
      for(int c=0;c<3;c++){float v=field[c*N+i]/spacing[c];sum+=v*v;}out0[i]=metal::sqrt(sum);''',
      {'field':field,'spacing':ctx.upload(spacing),'meta':ctx.meta(n)},[field.shape[1:]],n)
    return reduce_norms(ctx,values,mean_tolerance)


def reduce_norms(ctx,values,mean_tolerance=None):
    """Reduce already evaluated norms, retaining the threshold guard."""
    parallel='parallel_norms' in ctx.optimizations
    result=reductions.norms(ctx,values) if parallel else _serial_norms(ctx,values)
    mean,maximum=ctx.control(result,'inverse_norms',2)
    if parallel and mean_tolerance is not None:
        scale=max(float(mean),float(mean_tolerance),np.finfo(np.float32).tiny)
        if abs(float(mean)-mean_tolerance)<=32*np.finfo(np.float32).eps*scale:
            # Positive fixed-tree sums have no cancellation; the guard is wider
            # than their tree error. Preserve reference decisions near a limit.
            mean,maximum=ctx.control(_serial_norms(ctx,values,'norm_reduce_serial_guard'),'inverse_norms_guard',2)
    return values,float(mean),float(maximum)


def maximum_control(ctx,field,spacing):
    candidate=(reductions.maximum_vector(ctx,field,spacing) if 'parallel_max' in ctx.optimizations else
      serial.maximum_vector(ctx,field,spacing) if field.size//3>serial.SLICE else ctx.call('maximum_vector', 'float maximum=0;uint selected=0,N=meta[0];\n      for(uint i=0;i<N;i++){float squared=0;\n       for(int c=0;c<3;c++){float v=field[c*N+i]/spacing[c];squared+=v*v;}\n       if(squared>maximum){maximum=squared;selected=i;}}\n      for(int c=0;c<3;c++)out0[c]=field[c*N+selected];',
      {'field':field,'spacing':ctx.upload(spacing),'meta':ctx.meta(field.size//3)},[(3,)],1))
    # Three values only, using native CPU FP64; no software floating point.
    values=ctx.control(candidate,'maximum_vector',3).astype(np.float64)/np.asarray(spacing,np.float64)
    return float(np.sqrt(values@values))


def inverse_step(ctx,current,residual,norms,maximum,epsilon,enforce_boundary):
    _,nz,ny,nx=current.shape
    return ctx.call('inverse_step', '''uint i=thread_position_in_grid.x,N=meta[0]*meta[1]*meta[2];
      int3 n=int3(meta[0],meta[1],meta[2]),p=xyz(i,n);float maximum=params[0],eps=params[1];
      float factor=norms[i]>eps*maximum?eps*maximum/norms[i]:1.0f;
      for(int c=0;c<3;c++){float v=current[c*N+i]-residual[c*N+i]*factor*eps;
       out0[c*N+i]=meta[3]&&edge(p,n)?0.0f:v;}''',
      {'current':current,'residual':residual,'norms':norms,'params':ctx.upload([maximum,epsilon]),
       'meta':ctx.meta(nx,ny,nz,int(enforce_boundary))},[current.shape],nx*ny*nz)
