"""Device image kernels and geometry-aware preparation/resampling."""
import numpy as np
from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal import reductions
from fast_ants.backends.metal import serial
from fast_ants.backends.metal.types import MetalImage
from fast_ants.core.types import Image
from fast_ants.core.transforms import AffineTransform
from fast_ants.pipeline.preprocessing import window_parameters
from fast_ants.pipeline.gaussian import recursive_gaussian_coefficients


def zeros(ctx, shape):
    count=int(np.prod(shape))
    return ctx.call('zeros', 'out0[thread_position_in_grid.x]=0.0f;', {}, [shape], count)


def indices(ctx, shape):
    nz,ny,nx=shape; count=nx*ny*nz
    return ctx.call('indices', '''uint i=thread_position_in_grid.x; int3 p=xyz(i,int3(meta[0],meta[1],meta[2]));
        for(int c=0;c<3;c++) out0[3*i+c]=float(p[c]);''',
        {'meta':ctx.meta(nx,ny,nz)}, [(count,3)], count)


def range32(ctx, data):
    if 'parallel_range' in ctx.optimizations:
        return ctx.control(reductions.range32(ctx,data),'image_range',2)
    if data.size>serial.SLICE:return ctx.control(serial.range32(ctx,data),'image_range',2)
    result=ctx.call('range', '''float lo=data[0],hi=data[0];
        for(uint i=1;i<meta[0];i++){lo=metal::min(lo,data[i]);hi=metal::max(hi,data[i]);}
        out0[0]=lo;out0[1]=hi;''', {'data':data,'meta':ctx.meta(data.size)}, [(2,)], 1)
    return ctx.control(result,'image_range',2)


def window(ctx,data,low,high,scale):
    return ctx.call('window', '''uint i=thread_position_in_grid.x;float v=data[i];
        out0[i]=params[2]==0?0.0f:(v<params[0]?0.0f:(v>params[1]?1.0f:(v-params[0])*params[2]));''',
        {'data':data,'params':ctx.upload([low,high,scale])},[data.shape],data.size)


def warp(ctx,data,field,linear,offset,field_linear,nearest=False):
    _,nz,ny,nx=field.shape; iz,iy,ix=data.shape
    return ctx.call('warp', '''uint i=thread_position_in_grid.x;
        int3 n=int3(meta[0],meta[1],meta[2]), src=int3(meta[3],meta[4],meta[5]);
        int3 p=xyz(i,n);float3 q;uint count=meta[0]*meta[1]*meta[2];
        for(int c=0;c<3;c++){
          float v=offset[c]+linear[c*3]*float(p.x)+linear[c*3+1]*float(p.y)+linear[c*3+2]*float(p.z);
          for(int j=0;j<3;j++)v+=fl[c*3+j]*field[j*count+i];q[c]=v;
        }
        float v=0;
        if(inside(q,src))v=meta[6]?data[idx(int(metal::floor(q.x+0.5f)),int(metal::floor(q.y+0.5f)),int(metal::floor(q.z+0.5f)),src)]:sample1(data,q,src);
        out0[i]=v;''',
        {'data':data,'field':field,'linear':ctx.upload(linear),'offset':ctx.upload(offset),
         'fl':ctx.upload(field_linear),'meta':ctx.meta(nx,ny,nz,ix,iy,iz,int(nearest))},[(nz,ny,nx)],nx*ny*nz)


def gradient(ctx,data,shift,spacing):
    nz,ny,nx=data.shape
    return ctx.call('gradient', '''uint i=thread_position_in_grid.x;
        int3 n=int3(meta[0],meta[1],meta[2]);float3 p=float3(xyz(i,n));
        uint count=meta[0]*meta[1]*meta[2];
        for(int c=0;c<3;c++){
          float3 d=float3(shift[c],shift[3+c],shift[6+c]), a=p+d,b=p-d;
          out0[c*count+i]=inside(a,n)&&inside(b,n)?(sample1(data,a,n,data[i])-sample1(data,b,n,data[i]))/spacing[c]:0.0f;
        }''',{'data':data,'shift':ctx.upload(shift),'spacing':ctx.upload(spacing),'meta':ctx.meta(nx,ny,nz)},
        [(3,nz,ny,nx)],data.size)

# Geometry-aware image preparation and resampling, sharing the kernels above.
from fast_ants.backends.metal import dense


class MetalImageOps:
    """Methods for this registration stage; numerical definitions are unchanged."""

    def image(self,image,normalize=False):
        values=image.data if isinstance(image,MetalImage) else self.context.upload(image.data,'input_image')
        if normalize:
            lo,hi,scale,_=window_parameters(*dense.range32(self.context,values))
            values=dense.window(self.context,values,lo,hi,scale)
        return MetalImage(values,image.grid,dict(image.metadata))

    def download_image(self,image,reason='final_output'):
        return Image(self.context.download(image.data,reason),image.grid,image.metadata)

    def pull(self,image,grid,field=None,linear=None,nearest=False):
        field=self.zero(grid) if field is None else field
        if not field.grid.same_geometry(grid):raise ValueError('Pull field/output grid mismatch')
        linear=AffineTransform.identity() if linear is None else linear
        a=linear.matrix[:3,:3]; inverse=image.grid.lps_to_index[:3,:3]
        values=dense.warp(self.context,image.data,field.data,
                          inverse@a@grid.index_to_lps[:3,:3],
                          inverse@(linear.map_points(grid.origin)-image.grid.origin),inverse@a,nearest)
        return MetalImage(values,grid,image.metadata)

    def gradient(self,image):
        shift=image.grid.lps_to_index[:3,:3]@np.diag(image.grid.spacing_xyz*.5)
        return dense.gradient(self.context,image.data,shift,image.grid.spacing_xyz)

    def smooth_image(self,image,sigma,units='vox'):
        sigmas=np.full(3,float(sigma)) if np.ndim(sigma)==0 else np.array(sigma,np.float64)
        if sigmas.shape!=(3,) or not np.isfinite(sigmas).all() or np.any(sigmas<0):raise ValueError('Invalid Gaussian sigmas')
        if units=='mm':sigmas=sigmas/image.grid.spacing_xyz
        elif units!='vox':raise ValueError('Sigma units must be vox or mm')
        values=image.data
        for dimension in (2,0,1):
            if sigmas[dimension]==0:continue
            axis=2-dimension
            if values.shape[axis]<4:raise ValueError('Recursive Gaussian requires four pixels')
            moved=mx.moveaxis(values,axis,-1);shape=moved.shape
            lines=moved.reshape(-1,shape[-1])
            if 'gaussian_sections' in self.context.optimizations:
                from fast_ants.backends.cpu.fp32.gaussian_sections import coefficients32
                from fast_ants.backends.metal.gaussian_sections import section_lines32
                result=section_lines32(self.context,lines,coefficients32(float(sigmas[dimension])))
            else:
                result=dense.recursive(self.context,lines,recursive_gaussian_coefficients(sigmas[dimension]))
            values=mx.moveaxis(result.reshape(shape),-1,axis)
        return MetalImage(values,image.grid,image.metadata)
