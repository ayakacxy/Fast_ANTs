"""Device image kernels and geometry-aware FP32 resampling."""
import numpy as np
from fast_ants.backends.cuda.runtime import CUDAContext, cp
from fast_ants.backends.cuda.types import CUDAImage, CUDAField
from fast_ants.core.types import Image
from fast_ants.core.transforms import AffineTransform
from fast_ants.pipeline.preprocessing import window_parameters


def zeros(ctx,shape):
    count=int(np.prod(shape))
    return ctx.call('zeros','out0[i]=0.0f;',{},[shape],count)


def window(ctx,data,low,high,scale):
    return ctx.call('window','''float v=data[i];
        out0[i]=params[2]==0?0.0f:(v<params[0]?0.0f:(v>params[1]?1.0f:(v-params[0])*params[2]));''',
        {'data':data,'params':ctx.geometry([low,high,scale])},[data.shape],data.size)


def warp(ctx,data,field,linear,offset,field_linear,nearest=False):
    _,nz,ny,nx=field.shape;iz,iy,ix=data.shape
    return ctx.call('warp','''
        int3 n=make_int3(meta[0],meta[1],meta[2]),src=make_int3(meta[3],meta[4],meta[5]);
        int3 p=xyz(i,n);float q[3];unsigned int count=meta[0]*meta[1]*meta[2];
        for(int c=0;c<3;c++){
          float v=offset[c]+linear[c*3]*float(p.x)+linear[c*3+1]*float(p.y)+linear[c*3+2]*float(p.z);
          for(int j=0;j<3;j++)v+=fl[c*3+j]*field[j*count+i];q[c]=v;
        }
        float3 point=make_float3(q[0],q[1],q[2]);float v=0.0f;
        if(inside(point,src)){
          if(meta[6]){
            // Clamp after rounding: a singleton upper-support predecessor can round to one.
            int x=min(int(floorf(point.x+0.5f)),src.x-1);
            int y=min(int(floorf(point.y+0.5f)),src.y-1);
            int z=min(int(floorf(point.z+0.5f)),src.z-1);
            v=data[idx(x,y,z,src)];
          }else v=sample1(data,point,src);
        }
        out0[i]=v;''',
        {'data':data,'field':field,'linear':ctx.geometry(linear),'offset':ctx.geometry(offset),
         'fl':ctx.geometry(field_linear),'meta':ctx.meta(nx,ny,nz,ix,iy,iz,int(nearest))},[(nz,ny,nx)],nx*ny*nz)


def gradient(ctx,data,shift,spacing):
    nz,ny,nx=data.shape
    return ctx.call('gradient','''
        int3 n=make_int3(meta[0],meta[1],meta[2]),p=xyz(i,n);
        unsigned int count=meta[0]*meta[1]*meta[2];
        for(int c=0;c<3;c++){
          float3 a=make_float3(float(p.x)+shift[c],float(p.y)+shift[3+c],float(p.z)+shift[6+c]);
          float3 b=make_float3(float(p.x)-shift[c],float(p.y)-shift[3+c],float(p.z)-shift[6+c]);
          out0[c*count+i]=inside(a,n)&&inside(b,n)?(sample1(data,a,n,data[i])-sample1(data,b,n,data[i]))/spacing[c]:0.0f;
        }''',{'data':data,'shift':ctx.geometry(shift),'spacing':ctx.geometry(spacing),'meta':ctx.meta(nx,ny,nz)},
        [(3,nz,ny,nx)],data.size)


class CUDAImageOps:
    """Geometry-aware device image preparation and resampling."""

    def __init__(self,context=None):
        self.context=CUDAContext() if context is None else context

    def image(self,image,normalize=False):
        values=image.data if isinstance(image,CUDAImage) else self.context.upload(image.data,'input_image')
        self.context.require(values)
        if normalize:
            with self.context.device,self.context.stream:
                limits=cp.stack((cp.min(values),cp.max(values)))
            lo,hi,scale,_=window_parameters(*self.context.control(limits,'image_range',2))
            values=window(self.context,values,lo,hi,scale)
        return CUDAImage(values,image.grid,dict(image.metadata))

    def download_image(self,image,reason='final_output'):
        return Image(self.context.download(image.data,reason),image.grid,image.metadata)

    def zero(self,grid):
        return CUDAField(zeros(self.context,(3,*grid.shape_zyx)),grid)

    def pull(self,image,grid,field=None,linear=None,nearest=False):
        """Pull image at linear(x+field(x)), using local FP32 source indices."""
        if not isinstance(image,CUDAImage):raise TypeError('CUDA pull requires a CUDAImage')
        field=self.zero(grid) if field is None else field
        if not isinstance(field,CUDAField):raise TypeError('CUDA pull requires a CUDAField')
        if not field.grid.same_geometry(grid):raise ValueError('Pull field/output grid mismatch')
        linear=AffineTransform.identity() if linear is None else linear
        a=linear.matrix[:3,:3];inverse=image.grid.lps_to_index[:3,:3]
        # Subtract physical origins in small CPU FP64 geometry before uploading.
        values=warp(self.context,image.data,field.data,
                    inverse@a@grid.index_to_lps[:3,:3],
                    inverse@(linear.map_points(grid.origin)-image.grid.origin),inverse@a,nearest)
        return CUDAImage(values,grid,image.metadata)

    def gradient(self,image):
        if not isinstance(image,CUDAImage):raise TypeError('CUDA gradient requires a CUDAImage')
        shift=image.grid.lps_to_index[:3,:3]@np.diag(image.grid.spacing_xyz*.5)
        return gradient(self.context,image.data,shift,image.grid.spacing_xyz)

    def smooth_image(self,image,sigma,units='vox'):
        from fast_ants.backends.cuda.filtering import smooth_image
        return smooth_image(self.context,image,sigma,units)
