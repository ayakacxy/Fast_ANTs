"""Geometry-aware device fields, smoothing, composition, and inversion."""
import numpy as np
from fast_ants.backends.cuda import filtering
from fast_ants.backends.cuda.fields import kernels
from fast_ants.backends.cuda.types import CUDAField
from fast_ants.core.fields import DisplacementField
from fast_ants.core.results import InversionResult
from fast_ants.pipeline.gaussian import discrete_gaussian_kernel


class CUDAFieldOps:
    """Methods for this registration stage; numerical definitions are unchanged."""

    def field(self,field):
        if isinstance(field,CUDAField):
            self.context.require(field.data)
            return field
        return CUDAField(self.context.upload(field.data,'input_field'),field.grid)

    def download_field(self,field,reason='final_output'):
        return DisplacementField(self.context.download(field.data,reason),field.grid)

    def smooth_field(self,field,variance):
        if not np.isfinite(variance) or variance<0:raise ValueError('Invalid field variance')
        if variance==0:return CUDAField(field.data,field.grid)
        values=field.data;weight=np.float32(1-variance/.5 if variance<.5 else 1.)
        for dimension in range(3):
            kernel=discrete_gaussian_kernel(variance,field.grid.shape_xyz[dimension])
            values=filtering.convolve_field(self.context,values,kernel,dimension,
                field.data if dimension==2 else None,weight)
        return CUDAField(values,field.grid)

    def compose(self,outer,inner):
        inverse=outer.grid.lps_to_index[:3,:3]
        values=kernels.compose(self.context,outer.data,inner.data,
            inverse@inner.grid.index_to_lps[:3,:3],inverse@(inner.grid.origin-outer.grid.origin),inverse)
        return CUDAField(values,inner.grid)

    def resample(self,field,grid):
        return self.compose(field,self.zero(grid))

    def scale(self,field,learning_rate):
        if not np.isfinite(learning_rate) or learning_rate<0:raise ValueError('Invalid learning rate')
        maximum=kernels.maximum_control(self.context,field.data,field.grid.spacing_xyz)
        target=float(np.float32(learning_rate))
        factor=np.float32(target/maximum) if maximum>0 else np.float32(target)
        values=kernels.multiply(self.context,field.data,factor)
        return CUDAField(values,field.grid),{
            'maximum_spacing_scaled_norm':maximum,'scale':float(factor),'effective_target':target}

    def invert(self,field,initial=None,*,max_iterations=20,mean_tolerance=.001,max_tolerance=.1,enforce_boundary=True,
               _evaluate_final_residual=True):
        if not isinstance(max_iterations,int) or max_iterations<0:raise ValueError('Invalid inverse iteration count')
        if not np.isfinite([mean_tolerance,max_tolerance]).all() or min(mean_tolerance,max_tolerance)<0:raise ValueError('Invalid inverse tolerances')
        if initial is not None and not initial.grid.same_geometry(field.grid):raise ValueError('Initial inverse grid mismatch')
        current=self.zero(field.grid) if initial is None else initial
        history=[];reason='max_iterations';spacing=field.grid.spacing_xyz
        inverse=field.grid.lps_to_index[:3,:3]
        mean_limit,max_limit=float(mean_tolerance),float(max_tolerance)
        for i in range(max_iterations):
            residual,norms,mean,maximum=kernels.residual_norms(
                self.context,field.data,current.data,inverse,spacing,mean_limit)
            mean_control,max_control=mean,maximum
            tiny=np.finfo(np.float32).tiny;eps=np.finfo(np.float32).eps
            guard_mean=abs(mean-mean_limit)<=8*eps*max(mean,mean_limit,tiny)
            guard_max=abs(maximum-max_limit)<=8*eps*max(maximum,max_limit,tiny)
            if guard_max:max_control=kernels.maximum_control(self.context,residual,spacing)
            epsilon=np.float32(.75 if i==0 else .5)
            current=CUDAField(kernels.inverse_step(self.context,current.data,residual,norms,maximum,epsilon,enforce_boundary),field.grid)
            del residual,norms
            history.append({'iteration':i+1,'mean_scaled_error_before_update':mean,'max_scaled_error_before_update':maximum,
                'epsilon':float(epsilon),'mean_control_error':mean_control,'max_control_error':max_control,
                'distance_to_mean_threshold':abs(mean_control-mean_limit),'distance_to_max_threshold':abs(max_control-max_limit),
                'native_scalar_control_guard':{'mean':bool(guard_mean),'maximum':bool(guard_max)}})
            if max_control<=max_limit or mean_control<=mean_limit:
                reason='maximum_error_threshold' if max_control<=max_limit else 'mean_error_threshold';break
        if _evaluate_final_residual:
            _,_,mean,maximum=kernels.residual_norms(self.context,field.data,current.data,inverse,spacing)
            final_residual={'mean_spacing_scaled_norm':mean,'max_spacing_scaled_norm':maximum}
        else:
            final_residual={'evaluated':False}
        return InversionResult(current,history,reason,final_residual)
