"""Geometry-aware device fields, smoothing, composition, and inversion."""
import numpy as np
from fast_ants.backends.metal import dense
from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal.types import MetalField
from fast_ants.core.fields import DisplacementField
from fast_ants.core.results import InversionResult
from fast_ants.pipeline.gaussian import discrete_gaussian_kernel


class MetalFieldOps:
    """Methods for this registration stage; numerical definitions are unchanged."""

    def field(self,field):
        if isinstance(field,MetalField):return field
        return MetalField(self.context.upload(field.data,'input_field'),field.grid)

    def zero(self,grid):
        return MetalField(dense.zeros(self.context,(3,*grid.shape_zyx)),grid)

    def download_field(self,field,reason='final_output'):
        return DisplacementField(self.context.download(field.data,reason),field.grid)

    def smooth_field(self,field,variance):
        if not np.isfinite(variance) or variance<0:raise ValueError('Invalid field variance')
        if variance==0:return MetalField(field.data,field.grid)
        values=field.data
        weight=np.float32(1-variance/.5 if variance<.5 else 1.)
        if 'direct_field_convolution' in self.context.optimizations:
            for dimension in range(3):
                kernel=discrete_gaussian_kernel(variance,field.grid.shape_xyz[dimension])
                values=dense.convolve_field(self.context,values,kernel,dimension,
                    field.data if dimension==2 else None,weight)
            return MetalField(values,field.grid)
        for dimension in range(3):
            kernel=discrete_gaussian_kernel(variance,field.grid.shape_xyz[dimension])
            moved=mx.moveaxis(values,3-dimension,-1);shape=moved.shape
            result=dense.convolve(self.context,moved.reshape(-1,shape[-1]),kernel)
            values=mx.moveaxis(result.reshape(shape),-1,3-dimension)
        return MetalField(dense.mix_boundary(self.context,values,field.data,weight),field.grid)

    def compose(self,outer,inner):
        inverse=outer.grid.lps_to_index[:3,:3]
        values=dense.compose(self.context,outer.data,inner.data,
                             inverse@inner.grid.index_to_lps[:3,:3],
                             inverse@(inner.grid.origin-outer.grid.origin),inverse)
        return MetalField(values,inner.grid)

    def resample(self,field,grid):
        return self.compose(field,self.zero(grid))

    def scale(self,field,learning_rate):
        if not np.isfinite(learning_rate) or learning_rate<0:raise ValueError('Invalid learning rate')
        maximum=dense.maximum_control(self.context,field.data,field.grid.spacing_xyz)
        target=float(np.float32(learning_rate))
        factor=np.float32(target/maximum) if maximum>0 else np.float32(target)
        # SyN normalizes once. Rechecking the stored field and nudging the
        # factor changes the source algorithm and adds device/host controls.
        values=dense.multiply(self.context,field.data,factor)
        return MetalField(values,field.grid),{
            'maximum_spacing_scaled_norm':maximum,'scale':float(factor),'effective_target':target}

    def invert(self,field,initial=None,*,max_iterations=20,mean_tolerance=.001,max_tolerance=.1,enforce_boundary=True,
               _evaluate_final_residual=True):
        if not isinstance(max_iterations,int) or max_iterations<0:raise ValueError('Invalid inverse iteration count')
        if not np.isfinite([mean_tolerance,max_tolerance]).all() or min(mean_tolerance,max_tolerance)<0:raise ValueError('Invalid inverse tolerances')
        if initial is not None and not initial.grid.same_geometry(field.grid):raise ValueError('Initial inverse grid mismatch')
        current=self.zero(field.grid) if initial is None else initial
        history=[];reason='max_iterations';spacing=field.grid.spacing_xyz
        mean_limit,max_limit=float(mean_tolerance),float(max_tolerance)
        if 'fused_inverse_norms' in self.context.optimizations:
            inverse=field.grid.lps_to_index[:3,:3]
            linear=inverse@field.grid.index_to_lps[:3,:3]
            offset=inverse@(field.grid.origin-field.grid.origin)
            def residual_norms(current,limit=None):
                # same_geometry accepts a small tolerance; the initial inverse
                # may therefore carry a distinct (but accepted) grid. Preserve
                # its actual sampling geometry until the first update.
                mapping_linear,mapping_offset=linear,offset
                if current.grid is not field.grid:
                    mapping_linear=inverse@current.grid.index_to_lps[:3,:3]
                    mapping_offset=inverse@(current.grid.origin-field.grid.origin)
                residual,values=dense.compose(self.context,field.data,current.data,
                    mapping_linear,mapping_offset,inverse,spacing)
                norms,mean,maximum=dense.reduce_norms(self.context,values,limit)
                return residual,norms,mean,maximum
        else:
            def residual_norms(current,limit=None):
                residual=self.compose(field,current).data
                norms,mean,maximum=dense.norms(self.context,residual,spacing,limit)
                return residual,norms,mean,maximum
        for i in range(max_iterations):
            residual,norms,mean,maximum=residual_norms(current,mean_limit)
            mean_control,max_control=mean,maximum
            tiny=np.finfo(np.float32).tiny;eps=np.finfo(np.float32).eps
            guard_mean=abs(mean-mean_limit)<=8*eps*max(mean,mean_limit,tiny)
            guard_max=abs(maximum-max_limit)<=8*eps*max(maximum,max_limit,tiny)
            if guard_mean or guard_max:
                if guard_max:max_control=dense.maximum_control(self.context,residual,spacing)
            epsilon=np.float32(.75 if i==0 else .5)
            current=MetalField(dense.inverse_step(self.context,current.data,residual,norms,maximum,epsilon,enforce_boundary),field.grid)
            history.append({'iteration':i+1,'mean_scaled_error_before_update':mean,'max_scaled_error_before_update':maximum,
                'epsilon':float(epsilon),'mean_control_error':mean_control,'max_control_error':max_control,
                'distance_to_mean_threshold':abs(mean_control-mean_limit),'distance_to_max_threshold':abs(max_control-max_limit),
                'native_scalar_control_guard':{'mean':bool(guard_mean),'maximum':bool(guard_max)}})
            if max_control<=max_limit or mean_control<=mean_limit:
                reason='maximum_error_threshold' if max_control<=max_limit else 'mean_error_threshold';break
        if _evaluate_final_residual:
            _,_,mean,maximum=residual_norms(current)
            final_residual={'mean_spacing_scaled_norm':mean,'max_spacing_scaled_norm':maximum}
        else:
            # The last post-update residual is a separate diagnostic, unused
            # by the inverse update and stopping rules above.
            final_residual={'evaluated':False}
        return InversionResult(current,history,reason,final_residual)

