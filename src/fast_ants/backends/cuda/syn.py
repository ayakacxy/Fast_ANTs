"""Symmetric CUDA updates and level convergence orchestration."""
import numpy as np
from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.backends.cuda.runtime import cp
from fast_ants.core.state import SyNState
from fast_ants.core.results import SyNStepResult


def _syn_level(ops,fs,ms,initial,stage,linear,fm,mm,iterations,level,callback,_progress=None):
    state=initial;convergence=WindowConvergence(stage.convergence_window)
    records=[];reason='max_iterations'
    for i in range(iterations):
        step=ops.syn_step(fs,ms,state,linear_transform=linear,radius=stage.radius,
            gradient_step=stage.gradient_step,update_variance=stage.update_variance,
            total_variance=stage.total_variance,fixed_mask=fm,moving_mask=mm,
            _evaluate_final_residual=callback is not None)
        state=step.state;value=convergence.add(step.value)
        records.append({'iteration':i+1,'value':step.value,'convergence':value,**step.details})
        if _progress is not None:_progress(records[-1])
        if callback is not None:callback(level,i+1,step)
        if value<stage.convergence_threshold:reason='window_convergence';break
    return state,records,reason


class CUDASyNOps:
    """Four-field greedy SyN with independently stopped inverse solves."""

    def identity(self,grid):
        return SyNState(*(self.zero(grid) for _ in range(4)))

    def adapt(self,state,grid):
        return SyNState(*(self.resample(field,grid) for field in state.fields().values()))

    def combine(self,state):
        forward=self.compose(state.moving_inverse,state.fixed_to_middle)
        inverse=self.compose(state.fixed_inverse,state.moving_to_middle)
        forward.inverse_field=inverse;inverse.inverse_field=forward
        return forward,inverse

    def syn_step(self,fixed,moving,state,*,linear_transform=None,radius=2,gradient_step=.1,
                 update_variance=3.,total_variance=0.,fixed_mask=None,moving_mask=None,capture=False,
                 _evaluate_final_residual=True,_reference_inversions=False):
        fmid=self.pull(fixed,state.grid,state.fixed_inverse)
        mmid=self.pull(moving,state.grid,state.moving_inverse,linear_transform)
        valid=None
        for mask,field,linear in ((fixed_mask,state.fixed_inverse,None),(moving_mask,state.moving_inverse,linear_transform)):
            if mask is not None:
                sampled=self.pull(mask,state.grid,field,linear,nearest=True).data
                if valid is None:
                    valid=self.context.call('mask_valid','out0[i]=data[i]!=0;',
                        {'data':sampled},[sampled.shape],sampled.size,[cp.bool_])
                else:
                    valid=self.context.call('mask_and','out0[i]=valid[i]&&data[i]!=0;',
                        {'data':sampled,'valid':valid},[sampled.shape],sampled.size,[cp.bool_])
        fm,mf=self.cc_pair(fmid,mmid,radius,valid,
            _defer_control=not capture and not _reference_inversions)
        fs,ms=self.smooth_field(fm.update,update_variance),self.smooth_field(mf.update,update_variance)
        fu,fscale=self.scale(fs,gradient_step);mu,mscale=self.scale(ms,gradient_step)
        ft=self.smooth_field(self.compose(fu,state.fixed_to_middle),total_variance)
        mt=self.smooth_field(self.compose(mu,state.moving_to_middle),total_variance)
        evaluate_residual=capture or _evaluate_final_residual
        if capture or _reference_inversions:
            fi=self.invert(ft,state.fixed_inverse,_evaluate_final_residual=evaluate_residual)
            ff=self.invert(fi.field,ft,_evaluate_final_residual=evaluate_residual)
            mi=self.invert(mt,state.moving_inverse,_evaluate_final_residual=evaluate_residual)
            mm=self.invert(mi.field,mt,_evaluate_final_residual=evaluate_residual)
        else:
            from fast_ants.backends.cuda.fields.inverse import invert_pair
            fi,mi=invert_pair(self,(ft,mt),(state.fixed_inverse,state.moving_inverse),
                _evaluate_final_residual=evaluate_residual)
            ff,mm=invert_pair(self,(fi.field,mi.field),(ft,mt),
                _evaluate_final_residual=evaluate_residual)
        next_state=SyNState(ff.field,fi.field,mm.field,mi.field)
        details={'forward_value':fm.value,'reverse_value':mf.value,'fixed_scale':fscale,'moving_scale':mscale,
                 'fixed_valid_centres':fm.valid_points,'moving_valid_centres':mf.valid_points,
                 'inversions':{name:{'history':result.history,'stop_reason':result.stop_reason,'final_residual':result.final_residual}
                              for name,result in (('fixed_inverse',fi),('fixed_forward',ff),('moving_inverse',mi),('moving_forward',mm))}}
        intermediate={}
        if capture:
            intermediate={'fixed_midpoint':fmid,'moving_midpoint':mmid,'fixed_raw_update':fm.update,'moving_raw_update':mf.update,
                'fixed_smoothed_update':fs,'moving_smoothed_update':ms,'fixed_scaled_update':fu,'moving_scaled_update':mu,
                'fixed_total_before_inverse':ft,'moving_total_before_inverse':mt,**next_state.fields()}
        return SyNStepResult(next_state,float(np.float32(.5)*np.float32(fm.value+mf.value)),details,intermediate)
