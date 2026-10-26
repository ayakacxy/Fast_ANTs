"""Symmetric device updates and level convergence orchestration."""
import numpy as np
from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.backends.metal.runtime import mx
from fast_ants.core.state import SyNState
from fast_ants.core.results import SyNStepResult


def _syn_level(ops,fs,ms,initial,stage,linear,fm,mm,iterations,level,callback,_progress=None):
    """Run a level; legacy scalar-order replay is an explicit diagnostic flag.

    The update fields are independent of the CC scalar reduction. A fixed
    positive tree can perturb scores by a few FP32 ulps, so a conservative
    convergence guard replays the whole level from its unchanged initial state.
    This also reconstructs the complete convergence energy history. Callbacks
    use serial scores immediately, so candidate callbacks are never replayed.
    """
    ctx=ops.context;previous=ctx.serial_cc_guard;parallel='parallel_cc' in ctx.optimizations
    strict='reference_cc_convergence' in ctx.optimizations
    serial=previous or (strict and parallel and callback is not None);discarded=0
    if strict and parallel and callback is not None:ctx.control_guards['cc_callback_serial']+=1
    try:
        for attempt in range(2):
            if attempt and _progress is not None:
                _progress.message(f'REPLAY reason=reference_cc_convergence; discarded_iterations={discarded}')
            ctx.serial_cc_guard=serial or attempt>0
            state=initial;convergence=WindowConvergence(stage.convergence_window);records=[];reason='max_iterations';retry=False
            for i in range(iterations):
                step=ops.syn_step(fs,ms,state,linear_transform=linear,radius=stage.radius,
                    gradient_step=stage.gradient_step,update_variance=stage.update_variance,total_variance=stage.total_variance,
                    fixed_mask=fm,moving_mask=mm,_evaluate_final_residual=callback is not None)
                state=step.state;value=convergence.add(step.value)
                if strict and parallel and not ctx.serial_cc_guard and abs(value-stage.convergence_threshold)<=64*np.finfo(np.float32).eps:
                    retry=True;discarded=i+1;ctx.control_guards['cc_level_reference_retry']+=1;break
                records.append({'iteration':i+1,'value':step.value,'convergence':value,**step.details})
                if _progress is not None:_progress(records[-1])
                if callback is not None:callback(level,i+1,step)
                if value<stage.convergence_threshold:reason='window_convergence';break
            if not retry:return state,records,reason,discarded
        raise ArithmeticError('CC convergence reference replay did not finish')
    finally:ctx.serial_cc_guard=previous

# Device SyN update stages, sharing the same convergence-level orchestration.


class MetalSyNOps:
    """Methods for this registration stage; numerical definitions are unchanged."""

    def identity(self,grid):
        return SyNState(*(self.zero(grid) for _ in range(4)))

    def adapt(self,state,grid):
        return SyNState(*(self.resample(f,grid) for f in state.fields().values()))

    def combine(self,state):
        forward=self.compose(state.moving_inverse,state.fixed_to_middle)
        inverse=self.compose(state.fixed_inverse,state.moving_to_middle)
        forward.inverse_field=inverse;inverse.inverse_field=forward
        return forward,inverse

    def syn_step(self,fixed,moving,state,*,linear_transform=None,radius=2,gradient_step=.1,
                 update_variance=3.,total_variance=0.,fixed_mask=None,moving_mask=None,capture=False,
                 _evaluate_final_residual=True):
        fmid=self.pull(fixed,state.grid,state.fixed_inverse)
        mmid=self.pull(moving,state.grid,state.moving_inverse,linear_transform)
        valid=None
        for mask,field,linear in ((fixed_mask,state.fixed_inverse,None),(moving_mask,state.moving_inverse,linear_transform)):
            if mask is not None:
                sampled=self.pull(mask,state.grid,field,linear,nearest=True).data
                if valid is None:
                    valid=self.context.call('mask_valid','uint i=thread_position_in_grid.x;out0[i]=data[i]!=0;',
                        {'data':sampled},[sampled.shape],sampled.size,[mx.bool_])
                else:
                    valid=self.context.call('mask_and','uint i=thread_position_in_grid.x;out0[i]=valid[i]&&data[i]!=0;',
                        {'data':sampled,'valid':valid},[sampled.shape],sampled.size,[mx.bool_])
        if 'paired_cc' in self.context.optimizations:
            deferred=('deferred_syn_controls' in self.context.optimizations and not capture
                      and not self.context.serial_cc_guard)
            fm,mf=self.cc_pair(fmid,mmid,radius,valid,_defer_control=deferred)
        else:
            fm=self.cc(fmid,mmid,radius,valid);mf=self.cc(mmid,fmid,radius,valid)
        fs,ms=self.smooth_field(fm.update,update_variance),self.smooth_field(mf.update,update_variance)
        if 'fused_scaled_compose' in self.context.optimizations and not capture:
            from fast_ants.backends.metal.fields.composition import scaled_compose_pair
            ft,mt,fscale,mscale=scaled_compose_pair(self,(fs,ms),
                (state.fixed_to_middle,state.moving_to_middle),gradient_step)
            ft,mt=self.smooth_field(ft,total_variance),self.smooth_field(mt,total_variance)
        else:
            fu,fscale=self.scale(fs,gradient_step);mu,mscale=self.scale(ms,gradient_step)
            ft=self.smooth_field(self.compose(fu,state.fixed_to_middle),total_variance)
            mt=self.smooth_field(self.compose(mu,state.moving_to_middle),total_variance)
        evaluate_residual=capture or _evaluate_final_residual
        if 'batched_inverse' in self.context.optimizations:
            if 'overlapped_inverse' in self.context.optimizations:
                from fast_ants.backends.metal.fields.overlap import invert_pair
            else:
                from fast_ants.backends.metal.fields.inverse import invert_pair
            fi,mi=invert_pair(self,(ft,mt),(state.fixed_inverse,state.moving_inverse),
                              _evaluate_final_residual=evaluate_residual)
            ff,mm=invert_pair(self,(fi.field,mi.field),(ft,mt),
                              _evaluate_final_residual=evaluate_residual)
        else:
            fi=self.invert(ft,state.fixed_inverse,_evaluate_final_residual=evaluate_residual)
            ff=self.invert(fi.field,ft,_evaluate_final_residual=evaluate_residual)
            mi=self.invert(mt,state.moving_inverse,_evaluate_final_residual=evaluate_residual)
            mm=self.invert(mi.field,mt,_evaluate_final_residual=evaluate_residual)
        next_state=SyNState(ff.field,fi.field,mm.field,mi.field)
        details={'forward_value':fm.value,'reverse_value':mf.value,'fixed_scale':fscale,'moving_scale':mscale,
                 'fixed_valid_centres':fm.valid_points,'moving_valid_centres':mf.valid_points,
                 'inversions':{name:{'history':r.history,'stop_reason':r.stop_reason,'final_residual':r.final_residual}
                              for name,r in (('fixed_inverse',fi),('fixed_forward',ff),('moving_inverse',mi),('moving_forward',mm))}}
        intermediate={}
        if capture:
            intermediate={'fixed_midpoint':fmid,'moving_midpoint':mmid,'fixed_raw_update':fm.update,'moving_raw_update':mf.update,
                'fixed_smoothed_update':fs,'moving_smoothed_update':ms,'fixed_scaled_update':fu,'moving_scaled_update':mu,
                'fixed_total_before_inverse':ft,'moving_total_before_inverse':mt,**next_state.fields()}
        return SyNStepResult(next_state,float(np.float32(.5)*np.float32(fm.value+mf.value)),details,intermediate)
