"""Geometry-aware local CC dispatch and independent score controls."""
import numpy as np
from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal import dense
from fast_ants.backends.metal.types import MetalField
from fast_ants.core.results import CCEvaluation, NoValidSamplesError


class MetalCCOps:
    """Methods for this registration stage; numerical definitions are unchanged."""

    def cc(self,fixed,moving,radius=2,valid=None):
        if not fixed.grid.same_geometry(moving.grid):raise ValueError('CC grid mismatch')
        if not isinstance(radius,int) or radius<0:raise ValueError('CC radius must be a nonnegative integer')
        if valid is None:
            valid=self.context.call('valid_ones','out0[thread_position_in_grid.x]=true;',{},
                                    [fixed.data.shape],fixed.data.size,[mx.bool_])
        if valid.dtype!=mx.bool_ or valid.shape!=fixed.data.shape:raise ValueError('CC validity grid/type mismatch')
        update,value,count=dense.cc(self.context,fixed.data,moving.data,self.gradient(moving),valid,radius)
        if not count:raise NoValidSamplesError('No valid Metal CC centre')
        return CCEvaluation(value,MetalField(update,fixed.grid),count)

    def _cc_pair_arrays(self,fixed,moving,radius,valid):
        if 'cc_window_stats' in self.context.optimizations:
            from fast_ants.backends.metal.metrics.cc.window import cc_pair as window_cc_pair
            return window_cc_pair(self.context,fixed.data,moving.data,
                self.gradient(fixed),self.gradient(moving),valid,radius)
        if 'tiled_cc' in self.context.optimizations:
            from fast_ants.backends.metal.metrics.cc.tiled import cc_pair as tiled_cc_pair
            return tiled_cc_pair(self.context,fixed.data,moving.data,
                self.gradient(fixed),self.gradient(moving),valid,radius)
        if 'inline_cc_gradient' in self.context.optimizations:
            from fast_ants.backends.metal.metrics.cc.gradient import cc_pair_gradient
            spacing=fixed.grid.spacing_xyz
            shift=fixed.grid.lps_to_index[:3,:3]@np.diag(spacing*.5)
            return cc_pair_gradient(self.context,fixed.data,moving.data,valid,radius,shift,spacing)
        return dense.cc_pair(self.context,fixed.data,moving.data,
            self.gradient(fixed),self.gradient(moving),valid,radius)

    def cc_pair(self,fixed,moving,radius=2,valid=None,*,_defer_control=False):
        if not fixed.grid.same_geometry(moving.grid):raise ValueError('CC grid mismatch')
        if not isinstance(radius,int) or radius<0:raise ValueError('CC radius must be a nonnegative integer')
        if valid is None:
            valid=self.context.call('valid_ones','out0[thread_position_in_grid.x]=true;',{},
                                    [fixed.data.shape],fixed.data.size,[mx.bool_])
        if valid.dtype!=mx.bool_ or valid.shape!=fixed.data.shape:raise ValueError('CC validity grid/type mismatch')
        previous=self.context.defer_cc_score;self.context.defer_cc_score=bool(_defer_control)
        try:
            forward,reverse,value,count=self._cc_pair_arrays(fixed,moving,radius,valid)
        finally:
            self.context.defer_cc_score=previous
        if _defer_control:
            from fast_ants.backends.metal.metrics.cc.control import PendingCCScore
            from fast_ants.backends.metal.metrics.cc.control import DeferredCCEvaluation
            pending=PendingCCScore(self.context,value,count)
            self.context.submit(pending.packet,reason='independent_cc_score')
            return (DeferredCCEvaluation(MetalField(forward,fixed.grid),pending),
                    DeferredCCEvaluation(MetalField(reverse,moving.grid),pending))
        if not count:raise NoValidSamplesError('No valid Metal CC centre')
        return (CCEvaluation(value,MetalField(forward,fixed.grid),count),
                CCEvaluation(value,MetalField(reverse,moving.grid),count))

