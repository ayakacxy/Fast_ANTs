"""Geometry-aware local CC dispatch and shared score controls."""
import numpy as np
from fast_ants.backends.cuda.runtime import cp
from fast_ants.backends.cuda.types import CUDAField
from fast_ants.backends.cuda.metrics.cc import dense, tiled
from fast_ants.backends.cuda.metrics.cc.control import PendingCCScore, DeferredCCEvaluation
from fast_ants.core.results import CCEvaluation, NoValidSamplesError


class CUDACCOps:
    """Paired centered windows retain the serial FP32 sample order."""

    def _cc_valid(self,fixed,moving,radius,valid):
        if not fixed.grid.same_geometry(moving.grid):raise ValueError('CC grid mismatch')
        if not isinstance(radius,int) or radius<0:raise ValueError('CC radius must be a nonnegative integer')
        if valid is None:
            valid=self.context.call('valid_ones','out0[i]=true;',{},[fixed.data.shape],fixed.data.size,[cp.bool_])
        self.context.require(valid)
        if valid.dtype!=cp.bool_ or valid.shape!=fixed.data.shape:raise ValueError('CC validity grid/type mismatch')
        return valid

    def cc(self,fixed,moving,radius=2,valid=None):
        valid=self._cc_valid(fixed,moving,radius,valid)
        update,value,count=dense.cc(self.context,fixed.data,moving.data,self.gradient(moving),valid,radius)
        if not count:raise NoValidSamplesError('No valid CUDA CC centre')
        if not np.isfinite(value):raise ValueError('CUDA CC score must be finite')
        return CCEvaluation(value,CUDAField(update,fixed.grid),count)

    def cc_pair(self,fixed,moving,radius=2,valid=None,*,_defer_control=False):
        valid=self._cc_valid(fixed,moving,radius,valid)
        previous=getattr(self.context,'defer_cc_score',False)
        self.context.defer_cc_score=bool(_defer_control)
        try:
            forward,reverse,value,count=tiled.cc_pair(self.context,fixed.data,moving.data,
                self.gradient(fixed),self.gradient(moving),valid,radius)
        finally:
            self.context.defer_cc_score=previous
        if _defer_control:
            pending=PendingCCScore(self.context,value,count)
            return (DeferredCCEvaluation(CUDAField(forward,fixed.grid),pending),
                    DeferredCCEvaluation(CUDAField(reverse,moving.grid),pending))
        if not count:raise NoValidSamplesError('No valid CUDA CC centre')
        if not np.isfinite(value):raise ValueError('CUDA CC score must be finite')
        return (CCEvaluation(value,CUDAField(forward,fixed.grid),count),
                CCEvaluation(value,CUDAField(reverse,moving.grid),count))
