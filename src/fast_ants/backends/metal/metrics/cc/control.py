"""One deferred score/count read shared by the two CC update directions."""

import numpy as np

from fast_ants.core.results import NoValidSamplesError


class PendingCCScore:
    def __init__(self,ctx,value,count):
        from fast_ants.backends.metal.runtime import mx
        self.ctx=ctx;self.result=None
        self.packet=ctx.call('cc_packed_control32',
            'uint i=thread_position_in_grid.x;out0[i]=i==0u?as_type<uint>(value[0]):count[0];',
            {'value':value,'count':count},[(2,)],2,[mx.uint32])

    def resolve(self):
        if self.result is None:
            words=self.ctx.control(self.packet,'cc_final',2)
            value=float(words[:1].view(np.float32)[0]);count=int(words[1])
            if not count:raise NoValidSamplesError('No valid Metal CC centre')
            if not np.isfinite(value):raise ValueError('Metal CC score must be finite')
            self.result=value,count
        return self.result


class DeferredCCEvaluation:
    def __init__(self,update,pending):
        self.update=update;self.pending=pending;self.statistics=None

    @property
    def value(self):return self.pending.resolve()[0]

    @property
    def valid_points(self):return self.pending.resolve()[1]
