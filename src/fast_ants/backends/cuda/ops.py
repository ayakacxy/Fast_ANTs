"""Public CUDA device operators assembled by registration stage."""
from fast_ants.backends.cuda.runtime import CUDAContext
from fast_ants.backends.cuda.images import CUDAImageOps
from fast_ants.backends.cuda.fields.operators import CUDAFieldOps
from fast_ants.backends.cuda.metrics.cc.operators import CUDACCOps
from fast_ants.backends.cuda.syn import CUDASyNOps


class CUDAOps(CUDAImageOps,CUDAFieldOps,CUDACCOps,CUDASyNOps):
    """Geometry-aware GPU operations with explicit host/device boundaries."""

    def __init__(self,context=None):
        self.context=CUDAContext() if context is None else context
