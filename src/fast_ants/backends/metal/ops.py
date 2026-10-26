"""Public device-operator assembly with stage-owned implementations.

The static mixins retain the MetalOps interface without adding dynamic backend
routing, intermediate downloads, or per-voxel Python dispatch.
"""
from fast_ants.backends.metal.runtime import MetalContext
from fast_ants.backends.metal.types import MetalImage, MetalField
from fast_ants.backends.metal.images import MetalImageOps
from fast_ants.backends.metal.fields.operators import MetalFieldOps
from fast_ants.backends.metal.metrics.cc.operators import MetalCCOps
from fast_ants.backends.metal.syn import MetalSyNOps


class MetalOps(MetalImageOps, MetalFieldOps, MetalCCOps, MetalSyNOps):
    """Geometry-aware GPU operations assembled by registration stage."""

    def __init__(self,context=None):
        self.context=MetalContext() if context is None else context
