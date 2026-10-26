"""Validated GPU image and displacement-field data contracts."""
from dataclasses import dataclass, field as dataclass_field
from fast_ants.backends.metal.runtime import mx
from fast_ants.core.types import Grid

@dataclass
class MetalImage:
    data: object
    grid: Grid
    metadata: dict = dataclass_field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data,mx.array) or self.data.dtype!=mx.float32 or self.data.shape!=self.grid.shape_zyx:
            raise ValueError('Metal images require FP32 device arrays on their declared grid')


@dataclass
class MetalField:
    data: object
    grid: Grid
    inverse_field: object = None

    def __post_init__(self):
        if not isinstance(self.data,mx.array) or self.data.dtype!=mx.float32 or self.data.shape!=(3,*self.grid.shape_zyx):
            raise ValueError('Metal fields require (3,z,y,x) FP32 device arrays')

