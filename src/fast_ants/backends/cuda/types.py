"""Validated CUDA image and displacement-field data contracts."""
from dataclasses import dataclass, field as dataclass_field
from fast_ants.backends.cuda.runtime import cp
from fast_ants.core.types import Grid


@dataclass
class CUDAImage:
    data: object
    grid: Grid
    metadata: dict = dataclass_field(default_factory=dict)

    def __post_init__(self):
        if (not isinstance(self.data,cp.ndarray) or self.data.dtype!=cp.float32
                or self.data.shape!=self.grid.shape_zyx or not self.data.flags.c_contiguous):
            raise ValueError('CUDA images require contiguous FP32 device arrays on their declared grid')


@dataclass
class CUDAField:
    data: object
    grid: Grid
    inverse_field: object = None

    def __post_init__(self):
        if (not isinstance(self.data,cp.ndarray) or self.data.dtype!=cp.float32
                or self.data.shape!=(3,*self.grid.shape_zyx) or not self.data.flags.c_contiguous):
            raise ValueError('CUDA fields require contiguous (3,z,y,x) FP32 device arrays')
