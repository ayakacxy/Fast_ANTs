"""Four-field SyN state shared by CPU and Metal; CPU methods load on use."""
from dataclasses import dataclass
@dataclass
class SyNState:
    fixed_to_middle: object
    fixed_inverse: object
    moving_to_middle: object
    moving_inverse: object

    def __post_init__(self):
        grid = self.fixed_to_middle.grid
        if any(not grid.same_geometry(field.grid) for field in self.fields().values()):
            raise ValueError("All four SyN state fields must share a physical grid")

    @classmethod
    def identity(cls, grid):
        from fast_ants.backends.cpu.fp64.fields import zero_field
        return cls(*(zero_field(grid) for i in range(4)))

    @property
    def grid(self):
        return self.fixed_to_middle.grid

    def fields(self):
        return {name:getattr(self,name) for name in ('fixed_to_middle','fixed_inverse','moving_to_middle','moving_inverse')}

    def adapt(self, grid):
        from fast_ants.backends.cpu.fp64.fields import resample_field
        return SyNState(*(resample_field(field,grid) for field in self.fields().values()))

    def combined(self):
        from fast_ants.backends.cpu.fp64.fields import compose_fields
        forward = compose_fields(self.moving_inverse,self.fixed_to_middle)
        inverse = compose_fields(self.fixed_inverse,self.moving_to_middle)
        forward.inverse_field = inverse
        inverse.inverse_field = forward
        return forward,inverse

