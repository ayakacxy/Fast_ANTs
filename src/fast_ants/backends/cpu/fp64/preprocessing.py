"""CPU image preprocessing and construction of metric pyramid levels."""
import numpy as np
from fast_ants.core.types import Grid
from fast_ants.core.types import Image
from fast_ants.backends.cpu.common.reductions import rescale
from fast_ants.backends.cpu.common.reductions import minmax
from fast_ants.pipeline.preprocessing import window_parameters
from fast_ants.pipeline.preprocessing import window64
from fast_ants.backends.cpu.fp64.filtering import smooth_image
from fast_ants.backends.cpu.fp64.sampling import apply_image
from fast_ants.pipeline.pyramid import PyramidLevel
from fast_ants.pipeline.pyramid import shrink_grid
def shrink_image(image, factors):
    # ITK ShrinkImageFilter samples nearest centres on its centered output grid.
    return apply_image(image, shrink_grid(image.grid, factors), interpolation="NearestNeighbor")


def preprocess_image(image, winsorization=None, histogram_matching=False, profile="ants"):
    if winsorization is not None or histogram_matching:
        raise ValueError("The current registration subset disables winsorization and histogram matching")
    values=np.asarray(image.data,dtype=np.float64)
    if profile=="ants":
        low,high,scale,offset=window_parameters(*minmax(values.ravel()))
        output=window64(values,low,high,scale,offset)
    elif profile=="minmax":output=rescale(values)
    else:raise ValueError("Preprocessing profile must be ants or minmax")
    return Image(output,image.grid,{**image.metadata,"preprocessing_profile":profile})


def prepare_level(fixed, moving, shrink_factor, sigma, sigma_units="vox"):
    # RegistrationMethodv4 shrinks the virtual domain; it keeps both metric
    # images at full resolution after smoothing. SyN later resamples them onto
    # the virtual grid inside its own update calculation.
    return PyramidLevel(smooth_image(fixed, sigma, sigma_units), smooth_image(moving, sigma, sigma_units),
                        shrink_grid(fixed.grid, shrink_factor), shrink_factor, sigma, sigma_units)

