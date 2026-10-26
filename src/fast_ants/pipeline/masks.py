"""Binary image masks in their own physical geometry."""

import numpy as np

from fast_ants.backends.cpu.fp64.sampling import sample_image


def sample_mask(mask, points_lps):
    if not np.all((mask.data == 0) | (mask.data == 1)):
        raise ValueError("The current mask contract accepts binary 0/1 masks only")
    values, valid = sample_image(mask, points_lps, interpolation="NearestNeighbor", default_value=0)
    return valid & (values != 0)
