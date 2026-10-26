"""ANTs default endpoint-histogram windowing, without optional winsorization.

ANTs PreprocessImage uses ITK's 256-bin histogram with marginal scale 10 even
when the quantiles are 0 and 1. Its upper endpoint includes the histogram
margin, and both endpoint quantiles are stored as float before windowing.
This differs from dividing by the observed image min/max.
"""

import numpy as np
from numba import njit
from fast_ants.backends.cpu.common.reductions import minmax


def window_parameters(minimum,maximum):
    minimum,maximum=float(minimum),float(maximum)
    if maximum<=minimum:return minimum,maximum,0.,0.
    low=float(np.float32(minimum))
    high=float(np.float32(maximum+(maximum-minimum)/256./10.))
    if not np.isfinite([low,high]).all() or high<=low:raise ValueError("ANTs default window endpoints are not representable")
    scale=1./(high-low)
    return low,high,scale,-low*scale


@njit(cache=True,fastmath=False)
def window64(values,low,high,scale,offset):
    output=np.empty(values.shape,np.float64)
    src,dst=values.ravel(),output.ravel()
    for i in range(src.size):
        value=src[i]
        if scale==0.:dst[i]=0.
        elif value<low:dst[i]=0.
        elif value>high:dst[i]=1.
        else:dst[i]=value*scale+offset
    return output
