"""Native FP64 finalization of bounded FP32 Mattes histograms, Apache-2.0.

This control calculation has bins^2*(1+parameters) entries, not image voxels.
It uses the CPU's native FP64 instructions and never emulates floating point.
Its PDF degeneracy branches follow the official float1 metric, independently
of the host arithmetic dtype. This does not yet match every float1 rounding.
"""

import math
import numpy as np
from numba import njit
from fast_ants.backends.cpu.common.reductions import sum_values
from fast_ants.pipeline.float1_policy import FLOAT1_PDF_EPS


@njit(cache=True,fastmath=False)
def finalize_mattes_control(histogram,counts,derivatives,valid,bin_width,joint_scale=1.,derivative_scale=1.):
    bins=histogram.shape[0];parameters=derivatives.shape[2]
    joint=histogram.astype(np.float64)/joint_scale;derivative=derivatives.astype(np.float64)/derivative_scale
    total=sum_values(joint.ravel())
    fixed_pdf=np.zeros(bins,np.float64);moving_pdf=np.zeros(bins,np.float64)
    if valid==0 or total<FLOAT1_PDF_EPS:
        return np.finfo(np.float64).max,np.zeros(parameters),joint,fixed_pdf,moving_pdf,derivative
    factor=-1/(np.float64(bin_width)*valid)
    for i in range(bins):
        fixed_pdf[i]=counts[i]/total
        for j in range(bins):
            joint[i,j]/=total;moving_pdf[j]+=joint[i,j]
            for p in range(parameters):derivative[i,j,p]*=factor
    value=0.;descent=np.zeros(parameters,np.float64)
    for i in range(bins):
        if fixed_pdf[i]<=FLOAT1_PDF_EPS:continue
        log_fixed=math.log(fixed_pdf[i])
        for j in range(bins):
            if joint[i,j]>FLOAT1_PDF_EPS and moving_pdf[j]>FLOAT1_PDF_EPS:
                ratio=math.log(joint[i,j]/moving_pdf[j])
                value+=joint[i,j]*(ratio-log_fixed)
                for p in range(parameters):descent[p]+=derivative[i,j,p]*ratio
    return -value,descent,joint,fixed_pdf,moving_pdf,derivative
