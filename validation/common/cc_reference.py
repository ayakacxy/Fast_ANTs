"""Small, intentionally direct centred-window CC reference for CPU checks."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import numpy as np


def naive_cc(fixed, moving, gradient, valid, radius_xyz):
    radii = tuple(int(i) for i in radius_xyz[::-1])
    stats = np.zeros((7, *fixed.shape), dtype=np.float64)
    output = np.zeros((3, *fixed.shape), dtype=np.float64)
    values = []
    for index in np.ndindex(fixed.shape):
        if not valid[index]:
            continue
        slices = tuple(slice(max(0, i-r), min(n, i+r+1)) for i, r, n in zip(index, radii, fixed.shape))
        keep = valid[slices]
        f, m = fixed[slices][keep], moving[slices][keep]
        if f.size == 0:
            continue
        fc, mc = f-f.mean(), m-m.mean()
        ff, mm, fm = np.dot(fc, fc), np.dot(mc, mc), np.dot(fc, mc)
        product = ff*mm
        score = fm*fm/product if abs(product) > np.finfo(np.float64).eps else 1.0
        if ff > np.finfo(np.float64).eps and mm > np.finfo(np.float64).eps:
            weight = 2*fm/product*((fixed[index]-f.mean())-fm/mm*(moving[index]-m.mean()))
            output[(slice(None), *index)] = weight*gradient[(slice(None), *index)]
        stats[(slice(None), *index)] = [f.size, f.mean(), m.mean(), ff, mm, fm, score]
        values.append(score)
    if not values:
        raise ValueError("Empty reference CC")
    return -float(np.mean(values)), output, stats
