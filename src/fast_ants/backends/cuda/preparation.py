"""Bounded per-registration reuse of prepared device images."""
import numpy as np


class _PreparationCache:
    """Retain identical positive-sigma preparations within one fit."""

    def __init__(self,maximum_bytes=256*1024**2):
        self.maximum_bytes=maximum_bytes;self.bytes=0;self.hits=0;self.misses=0;self.entries={}

    def smooth(self,ops,image,sigma,units):
        sigmas=np.full(3,float(sigma)) if np.ndim(sigma)==0 else np.asarray(sigma,np.float64)
        if sigmas.shape!=(3,) or not np.isfinite(sigmas).all() or np.any(sigmas<0):
            return ops.smooth_image(image,sigma,units)
        key=(id(image.data),id(image.grid),tuple(sigmas),units)
        if key in self.entries:
            self.hits+=1;return self.entries[key][1]
        self.misses+=1;result=ops.smooth_image(image,sigma,units)
        if np.any(sigmas>0) and result.data.nbytes<=self.maximum_bytes-self.bytes:
            self.entries[key]=(image,result);self.bytes+=result.data.nbytes
        return result
