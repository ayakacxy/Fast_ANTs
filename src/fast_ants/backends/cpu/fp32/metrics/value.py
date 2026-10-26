"""Fused ordinary-FP32 affine sampling and value-only Mattes histograms.

Only bounded per-chunk histograms and small sampling scratch are
allocated. Pixels and mapped coordinates never become full temporary arrays.
Private Kahan sums and their fixed-order Kahan merge retain FP32 storage;
partitioning follows input samples rather than the old compacted sample array.
"""
import numpy as np
from numba import get_num_threads,njit,prange

from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import TWO
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import _local_affine_integer_point32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import _sample_inside32
from fast_ants.core.arithmetic32 import fma32
from fast_ants.backends.cpu.fp32.metrics.parzen import parzen_add4
from fast_ants.backends.cpu.fp32.metrics.coordinates import map_integer4,integer_plan_kind


class ValueWorkspace32:
    """Private reusable PDFs with the same sample partition and merge order."""
    def __init__(self,count,bins):
        if count<0 or bins<5:
            raise ValueError('Value workspace needs count>=0 and bins>=5')
        self.count,self.bins=int(count),int(bins)
        self.block=max(4096,(self.count+127)//128)
        self.chunks=(self.count+self.block-1)//self.block
        shape=(self.chunks,self.bins,self.bins)
        self.partial=np.empty(shape,np.float32)
        self.partial_correction=np.empty(shape,np.float32)
        self.counts=np.empty((self.chunks,self.bins),np.int64)
        self.sizes=np.empty((self.chunks,2),np.int64)
        self.joint=np.empty((self.bins,self.bins),np.float32)
        self.correction=np.empty_like(self.joint)
        self.total_counts=np.empty(self.bins,np.int64)


@njit(cache=True,fastmath=False)
def fixed_bins32(values,bins,minimum,maximum):
    width=(maximum-minimum)/np.float32(bins-4)
    normalized_min=minimum/width-TWO
    output=np.empty(values.size,np.int32)
    for i in range(values.size):
        output[i]=min(max(int(values[i]/width-normalized_min),2),bins-3)
    return output


@njit(cache=True,fastmath=False,inline='always')
def _accumulate_sample(data,base,offset,fixed_bin,bins,minimum,maximum,width,normalized_min,
                       joint,correction,counts):
    pixel,inside=_sample_inside32(data,base,offset)
    if not inside:
        return 0,0
    if pixel<minimum or pixel>maximum:
        return 1,0
    # Exactly the old rounded-pixel division/subtraction boundaries.
    term=pixel/width-normalized_min
    moving_bin=min(max(int(term),2),bins-3)
    fraction=term-np.float32(moving_bin)
    counts[fixed_bin]+=1
    parzen_add4(joint,correction,fixed_bin,moving_bin,fraction)
    return 1,1


@njit(cache=True,fastmath=False)
def _chunk(data,points,fixed_bins,integer_linear,residual_linear,integer_offset,
           residual_offset,origin,bins,minimum,maximum,width,normalized_min,
           start,end,joint,correction,counts,points_are_integer):
    joint.fill(ZERO);correction.fill(ZERO);counts.fill(0)
    base=np.empty(3,np.int32);offset=np.empty(3,np.float32)
    batch_bases=np.empty((3,4),np.int32);batch_offsets=np.empty((3,4),np.float32)
    vector_end=end-(end-start)%4
    geometric=valid=0
    # Select once per chunk, outside either point loop.
    if points_are_integer:
        for n in range(start,end):
            if points_are_integer==2 and n<vector_end:
                lane=(n-start)%4
                if lane==0:
                    map_integer4(points,n,integer_linear,residual_linear,integer_offset,
                                 residual_offset,origin,batch_bases,batch_offsets)
                for c in range(3):base[c]=batch_bases[c,lane];offset[c]=batch_offsets[c,lane]
            else:
                _local_affine_integer_point32(points,n,integer_linear,residual_linear,integer_offset,
                                              residual_offset,origin,base,offset)
            g,sample_valid=_accumulate_sample(data,base,offset,fixed_bins[n],bins,minimum,maximum,
                width,normalized_min,joint,correction,counts)
            geometric+=g;valid+=sample_valid
    else:
        for n in range(start,end):
            for c in range(3):
                b=integer_offset[c];v=residual_offset[c]
                for j in range(3):
                    q=points[n,j]-origin[j]
                    integer=int(np.floor(q));fraction=q-np.float32(integer)
                    b+=integer_linear[c,j]*integer
                    v=fma32(residual_linear[c,j],np.float32(integer),v)
                    v=fma32(np.float32(integer_linear[c,j])+residual_linear[c,j],fraction,v)
                base[c]=b;offset[c]=v
            g,sample_valid=_accumulate_sample(data,base,offset,fixed_bins[n],bins,minimum,maximum,
                width,normalized_min,joint,correction,counts)
            geometric+=g;valid+=sample_valid
    return geometric,valid


@njit(cache=True,fastmath=False,parallel=True)
def _histograms(data,points,fixed_bins,integer_linear,residual_linear,integer_offset,
                residual_offset,origin,bins,minimum,maximum,width,normalized_min,parallel,points_are_integer,
                block,partial,partial_correction,counts,sizes,joint,correction,total_counts):
    chunks=partial.shape[0]
    if parallel:
        for task in prange(chunks):
            chunk=np.int64(task)
            sizes[chunk,0],sizes[chunk,1]=_chunk(data,points,fixed_bins,integer_linear,
                residual_linear,integer_offset,residual_offset,origin,bins,minimum,maximum,
                width,normalized_min,chunk*block,min((chunk+1)*block,points.shape[0]),
                partial[chunk],partial_correction[chunk],counts[chunk],points_are_integer)
    else:
        for chunk in range(chunks):
            sizes[chunk,0],sizes[chunk,1]=_chunk(data,points,fixed_bins,integer_linear,
                residual_linear,integer_offset,residual_offset,origin,bins,minimum,maximum,
                width,normalized_min,chunk*block,min((chunk+1)*block,points.shape[0]),
                partial[chunk],partial_correction[chunk],counts[chunk],points_are_integer)
    joint.fill(ZERO);correction.fill(ZERO);total_counts.fill(0)
    geometric=valid=0
    for chunk in range(chunks):
        geometric+=sizes[chunk,0];valid+=sizes[chunk,1]
        for i in range(bins):
            total_counts[i]+=counts[chunk,i]
            for j in range(bins):
                joint[i,j],correction[i,j]=add32(joint[i,j],correction[i,j],partial[chunk,i,j])
    return joint,total_counts,valid,geometric


def value_histogram32(data,points,fixed_bins,mapping,bins,moving_range,width,normalized_min,*,points_are_integer=False,workspace=None):
    """Return private borrowed PDFs when supplied with a reusable workspace."""
    parallel=points.shape[0]>=16384 and get_num_threads()>1
    workspace=ValueWorkspace32(len(points),bins) if workspace is None else workspace
    if (workspace.count,workspace.bins)!=(len(points),bins):
        raise ValueError('Value workspace must match sample count and bins')
    return _histograms(data,points,fixed_bins,*mapping,bins,
        np.float32(moving_range[0]),np.float32(moving_range[1]),
        np.float32(width),np.float32(normalized_min),parallel,integer_plan_kind(points,mapping,points_are_integer),
        workspace.block,workspace.partial,workspace.partial_correction,workspace.counts,
        workspace.sizes,workspace.joint,workspace.correction,workspace.total_counts)
