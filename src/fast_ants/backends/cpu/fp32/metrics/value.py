"""Fused ordinary-FP32 affine sampling and value-only Mattes histograms.

Only bounded per-chunk histograms and three-element sampling scratch are
allocated. Pixels and mapped coordinates never become full temporary arrays.
Private Kahan sums and their fixed-order Kahan merge retain FP32 storage;
partitioning follows input samples rather than the old compacted sample array.
"""
import numpy as np
from numba import get_num_threads,njit,prange

from fast_ants.backends.cpu.fp32.kernels import ZERO
from fast_ants.backends.cpu.fp32.kernels import TWO
from fast_ants.backends.cpu.fp32.kernels import add32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import cubic32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import local_inside32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import local_sample32
from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import _local_affine_integer_point32
from fast_ants.core.arithmetic32 import fma32


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
    if not local_inside32(base,offset,data.shape):
        return 0,0
    pixel=local_sample32(data,base,offset,ZERO)
    if pixel<minimum or pixel>maximum:
        return 1,0
    # Exactly the old rounded-pixel division/subtraction boundaries.
    term=pixel/width-normalized_min
    moving_bin=min(max(int(term),2),bins-3)
    fraction=term-np.float32(moving_bin)
    counts[fixed_bin]+=1
    for j in range(moving_bin-1,moving_bin+3):
        argument=np.float32(j-moving_bin)-fraction
        joint[fixed_bin,j],correction[fixed_bin,j]=add32(
            joint[fixed_bin,j],correction[fixed_bin,j],cubic32(argument))
    return 1,1


@njit(cache=True,fastmath=False)
def _chunk(data,points,fixed_bins,integer_linear,residual_linear,integer_offset,
           residual_offset,origin,bins,minimum,maximum,width,normalized_min,
           start,end,joint,counts,points_are_integer):
    correction=np.zeros((bins,bins),np.float32)
    base=np.empty(3,np.int32);offset=np.empty(3,np.float32)
    geometric=valid=0
    # Select once per chunk, outside either point loop.
    if points_are_integer:
        for n in range(start,end):
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
                residual_offset,origin,bins,minimum,maximum,width,normalized_min,parallel,points_are_integer):
    block=max(4096,(points.shape[0]+127)//128)
    chunks=(points.shape[0]+block-1)//block
    partial=np.zeros((chunks,bins,bins),np.float32)
    counts=np.zeros((chunks,bins),np.int64);sizes=np.empty((chunks,2),np.int64)
    if parallel:
        for task in prange(chunks):
            chunk=np.int64(task)
            sizes[chunk,0],sizes[chunk,1]=_chunk(data,points,fixed_bins,integer_linear,
                residual_linear,integer_offset,residual_offset,origin,bins,minimum,maximum,
                width,normalized_min,chunk*block,min((chunk+1)*block,points.shape[0]),partial[chunk],counts[chunk],points_are_integer)
    else:
        for chunk in range(chunks):
            sizes[chunk,0],sizes[chunk,1]=_chunk(data,points,fixed_bins,integer_linear,
                residual_linear,integer_offset,residual_offset,origin,bins,minimum,maximum,
                width,normalized_min,chunk*block,min((chunk+1)*block,points.shape[0]),partial[chunk],counts[chunk],points_are_integer)
    joint=np.zeros((bins,bins),np.float32);correction=np.zeros((bins,bins),np.float32)
    total_counts=np.zeros(bins,np.int64);geometric=valid=0
    for chunk in range(chunks):
        geometric+=sizes[chunk,0];valid+=sizes[chunk,1]
        for i in range(bins):
            total_counts[i]+=counts[chunk,i]
            for j in range(bins):
                joint[i,j],correction[i,j]=add32(joint[i,j],correction[i,j],partial[chunk,i,j])
    return joint,total_counts,valid,geometric


def value_histogram32(data,points,fixed_bins,mapping,bins,moving_range,width,normalized_min,*,points_are_integer=False):
    """Consume the metric's validated FP32 data and bounded mapping parameters."""
    parallel=points.shape[0]>=16384 and get_num_threads()>1
    return _histograms(data,points,fixed_bins,*mapping,bins,
        np.float32(moving_range[0]),np.float32(moving_range[1]),
        np.float32(width),np.float32(normalized_min),parallel,bool(points_are_integer))
