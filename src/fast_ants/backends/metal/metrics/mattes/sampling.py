"""Mattes moving-image sampling and value-only PDF construction.

Both the fused sampling fallback and the streamed CSR value path retain their
original kernels and precision rules. Dispatch selection belongs to the metric.
"""
import math
import numpy as np
from fast_ants.backends.metal.runtime import mx

def sample_value_bins(ctx, data, points, linear, offset, center, bins, parameters):
    """Return moving values, spatial validity, bin indices and fractions.

    ``linear/offset/center`` use the same bounded host geometry setup as
    mattes.local_mapping. ``parameters`` is the device float1
    [BinSize, NormalizedMin] pair. Binning uses the FP32 interpolated pixel,
    never a recomputed residual or a reciprocal-width rearrangement.
    """
    if not isinstance(data, mx.array) or data.dtype != mx.float32 or data.ndim != 3:
        raise ValueError('Fused value sampling requires a 3D FP32 device image')
    if (not isinstance(points, mx.array) or points.dtype != mx.float32
            or points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1):
        raise ValueError('Fused value sampling requires nonempty FP32 device points')
    if (not isinstance(parameters, mx.array) or parameters.dtype != mx.float32
            or parameters.shape != (2,)):
        raise ValueError('Fused value sampling requires a device float1 bin pair')
    if not isinstance(bins, (int, np.integer)) or bins < 5:
        raise ValueError('Fused value sampling requires at least five bins')
    linear, offset, center = (np.asarray(value) for value in (linear, offset, center))
    if (linear.shape != (3, 3) or offset.shape != (3,) or center.shape != (3,)
            or not all(np.isfinite(value).all() for value in (linear, offset, center))):
        raise ValueError('Invalid bounded affine geometry for fused sampling')
    # Keep exactly the same host decomposition and FP32 upload boundaries as
    # local_mapping, rather than constructing global FP32 world positions.
    integer = np.rint(linear).astype(np.int32)
    integer_offset = np.rint(offset).astype(np.int32)
    nz, ny, nx = data.shape
    count = points.shape[0]
    source = '''
      uint i=thread_position_in_grid.x;
      int3 b=int3(0);float3 p=float3(0);
      for(int c=0;c<3;c++){
        int base=bo[c];float v=offset[c];
        for(int j=0;j<3;j++){
          float q=points[3*i+j]-center[j];
          int whole=int(metal::floor(q));float fraction=q-float(whole);
          base+=integer[3*c+j]*whole;
          v=metal::fma(residual[3*c+j],float(whole),v);
          v=metal::fma(float(integer[3*c+j])+residual[3*c+j],fraction,v);
        }
        b[c]=base;p[c]=v;
      }
      int3 n=int3(meta[0],meta[1],meta[2]);
      bool valid=local_inside(b,p,n);float value=0;
      if(valid)value=local_at(data,b,p,n);
      // value is an ordinary FP32 pixel, as in the separate sample output.
      float term=value/params[0]-params[1];
      int index=metal::clamp(int(term),2,int(meta[3])-3);
      out0[i]=value;out1[i]=valid;out2[i]=index;out3[i]=term-float(index);'''
    return ctx.call('fused_value_sampling32', source,
        {'data': data, 'points': points,
         'integer': ctx.upload(integer, dtype=np.int32),
         'bo': ctx.upload(integer_offset, dtype=np.int32),
         'residual': ctx.upload(linear-integer),
         'offset': ctx.upload(offset-integer_offset), 'center': ctx.upload(center),
         'params': parameters, 'meta': ctx.meta(nx, ny, nz, int(bins))},
        [(count,)]*4, count, [mx.float32, mx.bool_, mx.int32, mx.float32])


def stream_value_histograms(ctx, data, points, bins, moving_range, mapping,
                            fixed_plan, *, moving_parameters=None, return_device=False):
    """Return bounded host (joint, counts, empty derivative, count, width).

    Only fixed membership is cached. Moving support, rounded-pixel range and
    float1 Parzen bins are reevaluated for every transform. Per-lane Kahan
    accumulation and SIMD sums replace the old global-sample partitioning;
    no floating atomics or expanded-precision arithmetic are used.
    """
    from fast_ants.backends.metal.metrics.mattes.histogram import FixedBinPlan
    from fast_ants.backends.metal.metrics.mattes.histogram import merge_histogram_controls

    if (not isinstance(data, mx.array) or data.dtype != mx.float32
            or data.ndim != 3 or min(data.shape) < 1):
        raise ValueError('Streaming MI value needs a nonempty FP32 device image')
    if (not isinstance(points, mx.array) or points.dtype != mx.float32
            or points.ndim != 2 or points.shape[1] != 3
            or not 0 < points.shape[0] < 2**32):
        raise ValueError('Streaming MI value needs nonempty FP32 device indices')
    if not isinstance(bins, (int, np.integer)) or not 5 <= bins <= 32:
        raise ValueError('Streaming MI value supports 5..32 bins')
    size, bins = points.shape[0], int(bins)
    if (not isinstance(fixed_plan, FixedBinPlan)
            or fixed_plan.size != size or fixed_plan.bins != bins
            or not isinstance(fixed_plan.indices, mx.array)
            or fixed_plan.indices.dtype != mx.uint32
            or fixed_plan.indices.shape != (size,)
            or not isinstance(fixed_plan.offsets, mx.array)
            or fixed_plan.offsets.dtype != mx.uint32
            or fixed_plan.offsets.shape != (bins+1,)):
        raise ValueError('Streaming MI value fixed-bin plan must match the sample vector')
    bounds = np.asarray(moving_range, np.float32)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[1] <= bounds[0]):
        raise ValueError('Streaming MI value needs a finite nonconstant moving range')
    width = np.float32(np.float32(bounds[1]-bounds[0])/np.float32(bins-4))
    normalized = np.float32(np.float32(bounds[0]/width)-np.float32(2))
    if not np.isfinite(width) or width <= 0 or not np.isfinite(normalized):
        raise ValueError('Streaming MI value needs finite float1 bin parameters')
    if moving_parameters is None:
        moving_parameters = ctx.upload([width, normalized])
    if (not isinstance(moving_parameters, mx.array)
            or moving_parameters.dtype != mx.float32
            or moving_parameters.shape != (2,)):
        raise ValueError('Streaming MI value needs a device float1 bin pair')
    linear, offset, center = (np.asarray(value) for value in mapping)
    if (linear.shape != (3, 3) or offset.shape != (3,) or center.shape != (3,)
            or not all(np.isfinite(a).all() for a in (linear, offset, center))):
        raise ValueError('Invalid bounded affine geometry for streaming MI value')
    rounded_linear, rounded_offset = np.rint(linear), np.rint(offset)
    limits = np.iinfo(np.int32)
    if any(np.any((a < limits.min) | (a > limits.max))
           for a in (rounded_linear, rounded_offset)):
        raise ValueError('Streaming MI value affine decomposition exceeds int32')
    integer = rounded_linear.astype(np.int32)
    integer_offset = rounded_offset.astype(np.int32)
    # More independent parts keep the large background CSR row occupied even
    # though value-only PDFs have just one channel. Residency stays bounded.
    parts = min(512, max(1, math.ceil(size/2048)))
    nz, ny, nx = data.shape
    source = 'constexpr uint NB='+str(bins)+''';
      uint gid=thread_position_in_grid.x,lane=gid%32u,task=gid/32u;
      uint B=meta[3],parts=meta[4],row=task/parts,part=task%parts;
      uint row_begin=offsets[row],length=offsets[row+1]-row_begin;
      uint block=length/parts+uint(length%parts!=0);
      uint begin=row_begin+metal::min(length,part*block);
      uint end=begin+metal::min(block,row_begin+length-begin);
      int3 dimensions=int3(meta[0],meta[1],meta[2]);
      float2 accum[NB];for(uint j=0;j<NB;j++)accum[j]=float2(0);
      uint count=0,bad=0;
      for(uint logical=begin+lane;logical<end;logical+=32u){
        uint n=indices[logical];bool finite_point=true;
        for(int j=0;j<3;j++)finite_point=finite_point&&metal::isfinite(points[3*n+j]);
        if(!finite_point){bad++;continue;}
        int3 base=int3(0);float3 position=float3(0);
        for(int c=0;c<3;c++){
          int b=bo[c];float value=offset[c];
          for(int j=0;j<3;j++){
            float q=points[3*n+j]-center[j];
            int whole=int(metal::floor(q));float fraction=q-float(whole);
            b+=integer[3*c+j]*whole;
            value=metal::fma(residual[3*c+j],float(whole),value);
            value=metal::fma(float(integer[3*c+j])+residual[3*c+j],fraction,value);
          }
          base[c]=b;position[c]=value;
        }
        if(!local_inside(base,position,dimensions))continue;
        float pixel=local_at(data,base,position,dimensions);
        if(!metal::isfinite(pixel)){bad++;continue;}
        if(pixel<ranges[0]||pixel>ranges[1])continue;
        // Preserve the rounded FP32 interpolated pixel and float1 division.
        float term=pixel/params[0]-params[1];
        if(!metal::isfinite(term)){bad++;continue;}
        int moving_bin=metal::clamp(int(term),2,int(B)-3);
        float fraction=term-float(moving_bin);count++;
        for(int j=moving_bin-1;j<moving_bin+3;j++)
          accum[j]=compensated(accum[j],cubic(float(j-moving_bin)-fraction));
      }
      for(uint j=0;j<NB;j++){
        float value=metal::simd_sum(accum[j].x);
        if(lane==0)out0[part*B*B+row*B+j]=value;
      }
      uint selected=metal::simd_sum(count),invalid=metal::simd_sum(bad);
      if(lane==0){
        out1[part*(B+1u)+row]=selected;
        if(row==0)out1[part*(B+1u)+B]=0;
        out2[task]=invalid;
      }'''
    sums, counts, bad = ctx.call('stream_value_fixed_simd32_b'+str(bins), source,
        {'data': data, 'points': points, 'indices': fixed_plan.indices,
         'offsets': fixed_plan.offsets,
         'integer': ctx.upload(integer, dtype=np.int32),
         'bo': ctx.upload(integer_offset, dtype=np.int32),
         'residual': ctx.upload(linear-integer),
         'offset': ctx.upload(offset-integer_offset), 'center': ctx.upload(center),
         'params': moving_parameters, 'ranges': ctx.upload(bounds),
         'meta': ctx.meta(nx, ny, nz, bins, parts)},
        [(parts, bins, bins, 1), (parts, bins+1), (bins, parts)],
        bins*parts*32, [mx.float32, mx.uint32, mx.uint32])
    merged = merge_histogram_controls(
        ctx, sums, counts, bins, 0, parts, badflags=bad,
        derive_total_from_rows=True, return_device=return_device)
    if return_device:
        return merged, float(width)
    joint, fixed_counts, derivative, valid_count = merged
    return joint, fixed_counts, derivative, valid_count, float(width)

