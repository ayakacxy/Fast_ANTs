"""Two-stage FP32 Metal Mattes derivatives without voxel-by-parameter arrays.

Local mapping, rounded-pixel sampling, centered gradients and float1 bins share
one sample kernel. SIMD histogram channels evaluate their small Jacobians from
the shared gradient vector and original virtual index. No mapped-coordinate,
local-coordinate, cubic-weight or N-by-P intensity-derivative volume is stored.
"""

import math
import numpy as np

from fast_ants.backends.metal.runtime import mx


def _bin_parameters(value_range, bins):
    low, high = np.asarray(value_range, np.float32)
    width = np.float32(np.float32(high-low)/np.float32(bins-4))
    normalized = np.float32(np.float32(low/width)-np.float32(2))
    return np.asarray([width, normalized], np.float32)


def _jacobian_plan(linear, translation):
    """Recognize sparse affine columns and translations, in any ordering.

    A frozen preceding affine turns a unit local Jacobian into one active
    coordinate column. Keep that column's three products/FMA order rather than
    assuming it is still a unit basis after transform composition.
    """
    plan = np.zeros((linear.shape[0], 3), np.int32)
    for p in range(len(plan)):
        positions = np.argwhere(linear[p] != 0)
        active_translation = np.flatnonzero(translation[p] != 0)
        if (len(positions) == 1 and len(active_translation) == 0
                and linear[p, positions[0, 0], positions[0, 1]] == 1):
            plan[p] = (1, positions[0, 0], positions[0, 1])
        elif (len(positions) == 0 and len(active_translation) == 1
              and translation[p, active_translation[0]] == 1):
            plan[p] = (2, active_translation[0], 0)
        elif len(positions) == 0 and np.isfinite(translation[p]).all():
            plan[p] = (4, 0, 0)
        elif len(active_translation) == 0 and np.isfinite(linear[p]).all():
            columns = np.flatnonzero(np.any(linear[p] != 0, axis=0))
            if len(columns) == 1:
                plan[p] = (3, columns[0], 0)
    return plan


def sample_gradient_bins(ctx, data, points, mapping, bins, moving_range,
                         gradient_geometry, *, moving_parameters=None):
    """Return values/state/gradients/bin indices/fractions on the GPU.

    ``mapping`` is the existing (linear, offset, integer-floor centre) affine
    decomposition. Gradient geometry is (LPS half-spacing shifts, spacing).
    Geometry and gradient parameters are bounded host tables, never images.
    State is uint32: bit 0 is spatial validity, bit 1 denotes finite virtual
    indices bounded by 1e8, and bit 2 marks three exact-zero finite gradients.
    Nonfinite in-range gradients become a nonfinite moving-value sentinel so
    the histogram's existing bounded diagnostics reject them once.
    """
    if (not isinstance(data, mx.array) or data.dtype != mx.float32
            or data.ndim != 3 or min(data.shape) < 1):
        raise ValueError('Streaming MI needs a nonempty 3D FP32 device image')
    if (not isinstance(points, mx.array) or points.dtype != mx.float32
            or points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1):
        raise ValueError('Streaming MI needs nonempty FP32 device points')
    if not isinstance(bins, (int, np.integer)) or not 5 <= bins <= 32:
        raise ValueError('Streaming MI supports 5..32 bins')
    bounds = np.asarray(moving_range, np.float32)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[1] <= bounds[0]):
        raise ValueError('Streaming MI needs a finite nonconstant moving range')
    parameters = _bin_parameters(bounds, bins)
    if not np.isfinite(parameters).all() or parameters[0] <= 0:
        raise ValueError('Streaming MI float1 bin parameters must be finite')
    if moving_parameters is None:
        moving_parameters = ctx.upload(parameters)
    if (not isinstance(moving_parameters, mx.array)
            or moving_parameters.dtype != mx.float32
            or moving_parameters.shape != (2,)):
        raise ValueError('Streaming MI needs a device float1 bin pair')
    linear, offset, center = (np.asarray(value) for value in mapping)
    shift, spacing = (np.asarray(value, np.float32) for value in gradient_geometry)
    if (linear.shape != (3, 3) or offset.shape != (3,) or center.shape != (3,)
            or not all(np.isfinite(a).all() for a in (linear, offset, center))):
        raise ValueError('Invalid bounded affine geometry for streaming MI')
    if (shift.shape != (3, 3) or spacing.shape != (3,)
            or not np.isfinite(shift).all() or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise ValueError('Invalid physical gradient geometry for streaming MI')
    rounded_linear, rounded_offset = np.rint(linear), np.rint(offset)
    limit = np.iinfo(np.int32)
    if any(np.any((a < limit.min) | (a > limit.max))
           for a in (rounded_linear, rounded_offset)):
        raise ValueError('Streaming MI affine decomposition exceeds int32')
    integer, integer_offset = rounded_linear.astype(np.int32), rounded_offset.astype(np.int32)
    count = points.shape[0]
    nz, ny, nx = data.shape
    source = '''
      uint i=thread_position_in_grid.x;
      int3 b=int3(0);float3 p=float3(0);
      bool finite_point=true,bounded_point=true;
      for(int j=0;j<3;j++){
        finite_point=finite_point&&metal::isfinite(points[3*i+j]);
        bounded_point=bounded_point&&metal::abs(points[3*i+j])<=1e8f;
      }
      if(!finite_point){
        out0[i]=as_type<float>(0x7fc00000u);out1[i]=1u;
        for(int c=0;c<3;c++)out2[3*i+c]=0;
        out3[i]=2;out4[i]=0;return;
      }
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
      for(int c=0;c<3;c++)out2[3*i+c]=0;
      out3[i]=2;out4[i]=0;
      if(valid)value=local_at(data,b,p,n);
      uint state=(valid?1u:0u)|(bounded_point?2u:0u);
      out0[i]=value;out1[i]=state;
      // Out-of-range samples do not contribute to either PDF. Their image
      // gradients need not be evaluated, just as in the streaming CPU path.
      if(!valid||!metal::isfinite(value)||value<ranges[0]||value>ranges[1])return;
      bool finite_gradient=true,zero_gradient=true;
      for(int c=0;c<3;c++){
        float3 d=float3(shift[c],shift[3+c],shift[6+c]);
        float3 positive=p+d,negative=p-d;
        float component=0;
        if(local_inside(b,positive,n)&&local_inside(b,negative,n))
          component=(local_at(data,b,positive,n,value)-
                       local_at(data,b,negative,n,value))/spacing[c];
        out2[3*i+c]=component;
        finite_gradient=finite_gradient&&metal::isfinite(component);
        zero_gradient=zero_gradient&&component==0;
      }
      if(!finite_gradient){out0[i]=as_type<float>(0x7fc00000u);return;}
      if(zero_gradient)state|=4u;
      out1[i]=state;
      // Preserve the already-rounded pixel and explicit float1 division.
      float term=value/params[0]-params[1];
      int index=metal::clamp(int(term),2,int(meta[3])-3);
      out3[i]=index;out4[i]=term-float(index);'''
    return ctx.call('stream_derivative_sample32', source,
        {'data': data, 'points': points,
         'integer': ctx.upload(integer, dtype=np.int32),
         'bo': ctx.upload(integer_offset, dtype=np.int32),
         'residual': ctx.upload(linear-integer),
         'offset': ctx.upload(offset-integer_offset), 'center': ctx.upload(center),
         'shift': ctx.upload(shift), 'spacing': ctx.upload(spacing),
         'params': moving_parameters, 'ranges': ctx.upload(bounds),
         'meta': ctx.meta(nx, ny, nz, int(bins))},
        [(count,), (count,), (count, 3), (count,), (count,)], count,
        [mx.float32, mx.uint32, mx.float32, mx.int32, mx.float32])


def stream_derivative_histograms(ctx, data, points, fixed_bin_indices, bins,
                                 moving_range, mapping, gradient_geometry,
                                 local_geometry, linear_derivatives,
                                 translation_derivatives, *, moving_parameters=None,
                                 fixed_plan=None, return_device=False):
    """Return bounded host (joint, counts, derivative PDF, count, bin width).

    Moving masks and larger/custom parameter counts stay in the caller's
    existing general path. All volume work here remains on the GPU; only the
    final bounded PDFs, counts and nonfinite flags cross to the host. An
    optional reusable fixed-bin CSR plan visits only each row's membership;
    moving support and intensity validity are still checked on every call.
    """
    from fast_ants.backends.metal.metrics.mattes.histogram import merge_histogram_controls

    linear = np.ascontiguousarray(linear_derivatives, np.float32)
    translation = np.ascontiguousarray(translation_derivatives, np.float32)
    if (linear.ndim != 3 or linear.shape[1:] != (3, 3)
            or not 1 <= linear.shape[0] <= 12
            or translation.shape != (linear.shape[0], 3)):
        raise ValueError('Streaming MI supports 1..12 three-dimensional Jacobian parameters')
    if (not isinstance(fixed_bin_indices, mx.array)
            or fixed_bin_indices.dtype != mx.int32
            or fixed_bin_indices.shape != (points.shape[0],)):
        raise ValueError('Streaming MI needs cached fixed-bin indices')
    if fixed_plan is not None:
        from fast_ants.backends.metal.metrics.mattes.histogram import FixedBinPlan
        if (not isinstance(fixed_plan, FixedBinPlan)
                or fixed_plan.size != points.shape[0] or fixed_plan.bins != bins
                or not isinstance(fixed_plan.indices, mx.array)
                or fixed_plan.indices.dtype != mx.uint32
                or fixed_plan.indices.shape != (points.shape[0],)
                or not isinstance(fixed_plan.offsets, mx.array)
                or fixed_plan.offsets.dtype != mx.uint32
                or fixed_plan.offsets.shape != (bins+1,)):
            raise ValueError('Streaming MI fixed-bin plan must match this sample vector and bin count')
    local_linear, local_offset, local_center = (
        np.asarray(value, np.float32) for value in local_geometry)
    if (local_linear.shape != (3, 3) or local_offset.shape != (3,)
            or local_center.shape != (3,)):
        raise ValueError('Invalid local Jacobian geometry')
    # These conservative bounds guarantee finite local coordinates for the
    # bounded virtual-index points tested in the shader, even in unused axes.
    # Unsafe inputs retain every multiplication in the generic expression.
    safe_geometry = all(np.isfinite(a).all() and np.max(np.abs(a)) <= 1e8
                        for a in (local_linear, local_offset, local_center))
    safe_coefficients = all(np.isfinite(a).all() and np.max(np.abs(a)) <= 1e8
                           for a in (linear, translation))
    plan = _jacobian_plan(linear, translation)
    moving, states, gradients, bin_indices, fractions = sample_gradient_bins(
        ctx, data, points, mapping, bins, moving_range, gradient_geometry,
        moving_parameters=moving_parameters)
    count, parameters = points.shape[0], linear.shape[0]
    channels = parameters+1
    block = max(2048, math.ceil(count/128))
    chunks = math.ceil(count/block) if fixed_plan is None else 128
    source = 'constexpr uint NB='+str(int(bins))+''';
      uint gid=thread_position_in_grid.x,lane=gid%32u,task=gid/32u;
      uint B=meta[1],P=meta[2],K=P+1,channel=task%K;
      uint row=(task/K)%B,chunk=task/(B*K),N=meta[0],block=meta[3];
      uint S=B*B*K,base=chunk*S,cb=chunk*(B+1);
      float2 accum[NB];for(uint j=0;j<NB;j++)accum[j]=float2(0);
      uint row_count=0,total_count=0,bad=0;
      for(uint n=chunk*block+lane;n<metal::min(N,(chunk+1)*block);n+=32u){
        uint state=states[n];if(!(state&1u))continue;
        if(!metal::isfinite(moving[n])){if(row==0&&channel==0)bad++;continue;}
        if(moving[n]<ranges[0]||moving[n]>ranges[1])continue;
        if(channel==0)total_count++;
        uint fi=uint(fixed_bins[n]);if(fi!=row)continue;
        if(channel==0)row_count++;
        float intensity=0;
        if(channel>0){
          uint parameter=channel-1;
          bool planned=meta[4]&&meta[5]&&(state&2u);
          if(planned&&(state&4u))continue;
          float3 grad=float3(gradients[3*n],gradients[3*n+1],gradients[3*n+2]);
          int kind=plan[parameter*3];
          if(planned&&kind==2){
            intensity=grad[plan[parameter*3+1]];
          }else if(planned&&kind==1){
            int c=plan[parameter*3+1],j=plan[parameter*3+2];
            float local=local_offset[j];
            for(int k=0;k<3;k++)local+=local_linear[j*3+k]*(points[3*n+k]-local_center[k]);
            intensity=grad[c]*local;
          }else if(planned&&kind==3){
            int j=plan[parameter*3+1];
            float local=local_offset[j];
            for(int k=0;k<3;k++)local+=local_linear[j*3+k]*(points[3*n+k]-local_center[k]);
            for(int c=0;c<3;c++){
              float spatial=linear[parameter*9+c*3+j]*local;
              intensity=metal::fma(grad[c],spatial,intensity);
            }
          }else if(planned&&kind==4){
            for(int c=0;c<3;c++)
              intensity=metal::fma(grad[c],translation[parameter*3+c],intensity);
          }else{
            float local[3];
            for(int c=0;c<3;c++){
              float v=local_offset[c];
              for(int j=0;j<3;j++)v+=local_linear[c*3+j]*(points[3*n+j]-local_center[j]);
              local[c]=v;
            }
            for(int c=0;c<3;c++){
              float spatial=translation[parameter*3+c];
              for(int j=0;j<3;j++)spatial=metal::fma(linear[parameter*9+c*3+j],local[j],spatial);
              intensity=metal::fma(grad[c],spatial,intensity);
            }
          }
          if(!metal::isfinite(intensity)){bad++;continue;}
        }
        int mi=bin_indices[n];
        for(int j=mi-1;j<mi+3;j++){
          float argument=float(j-mi)-fractions[n];
          float value=channel==0?cubic(argument):intensity*cubic_derivative(argument);
          accum[j]=compensated(accum[j],value);
        }
      }
      for(uint j=0;j<NB;j++){
        float value=metal::simd_sum(accum[j].x);
        if(lane==0)out0[base+(row*B+j)*K+channel]=value;
      }
      uint selected=metal::simd_sum(row_count),total=metal::simd_sum(total_count);
      uint invalid=metal::simd_sum(bad);
      if(lane==0){
        out2[task]=invalid;
        if(channel==0){out1[cb+row]=selected;if(row==0)out1[cb+B]=total;}
      }'''
    name = 'stream_derivative_histogram_simd32_b'+str(int(bins))
    inputs = {'moving': moving, 'states': states, 'gradients': gradients,
              'points': points, 'fixed_bins': fixed_bin_indices,
              'bin_indices': bin_indices, 'fractions': fractions,
              'linear': ctx.upload(linear), 'translation': ctx.upload(translation),
              'plan': ctx.upload(plan, dtype=np.int32),
              'local_linear': ctx.upload(local_linear), 'local_offset': ctx.upload(local_offset),
              'local_center': ctx.upload(local_center), 'ranges': ctx.upload(moving_range),
              'meta': ctx.meta(count, bins, parameters, block,
                               int(safe_geometry), int(safe_coefficients))}
    if fixed_plan is not None:
        # Fixed membership is stable for a metric, unlike moving support and
        # intensity validity. Each row has its own 128 ordered parts; SIMD
        # lanes now visit only members rather than filtering the whole vector.
        # Lane grouping changes, while each sample's float1 expressions stay.
        # Keep the parts and channels of a long row adjacent, so a heavy
        # background bin fills nearby SIMD groups instead of being interleaved
        # with empty rows. Partial output addresses remain chunk/row/channel.
        source = source.replace(
            'uint row=(task/K)%B,chunk=task/(B*K),N=meta[0],block=meta[3];',
            'uint row=task/(128u*K),chunk=(task/K)%128u,N=meta[0],block=meta[3];')
        source = source.replace(
            'for(uint n=chunk*block+lane;n<metal::min(N,(chunk+1)*block);n+=32u){',
            '''uint row_begin=fixed_offsets[row];
      uint length=fixed_offsets[row+1]-row_begin;
      uint row_block=length/128u+uint(length%128u!=0);
      uint begin=metal::min(length,chunk*row_block);
      uint part_size=metal::min(row_block,length-begin);
      for(uint logical=lane;logical<part_size;logical+=32u){
        uint n=fixed_indices[row_begin+begin+logical];''')
        source = source.replace(
            'if(!metal::isfinite(moving[n])){if(row==0&&channel==0)bad++;continue;}',
            'if(!metal::isfinite(moving[n])){if(channel==0)bad++;continue;}')
        source = source.replace(
            '''if(channel==0)total_count++;
        uint fi=uint(fixed_bins[n]);if(fi!=row)continue;
        if(channel==0)row_count++;''',
            'if(channel==0)row_count++;')
        source = source.replace(
            'uint selected=metal::simd_sum(row_count),total=metal::simd_sum(total_count);',
            'uint selected=metal::simd_sum(row_count);')
        source = source.replace(
            'if(channel==0){out1[cb+row]=selected;if(row==0)out1[cb+B]=total;}',
            'if(channel==0){out1[cb+row]=selected;if(row==0)out1[cb+B]=0u;}')
        inputs.pop('fixed_bins')
        inputs.update(fixed_indices=fixed_plan.indices, fixed_offsets=fixed_plan.offsets)
        name += '_fixed_rows'
    sums, counts, bad = ctx.call(name, source, inputs,
        [(chunks, bins, bins, channels), (chunks, bins+1), (chunks, bins, channels)],
        chunks*bins*channels*32, [mx.float32, mx.uint32, mx.uint32])
    merged = merge_histogram_controls(
        ctx, sums, counts, bins, parameters, chunks, badflags=bad,
        derive_total_from_rows=fixed_plan is not None, return_device=return_device)
    width = float(_bin_parameters(moving_range, bins)[0])
    if return_device:
        return merged, width
    hist, fixed_counts, derivative, valid_count = merged
    return hist, fixed_counts, derivative, valid_count, width
