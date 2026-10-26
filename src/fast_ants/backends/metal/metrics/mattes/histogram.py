"""Stable fixed-bin plans, raw PDF merging, and packed MI controls.

Membership planning is reused by value and derivative stages. Packed uint32
transport preserves FP32 bits and integer counts; it does not emulate precision.
"""
from dataclasses import dataclass
import math
import numpy as np
from fast_ants.backends.metal.runtime import mx

@dataclass(frozen=True)
class FixedBinPlan:
    """Device CSR indices and row offsets for one fixed sample vector."""

    indices: object
    offsets: object
    size: int
    bins: int
    tile: int


def fixed_bin_plan(ctx, bin_indices, bins, tile=2048):
    """Group valid fixed-bin codes stably without downloading sample data.

    The codes must come from the metric's clamped fixed_bins operator and
    therefore lie in [0,bins). A (tile,row) worker scans one original tile;
    per-row prefixes and a second scan preserve original index order inside
    every bin. Integer states and output indices use uint32.
    """
    if (not isinstance(bin_indices, mx.array) or bin_indices.ndim != 1
            or bin_indices.dtype not in (mx.int32, mx.uint32)):
        raise ValueError('Fixed-bin plan requires a device integer vector')
    if not isinstance(bins, (int, np.integer)) or bins < 5:
        raise ValueError('Fixed-bin plan requires at least five bins')
    if not isinstance(tile, (int, np.integer)) or not 0 < tile < 2**32:
        raise ValueError('Fixed-bin tile size must be positive')
    size = bin_indices.size
    if not 0 < size < 2**32:
        raise ValueError('Fixed-bin sample count must fit positive uint32')
    bins, tile = int(bins), int(tile)
    tiles = math.ceil(size / tile)
    counts = ctx.call('grouped_fixed_bin_counts', '''
      uint task=thread_position_in_grid.x,B=meta[1],row=task%B,tile=task/B;
      uint count=0,begin=tile*meta[2],end=begin+metal::min(meta[2],meta[0]-begin);
      for(uint n=begin;n<end;n++)if(uint(bin_indices[n])==row)count++;
      out0[task]=count;''',
      {'bin_indices': bin_indices, 'meta': ctx.meta(size, bins, tile)},
      [(tiles, bins)], tiles*bins, [mx.uint32])
    local_offsets, row_counts = ctx.call('grouped_fixed_bin_row_prefix', '''
      uint row=thread_position_in_grid.x,B=meta[0],count=0;
      for(uint tile=0;tile<meta[1];tile++){
        uint i=tile*B+row;out0[i]=count;count+=counts[i];
      }
      out1[row]=count;''',
      {'counts': counts, 'meta': ctx.meta(bins, tiles)},
      [(tiles, bins), (bins,)], bins, [mx.uint32, mx.uint32])
    offsets = ctx.call('grouped_fixed_bin_row_offsets', '''
      uint count=0;
      for(uint row=0;row<meta[0];row++){out0[row]=count;count+=counts[row];}
      out0[meta[0]]=count;''',
      {'counts': row_counts, 'meta': ctx.meta(bins)}, [(bins+1,)], 1, [mx.uint32])
    indices = ctx.call('grouped_fixed_bin_scatter', '''
      uint task=thread_position_in_grid.x,B=meta[1],row=task%B,tile=task/B;
      uint target=offsets[row]+local_offsets[task];
      uint begin=tile*meta[2],end=begin+metal::min(meta[2],meta[0]-begin);
      for(uint n=begin;n<end;n++)if(uint(bin_indices[n])==row)out0[target++]=n;''',
      {'bin_indices': bin_indices, 'offsets': offsets, 'local_offsets': local_offsets,
       'meta': ctx.meta(size, bins, tile)}, [(size,)], tiles*bins, [mx.uint32])
    return FixedBinPlan(indices, offsets, size, bins, tile)


def histograms(ctx, plan, moving, valid, intensity, bin_indices, bin_fractions,
               moving_range, *, weights=None, moving_width):
    """Return the same bounded joint/count/derivative controls as Mattes.

    CSR rows are split into at most 128 ordered parts, independently for each
    row. Every part/channel uses compensated accumulation, followed by an
    ordered compensated merge. No floating atomic updates are used.
    """
    if not isinstance(plan, FixedBinPlan):
        raise ValueError('Grouped histogram requires a fixed-bin plan')
    size, bins = plan.size, plan.bins
    for data, dtype in ((moving, mx.float32), (valid, mx.bool_),
                        (bin_indices, mx.int32), (bin_fractions, mx.float32)):
        if not isinstance(data, mx.array) or data.dtype != dtype or data.shape != (size,):
            raise ValueError('Grouped histogram sample shape/type mismatch')
    if (not isinstance(intensity, mx.array) or intensity.dtype != mx.float32
            or intensity.ndim != 2 or intensity.shape[0] != size
            or not 0 <= intensity.shape[1] <= 12):
        raise ValueError('Grouped histogram needs FP32 derivative channels (0..12)')
    parameters = intensity.shape[1]
    channels = parameters + 1
    if weights is not None:
        width = 8 if parameters else 4
        if (not isinstance(weights, mx.array) or weights.dtype != mx.float32
                or weights.shape != (size, width)):
            raise ValueError('Grouped histogram cubic table shape/type mismatch')
    bounds = np.asarray(moving_range, np.float32)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not bounds[1] > bounds[0]
            or not np.isfinite(moving_width) or moving_width <= 0):
        raise ValueError('Grouped histogram needs finite nonconstant ranges and positive bin width')
    parts = min(128, max(1, math.ceil(size/2048)))
    shape = (parts, bins, bins, channels)
    inputs = {'indices': plan.indices, 'offsets': plan.offsets,
              'moving': moving, 'valid': valid, 'intensity': intensity,
              'bin_indices': bin_indices, 'bin_fractions': bin_fractions,
              'ranges': ctx.upload(bounds, 'mattes_grouped_range'),
              'meta': ctx.meta(bins, parameters, parts)}
    # Keep adjacent SIMD lanes in the same CSR row. Row lengths may differ
    # greatly (especially the background bin); grouping row/part/channel
    # prevents one long row from leaving unrelated short-row lanes idle.
    # Output addresses and every cell's sample/merge order stay unchanged.
    source = '''
      uint task=thread_position_in_grid.x,B=meta[0],P=meta[1],K=P+1;
      uint channel=task%K,part=(task/K)%meta[2],row=task/(meta[2]*K);
      uint base=part*B*B*K;
      for(uint j=0;j<B;j++){
        uint q=base+(row*B+j)*K+channel;out0[q]=0;out1[q]=0;
      }
      uint row_begin=offsets[row],length=offsets[row+1]-row_begin;
      uint block=length/meta[2]+uint(length%meta[2]!=0);
      uint begin=row_begin+metal::min(length,part*block);
      uint end=begin+metal::min(block,row_begin+length-begin),count=0;
      for(uint logical=begin;logical<end;logical++){
        uint n=indices[logical];
        if(!valid[n]||moving[n]<ranges[0]||moving[n]>ranges[1])continue;
        if(channel==0)count++;
        int mi=bin_indices[n];
        for(int j=mi-1;j<mi+3;j++){
          float argument=float(j-mi)-bin_fractions[n];
          float value;
          if(channel==0)value=cubic(argument);
          else value=intensity[n*P+channel-1]*cubic_derivative(argument);
          uint q=base+(row*B+uint(j))*K+channel;
          float2 total=compensated(float2(out0[q],out1[q]),value);
          out0[q]=total.x;out1[q]=total.y;
        }
      }
      if(channel==0)out2[part*B+row]=count;'''
    name = 'grouped_histogram_channels'
    if weights is not None:
        inputs['weights'] = weights
        source = source.replace('cubic(argument)', 'weights[n*'+str(weights.shape[1])+'+j-mi+1]')
        source = source.replace('cubic_derivative(argument)', 'weights[n*8+j-mi+5]' if parameters else '0.0f')
        name += '_cached_cubic'
    sums, corrections, counts = ctx.call(name, source, inputs,
        [shape, shape, (parts, bins)], parts*bins*channels,
        [mx.float32, mx.float32, mx.uint32])
    merged = ctx.call('grouped_histogram_merge', '''
      uint i=thread_position_in_grid.x,S=meta[0];float2 total=float2(0);
      for(uint part=0;part<meta[1];part++)total=compensated(total,data[part*S+i]);
      out0[i]=total.x;''',
      {'data': sums, 'meta': ctx.meta(bins*bins*channels, parts)},
      [(bins, bins, channels)], bins*bins*channels)
    merged_counts = ctx.call('grouped_histogram_counts', '''
      uint row=thread_position_in_grid.x,B=meta[0],count=0;
      if(row<B){for(uint part=0;part<meta[1];part++)count+=counts[part*B+row];}
      else{for(uint part=0;part<meta[1];part++)for(uint r=0;r<B;r++)count+=counts[part*B+r];}
      out0[row]=count;''',
      {'counts': counts, 'meta': ctx.meta(bins, parts)}, [(bins+1,)], bins+1, [mx.uint32])
    data = ctx.control(merged, 'mattes_histogram', bins*bins*13)
    count = ctx.control(merged_counts, 'mattes_counts', bins+1)
    return (data[:, :, 0].copy(), count[:bins].astype(np.int64),
            data[:, :, 1:].copy(), int(count[-1]), float(moving_width))


@dataclass(frozen=True)
class DeviceHistogram32:
    packed: object
    bins: int
    parameters: int
    bad_blocks: int


def merge_histogram_controls(ctx, sums, counts, bins, parameters, chunks, *,
                             sample_count=None, badflags=None,
                             derive_total_from_rows=False, return_device=False):
    """Return raw joint, integer counts, derivative PDF and valid sample count.

    Independently partitioned CSR rows do not have one common sample chunk.
    Their total must be derived from all row counts rather than the separate
    per-global-chunk total column, which is ignored in this mode.
    """
    if not 5 <= bins <= 32 or not 0 <= parameters <= 12 or chunks < 0:
        raise ValueError('Packed MI controls support 5..32 bins and 0..12 parameters')
    channels=parameters+1
    size=bins*bins*channels
    if (sums.dtype!=mx.float32 or sums.shape!=(chunks,bins,bins,channels)
            or counts.dtype!=mx.uint32 or counts.shape!=(chunks,bins+1)):
        raise ValueError('Invalid private MI histogram/control layout')
    if derive_total_from_rows and sample_count is not None:
        raise ValueError('Independent MI rows cannot use global sample-count partitioning')
    inputs={'data':sums,'counts':counts}
    prefix='uint active=meta[1];'
    name='histogram_packed_controls32'
    if sample_count is not None:
        inputs['sample_count']=sample_count
        prefix=('uint N=sample_count[0],block=metal::max(2048u,(N+127u)/128u);'
                'uint active=(N+block-1u)/block;')
        name+='_indexed'
    bad_size=0 if badflags is None else badflags.size
    bad_blocks=(bad_size+255)//256
    if badflags is not None:
        if badflags.dtype!=mx.uint32:
            raise ValueError('MI diagnostic flags must be uint32')
        inputs['badflags']=badflags
        name+='_checked'
    inputs['meta']=ctx.meta(size,chunks,bins,bad_size)
    source=prefix+'''
      uint i=thread_position_in_grid.x,S=meta[0],B=meta[2];
      if(i<S){
        float2 total=float2(0);
        for(uint c=0;c<active;c++)total=compensated(total,data[c*S+i]);
        out0[i]=as_type<uint>(total.x);
      }else if(i<S+B+1u){
        uint j=i-S,total=0;
        for(uint c=0;c<active;c++)total+=counts[c*(B+1u)+j];
        out0[i]=total;
      }'''
    if derive_total_from_rows:
        source=source.replace(
            'for(uint c=0;c<active;c++)total+=counts[c*(B+1u)+j];',
            '''if(j<B){
          for(uint c=0;c<active;c++)total+=counts[c*(B+1u)+j];
        }else{
          for(uint c=0;c<active;c++)
            for(uint row=0;row<B;row++)total+=counts[c*(B+1u)+row];
        }''')
        name+='_fixed_rows'
    if badflags is not None:
        source+='''else{
          uint begin=(i-S-B-1u)*256u,flag=0;
          for(uint j=begin;j<metal::min(meta[3],begin+256u);j++)flag|=badflags[j];
          out0[i]=flag;
        }'''
    packed_size=size+bins+1+bad_blocks
    packed=ctx.call(name,source,inputs,[(packed_size,)],packed_size,[mx.uint32])
    if return_device:
        return DeviceHistogram32(packed,int(bins),int(parameters),int(bad_blocks))
    values=ctx.control(packed,'mattes_packed',packed_size)
    if bad_blocks and np.any(values[size+bins+1:]):
        raise ValueError('Nonfinite Metal MI samples or intensity derivatives')
    data=values[:size].view(np.float32).reshape(bins,bins,channels)
    integer_counts=values[size:size+bins+1]
    return (data[:,:,0].copy(),integer_counts[:bins].astype(np.int64),
            data[:,:,1:].copy(),int(integer_counts[-1]))

