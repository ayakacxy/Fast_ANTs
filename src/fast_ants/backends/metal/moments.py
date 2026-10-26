"""Stable parallel centered image moments, without serial watchdog slices.

Each lane uses ordinary FP32 compensated sums; a fixed compensated tree
combines those sums, followed by a bounded block merge. Only the returned
four scalars need a host read. No image-sized scratch or precision emulation
is used, and reduction grouping need not reproduce the serial traversal.
"""

from fast_ants.backends.metal.runtime import mx


TILE = 256
MAX_BLOCKS = 4096


def moments(ctx, data):
    """Return device FP32 ``[mass, centered_x, centered_y, centered_z]``.

    The centered components are ``sum(value * (index - (size-1)/2))``.
    Division by mass and conversion to physical coordinates remain the
    caller's bounded host geometry work. Zero/signed masses are not clamped.
    """
    if not isinstance(data, mx.array) or data.dtype != mx.float32:
        raise ValueError('Parallel moments require an FP32 device image')
    if data.ndim != 3 or min(data.shape) < 1 or data.size >= 2**32:
        raise ValueError('Parallel moments require a nonempty 3D uint32-sized image')
    nz, ny, nx = data.shape
    per_lane = max(16, (data.size + TILE*MAX_BLOCKS-1)//(TILE*MAX_BLOCKS))
    block = TILE*per_lane
    blocks = (data.size+block-1)//block
    partial, correction = ctx.call('moments_parallel_tiles32', '''
      uint gid=thread_position_in_grid.x,t=thread_index_in_threadgroup;
      uint b=gid/256u,begin=b*meta[4];
      uint end=begin+metal::min(meta[4],meta[3]-begin);
      int3 n=int3(meta[0],meta[1],meta[2]);
      float2 sums[4];for(uint c=0;c<4;c++)sums[c]=float2(0);
      for(uint offset=t;offset<end-begin;offset+=256u){
        uint i=begin+offset;
        float value=data[i];int3 p=xyz(i,n);
        sums[0]=compensated(sums[0],value);
        for(uint c=0;c<3;c++){
          float coordinate=float(p[c])-float(n[c]-1)*0.5f;
          sums[c+1]=compensated(sums[c+1],value*coordinate);
        }
      }
      threadgroup float2 states[1024];
      for(uint c=0;c<4;c++)states[c*256u+t]=sums[c];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128u;stride>0;stride/=2u){
        if(t<stride){
          for(uint c=0;c<4;c++){
            uint at=c*256u+t;float2 other=states[at+stride];
            float2 total=compensated(states[at],other.x);
            states[at]=compensated(total,-other.y);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0){
        for(uint c=0;c<4;c++){
          out0[b*4u+c]=states[c*256u].x;
          out1[b*4u+c]=states[c*256u].y;
        }
      }''',
      {'data': data, 'meta': ctx.meta(nx, ny, nz, data.size, block)},
      [(blocks, 4), (blocks, 4)], blocks*TILE)
    return ctx.call('moments_parallel_merge32', '''
      uint c=thread_position_in_grid.x;float2 total=float2(0);
      for(uint b=0;b<meta[0];b++){
        total=compensated(total,sums[b*4u+c]);
        total=compensated(total,-correction[b*4u+c]);
      }
      out0[c]=total.x;''',
      {'sums': partial, 'correction': correction, 'meta': ctx.meta(blocks)},
      [(4,)], 4)
