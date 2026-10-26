"""Deterministic parallel reductions for ordinary FP32 device arrays.

Extrema preserve the earliest squared-norm argmax. Positive norm sums use a
fixed tree and a compensated block merge. Near an inverse stopping threshold,
the caller repeats the reference serial compensated reduction on the GPU.
"""

from fast_ants.backends.metal.runtime import mx

TILE = 256


def range32(ctx, data):
    blocks = (data.size + TILE - 1) // TILE
    partial = ctx.call('range_tiles', '''
      uint i=thread_position_in_grid.x,t=thread_index_in_threadgroup;
      threadgroup float lo[256],hi[256];
      lo[t]=i<meta[0]?data[i]:INFINITY;hi[t]=i<meta[0]?data[i]:-INFINITY;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128;stride>0;stride/=2){
        if(t<stride){lo[t]=metal::min(lo[t],lo[t+stride]);hi[t]=metal::max(hi[t],hi[t+stride]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0){out0[2*(i/256)]=lo[0];out0[2*(i/256)+1]=hi[0];}''',
      {'data': data, 'meta': ctx.meta(data.size)}, [(blocks, 2)], blocks*TILE)
    return ctx.call('range_merge', '''float lo=data[0],hi=data[1];
      for(uint b=1;b<meta[0];b++){lo=metal::min(lo,data[2*b]);hi=metal::max(hi,data[2*b+1]);}
      out0[0]=lo;out0[1]=hi;''',
      {'data': partial, 'meta': ctx.meta(blocks)}, [(2,)], 1)


def maximum_vector(ctx, field, spacing):
    n = field.size // 3
    blocks = (n + TILE - 1) // TILE
    scores, indices = ctx.call('maximum_vector_tiles', '''
      uint i=thread_position_in_grid.x,t=thread_index_in_threadgroup,N=meta[0];
      threadgroup float scores[256];threadgroup uint indices[256];
      float squared=0;
      if(i<N)for(int c=0;c<3;c++){float v=field[c*N+i]/spacing[c];squared+=v*v;}
      scores[t]=squared>0?squared:0.0f;indices[t]=i<N?i:0xffffffffu;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128;stride>0;stride/=2){
        if(t<stride){float other=scores[t+stride];uint index=indices[t+stride];
          if(other>scores[t]||(other==scores[t]&&index<indices[t])){scores[t]=other;indices[t]=index;}}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0){out0[i/256]=scores[0];out1[i/256]=indices[0];}''',
      {'field': field, 'spacing': ctx.upload(spacing), 'meta': ctx.meta(n)},
      [(blocks,), (blocks,)], blocks*TILE, [mx.float32, mx.uint32])
    return ctx.call('maximum_vector_merge', '''float maximum=0;uint selected=0;
      for(uint b=0;b<meta[1];b++){
        if(scores[b]>maximum||(scores[b]==maximum&&indices[b]<selected)){
          maximum=scores[b];selected=indices[b];}}
      for(int c=0;c<3;c++)out0[c]=field[c*meta[0]+selected];''',
      {'field': field, 'scores': scores, 'indices': indices, 'meta': ctx.meta(n, blocks)}, [(3,)], 1)


def norms(ctx, values):
    blocks = (values.size + TILE - 1) // TILE
    partial = ctx.call('norm_reduce_tiles', '''
      uint i=thread_position_in_grid.x,t=thread_index_in_threadgroup;
      threadgroup float sums[256],maxima[256];float v=i<meta[0]?data[i]:0.0f;
      sums[t]=v;maxima[t]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128;stride>0;stride/=2){
        if(t<stride){sums[t]+=sums[t+stride];maxima[t]=metal::max(maxima[t],maxima[t+stride]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0){out0[2*(i/256)]=sums[0];out0[2*(i/256)+1]=maxima[0];}''',
      {'data': values, 'meta': ctx.meta(values.size)}, [(blocks, 2)], blocks*TILE)
    return ctx.call('norm_reduce_merge', '''float2 total=float2(0);float maximum=0;
      for(uint b=0;b<meta[1];b++){total=compensated(total,data[2*b]);maximum=metal::max(maximum,data[2*b+1]);}
      out0[0]=total.x/float(meta[0]);out0[1]=maximum;''',
      {'data': partial, 'meta': ctx.meta(values.size, blocks)}, [(2,)], 1)


def cc_score(ctx,score,valid):
    blocks=(score.size+TILE-1)//TILE
    sums,counts=ctx.call('cc_reduce_tiles','''
      uint i=thread_position_in_grid.x,t=thread_index_in_threadgroup;
      threadgroup float sums[256];threadgroup uint counts[256];
      bool take=i<meta[0]&&valid[i];sums[t]=take?score[i]:0.0f;counts[t]=uint(take);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint stride=128;stride>0;stride/=2){
        if(t<stride){sums[t]+=sums[t+stride];counts[t]+=counts[t+stride];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(t==0){out0[i/256]=sums[0];out1[i/256]=counts[0];}''',
      {'score':score,'valid':valid,'meta':ctx.meta(score.size)},[(blocks,),(blocks,)],blocks*TILE,[mx.float32,mx.uint32])
    return ctx.call('cc_reduce_merge','''float2 total=float2(0);uint count=0;
      for(uint b=0;b<meta[0];b++){total=compensated(total,sums[b]);count+=counts[b];}
      out0[0]=count?total.x/float(count):0.0f;out1[0]=count;''',
      {'sums':sums,'counts':counts,'meta':ctx.meta(blocks)},[(1,),(1,)],1,[mx.float32,mx.uint32])
