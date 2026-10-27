"""Deterministic parallel reductions for ordinary FP32 device arrays.

Extrema preserve the earliest squared-norm argmax. Positive norm sums use a
fixed tree and a compensated block merge. Near an inverse stopping threshold,
the caller repeats the reference serial compensated reduction on the GPU.
"""

from fast_ants.backends.cuda.runtime import cp

TILE = 256


def range32(ctx, data):
    blocks = (data.size + TILE - 1) // TILE
    partial = ctx.call('range_tiles', '''
      unsigned int t=threadIdx.x;
      __shared__ float lo[256],hi[256];
      lo[t]=i<meta[0]?data[i]:INFINITY;hi[t]=i<meta[0]?data[i]:-INFINITY;
      __syncthreads();
      for(unsigned int stride=128;stride>0;stride/=2){
        if(t<stride){lo[t]=min(lo[t],lo[t+stride]);hi[t]=max(hi[t],hi[t+stride]);}
        __syncthreads();
      }
      if(t==0){out0[2*(i/256)]=lo[0];out0[2*(i/256)+1]=hi[0];}''',
      {'data': data, 'meta': ctx.meta(data.size)}, [(blocks, 2)], blocks*TILE)
    return ctx.call('range_merge', '''float lo=data[0],hi=data[1];
      for(unsigned int b=1;b<meta[0];b++){lo=min(lo,data[2*b]);hi=max(hi,data[2*b+1]);}
      out0[0]=lo;out0[1]=hi;''',
      {'data': partial, 'meta': ctx.meta(blocks)}, [(2,)], 1)


def maximum_vector(ctx, field, spacing):
    n = field.size // 3
    blocks = (n + TILE - 1) // TILE
    scores, indices = ctx.call('maximum_vector_tiles', '''
      unsigned int t=threadIdx.x,N=meta[0];
      __shared__ float scores[256];__shared__ unsigned int indices[256];
      float squared=0;
      if(i<N)for(int c=0;c<3;c++){float v=field[c*N+i]/spacing[c];squared+=v*v;}
      scores[t]=squared>0?squared:0.0f;indices[t]=i<N?i:0xffffffffu;
      __syncthreads();
      for(unsigned int stride=128;stride>0;stride/=2){
        if(t<stride){float other=scores[t+stride];unsigned int index=indices[t+stride];
          if(other>scores[t]||(other==scores[t]&&index<indices[t])){scores[t]=other;indices[t]=index;}}
        __syncthreads();
      }
      if(t==0){out0[i/256]=scores[0];out1[i/256]=indices[0];}''',
      {'field': field, 'spacing': ctx.geometry(spacing), 'meta': ctx.meta(n)},
      [(blocks,), (blocks,)], blocks*TILE, [cp.float32, cp.uint32])
    return ctx.call('maximum_vector_merge', '''unsigned int t=threadIdx.x;
      float maximum=0;unsigned int selected=0;
      __shared__ float maxima[256];__shared__ unsigned int selected_indices[256];
      for(unsigned int b=t;b<meta[1];b+=256){
        if(scores[b]>maximum||(scores[b]==maximum&&indices[b]<selected)){
          maximum=scores[b];selected=indices[b];}}
      maxima[t]=maximum;selected_indices[t]=selected;__syncthreads();
      for(unsigned int stride=128;stride>0;stride/=2){
        if(t<stride){float other=maxima[t+stride];unsigned int index=selected_indices[t+stride];
          if(other>maxima[t]||(other==maxima[t]&&index<selected_indices[t])){
            maxima[t]=other;selected_indices[t]=index;}}
        __syncthreads();
      }
      if(t<3)out0[t]=field[t*meta[0]+selected_indices[0]];''',
      {'field': field, 'scores': scores, 'indices': indices, 'meta': ctx.meta(n, blocks)}, [(3,)], TILE)


def norms(ctx, values):
    blocks = (values.size + TILE - 1) // TILE
    partial = ctx.call('norm_reduce_tiles', '''
      unsigned int t=threadIdx.x;
      __shared__ float sums[256],maxima[256];float v=i<meta[0]?data[i]:0.0f;
      sums[t]=v;maxima[t]=v;__syncthreads();
      for(unsigned int stride=128;stride>0;stride/=2){
        if(t<stride){sums[t]+=sums[t+stride];maxima[t]=max(maxima[t],maxima[t+stride]);}
        __syncthreads();
      }
      if(t==0){out0[2*(i/256)]=sums[0];out0[2*(i/256)+1]=maxima[0];}''',
      {'data': values, 'meta': ctx.meta(values.size)}, [(blocks, 2)], blocks*TILE)
    return ctx.call('norm_reduce_merge', '''float2 total=make_float2(0.0f,0.0f);float maximum=0;
      for(unsigned int b=0;b<meta[1];b++){total=compensated(total,data[2*b]);maximum=max(maximum,data[2*b+1]);}
      out0[0]=total.x/float(meta[0]);out0[1]=maximum;''',
      {'data': partial, 'meta': ctx.meta(values.size, blocks)}, [(2,)], 1)


def cc_score(ctx,score,valid):
    blocks=(score.size+TILE-1)//TILE
    sums,counts=ctx.call('cc_reduce_tiles','''
      unsigned int t=threadIdx.x;
      __shared__ float sums[256];__shared__ unsigned int counts[256];
      bool take=i<meta[0]&&valid[i];sums[t]=take?score[i]:0.0f;counts[t]=static_cast<unsigned int>(take);
      __syncthreads();
      for(unsigned int stride=128;stride>0;stride/=2){
        if(t<stride){sums[t]+=sums[t+stride];counts[t]+=counts[t+stride];}
        __syncthreads();
      }
      if(t==0){out0[i/256]=sums[0];out1[i/256]=counts[0];}''',
      {'score':score,'valid':valid,'meta':ctx.meta(score.size)},[(blocks,),(blocks,)],blocks*TILE,[cp.float32,cp.uint32])
    return ctx.call('cc_reduce_merge','''float2 total=make_float2(0.0f,0.0f);unsigned int count=0;
      for(unsigned int b=0;b<meta[0];b++){total=compensated(total,sums[b]);count+=counts[b];}
      out0[0]=count?total.x/float(count):0.0f;out1[0]=count;''',
      {'sums':sums,'counts':counts,'meta':ctx.meta(blocks)},[(1,),(1,)],1,[cp.float32,cp.uint32])


def _row_norm_partials(ctx,values):
    nz,ny,nx=values.shape;rows=nz*ny;blocks=(rows+TILE-1)//TILE
    partial=ctx.call('inverse_norm_rows','''float2 total=make_float2(0.0f,0.0f);float maximum=0.0f;
      if(i<meta[1])for(unsigned int x=0;x<meta[0];x++){
        float v=data[i*meta[0]+x];maximum=max(maximum,v);
        if(isfinite(v)&&isfinite(total.x))total=compensated(total,v);
        else total=make_float2(total.x+v,0.0f);
      }
      __shared__ float sums[256],maxima[256];unsigned int t=threadIdx.x;
      sums[t]=total.x;maxima[t]=maximum;__syncthreads();
      for(unsigned int stride=1;stride<256;stride*=2){
        if(t%(2*stride)==0&&i+stride<meta[1]){
          sums[t]+=sums[t+stride];maxima[t]=max(maxima[t],maxima[t+stride]);
        }
        __syncthreads();
      }
      if(t==0){out0[2*blockIdx.x]=sums[0];out0[2*blockIdx.x+1]=maxima[0];}''',
      {'data':values,'meta':ctx.meta(nx,rows)},[(blocks,2)],blocks*TILE)
    return partial


def row_norms(ctx,values):
    """Match CPU inverse row-wise Kahan sums and the fixed pairwise merge."""
    partial=_row_norm_partials(ctx,values);blocks=partial.shape[0]
    # A 256-row subtree in each block retains the CPU's first eight levels.
    # One block owns the scratch so every tree level can share one barrier.
    # Retaining the left row of each pair preserves the CPU addition order
    # without overwriting a row another thread still needs in this level.
    merged=ctx.call('inverse_norm_row_merge','''
      for(unsigned int row=i;row<meta[0];row+=256){out0[2*row]=data[2*row];out0[2*row+1]=data[2*row+1];}
      __syncthreads();
      for(unsigned int stride=1;stride<meta[0];stride*=2){
        for(unsigned int row=2*stride*i;row<meta[0];row+=2*stride*256){
          unsigned int other=row+stride;
          if(other<meta[0]){out0[2*row]+=out0[2*other];out0[2*row+1]=max(out0[2*row+1],out0[2*other+1]);}
        }
        __syncthreads();
      }
      if(i==0)out0[0]/=float(meta[1]);''',
      {'data':partial,'meta':ctx.meta(blocks,values.size)},[(blocks,2)],TILE)
    return merged.reshape(-1)[:2]


def row_norms_pair(ctx,first,second):
    """Merge two unchanged row trees in independent blocks and one packet."""
    a,b=_row_norm_partials(ctx,first),_row_norm_partials(ctx,second)
    blocks=max(a.shape[0],b.shape[0])
    _,controls=ctx.call('inverse_norm_pair_merge','''
      unsigned int side=blockIdx.x,t=threadIdx.x,count=meta[2*side];
      const float* data=side==0?first:second;float* scratch=out0+2*side*meta[4];
      for(unsigned int row=t;row<count;row+=256){scratch[2*row]=data[2*row];scratch[2*row+1]=data[2*row+1];}
      __syncthreads();
      for(unsigned int stride=1;stride<count;stride*=2){
        for(unsigned int row=2*stride*t;row<count;row+=2*stride*256){
          unsigned int other=row+stride;
          if(other<count){scratch[2*row]+=scratch[2*other];scratch[2*row+1]=max(scratch[2*row+1],scratch[2*other+1]);}
        }
        __syncthreads();
      }
      if(t==0){out1[2*side]=scratch[0]/float(meta[2*side+1]);out1[2*side+1]=scratch[1];}''',
      {'first':a,'second':b,'meta':ctx.meta(a.shape[0],first.size,b.shape[0],second.size,blocks)},
      [(2,blocks,2),(2,2)],2*TILE)
    return controls


def serial_norms(ctx,values):
    return ctx.call('norm_reduce_serial_guard','''float2 total=make_float2(0.0f,0.0f);float maximum=0.0f;
      for(unsigned int j=0;j<meta[0];j++){total=compensated(total,data[j]);maximum=max(maximum,data[j]);}
      out0[0]=total.x/float(meta[0]);out0[1]=maximum;''',
      {'data':values,'meta':ctx.meta(values.size)},[(2,)],1)


def serial_cc_score(ctx,score,valid):
    return ctx.call('cc_reduce_serial_guard','''float2 total=make_float2(0.0f,0.0f);unsigned int count=0;
      for(unsigned int j=0;j<meta[0];j++)if(valid[j]){total=compensated(total,score[j]);count++;}
      out0[0]=count?total.x/float(count):0.0f;out1[0]=count;''',
      {'score':score,'valid':valid,'meta':ctx.meta(score.size)},[(1,),(1,)],1,[cp.float32,cp.uint32])
