"""Bound long serial reductions without changing their floating-point order.

Each slice carries the exact FP32 Kahan/extremum state to the next dispatch.
Explicit completion bounds individual command-buffer duration. No volume is
downloaded, and no CPU arithmetic substitutes for a device reduction.
"""

from fast_ants.backends.metal.runtime import mx

SLICE=262144


def zero(ctx,n):
    return ctx.call('serial_state_zero','out0[thread_position_in_grid.x]=0.0f;',{},[(n,)],n)


def complete(ctx,*state):
    ctx.finish(*state);ctx.control_guards['serial_watchdog_slices']+=1


def range32(ctx,data):
    state=ctx.call('range_serial_init','out0[0]=data[0];out0[1]=data[0];',{'data':data},[(2,)],1)
    for start in range(1,data.size,SLICE):
        state=ctx.call('range_serial_slice','''float lo=state[0],hi=state[1];
          for(uint i=meta[0];i<meta[1];i++){lo=metal::min(lo,data[i]);hi=metal::max(hi,data[i]);}
          out0[0]=lo;out0[1]=hi;''',
          {'data':data,'state':state,'meta':ctx.meta(start,min(start+SLICE,data.size))},[(2,)],1)
        complete(ctx,state)
    return state


def maximum_vector(ctx,field,spacing):
    n=field.size//3;maximum=zero(ctx,1);selected=ctx.meta(0);gspacing=ctx.upload(spacing)
    for start in range(0,n,SLICE):
        maximum,selected=ctx.call('maximum_vector_serial_slice','''uint N=meta[0],selected=indices[0];float maximum=scores[0];
          for(uint i=meta[1];i<meta[2];i++){float squared=0;
            for(int c=0;c<3;c++){float v=field[c*N+i]/spacing[c];squared+=v*v;}
            if(squared>maximum){maximum=squared;selected=i;}}
          out0[0]=maximum;out1[0]=selected;''',
          {'field':field,'spacing':gspacing,'scores':maximum,'indices':selected,'meta':ctx.meta(n,start,min(start+SLICE,n))},
          [(1,),(1,)],1,[mx.float32,mx.uint32])
        complete(ctx,maximum,selected)
    return ctx.call('maximum_vector_serial_finish','for(int c=0;c<3;c++)out0[c]=field[c*meta[0]+selected[0]];',
                    {'field':field,'selected':selected,'meta':ctx.meta(n)},[(3,)],1)


def norms(ctx,values):
    state=zero(ctx,3)
    for start in range(0,values.size,SLICE):
        state=ctx.call('norm_reduce_serial_slice','''float2 sum=float2(state[0],state[1]);float maximum=state[2];
          for(uint i=meta[0];i<meta[1];i++){sum=compensated(sum,data[i]);maximum=metal::max(maximum,data[i]);}
          out0[0]=sum.x;out0[1]=sum.y;out0[2]=maximum;''',
          {'data':values,'state':state,'meta':ctx.meta(start,min(start+SLICE,values.size))},[(3,)],1)
        complete(ctx,state)
    return ctx.call('norm_reduce_serial_finish','out0[0]=state[0]/float(meta[0]);out0[1]=state[2];',
                    {'state':state,'meta':ctx.meta(values.size)},[(2,)],1)


def cc_score(ctx,score,valid):
    state=zero(ctx,2);count=ctx.meta(0)
    for start in range(0,score.size,SLICE):
        state,count=ctx.call('cc_reduce_serial_slice','''float2 total=float2(state[0],state[1]);uint count=counts[0];
          for(uint i=meta[0];i<meta[1];i++)if(valid[i]){total=compensated(total,score[i]);count++;}
          out0[0]=total.x;out0[1]=total.y;out1[0]=count;''',
          {'score':score,'valid':valid,'state':state,'counts':count,'meta':ctx.meta(start,min(start+SLICE,score.size))},
          [(2,),(1,)],1,[mx.float32,mx.uint32])
        complete(ctx,state,count)
    return ctx.call('cc_reduce_serial_finish','out0[0]=counts[0]?state[0]/float(counts[0]):0.0f;out1[0]=counts[0];',
                    {'state':state,'counts':count},[(1,),(1,)],1,[mx.float32,mx.uint32])


def moments(ctx,data):
    nz,ny,nx=data.shape;total=zero(ctx,4);correction=zero(ctx,4)
    for start in range(0,data.size,SLICE):
        total,correction=ctx.call('moments_serial_slice','''float2 sums[4];for(int c=0;c<4;c++)sums[c]=float2(totals[c],corrections[c]);
          int3 n=int3(meta[0],meta[1],meta[2]);
          for(uint i=meta[3];i<meta[4];i++){int3 p=xyz(i,n);float v=data[i];sums[0]=compensated(sums[0],v);
            for(int c=0;c<3;c++)sums[c+1]=compensated(sums[c+1],v*(float(p[c])-float(n[c]-1)*0.5f));}
          for(int c=0;c<4;c++){out0[c]=sums[c].x;out1[c]=sums[c].y;}''',
          {'data':data,'totals':total,'corrections':correction,'meta':ctx.meta(nx,ny,nz,start,min(start+SLICE,data.size))},[(4,),(4,)],1)
        complete(ctx,total,correction)
    return total
