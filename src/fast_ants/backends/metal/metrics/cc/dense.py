"""Local cross-correlation kernels and paired score/update reductions."""
from fast_ants.backends.metal.runtime import mx
from fast_ants.backends.metal import reductions
from fast_ants.backends.metal import serial
from fast_ants.pipeline.float1_policy import FLOAT1_CC_EPS_METAL
from fast_ants.backends.metal.images import gradient


_CC_FLOAT1_PREAMBLE = f"constexpr float cc_epsilon = {FLOAT1_CC_EPS_METAL};\n"


def cc(ctx,fixed,moving,gradient,valid,radius):
    nz,ny,nx=fixed.shape
    update,score=ctx.call('cc', _CC_FLOAT1_PREAMBLE + '''uint i=thread_position_in_grid.x;
      int3 n=int3(meta[0],meta[1],meta[2]),p=xyz(i,n);int r=int(meta[3]);uint N=meta[0]*meta[1]*meta[2];
      for(int c=0;c<3;c++)out0[c*N+i]=0.0f;out1[i]=0;
      if(!valid[i])return;
      float f0=fixed[i],m0=moving[i];float2 sf=float2(0),sm=float2(0);uint count=0;
      for(int z=metal::max(0,p.z-r);z<metal::min(n.z,p.z+r+1);z++)
       for(int y=metal::max(0,p.y-r);y<metal::min(n.y,p.y+r+1);y++)
        for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
         uint j=idx(x,y,z,n);if(valid[j]){sf=compensated(sf,fixed[j]-f0);sm=compensated(sm,moving[j]-m0);count++;}}
      float af=sf.x/float(count),am=sm.x/float(count);
      float2 ff=float2(0),mm=float2(0),fm=float2(0);
      for(int z=metal::max(0,p.z-r);z<metal::min(n.z,p.z+r+1);z++)
       for(int y=metal::max(0,p.y-r);y<metal::min(n.y,p.y+r+1);y++)
        for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
         uint j=idx(x,y,z,n);if(valid[j]){float f=(fixed[j]-f0)-af,m=(moving[j]-m0)-am;
          ff=compensated(ff,f*f);mm=compensated(mm,m*m);fm=compensated(fm,f*m);}}
      float product=ff.x*mm.x,eps=cc_epsilon;
      float value=metal::abs(product)>eps?fm.x*fm.x/product:1.0f;
      if(ff.x>eps&&mm.x>eps){float weight=2.0f*fm.x/product*(-af+fm.x/mm.x*am);
        for(int c=0;c<3;c++)out0[c*N+i]=weight*grad[c*N+i];}
      out1[i]=-value;''',{'fixed':fixed,'moving':moving,'grad':gradient,'valid':valid,
             'meta':ctx.meta(nx,ny,nz,radius)},[(3,nz,ny,nx),fixed.shape],fixed.size)
    value,count=_cc_score(ctx,score,valid)
    return update,value,count


def _cc_score(ctx,score,valid):
    value,count=(reductions.cc_score(ctx,score,valid) if 'parallel_cc' in ctx.optimizations and not ctx.serial_cc_guard else
      serial.cc_score(ctx,score,valid) if score.size>serial.SLICE else
      ctx.call('cc_reduce_serial_guard' if ctx.serial_cc_guard else 'cc_reduce', '''float2 total=float2(0);uint count=0;
      for(uint i=0;i<meta[0];i++)if(valid[i]){total=compensated(total,score[i]);count++;}
      out0[0]=count?total.x/float(count):0;out1[0]=count;''',
      {'score':score,'valid':valid,'meta':ctx.meta(score.size)},[(1,),(1,)],1,[mx.float32,mx.uint32]))
    if getattr(ctx,'defer_cc_score',False):return value,count
    return float(ctx.control(value,'cc_value')[0]),int(ctx.control(count,'cc_count')[0])


def cc_pair(ctx,fixed,moving,fixed_gradient,moving_gradient,valid,radius):
    """Share centered window statistics between the two SyN directions.

    The reverse direction exchanges sf/sm and ff/mm. The covariance and
    squared correlation are symmetric, so their reduction is needed once.
    Preserve the single-direction window order, compensation, cutoffs and
    update formulas; only redundant computations are removed. This candidate
    is opt-in until device and official float-1 validation have passed.
    """
    nz,ny,nx=fixed.shape
    forward,reverse,score=ctx.call('cc_pair', _CC_FLOAT1_PREAMBLE + '''uint i=thread_position_in_grid.x;
      int3 n=int3(meta[0],meta[1],meta[2]),p=xyz(i,n);int r=int(meta[3]);uint N=meta[0]*meta[1]*meta[2];
      for(int c=0;c<3;c++){out0[c*N+i]=0.0f;out1[c*N+i]=0.0f;}out2[i]=0;
      if(!valid[i])return;
      float f0=fixed[i],m0=moving[i];float2 sf=float2(0),sm=float2(0);uint count=0;
      for(int z=metal::max(0,p.z-r);z<metal::min(n.z,p.z+r+1);z++)
       for(int y=metal::max(0,p.y-r);y<metal::min(n.y,p.y+r+1);y++)
        for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
         uint j=idx(x,y,z,n);if(valid[j]){sf=compensated(sf,fixed[j]-f0);sm=compensated(sm,moving[j]-m0);count++;}}
      float af=sf.x/float(count),am=sm.x/float(count);
      float2 ff=float2(0),mm=float2(0),fm=float2(0);
      for(int z=metal::max(0,p.z-r);z<metal::min(n.z,p.z+r+1);z++)
       for(int y=metal::max(0,p.y-r);y<metal::min(n.y,p.y+r+1);y++)
        for(int x=metal::max(0,p.x-r);x<metal::min(n.x,p.x+r+1);x++){
         uint j=idx(x,y,z,n);if(valid[j]){float f=(fixed[j]-f0)-af,m=(moving[j]-m0)-am;
          ff=compensated(ff,f*f);mm=compensated(mm,m*m);fm=compensated(fm,f*m);}}
      float product=ff.x*mm.x,eps=cc_epsilon;
      float value=metal::abs(product)>eps?fm.x*fm.x/product:1.0f;
      if(ff.x>eps&&mm.x>eps){
        float forward_weight=2.0f*fm.x/product*(-af+fm.x/mm.x*am);
        float reverse_weight=2.0f*fm.x/product*(-am+fm.x/ff.x*af);
        for(int c=0;c<3;c++){
          out0[c*N+i]=forward_weight*moving_grad[c*N+i];
          out1[c*N+i]=reverse_weight*fixed_grad[c*N+i];}}
      out2[i]=-value;''',
      {'fixed':fixed,'moving':moving,'fixed_grad':fixed_gradient,'moving_grad':moving_gradient,'valid':valid,
       'meta':ctx.meta(nx,ny,nz,radius)},[(3,nz,ny,nx),(3,nz,ny,nx),fixed.shape],fixed.size)
    value,count=_cc_score(ctx,score,valid)
    return forward,reverse,value,count
