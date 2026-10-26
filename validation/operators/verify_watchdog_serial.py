"""Chunked device serial state must exactly match the original reduction."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import json
from pathlib import Path
import sys
import numpy as np

PROJECT=_TOOL_PROJECT;sys.path.insert(0,str(PROJECT/'src'))
from fast_ants.backends.metal.runtime import MetalContext,mx
from fast_ants.backends.metal import serial,dense,mattes


def main():
    ctx=MetalContext(());rng=np.random.default_rng(9801);checks=[];default=serial.SLICE
    for shape,cap in [((3,5,7),17),((37,64,64),10007),((75,128,128),default)]:
        data=rng.random(shape,dtype=np.float32);field=rng.normal(size=(3,*shape)).astype(np.float32);spacing=[.7,1.3,2.]
        gd=ctx.upload(data);gf=ctx.upload(field);mask=ctx.upload(rng.random(shape)>.2,dtype=np.bool_)
        serial.SLICE=data.size+1
        rr=dense.range32(ctx,gd);rm=mattes.moments(ctx,gd);rmax=dense.maximum_control(ctx,gf,spacing)
        values,rmean,rnormmax=dense.norms(ctx,gf,spacing)
        rv,rc=ctx.call('cc_serial_verification','''float2 total=float2(0);uint count=0;
          for(uint i=0;i<meta[0];i++)if(valid[i]){total=compensated(total,score[i]);count++;}
          out0[0]=count?total.x/float(count):0;out1[0]=count;''',
          {'score':gd,'valid':mask,'meta':ctx.meta(data.size)},[(1,),(1,)],1,[mx.float32,mx.uint32])
        rcc=ctx.control(rv,'verification');rcount=ctx.control(rc,'verification')
        serial.SLICE=cap
        cr=ctx.control(serial.range32(ctx,gd),'verification');cm=ctx.control(serial.moments(ctx,gd),'verification')
        candidate=ctx.control(serial.maximum_vector(ctx,gf,spacing),'verification').astype(np.float64)/np.asarray(spacing)
        cmax=float(np.sqrt(candidate@candidate));cn=ctx.control(serial.norms(ctx,values),'verification')
        cv,cc=serial.cc_score(ctx,gd,mask);ccc=ctx.control(cv,'verification');ccount=ctx.control(cc,'verification')
        exact=(np.array_equal(rr,cr) and np.array_equal(rm,cm) and rmax==cmax and
               np.array_equal(cn,np.array([rmean,rnormmax],np.float32)) and np.array_equal(rcc,ccc) and np.array_equal(rcount,ccount))
        if not exact:raise AssertionError((shape,rr,cr,rm,cm,rmax,cmax,cn,rmean,rnormmax,rcc,ccc))
        checks.append({'shape_zyx':shape,'slice_elements':cap,'range_moments_argmax_norm_cc_count_exact':True})
    serial.SLICE=default
    report={'status':'passed','checks':checks,'runtime':ctx.report(),'policy':'same serial FP32 operation order and Kahan state; explicit GPU completion between bounded slices; no CPU volume fallback'}
    (PROJECT/'validation/reports/operators/watchdog-serial-contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Watchdog-safe serial reductions passed:',len(checks),'exact shapes',flush=True)


if __name__=='__main__':main()
