"""Parallel extrema, norm accuracy and guarded inverse stopping decisions."""

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
from fast_ants import Grid,DisplacementField
from fast_ants.backends.metal.runtime import MetalContext
from fast_ants.backends.metal.ops import MetalOps
from fast_ants.backends.metal import dense


def main():
    flags=('parallel_range','parallel_max','parallel_norms')
    baseline=MetalContext(());candidate=MetalContext(flags);rng=np.random.default_rng(7803);checks=[]
    for n in (1,2,255,256,257,511,512,513,1023,1024,1025,151552):
        for mode in ('zero','random','ties'):
            values=(rng.normal(size=n)*10).astype(np.float32)
            field=rng.normal(size=(3,1,1,n)).astype(np.float32)
            if mode=='zero':values[:]=0;field[:]=0
            if mode=='ties':
                field[:]=0;field[0,0,0,:]=.3;field[1,0,0,:]=.4
                field[:2,0,0,::2]*=-1
                values[0]=-100;values[-1]=100
            v=baseline.upload(values);f=baseline.upload(field);spacing=np.array([.7,1.3,2.])
            r=dense.range32(baseline,v);c=dense.range32(candidate,v)
            if not np.array_equal(r,c):raise AssertionError(('range',n,mode,r,c))
            rm=dense.maximum_control(baseline,f,spacing);cm=dense.maximum_control(candidate,f,spacing)
            if rm!=cm:raise AssertionError(('maximum',n,mode,rm,cm))
            rv,rmean,rmax=dense.norms(baseline,f,spacing);cv,cmean,cmax=dense.norms(candidate,f,spacing)
            if rmax!=cmax:raise AssertionError(('norm maximum',n,mode,rmax,cmax))
            err=abs(rmean-cmean);limit=4*np.finfo(np.float32).eps*max(rmean,np.finfo(np.float32).tiny)
            if err>limit:raise AssertionError(('norm mean',n,mode,err,limit))
            if not np.array_equal(baseline.download(rv,'verification'),candidate.download(cv,'verification')):raise AssertionError('Pointwise norms changed')
            for target in (rmean,float(np.nextafter(np.float32(rmean),np.float32(np.inf)))):
                _,guarded,_=dense.norms(candidate,f,spacing,target)
                if guarded!=rmean:raise AssertionError(('guard',n,mode,rmean,guarded))
            checks.append({'n':n,'mode':mode,'range_exact':True,'argmax_control_exact':True,
                           'norm_max_exact':True,'norm_mean_absolute_error':err,'mean_limit_guard_exact':True})
    # Test true stop decisions around mean thresholds while max cannot stop.
    grid=Grid((8,10,13),np.eye(4));raw=np.zeros((3,*grid.shape_zyx),np.float32)
    raw[0]=.001;raw[0,3,4,5]=.8;raw[1]=.0001
    bops,cops=MetalOps(baseline),MetalOps(candidate)
    bfield=bops.field(DisplacementField(raw,grid));cfield=cops.field(DisplacementField(raw,grid))
    _,mean,_=dense.norms(baseline,bfield.data,grid.spacing_xyz)
    inversions=[]
    for target in (mean,float(np.nextafter(np.float32(mean),np.float32(0))),float(np.nextafter(np.float32(mean),np.float32(np.inf))),.0003):
        b=bops.invert(bfield,mean_tolerance=target,max_tolerance=0,enforce_boundary=False)
        c=cops.invert(cfield,mean_tolerance=target,max_tolerance=0,enforce_boundary=False)
        if len(b.history)!=len(c.history) or b.stop_reason!=c.stop_reason:raise AssertionError('Inverse stop decision changed')
        delta=float(np.abs(baseline.download(b.field.data,'verification')-candidate.download(c.field.data,'verification')).max())
        if delta!=0:raise AssertionError(('Inverse field changed',delta))
        inversions.append({'mean_limit':target,'iterations':len(b.history),'stop_reason':b.stop_reason,'maximum_field_difference_mm':delta})
    report={'status':'passed','checks':checks,'inversions':inversions,'candidate_runtime':candidate.report(),
            'numerical_policy':'ordinary FP32; exact extrema; fixed positive sum tree; GPU serial compensated guard near inverse mean thresholds'}
    out=PROJECT/'validation/reports/operators/parallel-reduction-contracts.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print('Parallel reduction contracts passed:',len(checks),'cases and',len(inversions),'inverse decisions',flush=True)


if __name__=='__main__':main()
