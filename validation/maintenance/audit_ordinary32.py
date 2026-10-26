"""Audit the current ordinary FP32 path; do not reuse the old expanded audit."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import importlib.abc
import json
from pathlib import Path
import sys
import hashlib

PROJECT=_TOOL_PROJECT
sys.path.insert(0,str(PROJECT/'src'))
FORBIDDEN=('scipy','itk','ants','fast_ants.backends.cpu.ds32','fast_ants.mattes_pair32')


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in FORBIDDEN):
            raise ImportError('Blocked formal runtime dependency: '+fullname)
        return None


def main():
    sys.meta_path.insert(0,Guard())
    import numpy as np
    from numba.core import types
    from numba.core.registry import CPUDispatcher
    from fast_ants import Grid,Image,LinearStage,SyNStage,register32
    from fast_ants.backends.cpu.fp32 import kernels as fp32
    from fast_ants.backends.cpu.fp32.metrics import mattes_kernels as mattes32
    grid=Grid((8,10,12),np.diag([.8,1.3,2.,1.]))
    z,y,x=np.indices(grid.shape_zyx)
    f=Image(np.exp(-((x-5.5)**2+(y-4.5)**2+(z-3.5)**2)/12).astype(np.float32),grid)
    m=Image(np.exp(-((x-5.7)**2+(y-4.4)**2+(z-3.5)**2)/12).astype(np.float32),grid)
    result=register32(f,m,linear_stages=[LinearStage('Rigid',iterations=(2,),shrink_factors=(1,),smoothing_sigmas=(1,)),
                    LinearStage('Affine',iterations=(2,),shrink_factors=(1,),smoothing_sigmas=(0,))],
                    syn_stage=SyNStage(iterations=(2,),shrink_factors=(1,),smoothing_sigmas=(0,)))
    # Exercise the large-sample parallel histogram branch as well as the small
    # serial branch used by the public smoke fixture.
    n=20001;sample=np.linspace(0,1,n,dtype=np.float32)
    mattes32.histograms32(sample,sample,np.zeros((n,12),np.float32),5,
        np.float32(0),np.float32(1),np.float32(0),np.float32(1),np.full(n,2,np.int32),np.zeros(n,np.float32))
    def fp64(t):
        if t==types.float64:return True
        if isinstance(t,types.Array):return fp64(t.dtype)
        if isinstance(t,types.BaseTuple):return any(fp64(v) for v in t.types)
        return False
    kernels={}
    for module in (fp32,mattes32):
        for name,fn in vars(module).items():
            if not isinstance(fn,CPUDispatcher) or fn.__module__!=module.__name__ or not fn.overloads:continue
            fn.recompile()  # Disk-cache hits may omit the actual inferred typemap.
            variants=[]
            for signature,overload in fn.overloads.items():
                bad=[str(k) for k,v in overload.type_annotation.typemap.items() if fp64(v)]
                variants.append({'signature':str(signature),'fp64_variables':bad})
                if bad:raise AssertionError('Hidden FP64 volume arithmetic: '+name+' '+str(bad))
            kernels[module.__name__+'.'+name]=variants
    loaded=[m for m in sys.modules if any(m==p or m.startswith(p+'.') for p in FORBIDDEN)]
    if loaded:raise AssertionError(loaded)
    source={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT/'src/fast_ants').rglob('*')) if p.suffix in ('.py','.metal')}
    report={'status':'passed','scope':'runtime dependency and actual inferred FP32 kernel types; not full-map numerical acceptance',
            'software_expanded_precision':False,'blocked_dependencies':list(FORBIDDEN),'forbidden_modules_loaded':loaded,
            'executed_kernel_count':len(kernels),'kernels':kernels,
            'public_registration_finite':bool(np.isfinite(result.warped.data).all()),
            'native_fp64_control':'bounded histograms, geometry/filter/bin setup, optimizer, scalar norm control, separate QA',
            'source_sha256':source}
    out=PROJECT/'validation/reports/ordinary-fp32-audit';out.mkdir(parents=True,exist_ok=True)
    (out/'ordinary-fp32-runtime-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','kernels')},indent=2))


if __name__=='__main__':main()
