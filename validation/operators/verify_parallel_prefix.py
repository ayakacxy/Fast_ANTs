"""Exact integer prefix/compaction tests at threadgroup boundary lengths."""

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
from fast_ants.backends.metal.runtime import MetalContext
from fast_ants.backends.metal.mattes import compact
from fast_ants.metrics import NoValidSamplesError


def main():
    ctx=MetalContext(('parallel_prefix',));rng=np.random.default_rng(92);checks=[]
    for n in (1,2,255,256,257,511,512,513,1023,1024,1025,151552):
        for pattern in ('all','alternating','random'):
            valid=np.ones(n,bool) if pattern=='all' else (np.arange(n)%2==0 if pattern=='alternating' else rng.uniform(size=n)>.4)
            valid[0]=True
            values=np.arange(n*3,dtype=np.int32).reshape(n,3)
            floats=np.linspace(-1,1,n,dtype=np.float32)
            a,b=compact(ctx,ctx.upload(valid,'test_validity',np.bool_),ctx.upload(values,'test_indices',np.int32),ctx.upload(floats,'test_values'))
            np.testing.assert_array_equal(ctx.download(a,'test_output'),values[valid]);np.testing.assert_array_equal(ctx.download(b,'test_output'),floats[valid])
            checks.append({'size':n,'pattern':pattern,'selected':int(valid.sum()),'exact':True})
    try:compact(ctx,ctx.upload(np.zeros(257,bool),'test_empty',np.bool_),ctx.upload(np.ones(257,np.float32),'test_data'))
    except NoValidSamplesError:pass
    else:raise AssertionError('Empty compaction did not signal no valid samples')
    p=PROJECT/'validation/reports/operators/parallel-prefix-contracts.json';p.write_text(json.dumps({'status':'passed','checks':checks,'empty_case':'NoValidSamplesError','runtime':ctx.report()},indent=2)+'\n')
    print('Exact parallel prefix and compaction passed',len(checks),'cases')


if __name__=='__main__':main()
