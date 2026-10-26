"""Cold-cache production tests with SciPy/ITK/ANTs imports and CLI calls blocked."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import importlib.abc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from scripts.maintenance.run_tests import selected_suite

PROJECT=_TOOL_PROJECT
FORBIDDEN={'scipy','itk','SimpleITK','ants','antspyx'}
attempts=[]


class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            attempts.append(fullname)
            raise ModuleNotFoundError('Independent core dependency guard: '+fullname)


def blocked_process(*args,**kwargs):
    raise RuntimeError('Production checks must not launch a reference executable')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=PROJECT/'src',
                        help='Source root containing fast_ants; enables isolated candidate checks')
    parser.add_argument('--output',type=Path,default=PROJECT/'validation/reports/core-dependencies',
                        help='Use a new directory to retain historical evidence')
    args=parser.parse_args()
    args.source=args.source.resolve()
    if not (args.source/'fast_ants/__init__.py').is_file():
        parser.error('--source must contain fast_ants')
    args.output=args.output.resolve()
    if args.output.exists():
        parser.error('Use a new output directory to retain historical evidence')
    (PROJECT/'validation/.numba-cache').mkdir(parents=True,exist_ok=True)
    cache_context=tempfile.TemporaryDirectory(prefix='cold-dependency-',dir=PROJECT/'validation/.numba-cache')
    os.environ['NUMBA_CACHE_DIR']=cache_context.name
    sys.path.insert(0,str(args.source))
    preloaded=FORBIDDEN.intersection(sys.modules)
    if preloaded:raise RuntimeError('Dependency test was not started in a fresh interpreter')
    sys.meta_path.insert(0,BlockImports())
    subprocess.run=blocked_process;subprocess.Popen=blocked_process
    os.system=blocked_process
    # Numerical core groups only: orchestration contracts intentionally launch
    # short mock Python children and do not belong under this process blocker.
    suite=unittest.TestSuite(selected_suite(group) for group in ('core','cpu-fp64','cpu-fp32'))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    # Exercise public FP32 registration, not just isolated functions.
    import numpy as np
    from fast_ants import Grid,Image,LinearStage,SyNStage,register32
    grid=Grid((8,10,12),np.diag([.8,1.3,2.,1.]))
    z,y,x=np.indices(grid.shape_zyx)
    fixed=Image(np.exp(-((x-5.5)**2+(y-4.5)**2+(z-3.5)**2)/10).astype(np.float32),grid)
    moving=Image(np.exp(-((x-5.7)**2+(y-4.4)**2+(z-3.5)**2)/10).astype(np.float32),grid)
    linear=[LinearStage('Rigid',iterations=(1,),shrink_factors=(1,),smoothing_sigmas=(0,)),
            LinearStage('Affine',iterations=(1,),shrink_factors=(1,),smoothing_sigmas=(0,))]
    stage=SyNStage(iterations=(1,),shrink_factors=(1,),smoothing_sigmas=(0,))
    registered=register32(fixed,moving,linear_stages=linear,syn_stage=stage)
    report={'status':'passed' if result.wasSuccessful() and not result.skipped else 'failed',
            'source':str(args.source),
            'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'test_groups':['core','cpu_fp64','cpu_fp32'],
            'forbidden_import_roots':sorted(FORBIDDEN),'forbidden_modules_loaded':[m for m in sys.modules if m.split('.')[0] in FORBIDDEN],
            'optional_import_attempts_blocked':sorted(set(attempts)),
            'reference_process_launch':'blocked','numba_cache':'new temporary empty cache directory per invocation',
            'public_fp32_registration':{'finite':bool(np.isfinite(registered.warped.data).all()),
                                        'output_dtype':str(registered.warped.data.dtype)}}
    if report['forbidden_modules_loaded']:report['status']='failed'
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'dependency-guard.log').write_text(stream.getvalue())
    (out/'dependency-guard.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise AssertionError('Core dependency guard failed; see its log')


if __name__=='__main__':main()
