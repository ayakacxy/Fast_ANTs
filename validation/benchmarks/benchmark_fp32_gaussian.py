"""Warm matched-operator FP32 Gaussian comparison; not registration speedup."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION

import argparse
import hashlib
import json
from pathlib import Path
import time
from validation.benchmarks.measure_fp64_cpu import PROJECT, import_implementation, write_json
from validation.quality.verify_float1_alignment import hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PROJECT/'src')
    p.add_argument('--shape',type=int,default=64)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--metal',action='store_true')
    p.add_argument('--repeat',type=int,default=4)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();source=args.source.resolve()
    if out.exists():p.error('Use a new output directory')
    if args.shape<4 or args.repeat<2 or args.warmups<1:p.error('At least four pixels, two repeats and one warmup required')
    api,numba=import_implementation(source,args.threads)
    import numpy as np
    from fast_ants.filtering import smooth_image
    from fast_ants.precision32 import smooth_image32
    values=np.random.default_rng(20261023).uniform(.1,.9,(args.shape,)*3).astype(np.float32)
    image=api.Image(values,api.Grid(values.shape,np.eye(4)))
    operations={'recursive':lambda sigma:smooth_image32(image,sigma,method='recursive'),
                'sections':lambda sigma:smooth_image32(image,sigma)}
    synchronize=lambda result:None
    download=lambda result:result.data
    contexts={}
    if args.metal:
        from fast_ants.backends.metal.runtime import MetalContext,DEFAULT_OPTIMIZATIONS
        from fast_ants.backends.metal.ops import MetalOps
        contexts={name:MetalContext(tuple(x for x in DEFAULT_OPTIMIZATIONS if x!='gaussian_sections')+
                    (() if name=='recursive' else ('gaussian_sections',))) for name in operations}
        operators={name:MetalOps(ctx) for name,ctx in contexts.items()}
        inputs={name:ops.image(image) for name,ops in operators.items()}
        operations={name:(lambda sigma,name=name:operators[name].smooth_image(inputs[name],sigma)) for name in operations}
        synchronize=lambda result:contexts['sections'].finish(result.data)
        download=lambda result:contexts['sections'].download(result.data,'untimed_accuracy')
    report={'scope':__doc__,'source':str(source),'source_sha256':hashes(source),'backend':'metal' if args.metal else 'cpu',
        'shape_zyx':list(values.shape),'dtype':'float32','threads':numba.get_num_threads(),'repeat':args.repeat,'warmups':args.warmups,
        'benchmark_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'timing':'input upload and warmup excluded; three axis passes and coefficient-table dispatch included; GPU evaluation and synchronization included',
        'cases':[]}
    out.mkdir(parents=True)
    for sigma in (1.,3.):
        for _ in range(args.warmups):
            for operation in operations.values():
                result=operation(sigma);synchronize(result)
        durations={name:[] for name in operations};results={}
        for repeat in range(args.repeat):
            order=('recursive','sections') if repeat%2==0 else ('sections','recursive')
            for name in order:
                started=time.perf_counter();result=operations[name](sigma);synchronize(result)
                durations[name].append(time.perf_counter()-started);results[name]=result
        truth=smooth_image(image,sigma).data
        errors={}
        for name,result in results.items():
            actual=download(result);delta=actual.astype(np.float64)-truth
            errors[name]={'max_abs':float(np.max(np.abs(delta))),'rms':float(np.sqrt(np.mean(delta*delta))),
                          'finite':bool(np.isfinite(actual).all())}
        median={name:float(np.median(t)) for name,t in durations.items()}
        row={'sigma_vox':sigma,'seconds':durations,'median_seconds':median,
             'speedup_recursive_over_sections':median['recursive']/median['sections'],
             'error_vs_double_recurrence_with_per_axis_float_storage':errors}
        report['cases'].append(row);write_json(out/'report.json',report)
        print(json.dumps(row),flush=True)
    report.update(status='operator_benchmark_complete',source_unchanged_during_run=report['source_sha256']==hashes(source),
                  device_reports={name:ctx.report() for name,ctx in contexts.items()})
    write_json(out/'report.json',report)


if __name__=='__main__':main()
