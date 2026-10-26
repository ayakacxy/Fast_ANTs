"""Warmed paired-CC operator A/B on existing MRI arrays, outside registration.

Both implementations receive identical normalized FP32 images, physical image
gradients and an all-valid mask. No upstream operators or SciPy are called.
These are operator timings, not whole-pipeline or official ANTs speed ratios.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, import_implementation, load_reference_image, write_json
from validation.benchmarks.measure_fp32_pipeline import selected_case, source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PROJECT/'src')
    p.add_argument('--resolution',choices=('small','half','native'),default='half')
    p.add_argument('--threads',type=int,default=10)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if args.output.exists():p.error('Use a new output directory')
    if args.threads<1 or args.warmups<1 or args.repeats<2:p.error('Positive threads/warmups and at least two repeats required')
    case=selected_case('ixi002_to_ixi012',args.resolution)
    pinned=source_hashes(args.source)
    inputs={case[s]:digest(case[s]) for s in ('fixed','moving')}
    api,numba=import_implementation(args.source,args.threads)
    import numpy as np
    from fast_ants.precision32 import image32,gradient_image32
    from fast_ants.backends.cpu.cc_pair32 import cc_pair32
    from fast_ants.backends.cpu.cc_queue_pair32 import cc_queue_pair32,_work32,_row32,_plane_stats32
    images=[image32(load_reference_image(api,case[s]),normalize=True) for s in ('fixed','moving')]
    if images[0].data.shape!=images[1].data.shape:raise ValueError('Equal array shapes required for this operator fixture')
    data=tuple(i.data for i in images)+tuple(gradient_image32(i) for i in images)+(np.ones(images[0].data.shape,bool),2)
    methods={'direct':cc_pair32,'queue':cc_queue_pair32};values={};times={name:[] for name in methods};order=[]
    for _ in range(args.warmups):
        for name,method in methods.items():values[name]=method(*data)
    for r in range(args.repeats):
        for name in (('direct','queue') if r%2==0 else ('queue','direct')):
            start=time.perf_counter();values[name]=methods[name](*data);seconds=time.perf_counter()-start
            times[name].append(seconds);order.append({'round':r+1,'method':name,'seconds':seconds})
    baseline,candidate=values['direct'],values['queue']
    if baseline[1]!=candidate[1]:raise AssertionError('Valid centre counts differ')
    differences={}
    for index,name in ((2,'forward_update'),(3,'reverse_update')):
        a,b=baseline[index],candidate[index]
        if a.dtype!=np.float32 or b.dtype!=np.float32 or not np.isfinite(b).all():raise AssertionError('Nonfinite or non-FP32 updates')
        delta=a.astype(np.float64)-b.astype(np.float64)
        differences[name]={'maximum_component_absolute':float(np.max(np.abs(delta))),'rms_component':float(np.sqrt(np.mean(delta*delta)))}
    if source_hashes(args.source)!=pinned or any(digest(path)!=h for path,h in inputs.items()):raise AssertionError('Source/input changed during measurement')
    medians={name:statistics.median(rows) for name,rows in times.items()}
    args.output.mkdir(parents=True)
    stream=io.StringIO()
    for function in (_work32,_row32,_plane_stats32):
        stream.write('\n'+function.py_func.__name__+'\n');function.inspect_types(file=stream)
    (args.output/'numba-types.txt').write_text(stream.getvalue())
    report={'status':'operator_timings_recorded_not_pipeline_acceptance','source':str(args.source),'source_sha256':pinned,'input_sha256':inputs,
            'resolution':args.resolution,'shape_zyx':list(images[0].data.shape),'numba_threads':numba.get_num_threads(),'radius':2,
            'warmups_per_method':args.warmups,'repeats_per_method':args.repeats,'order':order,'seconds':times,'median_seconds':medians,
            'queue_speedup_over_direct':medians['direct']/medians['queue'],'valid_centres':candidate[1],
            'scores':{name:float(value[0]) for name,value in values.items()},'update_differences_diagnostic':differences,
            'scope':__doc__,'source_unchanged':True,'inputs_unchanged':True,'numba_type_audit_sha256':digest(args.output/'numba-types.txt')}
    write_json(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('median_seconds','queue_speedup_over_direct','update_differences_diagnostic')},indent=2),flush=True)


if __name__=='__main__':main()
