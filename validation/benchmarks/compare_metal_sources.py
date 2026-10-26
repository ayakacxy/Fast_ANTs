"""Sequential fresh-process Metal A/B from two recorded source trees.

Warmups are excluded, then measured source order alternates AB/BA. Target
startup/imports, reads, registration, built-in host QA, device synchronization
and four output writes remain inside the shared whole-process timer. Every
run retains its worker record, saved products and provenance receipt. Timing
and receipt integrity do not establish registration-quality acceptance.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import validation.benchmarks.measure_fp32_pipeline as pipeline
from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, write_json


ARMS=('baseline','candidate')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',type=Path,required=True)
    parser.add_argument('--candidate-source',type=Path,required=True)
    parser.add_argument('--case',default='ixi002_to_ixi012')
    parser.add_argument('--swap',action='store_true')
    parser.add_argument('--resolution',choices=('small','half','native'),default='half')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=10)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--candidate-optimizations',
                        help='Complete comma-separated candidate flag set; omitted uses its source defaults')
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if (args.threads<1 or args.warmups<0 or args.repeats<1 or
            not math.isfinite(args.timeout) or args.timeout<=0):
        parser.error('threads/repeats/timeout must be positive; warmups must be nonnegative')
    for name in ('baseline_source','candidate_source','config','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.output.exists():parser.error('Refusing to overwrite an existing output directory')
    if args.candidate_optimizations is not None:
        args.candidate_optimizations=','.join(value.strip() for value in
            args.candidate_optimizations.split(',') if value.strip())
    return args


def provenance(args,case):
    binary=digest(sys.executable)
    common={'input_sha256':{case[side]:digest(case[side]) for side in ('fixed','moving')},
            'recipe_sha256':digest(args.config),'case_manifest_sha256':digest(pipeline.MANIFEST_PATH),
            'quality_policy_sha256':digest(pipeline.POLICY_PATH),'python_binary_sha256':binary,
            'binary_sha256':{'metal':binary},'threads':args.threads}
    result={}
    for arm in ARMS:
        source=getattr(args,arm+'_source');hashes=pipeline.source_hashes(source)
        if not (source/'fast_ants/__init__.py').is_file() or not hashes:
            raise ValueError('Source must contain the fast_ants package: '+str(source))
        result[arm]={**common,'source':str(source),'source_sha256':hashes}
    return result


def worker_args(args,arm):
    return argparse.Namespace(source=getattr(args,arm+'_source'),output=args.output/arm,
        case=args.case,swap=args.swap,resolution=args.resolution,config=args.config,
        threads=args.threads,timeout=args.timeout,engine=['metal'],
        optimizations=args.candidate_optimizations if arm=='candidate' else None,
        ants_registration=Path(sys.executable).parent/'antsRegistration')


def summarize(runs):
    sources={}
    for arm,rows in runs.items():
        seconds=[row['whole_process_seconds'] for row in rows]
        if not seconds or not all(math.isfinite(value) and value>0 for value in seconds):
            raise ValueError('Completed runs require positive finite whole-process times')
        rss=[row['peak_process_rss_bytes'] for row in rows if row.get('peak_process_rss_bytes') is not None]
        gpu=[row['worker']['gpu_peak_active_bytes'] for row in rows
             if row['worker'].get('gpu_peak_active_bytes') is not None]
        sources[arm]={'median_whole_process_seconds':statistics.median(seconds),
                      'minimum_whole_process_seconds':min(seconds),'maximum_whole_process_seconds':max(seconds),
                      'maximum_peak_process_rss_bytes':max(rss,default=None),
                      'maximum_gpu_peak_active_bytes':max(gpu,default=None)}
    baseline=sources['baseline']['median_whole_process_seconds']
    candidate=sources['candidate']['median_whole_process_seconds']
    return {'sources':sources,'baseline_time_divided_by_candidate_time':baseline/candidate,
            'candidate_time_divided_by_baseline_time':candidate/baseline,
            'candidate_time_reduction_percent':100*(1-candidate/baseline),
            'paired_round_baseline_time_divided_by_candidate_time':[
                left['whole_process_seconds']/right['whole_process_seconds']
                for left,right in zip(runs['baseline'],runs['candidate'])]}


def run_comparison(args,config,case,sources):
    scripts=(Path(__file__).resolve(),Path(pipeline.__file__).resolve(),
             PROJECT/'validation/benchmarks/measure_fp64_cpu.py',PROJECT/'validation/common/time_process.py')
    script_hashes={str(path):digest(path) for path in scripts}
    report={'schema_version':1,'status':'running','engine':'metal','sources':sources,
            'scope':__doc__,'recipe':config,'config_path':str(args.config),'case':case,
            'resolution':args.resolution,'swap':args.swap,'annotations_already_directional':True,
            'threads':args.threads,'platform':platform.platform(),'python':sys.version,
            'measurement_script_sha256':script_hashes,'warmup_runs_per_source':args.warmups,
            'repeats_per_source':args.repeats,'warmup_diagnostics':{arm:[] for arm in ARMS},
            'runs':{arm:[] for arm in ARMS},'execution_order':[],
            'requested_optimizations':{'baseline':None,'candidate':args.candidate_optimizations},
            'optimization_scope':'None means that source defaults; candidate override is a complete flag set',
            'output_contract':pipeline.OWN_ARTIFACTS,
            'timing_scope':'whole fresh synchronized Metal target process including startup/imports, reads, registration, built-in host output QA and writes; runner startup excluded',
            'cache_policy':'warmups excluded; sequential fresh workers retain disk/JIT/driver cache warming',
            'memory_scope':'process RSS includes shared CPU/GPU address space; gpu_peak_active_bytes is separately recorded by the worker',
            'power_background_load':'uncontrolled; all source workers run sequentially',
            'numerical_or_iteration_equality_required':False,
            'quality_status':'not_assessed; worker built-in QA is preserved, but independent saved-output quality/finite/geometry assessment is separate'}
    path=args.output/'report.json';write_json(path,report)
    try:
        schedule=[(arm,'warmup-'+str(i+1),'warmup_diagnostics') for arm in ARMS for i in range(args.warmups)]
        schedule.extend((arm,'repeat-'+str(i+1),'runs') for i in range(args.repeats)
                        for arm in (ARMS if i%2==0 else ARMS[::-1]))
        for arm,label,group in schedule:
            report['execution_order'].append({'source':arm,'label':label});write_json(path,report)
            row=pipeline.timed_run(worker_args(args,arm),'metal',label,config,case,sources[arm])
            row['source_arm']=arm;report[group][arm].append(row);write_json(path,report)
            if row['status']!='completed' or not row.get('integrity_passed'):
                raise RuntimeError(arm+' '+label+' failed; inspect '+row['directory'])
            if row['worker'].get('engine')!='metal' or not row['worker'].get('gpu_synchronized_before_worker_completion'):
                raise RuntimeError('Missing synchronized Metal worker record: '+row['directory'])
        report['summary']=summarize(report['runs'])
        report['starting_position_counts']={arm:sum((ARMS if i%2==0 else ARMS[::-1])[0]==arm
                                                   for i in range(args.repeats)) for arm in ARMS}
        report['starting_positions_exactly_balanced']=len(set(report['starting_position_counts'].values()))==1
        report['source_unchanged_during_comparison']={arm:pipeline.source_hashes(getattr(args,arm+'_source'))==sources[arm]['source_sha256'] for arm in ARMS}
        report['inputs_unchanged_during_comparison']=all(digest(name)==value for name,value in sources['baseline']['input_sha256'].items())
        report['recipe_unchanged_during_comparison']=digest(args.config)==sources['baseline']['recipe_sha256']
        report['case_manifest_unchanged_during_comparison']=digest(pipeline.MANIFEST_PATH)==sources['baseline']['case_manifest_sha256']
        report['quality_policy_unchanged_during_comparison']=digest(pipeline.POLICY_PATH)==sources['baseline']['quality_policy_sha256']
        report['python_binary_unchanged_during_comparison']=digest(sys.executable)==sources['baseline']['python_binary_sha256']
        report['measurement_scripts_unchanged']=all(digest(name)==value for name,value in script_hashes.items())
        if not (all(report['source_unchanged_during_comparison'].values()) and
                all(report[key] for key in ('inputs_unchanged_during_comparison','recipe_unchanged_during_comparison',
                    'case_manifest_unchanged_during_comparison','quality_policy_unchanged_during_comparison',
                    'python_binary_unchanged_during_comparison','measurement_scripts_unchanged'))):
            raise RuntimeError('Source, input, recipe, policy, binary or measurement script changed')
        report['status']='timing_and_integrity_completed_registration_quality_review_pending'
        write_json(path,report);print(json.dumps(report['summary'],indent=2),flush=True)
        return report
    except BaseException as error:
        report.update(status='failed_outputs_preserved',failure={'type':type(error).__name__,'message':str(error)})
        write_json(path,report)
        raise


def main(argv=None):
    args=parse_args(argv);config=json.loads(args.config.read_text());pipeline.validate_recipe(config)
    case=pipeline.selected_case(args.case,args.resolution,args.swap)
    sources=provenance(args,case)
    args.output.mkdir(parents=True,exist_ok=False)
    run_comparison(args,config,case,sources)


if __name__=='__main__':main()
