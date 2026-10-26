"""Sequential small-case CPU/Metal quality expansion with existing three MRIs."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from validation.benchmarks.measure_fp64_cpu import PROJECT, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--skip-metal',action='store_true')
    args=p.parse_args();out=args.output.resolve();source=args.source.resolve()
    if out.exists():p.error('Use a new output directory; existing reports are immutable')
    out.mkdir(parents=True);started=time.perf_counter()
    jobs=[(name,name,False,None) for name in ('synthetic_identity','synthetic_rigid','synthetic_affine','synthetic_smooth')]
    for case in ('ixi002_to_ixi012','ixi002_to_ixi013','ixi012_to_ixi013'):
        custom='validation/configs/quality_cases_20261024.json' if case=='ixi012_to_ixi013' else None
        jobs.extend([(case,case,False,custom),(case+'-swap',case,True,custom)])
    report={'status':'running','source':str(source),'scope':__doc__,'rows':[]}
    def run(command,log):
        report['active_command']=command;report['elapsed_seconds']=time.perf_counter()-started
        write_json(out/'report.json',report)
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise RuntimeError(f'Command exit {result.returncode}; see {log}')
    try:
        for tag,case,swap,custom in jobs:
            directory=out/tag;directory.mkdir()
            native=PROJECT/'validation/baseline/p00'/case/'float1'
            reuse=native.exists() and not swap and custom is None
            for backend in ('cpu',) if args.skip_metal else ('cpu','metal'):
                fit=directory/(backend+'-fit');quality=directory/(backend+'-quality')
                command=[sys.executable,'-B',str(PROJECT/'validation/quality/verify_float1_alignment.py'),
                    '--source',str(source),'--backend',backend,'--case',case,'--threads',str(args.threads),'--output',str(fit)]
                if swap:command+=['--swap']
                if custom:command+=['--case-manifest',str(PROJECT/custom)]
                if reuse:command+=['--native-dir',str(native)]
                print(f'{tag} {backend} fit; elapsed {time.perf_counter()-started:.1f}s',flush=True)
                run(command,directory/(backend+'-fit.log'))
                if not reuse:
                    native=fit/'native-float1';reuse=True
                run([sys.executable,'-B',str(PROJECT/'validation/quality/assess_saved_registration.py'),
                     '--report',str(fit/'report.json'),'--output',str(quality)],directory/(backend+'-quality.log'))
                fit_report=json.loads((fit/'report.json').read_text());q=json.loads((quality/'report.json').read_text())
                report['rows'].append({'case':case,'swap':swap,'backend':backend,'fitting_report':str(fit/'report.json'),
                    'quality_report':str(quality/'report.json'),'integrity_status':fit_report['status'],
                    'reference_map_errors_diagnostic':fit_report['complete_map_errors'],
                    'metrics':q['metrics'],'regularity':q['regularity'],'landmark_scope':q['landmark_scope']})
                report['elapsed_seconds']=time.perf_counter()-started;write_json(out/'report.json',report)
                print(f'{tag} {backend} quality recorded',flush=True)
        report['status']='quality_expansion_recorded'
    except BaseException as exc:
        report['status']='interrupted_or_failed';report['error']=str(exc)
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        if report['status']=='quality_expansion_recorded':report.pop('active_command',None)
        write_json(out/'report.json',report)


if __name__=='__main__':main()
