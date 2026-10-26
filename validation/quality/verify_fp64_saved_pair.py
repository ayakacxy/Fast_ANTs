"""Compare saved FP64 whole-process outputs from matching recorded recipes.

Supports reusing an unchanged official run without pretending it was timed in
the new Python batch. Source, inputs, recipe and saved-artifact provenance are
recorded; this script's own numerical work is outside all process timers.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from validation.benchmarks.measure_fp64_cpu import compare_whole_saved, digest, source_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--native-engine", choices=("ants-default", "ants-1", "ants-10"), default="ants-default")
    parser.add_argument("--tolerance-mm", type=float, default=.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.python_report, args.native_report = args.python_report.resolve(), args.native_report.resolve()
    args.output = args.output.resolve()
    if args.output.exists() or args.tolerance_mm <= 0:
        parser.error("Use a new output file and a positive tolerance")
    own = json.loads(args.python_report.read_text())
    native = json.loads(args.native_report.read_text())
    for key in ("configuration", "case", "resolution", "input_sha256", "native_binary_sha256"):
        if own[key] != native[key]:
            raise ValueError("Recorded conditions differ: "+key)
    for side, path in own["case"].items():
        if side in own["input_sha256"] and digest(path) != own["input_sha256"][side]:
            raise ValueError("Input changed: "+side)
    source = Path(own["source"])
    if source_hashes(source) != own["source_sha256"]:
        raise ValueError("Python source changed since the recorded run")
    if digest(Path(sys.executable).parent/"antsRegistration") != native["native_binary_sha256"]:
        raise ValueError("Official binary changed since the recorded run")
    python_count, native_count = len(own["runs"]["python"]), len(native["runs"][args.native_engine])
    folders = {"python": args.python_report.parent/"python"/f"repeat-{python_count}",
               args.native_engine: args.native_report.parent/args.native_engine/f"repeat-{native_count}"}
    options = SimpleNamespace(source=source, threads=own["requested_python_numba_threads"],
                              output=args.output.parent, repeats=python_count,
                              native_tolerance_mm=args.tolerance_mm)
    comparison = compare_whole_saved(options, ["python", args.native_engine], own["case"], folders)
    artifact_hashes = {engine: {name: digest(folder/name) for name in
        (("affine.mat", "warp.nii.gz", "inverse-warp.nii.gz", "warped.nii.gz") if engine == "python" else
         ("reg_0GenericAffine.mat", "reg_1Warp.nii.gz", "reg_1InverseWarp.nii.gz", "warped.nii.gz"))}
        for engine, folder in folders.items()}
    pseconds = own["summary"]["python"]["median_whole_process_seconds"]
    nseconds = native["summary"][args.native_engine]["median_whole_process_seconds"]
    passed = (comparison["all_saved_output_checks_passed"] and
              comparison["all_requested_numeric_gates_passed"] is True)
    report = {"status": "passed" if passed else "numeric_gate_failed", "scope": __doc__,
              "python_report": str(args.python_report), "python_report_sha256": digest(args.python_report),
              "native_report": str(args.native_report), "native_report_sha256": digest(args.native_report),
              "saved_numeric_comparison": comparison, "saved_artifact_sha256": artifact_hashes,
              "timings": {"python_seconds": pseconds, "native_seconds": nseconds,
                "native_to_python_ratio": nseconds/pseconds, "python_repeats": python_count,
                "native_repeats": native_count,
                "scope": "separately recorded batches; not a new interleaved benchmark; report single-run pilots as such"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "timings": report["timings"],
                      "errors": comparison["complete_map_comparisons"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
