"""Warmed whole-process CPU FP32 comparison of two pinned source trees.

Each arm runs in a fresh target process through measure_fp32_pipeline.timed_run.
One warmup per source precedes alternating AB/BA measured rounds by default.
Startup/imports, reads, registration, built-in host QA and all four output
writes are timed. Every warmup and repeat retains its products and receipt.
After all timings, finite/geometry/provenance checks cover every saved run;
last-repeat complete maps, warped arrays, affine and iteration traces are
compared diagnostically. Numerical or trajectory equality is not a gate.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import validation.benchmarks.measure_fp32_pipeline as pipeline
from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, import_implementation, load_reference_image, write_json


ARMS = ("baseline", "candidate")
INTEGRITY_FIELDS = ("source_unchanged_during_run", "inputs_unchanged_during_run",
                    "recipe_unchanged_during_run", "binary_unchanged_during_run")
PROVENANCE_FIELDS = ("source", "source_sha256", "input_sha256", "recipe_sha256",
                     "case_manifest_sha256", "quality_policy_sha256",
                     "python_binary_sha256", "threads")


def run_order(index):
    return ARMS if index % 2 == 0 else ARMS[::-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--case", default="ixi002_to_ixi012")
    parser.add_argument("--resolution", choices=("small", "half", "native"), default="half")
    parser.add_argument("--config", type=Path, default=PROJECT / "validation/configs/p00_dense.json")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4,
                        help="Repeats per source; an even count balances AB and BA starting positions")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("threads/repeats/timeout must be positive; warmups must be nonnegative")
    for name in ("baseline_source", "candidate_source", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    return args


def source_provenance(args, source, case):
    hashes = pipeline.source_hashes(source)
    if not (source / "fast_ants/__init__.py").is_file() or not hashes:
        raise ValueError("Source must contain the fast_ants package: " + str(source))
    return {"source": str(source), "source_sha256": hashes,
            "input_sha256": {case[side]: digest(case[side]) for side in ("fixed", "moving")},
            "recipe_sha256": digest(args.config),
            "case_manifest_sha256": digest(pipeline.MANIFEST_PATH),
            "quality_policy_sha256": digest(pipeline.POLICY_PATH),
            "threads": args.threads, "python_binary_sha256": digest(sys.executable),
            "binary_sha256": {"cpu": digest(sys.executable)}}


def worker_args(args, arm):
    """Adapt the shared whole-process runner without changing its CPU worker."""
    return argparse.Namespace(
        source=getattr(args, arm + "_source"), output=args.output / arm,
        case=args.case, resolution=args.resolution, config=args.config,
        threads=args.threads, swap=args.swap, timeout=args.timeout,
        engine=["cpu"], optimizations=None,
        ants_registration=Path(sys.executable).parent / "antsRegistration")


def summarize(runs):
    summary = {}
    for arm in ARMS:
        seconds = [row["whole_process_seconds"] for row in runs[arm]]
        if not seconds or not all(math.isfinite(value) and value > 0 for value in seconds):
            raise ValueError("Completed runs require positive finite whole-process times")
        memory = [row["peak_process_rss_bytes"] for row in runs[arm]
                  if row.get("peak_process_rss_bytes") is not None]
        summary[arm] = {"median_whole_process_seconds": statistics.median(seconds),
                        "minimum_whole_process_seconds": min(seconds),
                        "maximum_whole_process_seconds": max(seconds),
                        "maximum_peak_process_rss_bytes": max(memory, default=None)}
    baseline = summary["baseline"]["median_whole_process_seconds"]
    candidate = summary["candidate"]["median_whole_process_seconds"]
    return {"sources": summary, "baseline_time_divided_by_candidate_time": baseline / candidate,
            "candidate_time_divided_by_baseline_time": candidate / baseline,
            "candidate_time_reduction_percent": 100 * (1 - candidate / baseline),
            "paired_round_baseline_time_divided_by_candidate_time": [
                left["whole_process_seconds"] / right["whole_process_seconds"]
                for left, right in zip(runs["baseline"], runs["candidate"])]}


def array_difference(np, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("Diagnostic arrays must have matching geometry/shape")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta * delta))),
            "bitwise_values_equal_diagnostic": bool(actual.dtype == expected.dtype
                                                      and actual.tobytes(order="C") == expected.tobytes(order="C")),
            "baseline_values_sha256": hashlib.sha256(expected.tobytes(order="C")).hexdigest(),
            "candidate_values_sha256": hashlib.sha256(actual.tobytes(order="C")).hexdigest()}


def iteration_trace(history):
    return [{"kind": row["kind"], "level": row["level"],
             "iterations": len(row["iterations"]), "stop_reason": row.get("stop_reason"),
             "inverse_stop_decisions": [
                 {name: {"iterations": len(item.get("history", [])), "stop_reason": item.get("stop_reason")}
                  for name, item in step.get("inversions", {}).items()}
                 for step in row["iterations"]] if row["kind"] == "SyN" else []}
            for row in history]


def load_saved(api, np, row, provenance, fixed_grid):
    directory = Path(row["directory"])
    receipt = json.loads(Path(row["receipt_path"]).read_text())
    provenance_ok = (all(receipt.get(key) == provenance[key] for key in PROVENANCE_FIELDS)
                     and all(receipt.get(key) is True for key in INTEGRITY_FIELDS)
                     and receipt.get("engine") == "cpu"
                     and receipt.get("binary_sha256") == provenance["binary_sha256"]["cpu"])
    expected = set(pipeline.OWN_ARTIFACTS.values())
    artifacts_ok = (set(receipt["artifacts"]) == expected
                    and all(digest(directory / name) == value for name, value in receipt["artifacts"].items())
                    and all(digest(directory / name) == value for name, value in receipt["metadata_artifact_sha256"].items()))
    affine = api.load_affine(directory / pipeline.OWN_ARTIFACTS["affine"])
    forward = api.load_field(directory / pipeline.OWN_ARTIFACTS["forward_warp"])
    inverse = api.load_field(directory / pipeline.OWN_ARTIFACTS["inverse_warp"])
    warped = load_reference_image(api, directory / pipeline.OWN_ARTIFACTS["warped"])
    finite = bool(all(np.isfinite(array).all() for array in
                      (affine.matrix, affine.center, forward.data, inverse.data, warped.data)))
    geometry = bool(all(value.grid.same_geometry(fixed_grid) for value in (forward, inverse, warped)))
    checks = {"directory": str(directory), "receipt_provenance_matches": provenance_ok,
              "artifact_hashes_unchanged": artifacts_ok, "finite_outputs": finite,
              "output_geometry_matches_fixed": geometry,
              "integrity_passed": bool(provenance_ok and artifacts_ok and finite and geometry)}
    return (affine, forward, inverse, warped), checks


def assess_saved(args, case, provenance, report):
    # Registration workers have finished before importing either implementation
    # here. One pinned implementation evaluates both sets of saved maps.
    api, _ = import_implementation(args.candidate_source, args.threads)
    import numpy as np
    from fast_ants.quality import field_difference
    fixed, moving = [load_reference_image(api, case[side]) for side in ("fixed", "moving")]
    checks, last = {arm: [] for arm in ARMS}, {}
    for arm in ARMS:
        final_directory = report["runs"][arm][-1]["directory"]
        for row in report["warmup_diagnostics"][arm] + report["runs"][arm]:
            outputs, check = load_saved(api, np, row, provenance[arm], fixed.grid)
            checks[arm].append(check)
            if row["directory"] == final_directory:
                last[arm] = outputs
            if not check["integrity_passed"]:
                raise ValueError("Saved output integrity failed: " + row["directory"])
    transforms = {}
    for arm, (affine, forward, inverse, _) in last.items():
        transforms[arm] = (api.CompositeTransform((forward, affine)),
                           api.CompositeTransform((affine.inverse(), inverse)))
    maps = {}
    for side, grid, index in (("forward", fixed.grid, 0), ("inverse", moving.grid, 1)):
        baseline = api.transform_to_field(grid, transforms["baseline"][index])
        candidate = api.transform_to_field(grid, transforms["candidate"][index])
        if not all(np.isfinite(field.data).all() for field in (baseline, candidate)):
            raise ValueError("Nonfinite complete " + side + " map")
        maps[side] = field_difference(candidate, baseline)
        maps[side]["bitwise_values_equal_diagnostic"] = bool(
            candidate.data.dtype == baseline.data.dtype and candidate.data.tobytes() == baseline.data.tobytes())
    traces = {arm: iteration_trace(json.loads((Path(report["runs"][arm][-1]["directory"]) / "history.json").read_text()))
              for arm in ARMS}
    return {"scope": "all saved runs checked; last measured outputs compared after all timings; complete forward pull map on fixed grid and inverse pull map on moving grid, including affine",
            "diagnostic_source": provenance["candidate"]["source"],
            "saved_output_checks": checks, "complete_map_differences_diagnostic": maps,
            "warped_difference_diagnostic": array_difference(np, last["candidate"][3].data, last["baseline"][3].data),
            "affine_difference_diagnostic": {
                "matrix": array_difference(np, last["candidate"][0].matrix, last["baseline"][0].matrix),
                "center": array_difference(np, last["candidate"][0].center, last["baseline"][0].center)},
            "iteration_traces_diagnostic": traces,
            "iteration_trace_equal_diagnostic": traces["baseline"] == traces["candidate"],
            "map_array_affine_or_iteration_equality_required": False,
            "integrity_passed": True}


def run_comparison(args, config, case):
    provenance = {arm: source_provenance(args, getattr(args, arm + "_source"), case) for arm in ARMS}
    scripts = (Path(__file__).resolve(), Path(pipeline.__file__).resolve(),
               PROJECT / "validation/benchmarks/measure_fp64_cpu.py", PROJECT / "validation/common/time_process.py")
    script_hashes = {str(path): digest(path) for path in scripts}
    report = {"schema_version": 1, "status": "running", "engine": "cpu", "sources": provenance,
              "scope": __doc__, "recipe": config, "config_path": str(args.config),
              "case": case, "resolution": args.resolution, "swap": args.swap,
              "annotations_already_directional": True, "threads": args.threads,
              "platform": platform.platform(), "python": sys.version,
              "measurement_script_sha256": script_hashes,
              "warmup_runs_per_source": args.warmups, "repeats_per_source": args.repeats,
              "warmup_diagnostics": {arm: [] for arm in ARMS}, "runs": {arm: [] for arm in ARMS},
              "execution_order": [], "output_contract": pipeline.OWN_ARTIFACTS,
              "timing_scope": "whole fresh CPU target process: startup/imports, reads, registration, built-in output QA and writes; timing-runner startup excluded",
              "cache_policy": "warmups excluded from medians; fresh target processes retain filesystem/Numba cache warming",
              "power_background_load": "not controlled; source arms run sequentially in alternating order",
              "numerical_or_iteration_equality_required": False,
              "quality_status": "saved-output differences are diagnostic; registration quality acceptance is separate"}
    path = args.output / "report.json"
    write_json(path, report)
    try:
        for phase, count, group in (("warmup", args.warmups, "warmup_diagnostics"), ("repeat", args.repeats, "runs")):
            for repeat in range(count):
                for arm in run_order(repeat):
                    label = phase + "-" + str(repeat + 1)
                    report["execution_order"].append({"source": arm, "phase": phase, "repeat": repeat + 1})
                    write_json(path, report)
                    row = pipeline.timed_run(worker_args(args, arm), "cpu", label, config, case, provenance[arm])
                    row["source_arm"] = arm
                    report[group][arm].append(row)
                    write_json(path, report)
                    if row["status"] != "completed" or not row.get("integrity_passed"):
                        raise RuntimeError(arm + " " + label + " failed; inspect " + row["directory"])
        report["summary"] = summarize(report["runs"])
        report["starting_position_counts"] = {arm: sum(run_order(i)[0] == arm for i in range(args.repeats)) for arm in ARMS}
        report["starting_positions_exactly_balanced"] = len(set(report["starting_position_counts"].values())) == 1
        write_json(path, report)
        print("post-timing saved-output and complete-map diagnostics", flush=True)
        report["saved_diagnostics"] = assess_saved(args, case, provenance, report)
        report["source_unchanged_during_comparison"] = {
            arm: pipeline.source_hashes(getattr(args, arm + "_source")) == provenance[arm]["source_sha256"] for arm in ARMS}
        report["inputs_unchanged_during_comparison"] = all(
            digest(path) == value for path, value in provenance["baseline"]["input_sha256"].items())
        report["recipe_unchanged_during_comparison"] = digest(args.config) == provenance["baseline"]["recipe_sha256"]
        report["measurement_scripts_unchanged"] = all(digest(path) == value for path, value in script_hashes.items())
        if not (all(report["source_unchanged_during_comparison"].values())
                and report["inputs_unchanged_during_comparison"] and report["recipe_unchanged_during_comparison"]
                and report["measurement_scripts_unchanged"] and report["saved_diagnostics"]["integrity_passed"]):
            raise RuntimeError("Source/input/recipe/script or saved-output integrity failed")
        report["status"] = "timing_and_integrity_completed_registration_quality_review_pending"
        write_json(path, report)
        print(json.dumps(report["summary"], indent=2), flush=True)
        return report
    except BaseException as error:
        report.update(status="failed_outputs_preserved", failure={"type": type(error).__name__, "message": str(error)})
        write_json(path, report)
        raise


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text())
    pipeline.validate_recipe(config)
    case = pipeline.selected_case(args.case, args.resolution, args.swap)
    # Resolve and hash all required inputs/sources before creating outputs.
    for arm in ARMS:
        source_provenance(args, getattr(args, arm + "_source"), case)
    args.output.mkdir(parents=True)
    run_comparison(args, config, case)


if __name__ == "__main__":
    main()
