"""Warmed in-process Metal FP32 A/B, with quality and complete-map diagnostics.

Baseline uses this source's default flags unless --baseline-optimizations is
given; the default candidate adds paired_cc.
Repeat --variant label=comma,flags to replace the candidate(s); a `default` token
expands to the selected source's defaults, e.g. --variant paired=default,paired_cc.
Inputs/stages are preloaded. Timed register_metal calls include final resampling,
download, built-in host QA and explicit GPU synchronization. Serialization and
extra saved-output checks are excluded. No map/trace equality gate is applied.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import gc
import json
from numbers import Real
from pathlib import Path
import re
import resource
import statistics
import sys
import time

from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, environment, import_implementation, load_reference_image, stages_for, write_json
from validation.benchmarks.measure_fp32_pipeline import MANIFEST_PATH, OWN_ARTIFACTS, POLICY_PATH, selected_case, source_hashes, validate_recipe


def variant_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Variant must be label=comma,separated,flags")
    label, flags = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) or label == "baseline":
        raise argparse.ArgumentTypeError("Use a safe unique label; baseline is reserved")
    tokens = tuple(x.strip() for x in flags.split(",") if x.strip())
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) for token in tokens):
        raise argparse.ArgumentTypeError("Flags must be comma-separated identifiers; default expands source defaults")
    return label, tokens


def resolve_variants(specs, default_flags, available_flags):
    variants = {"baseline": tuple(default_flags)}
    for label, tokens in specs or [("paired-cc", ("default", "paired_cc"))]:
        if label in variants:
            raise ValueError("Duplicate variant label: " + label)
        flags = []
        for token in tokens:
            flags.extend(default_flags if token == "default" else (token,))
        flags = tuple(dict.fromkeys(flags))
        unknown = set(flags) - set(available_flags)
        if unknown:
            raise ValueError("Unknown Metal flags: " + ",".join(sorted(unknown)))
        variants[label] = flags
    return variants


def run_order(labels, index):
    """Rotate starting position; reverse each complete rotation for >2 arms."""
    labels = list(labels)
    cycle = labels if len(labels) == 2 or index // len(labels) % 2 == 0 else list(reversed(labels))
    offset = index % len(labels)
    return cycle[offset:] + cycle[:offset]


def discrete_trace(history):
    output = []
    for level in history:
        item = {"kind": level["kind"], "level": level["level"], "iterations": len(level["iterations"]),
                "stop_reason": level.get("stop_reason")}
        if level["kind"] == "SyN":
            item["inverse_stop_decisions"] = [
                {name: {"iterations": len(inversion.get("history", [])), "stop_reason": inversion.get("stop_reason")}
                 for name, inversion in step.get("inversions", {}).items()} for step in level["iterations"]]
        output.append(item)
    return output


def memory_value(mx, name):
    function = getattr(mx, name, None)
    return int(function()) if function is not None else None


def export_history(value):
    """Encode only the optimizer's unfilled convergence window as JSON null.

    This copies the diagnostic history; it does not change optimization state.
    All other non-finite values remain untouched so strict write_json rejects
    them, including NaN/-inf convergence or non-finite objective values.
    """
    if isinstance(value, dict):
        return {key: (None if key == "convergence" and isinstance(item, Real)
                      and item == float("inf") else export_history(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [export_history(item) for item in value]
    return value


def save_outputs(api, result, directory):
    directory.mkdir(parents=True)
    api.save_affine(result.linear_transform, directory / OWN_ARTIFACTS["affine"])
    api.save_field(result.forward_field, directory / OWN_ARTIFACTS["forward_warp"])
    api.save_field(result.inverse_field, directory / OWN_ARTIFACTS["inverse_warp"])
    api.save_image(result.warped, directory / OWN_ARTIFACTS["warped"])
    for name, value in (("history", export_history(result.history)), ("configuration", result.configuration), ("quality", result.quality)):
        write_json(directory / (name + ".json"), value)
    write_json(directory / "history-encoding.json", {
        "convergence_null_interpretation": "The convergence monitor window was not yet filled; the in-memory +inf diagnostic is encoded as JSON null. Optimization state and calculations are unchanged.",
        "other_nonfinite_values": "Rejected by strict JSON serialization."})
    return {path.name: digest(path) for path in sorted(directory.iterdir()) if path.is_file()}


def timed_registration(api, mx, np, fixed, moving, options, flags):
    gc.collect()
    mx.synchronize(mx.gpu)
    before = {"active_bytes": memory_value(mx, "get_active_memory"), "cache_bytes": memory_value(mx, "get_cache_memory")}
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    started = time.perf_counter()
    result = api.register_metal(fixed, moving, optimizations=flags, **options)
    mx.synchronize(mx.gpu)
    seconds = time.perf_counter() - started
    # These scans inspect already materialized host outputs after the timer.
    finite = bool(all(np.isfinite(array).all() for array in
                      (result.linear_transform.matrix, result.linear_transform.center,
                       result.forward_field.data, result.inverse_field.data, result.warped.data)))
    quality = result.quality
    minimum = quality.get("minimum_nonlinear_jacobian")
    jacobian_positive = bool(minimum is not None and np.isfinite(minimum) and minimum > 0
                             and quality.get("nonpositive_jacobian_points") == 0)
    row = {"seconds": seconds, "finite_outputs": finite, "positive_forward_nonlinear_jacobian": jacobian_positive,
           "minimum_nonlinear_jacobian": minimum, "nonpositive_jacobian_points": quality.get("nonpositive_jacobian_points"),
           "memory_before": before, "gpu_peak_active_bytes": memory_value(mx, "get_peak_memory"),
           "gpu_active_bytes_after_with_retained_state": memory_value(mx, "get_active_memory"),
           "gpu_cache_bytes_after": memory_value(mx, "get_cache_memory"),
           "process_peak_rss_bytes_cumulative": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
           "discrete_trace_diagnostic": discrete_trace(result.history), "quality": quality,
           "linear_seconds_diagnostic": quality.get("linear_seconds_diagnostic_not_benchmark"),
           "syn_seconds_diagnostic": sum(level.get("seconds_diagnostic_not_benchmark", 0.) for level in result.history if level["kind"] == "SyN"),
           "runtime": result.configuration.get("runtime"), "integrity_passed": finite and jacobian_positive}
    return result, row


def assess_saved(api, np, fixed, moving, output, variants, artifact_hashes):
    from fast_ants.fields import jacobian_determinant
    from fast_ants.quality import field_difference
    maps, checks, traces = {}, {}, {}
    for label in variants:
        directory = output / label
        integrity = all(digest(directory / name) == expected for name, expected in artifact_hashes[label].items())
        affine = api.load_affine(directory / OWN_ARTIFACTS["affine"])
        forward = api.load_field(directory / OWN_ARTIFACTS["forward_warp"])
        inverse = api.load_field(directory / OWN_ARTIFACTS["inverse_warp"])
        warped = load_reference_image(api, directory / OWN_ARTIFACTS["warped"])
        finite = bool(all(np.isfinite(a).all() for a in (affine.matrix, affine.center, forward.data, inverse.data, warped.data)))
        regularity = {}
        for side, field in (("forward", forward), ("inverse", inverse)):
            jacobian = jacobian_determinant(field)
            regularity[side] = {"minimum_nonlinear_jacobian": float(jacobian.min()),
                                "nonpositive_jacobian_points": int((jacobian <= 0).sum()),
                                "finite": bool(np.isfinite(jacobian).all())}
        geometry = bool(forward.grid.same_geometry(fixed.grid) and inverse.grid.same_geometry(fixed.grid)
                        and warped.grid.same_geometry(fixed.grid))
        determinant = float(np.linalg.det(affine.matrix[:3, :3]))
        positive = all(item["finite"] and item["minimum_nonlinear_jacobian"] > 0
                       and item["nonpositive_jacobian_points"] == 0 for item in regularity.values())
        checks[label] = {"artifact_hashes_unchanged": integrity, "finite_outputs": finite,
                         "output_geometry_matches_fixed": geometry, "regularity": regularity,
                         "affine_determinant": determinant,
                         "integrity_passed": bool(integrity and finite and geometry and positive and np.isfinite(determinant) and determinant > 0)}
        maps[label] = (api.CompositeTransform((forward, affine)), api.CompositeTransform((affine.inverse(), inverse)))
        traces[label] = discrete_trace(json.loads((directory / "history.json").read_text()))
    comparisons = {label: {} for label in variants if label != "baseline"}
    for side, grid, index in (("forward", fixed.grid, 0), ("inverse", moving.grid, 1)):
        expected = api.transform_to_field(grid, maps["baseline"][index])
        for label in comparisons:
            actual = api.transform_to_field(grid, maps[label][index])
            comparison = field_difference(actual, expected)
            comparison["bitwise_values_equal_diagnostic"] = bool(actual.data.dtype == expected.data.dtype
                                                                  and actual.data.tobytes() == expected.data.tobytes())
            comparisons[label][side] = comparison
            del actual
        del expected
    return {"scope": "last measured output per variant; complete pull maps including affine, forward on fixed grid and inverse on moving grid; post-timing QA",
            "saved_output_checks": checks, "complete_map_differences_diagnostic": comparisons,
            "discrete_trace_equal_diagnostic": {label: traces[label] == traces["baseline"] for label in comparisons},
            "map_or_trace_equality_required": False, "integrity_passed": all(row["integrity_passed"] for row in checks.values())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "src")
    parser.add_argument("--case", default="ixi002_to_ixi012")
    parser.add_argument("--resolution", choices=("small", "half", "native"), default="small")
    parser.add_argument("--config", type=Path, default=PROJECT / "validation/configs/p00_dense.json")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4, help="Per-variant repeats; even counts fully balance the default two-arm order")
    parser.add_argument("--variant", action="append", type=variant_spec)
    parser.add_argument("--baseline-optimizations",
                        help="Complete baseline flag set; default tokens also expand to this set")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("threads/repeats must be positive and warmups nonnegative")
    if args.variant and len({label for label, _ in args.variant}) != len(args.variant):
        parser.error("Variant labels must be unique")
    for name in ("source", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    config_hash = digest(args.config)
    config = json.loads(args.config.read_text())
    validate_recipe(config)
    case = selected_case(args.case, args.resolution)
    input_hashes = {case[side]: digest(case[side]) for side in ("fixed", "moving")}
    hashes = source_hashes(args.source)
    if not hashes:
        raise ValueError("Source must contain the fast_ants package")
    api, numba = import_implementation(args.source, args.threads)
    import numpy as np
    from fast_ants.backends.metal.runtime import AVAILABLE_OPTIMIZATIONS, DEFAULT_OPTIMIZATIONS, mx
    baseline_flags=(DEFAULT_OPTIMIZATIONS if args.baseline_optimizations is None else
                    tuple(flag.strip() for flag in args.baseline_optimizations.split(',') if flag.strip()))
    if set(baseline_flags)-AVAILABLE_OPTIMIZATIONS:
        raise ValueError('Unknown baseline Metal flags: '+str(set(baseline_flags)-AVAILABLE_OPTIMIZATIONS))
    variants = resolve_variants(args.variant, baseline_flags, AVAILABLE_OPTIMIZATIONS)
    fixed, moving = [load_reference_image(api, case[side]) for side in ("fixed", "moving")]
    linear, syn = stages_for(api, config, fixed.grid)
    options = dict(linear_stages=linear, syn_stage=syn, seed=config["random_seed"],
                   initialization_feature=config["initialization_feature"])
    args.output.mkdir(parents=True)
    report = {"schema_version": 1, "status": "running", "scope": __doc__,
              "source": str(args.source), "source_sha256": hashes,
              "script_sha256": digest(Path(__file__).resolve()), "recipe": config, "config_path": str(args.config),
              "recipe_sha256": config_hash, "case": case, "resolution": args.resolution,
              "input_sha256": input_hashes,
              "case_manifest_sha256": digest(MANIFEST_PATH), "quality_policy": json.loads(POLICY_PATH.read_text()),
              "quality_policy_sha256": digest(POLICY_PATH), "variants": variants,
              "warmups_per_variant": args.warmups, "repeats_per_variant": args.repeats,
              "environment": environment(numba), "device": mx.device_info(),
              "timing_scope": "preloaded API register_metal, including final download/resampling, built-in host QA and explicit GPU synchronization; serialization and extra checks excluded",
              "memory_scope": "MLX active/cache allocations, peak reset per call; process RSS is cumulative process high-water mark, not independent per-run peaks",
              "power_background_load": "not controlled; sequential in-process warmed A/B; allocator/driver caches retained",
              "map_or_iteration_equality_required": False,
              "warmup_diagnostics": {label: [] for label in variants}, "runs": {label: [] for label in variants},
              "execution_order": [], "artifact_sha256": {}}
    report_path = args.output / "report.json"
    write_json(report_path, report)
    try:
        for phase, count, group in (("warmup", args.warmups, "warmup_diagnostics"), ("repeat", args.repeats, "runs")):
            for repeat in range(count):
                for label in run_order(variants, repeat):
                    print(f"start {phase}-{repeat + 1} {label}", flush=True)
                    report["execution_order"].append({"phase": phase, "repeat": repeat + 1, "variant": label})
                    result, row = timed_registration(api, mx, np, fixed, moving, options, variants[label])
                    row["repeat"] = repeat + 1
                    report[group][label].append(row)
                    if (phase == "repeat" and repeat == count - 1) or not row["integrity_passed"]:
                        report["artifact_sha256"][label] = save_outputs(api, result, args.output / label)
                    # Returned state retains GPU arrays. Release it before the
                    # next variant; saved products and report values are host data.
                    result.state = None
                    del result
                    gc.collect()
                    mx.synchronize(mx.gpu)
                    row["gpu_active_bytes_after_state_release"] = memory_value(mx, "get_active_memory")
                    write_json(report_path, report)
                    print(f"done {phase}-{repeat + 1} {label}: {row['seconds']:.3f}s; integrity={row['integrity_passed']}", flush=True)
                    if not row["integrity_passed"]:
                        raise RuntimeError("Finite/Jacobian check failed for " + label)
        report["summary"] = {label: {"median_api_seconds": statistics.median(row["seconds"] for row in rows),
                                       "minimum_api_seconds": min(row["seconds"] for row in rows),
                                       "maximum_api_seconds": max(row["seconds"] for row in rows),
                                       "peak_gpu_active_bytes": max((row["gpu_peak_active_bytes"] for row in rows if row["gpu_peak_active_bytes"] is not None), default=None)}
                             for label, rows in report["runs"].items()}
        baseline = report["summary"]["baseline"]["median_api_seconds"]
        for item in report["summary"].values():
            item["baseline_time_divided_by_variant_time"] = baseline / item["median_api_seconds"]
        report["starting_position_counts"] = {label: sum(run_order(variants, i)[0] == label for i in range(args.repeats)) for label in variants}
        report["starting_positions_exactly_balanced"] = len(set(report["starting_position_counts"].values())) == 1
        write_json(report_path, report)
        print("post-timing complete-map and saved-output checks", flush=True)
        report["saved_diagnostics"] = assess_saved(api, np, fixed, moving, args.output, variants, report["artifact_sha256"])
        report["source_unchanged_during_run"] = source_hashes(args.source) == hashes
        report["inputs_unchanged_during_run"] = all(digest(path) == expected for path, expected in report["input_sha256"].items())
        report["recipe_unchanged_during_run"] = digest(args.config) == report["recipe_sha256"]
        integrity = all(report[key] for key in ("source_unchanged_during_run", "inputs_unchanged_during_run", "recipe_unchanged_during_run"))
        if not integrity or not report["saved_diagnostics"]["integrity_passed"]:
            raise RuntimeError("Saved-output or provenance integrity failed")
        report["status"] = "timing_and_integrity_completed_registration_quality_review_pending"
        write_json(report_path, report)
        print(json.dumps(report["summary"], indent=2), flush=True)
    except BaseException as error:
        report.update(status="failed_outputs_preserved", failure={"type": type(error).__name__, "message": str(error)})
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
