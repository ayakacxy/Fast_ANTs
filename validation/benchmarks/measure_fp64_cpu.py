"""FP64 CPU profiling, numerical checks, and process timings.

Examples (use an environment with the required dependencies):
  python -B validation/benchmarks/measure_fp64_cpu.py --source src --resolution half --threads 10 --profile --output validation/reports/cpu64-profile
  python -B validation/benchmarks/measure_fp64_cpu.py --mode whole --source src --resolution native --threads 10 --engine python --engine ants-default --output validation/reports/cpu64-native

API times use preloaded images and include register()'s final resampling and QA,
but exclude imports, reads, serialization, and the optional profiling run.
Whole-process times include startup/imports, input reads, registration, and
NIfTI/JSON writes; Python also performs its built-in output QA.
CPU FP32 and Metal registration use measure_fp32_pipeline.py.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION
from validation.common.runtime import digest, write_json


import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import platform
import pstats
import re
import statistics
import subprocess
import sys
import time

PROJECT = _TOOL_PROJECT






def source_hashes(source):
    return {str(p.relative_to(source)): digest(p)
            for p in sorted((source / "fast_ants").rglob("*.py"))}


def case_for(resolution):
    case = dict(next(c for c in json.loads((PROJECT / "validation/cases.json").read_text())["cases"]
                     if c["id"] == "ixi002_to_ixi012"))
    folder = {"small": "prepared", "half": "p11-half", "native": "public"}[resolution]
    for side in ("fixed", "moving"):
        case[side] = str(PROJECT / "validation/data" / folder / Path(case[side]).name)
    return case


def native_command(config, case, output, binary):
    """Build the exact recorded recipe, including native-only stock options."""
    from validation.reference_tools.run_p00_baseline import command_for
    command = command_for(config, case, 0, output, {"antsRegistration": binary})
    locations = [i+1 for i, arg in enumerate(command) if arg == "--metric"]
    fixed, moving = case["fixed"], case["moving"]
    for i, stage in enumerate(config["stages"]):
        sampling = config["sampling"] if i < 2 else {"strategy": "None", "fraction": 1., "gradient_filter": False}
        if sampling != {"strategy": "None", "fraction": 1., "gradient_filter": False}:
            command[locations[i]] = (f"{stage['metric']}[{fixed},{moving},1,{stage['bins_or_radius']},"
                                     f"{sampling['strategy']},{sampling['fraction']},{int(sampling['gradient_filter'])}]")
    if config["winsorization"] is not None:
        low, high = config["winsorization"]
        command += ["--winsorize-image-intensities", f"[{low},{high}]"]
    if config.get("write_inverse_warped", False):
        command[command.index("--output")+1] = f"[{output/'reg_'},{output/'warped.nii.gz'},{output/'inverse-warped.nii.gz'}]"
    return command


def import_implementation(source, threads):
    # Set all pools before importing NumPy/Numba. Kernels, rather than the whole
    # process, are what may remain serial; the report records these controls.
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(source))
    import fast_ants
    import numba
    numba.set_num_threads(threads)
    if not Path(fast_ants.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Wrong fast_ants source imported: " + fast_ants.__file__)
    return fast_ants, numba


def _shrink_schedule(stage, config, fixed_grid):
    schedule = tuple(stage["shrink_factors"])
    if config.get("shrink_factor_policy", "explicit") == "explicit":
        return schedule
    if config["shrink_factor_policy"] != "ants_physical_spacing" or fixed_grid is None:
        raise ValueError("ANTs physical-spacing shrink schedule needs the fixed grid")
    spacing = fixed_grid.spacing_xyz
    minimum = min(range(3), key=lambda d: spacing[d])
    output = []
    # RegistrationHelper::CalculateShrinkFactorsPerDimension, including its
    # decreasing search and <= tie rule. Core shrink_grid accepts these explicit
    # per-axis factors; no scalar-isotropic approximation enters this benchmark.
    for factor in schedule:
        factors = [factor]*3
        target = spacing[minimum]*factor
        for d in range(3):
            difference = abs(spacing[d]*factor-target)
            if d == minimum:
                continue
            for candidate in range(factor-1, 0, -1):
                delta = abs(spacing[d]*candidate-target)
                if delta <= difference:
                    difference, factors[d] = delta, candidate
        output.append(tuple(factors))
    return tuple(output)


def stages_for(api, config, fixed_grid=None):
    common = config["convergence"]
    linear = []
    for stage in config["stages"][:2]:
        kind, parameter = stage["transform"].split("[")
        linear.append(api.LinearStage(
            kind=kind, iterations=tuple(stage["iterations"]),
            shrink_factors=_shrink_schedule(stage, config, fixed_grid),
            smoothing_sigmas=tuple(stage["smoothing_sigmas"]),
            smoothing_units=stage["smoothing_units"], gradient_step=float(parameter[:-1]),
            metric=stage["metric"], bins=stage["bins_or_radius"],
            convergence_threshold=common["threshold"], convergence_window=common["window"],
            sampling_strategy=config["sampling"]["strategy"],
            sampling_fraction=config["sampling"]["fraction"]))
    stage = config["stages"][2]
    gradient_step, update_variance, total_variance = map(float, stage["transform"][4:-1].split(","))
    syn = api.SyNStage(
        iterations=tuple(stage["iterations"]), shrink_factors=_shrink_schedule(stage, config, fixed_grid),
        smoothing_sigmas=tuple(stage["smoothing_sigmas"]), smoothing_units=stage["smoothing_units"],
        gradient_step=gradient_step, radius=stage["bins_or_radius"],
        update_variance=update_variance, total_variance=total_variance,
        convergence_threshold=common["threshold"], convergence_window=common["window"])
    return tuple(linear), syn


def load_reference_image(api, path):
    """Use matching ITK IO while retaining source-selectable old snapshots.

    Old implementations lack scaling_policy and keep their historical reader.
    Their metadata records the absence of the explicit compatibility policy.
    The unscaled prepared/half fixtures are unaffected by that distinction.
    """
    options = {"geometry_policy": "itk_scanner"}
    if "scaling_policy" in inspect.signature(api.load_image).parameters:
        options["scaling_policy"] = "itk"
    return api.load_image(path, **options)


def without_timers(value):
    if isinstance(value, dict):
        return {k: without_timers(v) for k, v in value.items() if "seconds" not in k}
    if isinstance(value, list):
        return [without_timers(v) for v in value]
    return value


def result_arrays(result):
    return {"forward": result.forward_field.data, "inverse": result.inverse_field.data,
            "affine": result.linear_transform.matrix, "center": result.linear_transform.center,
            "warped": result.warped.data}


def array_hashes(result):
    return {k: {"sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest(),
                "dtype": str(v.dtype), "shape": list(v.shape)}
            for k, v in result_arrays(result).items()}


def save_result(result, output):
    import numpy as np
    np.savez(output / "arrays.npz", **result_arrays(result))
    write_json(output / "history.json", result.history)
    write_json(output / "history-without-timers.json", without_timers(result.history))
    write_json(output / "quality.json", result.quality)
    write_json(output / "configuration.json", result.configuration)


def compare_result(result, reference, tolerance):
    import numpy as np
    with np.load(reference / "arrays.npz") as expected:
        comparison = {}
        for key, actual in result_arrays(result).items():
            wanted = expected[key]
            if actual.shape != wanted.shape:
                comparison[key] = {"shape_matches": False, "passed": False}
                continue
            delta = actual.astype(np.float64) - wanted.astype(np.float64)
            comparison[key] = {
                "shape_matches": True,
                "bitwise_values_equal": bool(actual.dtype == wanted.dtype and
                                              actual.tobytes(order="C") == wanted.tobytes(order="C")),
                "max_abs": float(np.max(np.abs(delta))),
                "rms": float(np.sqrt(np.mean(delta * delta))),
                "absolute_tolerance": tolerance,
                "passed": bool(np.isfinite(delta).all() and np.max(np.abs(delta)) <= tolerance)}
    # Per-axis shrink schedules are tuples in memory and JSON arrays on disk.
    # Compare canonical JSON values, as the saved history hash already does.
    history_equal = json.loads(json.dumps(without_timers(result.history))) == json.loads((reference / "history-without-timers.json").read_text())
    return {"reference": str(reference), "arrays": comparison,
            "history_without_timers_exact": history_equal,
            "passed": history_equal and all(v["passed"] for v in comparison.values())}


def environment(numba):
    try:
        layer = numba.threading_layer()
    except ValueError:
        layer = "not initialized (no parallel kernel called)"
    return {"platform": platform.platform(), "python": sys.version,
            "logical_cpus": os.cpu_count(), "numba_threads": numba.get_num_threads(),
            "numba_threading_layer": layer,
            "thread_environment": {k: v for k, v in os.environ.items()
                                   if k.endswith("NUM_THREADS") or k == "VECLIB_MAXIMUM_THREADS"},
            "packages": {k: importlib.metadata.version(k) for k in ("numpy", "numba", "nibabel")},
            "power_background_load": "not controlled; runs coordinated and sequential"}


def run_api(args, config, case):
    api, numba = import_implementation(args.source, args.threads)
    fixed = load_reference_image(api, case["fixed"])
    moving = load_reference_image(api, case["moving"])
    linear, syn = stages_for(api, config, fixed.grid)
    report = {"scope": "preloaded FP64 register API, final resampling and built-in QA included; I/O/imports/serialization excluded",
              "source": str(args.source), "source_sha256": source_hashes(args.source),
              "config_path": str(args.config), "config_sha256": digest(args.config),
              "case": case, "resolution": args.resolution, "shape_xyz": list(fixed.grid.shape_xyz),
              "configuration": config, "requested_numba_threads": args.threads,
              "input_scaling_policies": {"fixed": fixed.metadata.get("scaling_policy", "legacy_nifti"),
                                         "moving": moving.metadata.get("scaling_policy", "legacy_nifti")},
              "thread_policy": "requested NumPy/BLAS pools fixed to 1; Numba controlled separately; recipe threads field is historical P00 metadata",
              "input_sha256": {s: digest(case[s]) for s in ("fixed", "moving")},
              "warmup_runs": args.warmups, "repeats": args.repeats, "runs": []}

    def run(label, profile=None):
        gc.collect()
        print("start", args.resolution, args.threads, label, flush=True)
        started = time.perf_counter()
        if profile is not None:
            profile.enable()
        try:
            result = api.register(fixed, moving, linear_stages=linear, syn_stage=syn,
                                  seed=config["random_seed"], initialization_feature=config["initialization_feature"])
        finally:
            if profile is not None:
                profile.disable()
        elapsed = time.perf_counter() - started
        row = {"seconds": elapsed, "array_hashes": array_hashes(result),
               "history_sha256_without_timers": hashlib.sha256(json.dumps(without_timers(result.history), sort_keys=True).encode()).hexdigest(),
               "history_counts": [(h["kind"], h["level"], len(h["iterations"])) for h in result.history],
               "diagnostic_quality_timers": {k: v for k, v in result.quality.items() if "seconds" in k}}
        print("done", label, format(elapsed, ".6f"), "s", flush=True)
        return result, row

    for i in range(args.warmups):
        result, row = run("warmup-" + str(i + 1))
        report.setdefault("warmup_diagnostics", []).append(row)
        del result
    for i in range(args.repeats):
        result, row = run("repeat-" + str(i + 1))
        report["runs"].append(row)
        write_json(args.output / "report.json", report)
        if i + 1 < args.repeats:
            del result
    report["summary"] = {"median_seconds": statistics.median(r["seconds"] for r in report["runs"]),
                         "minimum_seconds": min(r["seconds"] for r in report["runs"]),
                         "maximum_seconds": max(r["seconds"] for r in report["runs"])}
    report["repeat_arrays_exact"] = all(r["array_hashes"] == report["runs"][0]["array_hashes"] for r in report["runs"])
    report["repeat_histories_exact_without_timers"] = len({r["history_sha256_without_timers"] for r in report["runs"]}) == 1
    save_result(result, args.output)
    if args.compare_with is not None:
        report["numeric_regression"] = compare_result(result, args.compare_with, args.compare_tolerance)
    del result
    if args.profile:
        profile = cProfile.Profile()
        result, row = run("profile-excluded-from-summary", profile)
        profile.dump_stats(str(args.output / "profile.pstats"))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(80)
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("tottime").print_stats(60)
        (args.output / "profile.txt").write_text(stream.getvalue())
        report["profile_diagnostic"] = row
        del result
    report["environment"] = environment(numba)
    report["source_unchanged_during_run"] = source_hashes(args.source) == report["source_sha256"]
    report["status"] = "completed" if report["source_unchanged_during_run"] else "source_changed_during_run"
    if args.compare_with is not None and not report["numeric_regression"]["passed"]:
        report["status"] = "numeric_gate_failed"
    write_json(args.output / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def worker(args, config, case):
    api, numba = import_implementation(args.source, args.threads)
    started = time.perf_counter()
    fixed = load_reference_image(api, case["fixed"])
    moving = load_reference_image(api, case["moving"])
    linear, syn = stages_for(api, config, fixed.grid)
    print("CPU FP64", args.resolution, "start Rigid/Affine; SyN follows", flush=True)

    def progress(level, iteration, value):
        if iteration == 1 or iteration % 10 == 0:
            print(f"SyN level {level+1} iteration {iteration}; elapsed {time.perf_counter()-started:.1f}s",
                  flush=True)

    # The value-only observer does not expose state arrays. Keep the ordinary
    # callback's independent-result contract without disabling private buffer
    # reuse merely to print benchmark progress. Older snapshots use that callback.
    progress_options = ({"_progress_callback": progress}
                        if "_progress_callback" in inspect.signature(api.register).parameters
                        else {"callback": lambda level, iteration, result: progress(level, iteration, result.value)})
    result = api.register(fixed, moving, linear_stages=linear, syn_stage=syn,
                          seed=config["random_seed"], initialization_feature=config["initialization_feature"],
                          output_directory=args.worker, **progress_options)
    elapsed = time.perf_counter() - started
    write_json(args.worker / "worker.json", {"load_register_qa_write_seconds": elapsed,
               "environment": environment(numba), "history_counts": [(h["kind"], h["level"], len(h["iterations"])) for h in result.history]})


def compare_whole_saved(args, engines, case, folders=None):
    """Inspect saved final maps after all process timers have stopped.

    The complete image pull maps include the affine in the proper order. The
    optional tolerance is explicit: no newly invented default quality threshold
    is used, and numerical agreement is not an independent registration-quality
    claim. Results use the last repeat of each engine, not just one component.
    """
    import numpy as np
    api, _ = import_implementation(args.source, args.threads)
    from fast_ants.quality import field_difference
    fixed = load_reference_image(api, case["fixed"])
    moving = load_reference_image(api, case["moving"])
    pairs, checks = {}, {}
    for engine in engines:
        folder = (folders[engine] if folders is not None else
                  args.output / engine / ("repeat-" + str(args.repeats)))
        native = engine.startswith("ants")
        affine = api.load_affine(folder / ("reg_0GenericAffine.mat" if native else "affine.mat"))
        forward = api.load_field(folder / ("reg_1Warp.nii.gz" if native else "warp.nii.gz"))
        inverse = api.load_field(folder / ("reg_1InverseWarp.nii.gz" if native else "inverse-warp.nii.gz"))
        warped = load_reference_image(api, folder / "warped.nii.gz")
        checks[engine] = {
            "warped_geometry_matches_fixed": warped.grid.same_geometry(fixed.grid),
            "forward_grid_matches_fixed": forward.grid.same_geometry(fixed.grid),
            "inverse_field_grid_matches_fixed": inverse.grid.same_geometry(fixed.grid),
            "finite_saved_outputs": bool(all(np.isfinite(a).all()
                for a in (affine.matrix, forward.data, inverse.data, warped.data)))}
        pairs[engine] = (api.CompositeTransform((forward, affine)),
                         api.CompositeTransform((affine.inverse(), inverse)))
    reference = next((e for e in ("ants-1", "ants-default", "ants-10") if e in engines), None)
    comparisons = {e: {} for e in engines if e != reference}
    if reference is not None:
        for side, grid, index in (("forward", fixed.grid, 0), ("inverse", moving.grid, 1)):
            expected = api.transform_to_field(grid, pairs[reference][index])
            for engine in comparisons:
                actual = api.transform_to_field(grid, pairs[engine][index])
                item = field_difference(actual, expected, args.native_tolerance_mm)
                comparisons[engine][side] = item
                del actual
            del expected
    gated = [side["passed"] for item in comparisons.values() for side in item.values()
             if "passed" in side]
    return {"timing": "all checks excluded from whole-process timers",
            "scope": "last completed repeat per engine; complete maps, not only nonlinear fields",
            "reference_engine": reference, "saved_output_checks": checks,
            "complete_map_comparisons": comparisons if reference is not None else {},
            "explicit_tolerance_mm": args.native_tolerance_mm,
            "all_saved_output_checks_passed": all(all(c.values()) for c in checks.values()),
            "all_requested_numeric_gates_passed": all(gated) if gated else None}


def run_whole(args, config, case):
    engines = args.engine or ["python", "ants-default", "ants-1"]
    binary = str(Path(sys.executable).parent / "antsRegistration")
    report = {"scope": "whole fresh process: imports/startup, input reads, full FP64 registration, output writes; Python additionally performs built-in output QA",
              "process_measurement": "one fresh RUSAGE_CHILDREN runner per target; runner startup excluded from target wall time",
              "config_path": str(args.config), "config_sha256": digest(args.config),
              "warmup": "one excluded full fresh process per engine by default; caches may be warm",
              "warmup_runs_per_engine": args.warmups, "repeats_per_engine": args.repeats,
              "resolution": args.resolution, "case": case, "configuration": config,
              "requested_python_numba_threads": args.threads,
              "source": str(args.source), "source_sha256": source_hashes(args.source),
              "native_binary_sha256": digest(binary), "input_sha256": {s: digest(case[s]) for s in ("fixed", "moving")},
              "power_background_load": "not controlled; coordinated sequential runs", "runs": {e: [] for e in engines}}

    def run(engine, label):
        folder = args.output / engine / label
        folder.mkdir(parents=True)
        env = os.environ.copy()
        for key in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS"):
            env.pop(key, None)
        env["ANTS_RANDOM_SEED"] = str(config["random_seed"])
        if engine == "python":
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(folder),
                       "--source", str(args.source), "--resolution", args.resolution, "--threads", str(args.threads),
                       "--config", str(args.config)]
        else:
            if engine == "ants-1":
                env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
            elif engine == "ants-10":
                env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "10"
            command = native_command(config, case, folder, binary)
        print("start", engine, label, flush=True)
        stats_path = folder / "process.json"
        runner = [sys.executable, "-B", str(PROJECT / "validation/common/time_process.py"),
                  "--output", str(stats_path), "--timeout", str(args.timeout), "--", *command]
        with (folder / "console.log").open("w") as stream:
            process = subprocess.run(runner, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=args.timeout+30)
        if process.returncode:
            raise RuntimeError("Run failed; see " + str(folder / "console.log"))
        stats = json.loads(stats_path.read_text())
        seconds = stats["wall_seconds"]
        row = {"whole_process_seconds": seconds, "command": command,
               "peak_process_rss_bytes": stats["peak_process_rss_bytes"],
               "measurement": stats,
               "thread_environment": {k: v for k, v in env.items() if k.startswith("ITK_GLOBAL")}}
        if engine == "python":
            row["worker"] = json.loads((folder / "worker.json").read_text())
        else:
            log = (folder / "console.log").read_text()
            row["native_stage_seconds"] = [float(v) for v in re.findall(r"Elapsed time \(stage \d+\):\s*([\deE+.-]+)", log)]
            total = re.search(r"Total elapsed time:\s*([\deE+.-]+)", log)
            row["native_internal_seconds"] = float(total[1]) if total else None
        print("done", engine, label, format(seconds, ".6f"), "s", flush=True)
        return row

    for engine in engines:
        for i in range(args.warmups):
            report.setdefault("warmup_diagnostics", {}).setdefault(engine, []).append(run(engine, "warmup-" + str(i + 1)))
    for i in range(args.repeats):
        order = engines if i % 2 == 0 else list(reversed(engines))
        for engine in order:
            report["runs"][engine].append(run(engine, "repeat-" + str(i + 1)))
            write_json(args.output / "report.json", report)
    report["summary"] = {e: {"median_whole_process_seconds": statistics.median(r["whole_process_seconds"] for r in rows),
                              "minimum_whole_process_seconds": min(r["whole_process_seconds"] for r in rows),
                              "maximum_whole_process_seconds": max(r["whole_process_seconds"] for r in rows)}
                         for e, rows in report["runs"].items()}
    # Save timings before the optional, possibly memory-heavy validation. A
    # validation failure must leave its speed measurements and outputs visible.
    write_json(args.output / "report.json", report)
    if args.check_saved:
        report["saved_numeric_comparison"] = compare_whole_saved(args, engines, case)
    report["source_unchanged_during_run"] = source_hashes(args.source) == report["source_sha256"]
    report["timing_status"] = "completed"
    comparison = report.get("saved_numeric_comparison", {})
    numeric_passed = comparison.get("all_requested_numeric_gates_passed")
    report["numeric_status"] = ("passed_explicit_gate" if numeric_passed is True else
                                 "failed_explicit_gate" if numeric_passed is False else "not_gated")
    report["status"] = ("numeric_gate_failed" if numeric_passed is False else
                         "timing_and_numeric_gates_passed" if numeric_passed is True else
                         "timing_completed_numeric_not_gated")
    if not report["source_unchanged_during_run"]:
        report["status"] = "source_changed_during_run"
    write_json(args.output / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "src")
    parser.add_argument("--config", type=Path, default=PROJECT / "validation/configs/p00_dense.json",
                        help="Frozen dense MI/CC recipe; unsupported options are rejected")
    parser.add_argument("--resolution", choices=("small", "half", "native"), default="small")
    parser.add_argument("--mode", choices=("api", "whole"), default="api")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--engine", choices=("python", "ants-default", "ants-1", "ants-10"), action="append")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--compare-tolerance", type=float, default=0.0)
    parser.add_argument("--check-saved", action="store_true", help="after whole timings, compare complete saved maps with an original FP64 engine")
    parser.add_argument("--native-tolerance-mm", type=float, help="explicit complete-map gate for --check-saved; omitted means report errors without imposing a gate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = args.source.resolve()
    if not (args.source / "fast_ants/__init__.py").is_file():
        parser.error("--source must contain the fast_ants package")
    if (args.threads < 1 or args.repeats < 1 or args.warmups < 0 or args.compare_tolerance < 0 or
            (args.native_tolerance_mm is not None and args.native_tolerance_mm <= 0)):
        parser.error("threads/repeats must be positive; warmups and comparison tolerance nonnegative")
    args.config = args.config.resolve()
    config = json.loads(args.config.read_text())
    python_requested = (args.worker is not None or args.mode == "api" or args.engine is None or "python" in args.engine)
    if config["histogram_matching"] or config["masks"] is not None:
        parser.error("Histogram matching and masks are outside this benchmark's recorded recipes")
    if python_requested and (config["sampling"] != {"strategy": "None", "fraction": 1.0, "gradient_filter": False}
                             or config["winsorization"] is not None or config.get("write_inverse_warped", False)):
        parser.error("This strict comparison supports dense sampling with gradient filtering, winsorization, histogram matching and masks disabled; it must not silently approximate the stock SyN preset")
    case = case_for(args.resolution)
    if args.worker is not None:
        args.worker.mkdir(parents=True, exist_ok=True)
        worker(args, config, case)
        return
    if args.output is None:
        parser.error("--output is required; use a new directory for each run")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    args.output.mkdir(parents=True)
    if args.compare_with is not None:
        args.compare_with = args.compare_with.resolve()
    if args.mode == "api":
        run_api(args, config, case)
    else:
        run_whole(args, config, case)


if __name__ == "__main__":
    main()
