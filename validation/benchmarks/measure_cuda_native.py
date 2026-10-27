"""Fresh-process native CUDA/CPU FP32/ANTs float1 comparison and saved-output QA.

The exact public IXI002/IXI012 pair and brain-syn recipe match the Metal native
comparison. CUDA cold compilation, excluded full warmups, and formal repeats
are recorded separately. Historical Metal timings are not speed references.
"""

from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import statistics
import subprocess
import sys
import time

from validation.benchmarks import measure_fp32_pipeline as pipeline
from validation.benchmarks.measure_current_native import read_observation, stop_group
from validation.common.paths import PROJECT
from validation.common.runtime import digest, write_json


TASK = PROJECT / "setup/cuda-complete"
ENGINES = ("cuda", "cpu", "ants-float1")
FILES = ("IXI002-Guys-0828-T1.nii.gz", "IXI012-HH-1211-T1.nii.gz")


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes(source):
    return {str(path.relative_to(source)): digest(path)
            for path in sorted((source / "fast_ants").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".metal", ".cuh", ".json")}


def task_environment(root, threads, seed, source):
    """Keep every configurable runtime cache and scratch path in this run."""
    env = pipeline.thread_environment(threads, seed)
    paths = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache/xdg",
             "PIP_CACHE_DIR": "cache/pip", "CUPY_CACHE_DIR": "cache/cupy",
             "CUDA_CACHE_PATH": "cache/cuda", "NUMBA_CACHE_DIR": "cache/numba",
             "CONDA_PKGS_DIRS": "cache/conda", "CONDA_ENVS_PATH": "envs"}
    for name, relative in paths.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(TMP=env["TMPDIR"], TEMP=env["TMPDIR"], CONDA_REGISTER_ENVS="false",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(source))
    return env


def selected_case(directory):
    case = pipeline.selected_case("ixi002_to_ixi012", "native")
    recorded = json.loads((PROJECT / "validation/sources/downloaded-files.json").read_text())
    expected = {row["name"]: row for row in recorded["files"]}
    for side, name in zip(("fixed", "moving"), FILES):
        path = directory / name
        if path.stat().st_size != expected[name]["bytes"] or digest(path) != expected[name]["sha256"]:
            raise ValueError("Native MRI differs from pinned input: " + str(path))
        case[side] = str(path)
    return case


def command_for(args, config, case, engine, output):
    if engine == "ants-float1":
        return pipeline.float1_command(config, case, output, args.ants_registration)
    return [sys.executable, "-B", "-m", "fast_ants", "register", case["fixed"], case["moving"],
            "--output", str(output), "--backend", engine, "--precision", "fp32",
            "--config", str(args.config), "--threads", str(args.threads),
            "--geometry-policy", "itk_scanner", "--scaling-policy", "itk"]


def schedule(repeats, official_warmup=True):
    rows = [("cuda", "cold"), ("cuda", "warmup"), ("cpu", "warmup")]
    if official_warmup:
        rows.append(("ants-float1", "warmup"))
    for repeat in range(1, repeats + 1):
        order = ENGINES if repeat % 2 else tuple(reversed(ENGINES))
        rows.extend((engine, "repeat-" + str(repeat)) for engine in order)
    return rows


def tools_hashes():
    paths = [Path(__file__), PROJECT / "validation/common/time_process.py",
             PROJECT / "validation/common/runtime.py", PROJECT / "validation/common/paths.py",
             PROJECT / "validation/benchmarks/measure_current_native.py",
             PROJECT / "validation/benchmarks/measure_fp32_pipeline.py",
             PROJECT / "validation/benchmarks/measure_fp64_cpu.py",
             PROJECT / "validation/reference_tools/run_p00_baseline.py",
             PROJECT / "validation/quality/assess_saved_registration.py"]
    return {str(path): digest(path) for path in paths}


def integrity(provenance):
    return {"runtime_unchanged": source_hashes(Path(provenance["source"])) == provenance["source_sha256"],
            "inputs_unchanged": all(digest(path) == expected for path, expected in provenance["input_sha256"].items()),
            "files_unchanged": all(digest(path) == expected for path, expected in provenance["file_sha256"].items()),
            "tools_unchanged": tools_hashes() == provenance["tool_sha256"]}


def artifacts(output, engine):
    names = pipeline.ANTS_ARTIFACTS if engine == "ants-float1" else pipeline.OWN_ARTIFACTS
    result = {}
    for role, name in names.items():
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Required output is missing: " + str(path))
        result[role] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def run_one(args, config, report, engine, label):
    directory = args.output / engine / label
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "outputs"
    if engine == "ants-float1":
        output.mkdir()
    command = command_for(args, config, report["case"], engine, output)
    # Cold CUDA starts with empty CuPy, driver and Numba caches. Subsequent
    # warmup and repeats share a different cache populated by the full warmup.
    cache_root = directory if label == "cold" else args.output
    env = task_environment(cache_root, args.threads, config["random_seed"], args.source)
    stats = directory / "process.json"
    runner = [sys.executable, "-B", str(PROJECT / "validation/common/time_process.py"),
              "--output", str(stats), "--timeout", str(args.timeout), "--", *command]
    row = {"engine": engine, "label": label, "status": "running", "started_at_utc": utc(),
           "directory": str(directory), "output_directory": str(output), "command": command,
           "measurement_command": runner,
           "environment": {name: env[name] for name in (*pipeline.THREAD_KEYS, "NUMBA_NUM_THREADS",
               "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS",
               "ANTS_RANDOM_SEED", "CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR", "TMPDIR")}}
    report["runs"].append(row)
    write_json(args.output / "report.json", report)
    print(f"START {engine} {label}", flush=True)
    started = time.perf_counter()
    try:
        with (directory / "console.log").open("wb") as log:
            process = subprocess.Popen(runner, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=PROJECT, start_new_session=True)
            active = {"status": "running", "supervisor_pid": os.getpid(), "process_group_id": process.pid,
                      "runner_pid": process.pid, "engine": engine, "label": label, "command": command}
            write_json(args.output / "active-process.json", active)
            try:
                while True:
                    try:
                        returncode = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = time.perf_counter() - started
                        print(f"RUNNING {engine} {label} elapsed={elapsed:.0f}s", flush=True)
                        active["elapsed_seconds_diagnostic"] = elapsed
                        write_json(args.output / "active-process.json", active)
                        if elapsed > args.timeout + 30:
                            raise TimeoutError("Target exceeded timeout plus runner grace period")
                if returncode:
                    raise RuntimeError(f"{engine} {label} exited {returncode}; see {directory / 'console.log'}")
            except BaseException:
                stop_group(process)
                raise
            finally:
                active["status"] = "stopped"
                write_json(args.output / "active-process.json", active)
        measurement = json.loads(stats.read_text())
        if measurement["returncode"] != 0 or measurement["timed_out"]:
            raise RuntimeError("Unsuccessful target measurement: " + str(stats))
        row.update(measurement=measurement, whole_process_seconds=measurement["wall_seconds"],
                   peak_process_rss_bytes=measurement["peak_process_rss_bytes"],
                   artifacts=artifacts(output, engine), integrity=integrity(report["provenance"]))
        if not all(row["integrity"].values()):
            raise RuntimeError("Source, input, tool or binary changed during measurement")
        if engine != "ants-float1":
            status = json.loads((output / "run.json").read_text())
            if status.get("status") != "registration_completed_quality_metrics_recorded":
                raise RuntimeError("CLI did not report completed registration")
            row["cli_status"] = status
        row.update(status="completed", completed_at_utc=utc())
    except BaseException as error:
        row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed_outputs_preserved",
                   stopped_at_utc=utc(), failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(directory / "receipt.json", row)
        write_json(args.output / "report.json", report)
    print(f"DONE {engine} {label} seconds={row['whole_process_seconds']:.6f}", flush=True)


def summarize(report):
    summary = {}
    for engine in ENGINES:
        rows = [row for row in report["runs"] if row["engine"] == engine and row["label"].startswith("repeat-")]
        values = [row["whole_process_seconds"] for row in rows]
        summary[engine] = {"formal_repeats": len(values), "whole_process_seconds": values,
                           "median_seconds": statistics.median(values), "minimum_seconds": min(values),
                           "maximum_seconds": max(values),
                           "peak_process_rss_bytes": max(row["peak_process_rss_bytes"] for row in rows)}
    reference = summary["ants-float1"]["median_seconds"]
    for engine in ("cuda", "cpu"):
        summary[engine]["official_median_divided_by_backend_median"] = reference / summary[engine]["median_seconds"]
    return summary


def benchmark(args):
    config = json.loads(args.config.read_text())
    pipeline.validate_recipe(config)
    case = selected_case(args.data)
    import nibabel as nib
    headers = {side: list(nib.load(case[side]).shape) for side in ("fixed", "moving")}
    if any(shape != [256, 256, 150] for shape in headers.values()):
        raise ValueError("Expected the unchanged native 256x256x150 IXI pair")
    version = subprocess.check_output([str(args.ants_registration), "--version"], text=True,
                                      stderr=subprocess.STDOUT, timeout=15).strip()
    if "2.6.5" not in version:
        raise ValueError("Expected official ANTs 2.6.5: " + version)
    pinned = [args.config, args.ants_registration, Path(sys.executable),
              PROJECT / "validation/cases.json", pipeline.POLICY_PATH,
              PROJECT / "validation/sources/downloaded-files.json"]
    compressor=shutil.which('pigz')
    if compressor is not None:pinned.append(Path(compressor))
    pinned += sorted((args.ants_registration.parent.parent / "conda-meta").glob("*.json"))
    order = schedule(args.repeats, not args.skip_official_warmup)
    report = {"schema_version": 1, "status": "planned" if args.plan_only else "running",
        "started_at_utc": utc(), "case": case, "shape_xyz": headers, "recipe": config,
        "threads": args.threads, "repeats_per_engine": args.repeats,
        "provenance": {"source": str(args.source), "source_sha256": source_hashes(args.source),
            "input_sha256": {case[side]: digest(case[side]) for side in ("fixed", "moving")},
            "file_sha256": {str(path): digest(path) for path in pinned}, "tool_sha256": tools_hashes(),
            "ants_version": version},
        "environment": {"platform": platform.platform(), "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "nibabel", "cupy-cuda12x")},
            "cpu": read_observation(["lscpu"]), "gpu": read_observation(["nvidia-smi"]),
            "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES", "all visible devices")},
        "measurement_scope": "Fresh target process startup/imports, image reads, registration, GPU synchronization, built-in QA and final writes; runner startup excluded",
        "qa_timing_asymmetry": "Fast ANTs includes built-in host output QA; official ANTs omits that extra QA; no time subtraction",
        "logging": "Quiet Fast ANTs CLI; official ANTs verbose log retained, matching the historical native benchmark policy",
        "cold_scope": "Empty task-owned CuPy, driver and Numba caches; filesystem input/page caches and system GPU state are not reset",
        "warmup_policy": "One full excluded CUDA and CPU warmup; official warmup " + ("omitted explicitly" if args.skip_official_warmup else "included"),
        "statistics_scope": "Single native MRI direction, alternating serial formal order; no cross-platform Metal speed ratio",
        "retention": "All cold, warmup and formal outputs retained for inspection",
        "quality_status": "saved_output_assessment_pending", "planned_order": [
            {"engine": engine, "label": label, "command": command_for(args, config, case, engine, args.output / engine / label / "outputs")}
            for engine, label in order], "runs": []}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    if args.plan_only:
        print(str(args.output / "report.json"))
        return
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Benchmark received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for engine, label in order:
            run_one(args, config, report, engine, label)
        report.update(status="timing_completed_quality_assessment_pending", completed_at_utc=utc(),
                      summary=summarize(report), final_integrity=integrity(report["provenance"]))
        if not all(report["final_integrity"].values()):
            raise RuntimeError("Final provenance check failed")
        write_json(args.output / "report.json", report)
        print(json.dumps(report["summary"], indent=2), flush=True)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed_outputs_preserved",
                      stopped_at_utc=utc(), failure={"type": type(error).__name__, "message": str(error)})
        write_json(args.output / "report.json", report)
        raise


def assess(args):
    """Apply the existing CPU saved-output QA formulas to every formal run."""
    report = json.loads(args.assess_report.read_text())
    if report["status"] != "timing_completed_quality_assessment_pending":
        raise ValueError("A completed timing report is required")
    if not all(integrity(report["provenance"]).values()):
        raise ValueError("Benchmark provenance has changed")
    for row in report["runs"]:
        if row["status"] != "completed" or not all(row["integrity"].values()):
            raise ValueError("Incomplete or unverified run")
        directory = Path(row["directory"])
        if json.loads((directory / "receipt.json").read_text()) != row:
            raise ValueError("Run receipt differs from benchmark report")
        if json.loads((directory / "process.json").read_text()) != row["measurement"]:
            raise ValueError("Process measurement differs from run receipt")
        for item in row["artifacts"].values():
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Saved output changed: " + item["path"])
    from validation.benchmarks.measure_fp64_cpu import import_implementation
    api, _ = import_implementation(Path(report["provenance"]["source"]), args.threads)
    from validation.quality.assess_saved_registration import deformation_regularity
    import numpy as np
    fixed, moving = api.load_images([report["case"][side] for side in ("fixed", "moving")],
                                    geometry_policy="itk_scanner", scaling_policy="itk")
    fmin, fspan = float(fixed.data.min()), float(np.ptp(fixed.data))
    mmin, mspan = float(moving.data.min()), float(np.ptp(moving.data))
    if min(fspan, mspan) <= 0:
        raise ValueError("Nonconstant images required")
    mask = fixed.data > fmin + .05 * fspan
    target = (fixed.data[mask] - fmin) / fspan
    def similarity(transform):
        warped = api.apply_image(moving, fixed, transform).data
        values = (warped[mask] - mmin) / mspan
        x, y = target - target.mean(), values - values.mean()
        norm = np.sqrt(np.sum(x*x) * np.sum(y*y))
        points, truth = [np.asarray(report["case"][side + "_landmarks_lps_mm"])
                         for side in ("fixed", "moving")]
        error = np.linalg.norm(transform.map_points(points) - truth, axis=1)
        return {"foreground_ncc": None if norm == 0 else float(np.sum(x*y) / norm),
                "foreground_normalized_mse": float(np.mean((values-target)**2)),
                "foreground_points": int(mask.sum()), "finite": bool(np.isfinite(warped).all()),
                "landmark_error_mm": {"mean": float(error.mean()), "rms": float(np.sqrt(np.mean(error**2))),
                                      "maximum": float(error.max()), "per_point": error.tolist()}}
    assessment = {"status": "quality_metrics_running", "benchmark_report": str(args.assess_report),
        "benchmark_report_sha256": digest(args.assess_report), "before_identity": similarity(api.AffineTransform.identity()),
        "similarity_domain": "fixed intensity > minimum + 5% of fixed range; identical CPU FP64 sampler for every map",
        "landmark_scope": "Four coarse non-expert MRI annotations; exploratory, not clinical acceptance",
        "scope": "Saved products only, outside registration timers; finite fields, Jacobians and inverse consistency; no map-equality gate",
        "runs": []}
    args.output.mkdir(parents=True, exist_ok=False)
    for row in report["runs"]:
        if not row["label"].startswith("repeat-"):
            continue
        products = row["artifacts"]
        affine = api.load_affine(products["affine"]["path"])
        forward = api.load_field(products["forward_warp"]["path"])
        inverse = api.load_field(products["inverse_warp"]["path"])
        if not forward.grid.same_geometry(fixed.grid) or not inverse.grid.same_geometry(fixed.grid):
            raise ValueError("Saved field geometry differs from fixed grid")
        saved_warped = api.load_image(products["warped"]["path"])
        if not saved_warped.grid.same_geometry(fixed.grid):
            raise ValueError("Saved warped image geometry differs from fixed grid")
        result = {"engine": row["engine"], "label": row["label"], "artifacts": products,
                  "saved_warped_finite": bool(np.isfinite(saved_warped.data).all()),
                  "saved_warped_geometry_matches_fixed": True,
                  "metrics": similarity(api.CompositeTransform((forward, affine))),
                  "regularity": deformation_regularity(forward, inverse)}
        del saved_warped, forward, inverse
        assessment["runs"].append(result)
        write_json(args.output / "report.json", assessment)
        print(json.dumps(result, indent=2), flush=True)
    assessment.update(status="quality_metrics_recorded_not_full_acceptance", completed_at_utc=utc())
    write_json(args.output / "report.json", assessment)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=PROJECT / "src")
    parser.add_argument("--data", type=Path, default=TASK / "data")
    parser.add_argument("--config", type=Path, default=PROJECT / "src/fast_ants/presets/brain-syn.json")
    parser.add_argument("--ants-registration", type=Path, default=TASK / "reference-env/bin/antsRegistration")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--skip-official-warmup", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--assess-report", type=Path)
    args = parser.parse_args(argv)
    for name in ("output", "source", "data", "config", "ants_registration", "assess_report"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error("Use a new output directory")
    if not args.output.is_relative_to(PROJECT):
        parser.error("Output and caches must remain inside this project")
    if args.threads < 1 or args.repeats < 1 or args.timeout <= 0:
        parser.error("threads, repeats and timeout must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.assess_report is not None:
        assess(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
