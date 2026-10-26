"""Fresh-process CPU FP32 / Metal / official ANTs --float 1 benchmark.

All engines write an affine, forward/inverse nonlinear fields and a warped
image. Startup, imports, input reads, registration and output writes are timed.
The independent engines additionally perform their built-in host output QA;
ANTs does not perform that extra QA. This is disclosed, not subtracted.

Warmups run sequentially, followed by alternating interleaved repeat order.
Reports preserve all runs and receipts in a new directory. Timing completion
does not assert algorithm equivalence or registration-quality acceptance; use
registration_quality_policy.json and assess the saved outputs separately.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import codecs
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, environment, import_implementation, load_reference_image, native_command, stages_for, write_json

ENGINES = ("cpu", "metal", "ants-float1")
OWN_ARTIFACTS = {"affine": "affine.mat", "forward_warp": "warp.nii.gz",
                 "inverse_warp": "inverse-warp.nii.gz", "warped": "warped.nii.gz"}
ANTS_ARTIFACTS = {"affine": "reg_0GenericAffine.mat", "forward_warp": "reg_1Warp.nii.gz",
                  "inverse_warp": "reg_1InverseWarp.nii.gz", "warped": "warped.nii.gz"}
POLICY_PATH = PROJECT / "validation/configs/registration_quality_policy.json"
MANIFEST_PATH = PROJECT / "validation/cases.json"
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
MEASUREMENT_SCRIPTS = {"measure_fp32_pipeline.py": Path(__file__).resolve(),
                       "time_process.py": PROJECT / "validation/common/time_process.py"}


def source_hashes(source):
    return {str(p.relative_to(source)): digest(p)
            for p in sorted((source / "fast_ants").rglob("*"))
            if p.is_file() and p.suffix in (".py", ".metal")}


def validate_recipe(config):
    """Reject options that the independent stage builder would approximate."""
    if (config.get("dimensionality") != 3 or config["interpolation"] != "Linear"
            or config["sampling"] != {"strategy": "None", "fraction": 1., "gradient_filter": False}
            or config["winsorization"] is not None or config["histogram_matching"]
            or config["masks"] is not None or config.get("write_inverse_warped", False)):
        raise ValueError("Supported recipe is 3D, dense, unmasked, Linear, without histogram matching, winsorization or inverse-warped image")
    if [s["transform"].split("[")[0] for s in config["stages"]] != ["Rigid", "Affine", "SyN"]:
        raise ValueError("Expected Rigid, Affine, SyN stages")
    if [s["metric"] for s in config["stages"]] != ["MI", "MI", "CC"]:
        raise ValueError("Expected MI, MI, CC metrics")
    for stage in config["stages"]:
        if not (len(stage["iterations"]) == len(stage["shrink_factors"]) == len(stage["smoothing_sigmas"])):
            raise ValueError("Stage iteration, shrink and smoothing schedules must have matching lengths")
    if config.get("shrink_factor_policy", "explicit") not in ("explicit", "ants_physical_spacing"):
        raise ValueError("Unsupported shrink factor policy")


def selected_case(name, resolution, swap=False):
    cases = json.loads(MANIFEST_PATH.read_text())["cases"]
    matches = [case for case in cases if case["id"] == name]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous case: " + name)
    case = dict(matches[0])
    folder = {"small": "prepared", "half": "p11-half", "native": "public"}[resolution]
    for side in ("fixed", "moving"):
        path = PROJECT / case[side]
        if case["kind"] != "analytic":
            path = PROJECT / "validation/data" / folder / path.name
        case[side] = str(path.resolve())
    if swap:
        case["fixed"], case["moving"] = case["moving"], case["fixed"]
        # Store landmarks in the actual selected direction. The post-timing
        # adapter marks this explicitly so evaluators do not reverse twice.
        fixed, moving = "fixed_landmarks_lps_mm", "moving_landmarks_lps_mm"
        if fixed in case and moving in case:
            case[fixed], case[moving] = case[moving], case[fixed]
    return case


def float1_command(config, case, directory, binary):
    command = native_command(config, case, directory, str(binary))
    command[command.index("--float") + 1] = "1"
    return command


def worker_command(args, engine, directory):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(directory),
               "--source", str(args.source), "--case", args.case, "--resolution", args.resolution,
               "--config", str(args.config), "--threads", str(args.threads), "--engine", engine]
    if args.swap:
        command.append("--swap")
    if engine == "metal" and args.optimizations is not None:
        command += ["--optimizations", args.optimizations]
    verbose, log_every = bool(getattr(args, "verbose", False)), getattr(args, "log_every", 1)
    if verbose:
        command.append("--verbose")
    if verbose or log_every != 1:
        command += ["--log-every", str(log_every)]
    return command


def metal_logging_options(verbose, log_every):
    """Inspect the implementation, not the public *args/**kwargs wrapper."""
    if not verbose:
        return {}
    implementation = importlib.import_module("fast_ants.registration_metal").register_metal
    parameters = inspect.signature(implementation).parameters
    return {name: value for name, value in (("verbose", True), ("log_every", log_every))
            if name in parameters and parameters[name].kind in
            (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}


def worker_log(verbose, stage, message):
    if verbose:
        print(f"[fast_ants][{stage}] {time.strftime('%H:%M:%S')} | INFO | {message}", flush=True)


class _LiveLog:
    """Tail the regular child log without pipes, threads or split UTF-8 loss."""
    def __init__(self, path, terminal=None):
        self.reader = Path(path).open("rb")
        self.terminal = sys.stdout if terminal is None else terminal
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def drain(self, final=False):
        # Bound each live poll so a noisy orphan cannot hide runner failure
        # or prevent the parent timeout/KeyboardInterrupt cleanup checks.
        chunks = 0
        while final or chunks < 4:
            data = self.reader.read(64 * 1024)
            if not data:
                break
            chunks += 1
            text = self.decoder.decode(data)
            if text:
                self.terminal.write(text)
        if final:
            text = self.decoder.decode(b"", final=True)
            if text:
                self.terminal.write(text)
        self.terminal.flush()

    def close(self):
        self.reader.close()


def _terminate_process_group(process):
    """Kill surviving targets and reap the timing runner itself."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def thread_environment(threads, seed):
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_KEYS})
    env.update(NUMBA_NUM_THREADS=str(threads), ANTS_RANDOM_SEED=str(seed),
               ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads),
               ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS=str(threads))
    return env


def artifact_hashes(directory, engine):
    names = ANTS_ARTIFACTS if engine == "ants-float1" else OWN_ARTIFACTS
    return {name: digest(directory / name) for name in names.values()}


def make_receipt(directory, engine, command, provenance, measurement):
    """Hash the executed recipe and four required products after timing stops."""
    return {"schema_version": 1, "engine": engine, "command": list(command),
            "float_option": 1 if engine == "ants-float1" else None,
            "source": provenance["source"], "source_sha256": provenance["source_sha256"],
            "input_sha256": provenance["input_sha256"],
            "recipe_sha256": provenance["recipe_sha256"],
            "case_manifest_sha256": provenance["case_manifest_sha256"],
            "quality_policy_sha256": provenance["quality_policy_sha256"],
            "binary_sha256": provenance["binary_sha256"].get(engine),
            "python_binary_sha256": provenance["python_binary_sha256"],
            "threads": provenance["threads"], "artifacts": artifact_hashes(directory, engine),
            "metadata_artifact_sha256": {p.name: digest(p) for p in directory.glob("*.json")
                                         if p.name in ("worker.json", "configuration.json", "quality.json", "history.json")},
            "output_contract": ANTS_ARTIFACTS if engine == "ants-float1" else OWN_ARTIFACTS,
            "measurement": measurement,
            "quality_status": "not_assessed; timing/receipts are not registration acceptance"}


def worker(args, config, case):
    api, numba = import_implementation(args.source, args.threads)
    verbose, log_every = bool(getattr(args, "verbose", False)), getattr(args, "log_every", 1)
    started = time.perf_counter()
    read_started=time.perf_counter()
    if verbose:
        for side in ("fixed", "moving"):
            worker_log(True, "read", f"Loading {side}: {case[side]}")
    if hasattr(api,"load_images"):
        fixed,moving=api.load_images([case[side] for side in ("fixed","moving")],
                                     geometry_policy="itk_scanner",scaling_policy="itk")
        input_read_policy="bounded parallel library reads; unchanged FP64 host storage and ITK policies"
    else:
        fixed,moving=[load_reference_image(api,case[side]) for side in ("fixed","moving")]
        input_read_policy="sequential legacy reads"
    input_read_seconds=time.perf_counter()-read_started
    if verbose:
        for side, image in (("fixed", fixed), ("moving", moving)):
            worker_log(True, "read", f"Loaded {side}: shape_xyz={tuple(image.grid.shape_xyz)} | "
                       f"path={case[side]} | input batch elapsed={input_read_seconds:.3f}s")
    linear, syn = stages_for(api, config, fixed.grid)
    options = dict(linear_stages=linear, syn_stage=syn, seed=config["random_seed"],
                   initialization_feature=config["initialization_feature"], output_directory=args.worker)
    engine = args.engine[0]
    mx = None
    core_logging_options = {}
    if engine == "metal":
        from fast_ants.backends.metal.runtime import mx
        core_logging_options = metal_logging_options(verbose, log_every)
        options.update(core_logging_options)
        if args.optimizations is not None:
            options["optimizations"] = tuple(x for x in args.optimizations.split(",") if x)
        mx.synchronize(mx.gpu)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
    if verbose:
        engine_started = time.perf_counter()
        worker_log(True, engine, f"Starting {engine} registration")
    result = (api.register32 if engine == "cpu" else api.register_metal)(fixed, moving, **options)
    if mx is not None:
        mx.synchronize(mx.gpu)
    elapsed = time.perf_counter() - started
    if verbose:
        worker_log(True, engine, f"Completed {engine} registration | engine elapsed="
                   f"{time.perf_counter()-engine_started:.3f}s | read/register/QA/write elapsed={elapsed:.3f}s")
    write_json(args.worker / "worker.json", {
        "engine": engine, "load_register_qa_write_seconds_diagnostic": elapsed,
        "verbose": verbose, "log_every": log_every,
        "core_logging_options": core_logging_options,
        "input_read_policy":input_read_policy,
        "input_read_seconds_diagnostic_not_benchmark":input_read_seconds,
        "built_in_host_output_qa_included": True,
        "gpu_synchronized_before_worker_completion": mx is not None,
        "gpu_peak_active_bytes": mx.get_peak_memory() if mx is not None and hasattr(mx, "get_peak_memory") else None,
        "gpu_device": mx.device_info() if mx is not None else None,
        "environment": environment(numba), "configuration": result.configuration,
        "quality": result.quality,
        "shape_xyz": list(fixed.grid.shape_xyz),
        "input_scaling_policies": {"fixed": fixed.metadata.get("scaling_policy", "legacy_nifti"),
                                   "moving": moving.metadata.get("scaling_policy", "legacy_nifti")},
        "history_counts_diagnostic_only": [{"kind": row["kind"], "level": row["level"],
                                              "iterations": len(row["iterations"])} for row in result.history]})


def timed_run(args, engine, label, config, case, provenance):
    directory = args.output / engine / label
    directory.mkdir(parents=True)
    command = (float1_command(config, case, directory, args.ants_registration) if engine == "ants-float1"
               else worker_command(args, engine, directory))
    env = thread_environment(args.threads, config["random_seed"])
    verbose, log_every = bool(getattr(args, "verbose", False)), getattr(args, "log_every", 1)
    if verbose:
        env["PYTHONUNBUFFERED"] = "1"
    stats_path = directory / "process.json"
    runner = [sys.executable, "-B", str(PROJECT / "validation/common/time_process.py"),
              "--output", str(stats_path), "--timeout", str(args.timeout), "--", *command]
    print(f"start {engine} {label} ({args.case}, {args.resolution}, threads={args.threads})", flush=True)
    started = time.perf_counter()
    with (directory / "console.log").open("wb") as stream:
        live = _LiveLog(directory / "console.log") if verbose else None
        try:
            with subprocess.Popen(runner, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    heartbeat = started
                    while True:
                        try:
                            returncode = process.wait(timeout=.05 if verbose else 15)
                            # A crashed runner can leave its target alive with
                            # the log open. No pipe reader waits for that EOF.
                            if returncode:
                                _terminate_process_group(process)
                            break
                        except subprocess.TimeoutExpired:
                            if live is not None:
                                live.drain()
                            now = time.perf_counter()
                            elapsed = now - started
                            if not verbose or now-heartbeat >= 15:
                                print(f"running {engine} {label}: {elapsed:.0f}s; log={directory / 'console.log'}", flush=True)
                                heartbeat = now
                            if elapsed > args.timeout + 30:
                                raise TimeoutError("Timing runner exceeded target timeout plus 30 seconds")
                except BaseException:
                    _terminate_process_group(process)
                    raise
        finally:
            if live is not None:
                try:
                    live.drain(final=True)
                finally:
                    live.close()
    measurement = json.loads(stats_path.read_text()) if stats_path.exists() else {"returncode": returncode}
    row = {"engine": engine, "label": label, "directory": str(directory), "command": command,
           "verbose": verbose, "log_every": log_every,
           "measurement": measurement, "thread_environment": {k: env[k] for k in (*THREAD_KEYS,
               "NUMBA_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS", "ANTS_RANDOM_SEED")},
           "whole_process_seconds": measurement.get("wall_seconds"),
           "peak_process_rss_bytes": measurement.get("peak_process_rss_bytes"),
           "returncode": returncode, "status": "failed" if returncode else "completed"}
    write_json(directory / "run.json", row)
    if returncode:
        return row
    receipt = make_receipt(directory, engine, command, provenance, measurement)
    receipt["source_unchanged_during_run"] = source_hashes(args.source) == provenance["source_sha256"]
    receipt["inputs_unchanged_during_run"] = all(digest(path) == value for path, value in provenance["input_sha256"].items())
    receipt["recipe_unchanged_during_run"] = digest(args.config) == provenance["recipe_sha256"]
    receipt["binary_unchanged_during_run"] = (digest(args.ants_registration) == provenance["binary_sha256"][engine]
                                            if engine == "ants-float1" else digest(sys.executable) == provenance["python_binary_sha256"])
    write_json(directory / "receipt.json", receipt)
    if engine == "ants-float1":
        write_json(directory / "reference.json", receipt)
        log = (directory / "console.log").read_text()
        row["native_stage_seconds_diagnostic"] = [float(v) for v in re.findall(r"Elapsed time \(stage \d+\):\s*([\deE+.-]+)", log)]
    else:
        row["worker"] = json.loads((directory / "worker.json").read_text())
    row.update(receipt_path=str(directory / "receipt.json"), artifact_sha256=receipt["artifacts"],
               integrity_passed=all(receipt[k] for k in ("source_unchanged_during_run", "inputs_unchanged_during_run",
                                                        "recipe_unchanged_during_run", "binary_unchanged_during_run")))
    if not row["integrity_passed"]:
        row["status"] = "integrity_failed"
    write_json(directory / "run.json", row)
    print(f"done {engine} {label}: {row['whole_process_seconds']:.3f}s; {row['status']}", flush=True)
    return row


def run_benchmark(args, config, case):
    engines = args.engine or list(ENGINES)
    provenance = {"source": str(args.source), "source_sha256": source_hashes(args.source),
                  "input_sha256": {case[side]: digest(case[side]) for side in ("fixed", "moving")},
                  "recipe_sha256": digest(args.config), "case_manifest_sha256": digest(MANIFEST_PATH),
                  "quality_policy_sha256": digest(POLICY_PATH), "threads": args.threads,
                  "python_binary_sha256": digest(sys.executable),
                  "binary_sha256": {engine: digest(args.ants_registration if engine == "ants-float1" else sys.executable)
                                    for engine in engines}}
    report = {"schema_version": 1, "status": "running", **provenance,
              "verbose": bool(getattr(args, "verbose", False)), "log_every": getattr(args, "log_every", 1),
              "logging_scope": "opt-in worker phases and supported Metal core; live tee can affect timings; native command output unchanged",
              "scope": "whole fresh target process: startup/imports, reads, full registration and output writes; timing runner startup excluded",
              "qa_timing_asymmetry": "CPU FP32 and Metal include built-in native CPU FP64 output QA; official float1 does not run that extra QA; no subtraction",
              "measurement_method": "fresh time_process.py runner and RUSAGE_CHILDREN per target; macOS process RSS includes shared CPU/GPU address space and is not a standalone device-memory measurement",
              "quality_policy": json.loads(POLICY_PATH.read_text()), "recipe": config, "config_path": str(args.config),
              "case": case, "resolution": args.resolution, "swap": args.swap,
              "annotations_already_directional": True,
              "platform": platform.platform(), "python": sys.version,
              "measurement_script_sha256": {name: digest(path) for name, path in MEASUREMENT_SCRIPTS.items()},
              "engines": engines, "warmup_runs_per_engine": args.warmups, "repeats_per_engine": args.repeats,
              "execution_order": [], "warmup_diagnostics": {e: [] for e in engines}, "runs": {e: [] for e in engines},
              "thread_policy": "same requested Numba/ITK threads; BLAS pools fixed to 1; Metal GPU parallelism separately determined by backend",
              "power_background_load": "not controlled; sequential coordinated runs; disk/JIT/driver caches may be warm",
              "output_contract": {e: ANTS_ARTIFACTS if e == "ants-float1" else OWN_ARTIFACTS for e in engines},
              "quality_status": "not_assessed; algorithm logic, stability and final quality need saved-output assessment; map and iteration equality are not acceptance gates"}
    path = args.output / "report.json"
    write_json(path, report)
    try:
        schedule = [(e, "warmup-" + str(i + 1), "warmup_diagnostics") for e in engines for i in range(args.warmups)]
        schedule += [(e, "repeat-" + str(i + 1), "runs") for i in range(args.repeats)
                     for e in (engines if i % 2 == 0 else list(reversed(engines)))]
        for engine, label, group in schedule:
            report["execution_order"].append({"engine": engine, "label": label})
            write_json(path, report)
            row = timed_run(args, engine, label, config, case, provenance)
            report[group][engine].append(row)
            write_json(path, report)
            if row["status"] != "completed":
                raise RuntimeError(f"{engine} {label} {row['status']}; inspect {row['directory']}")
        report["summary"] = {engine: {
            "median_whole_process_seconds": statistics.median(r["whole_process_seconds"] for r in rows),
            "minimum_whole_process_seconds": min(r["whole_process_seconds"] for r in rows),
            "maximum_whole_process_seconds": max(r["whole_process_seconds"] for r in rows),
            "maximum_peak_process_rss_bytes": max(r["peak_process_rss_bytes"] for r in rows)}
            for engine, rows in report["runs"].items()}
        if "ants-float1" in report["summary"]:
            baseline = report["summary"]["ants-float1"]["median_whole_process_seconds"]
            for engine in engines:
                report["summary"][engine]["official_float1_time_divided_by_engine_time"] = baseline / report["summary"][engine]["median_whole_process_seconds"]
        report["source_unchanged_during_run"] = source_hashes(args.source) == provenance["source_sha256"]
        report["status"] = "timing_completed_quality_assessment_pending"
        if not report["source_unchanged_during_run"]:
            report["status"] = "integrity_failed"
        write_json(path, report)
        print(json.dumps(report["summary"], indent=2), flush=True)
    except BaseException as error:
        report.update(status="failed_outputs_preserved", failure={"type": type(error).__name__, "message": str(error)})
        write_json(path, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "src")
    parser.add_argument("--case", default="ixi002_to_ixi012")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--resolution", choices=("small", "half", "native"), default="small")
    parser.add_argument("--config", type=Path, default=PROJECT / "validation/configs/p00_dense.json")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engine", choices=ENGINES, action="append", help="Repeat to select engines; default is all three")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--optimizations", help="Metal comma-separated flags; omitted uses source defaults")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream worker/native logs live and retain console.log; logging can affect timing")
    parser.add_argument("--log-every", type=int, default=1, help="Positive iteration log interval for supported Metal sources (default: 1)")
    parser.add_argument("--ants-registration", type=Path, default=Path(sys.executable).parent / "antsRegistration")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.repeats < 1 or args.warmups < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.log_every < 1:
        parser.error("threads/repeats/timeout/log-every must be positive; warmups must be nonnegative")
    if args.engine and len(args.engine) != len(set(args.engine)):
        parser.error("Each engine may be selected only once")
    if args.worker is not None:
        if args.engine not in (["cpu"], ["metal"]):
            parser.error("Worker requires exactly one independent engine")
    elif args.output is None:
        parser.error("--output is required; use a new directory")
    for name in ("source", "config", "ants_registration", "output", "worker"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not (args.source / "fast_ants/__init__.py").is_file():
        raise ValueError("--source must contain the fast_ants package")
    config = json.loads(args.config.read_text())
    validate_recipe(config)
    case = selected_case(args.case, args.resolution, args.swap)
    if args.worker is not None:
        args.worker.mkdir(parents=True, exist_ok=True)
        worker(args, config, case)
        return
    if args.engine is None or "ants-float1" in args.engine:
        if not args.ants_registration.is_file():
            found = shutil.which("antsRegistration")
            if found is None:
                raise FileNotFoundError("antsRegistration not found; select the ANTs environment or pass --ants-registration")
            args.ants_registration = Path(found).resolve()
    # Check all inputs before creating a new immutable report directory.
    for side in ("fixed", "moving"):
        if not Path(case[side]).is_file():
            raise FileNotFoundError(case[side])
    args.output.mkdir(parents=True)
    run_benchmark(args, config, case)


if __name__ == "__main__":
    main()
