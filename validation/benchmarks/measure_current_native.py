"""One serial, fresh five-engine native-resolution comparison of current code.

The existing CPU FP64 and CPU FP32/Metal workers perform registration. The
existing time_process runner measures each complete target process. This tool
adds provenance, one excluded warmup per Python backend, bounded retention,
power observations, and precision-matched official speed ratios.
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
import subprocess
import sys
import time

from validation.benchmarks import measure_fp32_pipeline as pipeline
from validation.benchmarks import measure_fp64_cpu as fp64
from validation.common.paths import PROJECT
from validation.common.runtime import digest, write_json


ORDER = (("cpu-fp64", "warmup"), ("cpu-fp64", "repeat-1"),
         ("ants-float0", "repeat-1"), ("cpu-fp32", "warmup"),
         ("cpu-fp32", "repeat-1"), ("metal-fp32", "warmup"),
         ("metal-fp32", "repeat-1"), ("ants-float1", "repeat-1"))
OWN = {"affine": "affine.mat", "forward_warp": "warp.nii.gz",
       "inverse_warp": "inverse-warp.nii.gz", "warped": "warped.nii.gz"}
NATIVE = {"affine": "reg_0GenericAffine.mat", "forward_warp": "reg_1Warp.nii.gz",
          "inverse_warp": "reg_1InverseWarp.nii.gz", "warped": "warped.nii.gz"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def runtime_hashes(source):
    return {str(path.relative_to(source)): digest(path)
            for path in sorted((source / "fast_ants").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".metal", ".json")
            and "__pycache__" not in path.parts}


def read_observation(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "unavailable": str(error)}


def power_observation():
    return {"timestamp_utc": utc(), "battery": read_observation(["/usr/bin/pmset", "-g", "batt"]),
            "thermal": read_observation(["/usr/bin/pmset", "-g", "therm"]),
            "scope": "Read-only observations; no thermal/power/background control and no settings changed"}


def command_for(args, config, case, engine, directory):
    if engine.startswith("ants-"):
        command = fp64.native_command(config, case, directory, str(args.ants_registration))
        command[command.index("--float") + 1] = "0" if engine == "ants-float0" else "1"
        return command
    script = (PROJECT / "validation/benchmarks/measure_fp64_cpu.py" if engine == "cpu-fp64"
              else PROJECT / "validation/benchmarks/measure_fp32_pipeline.py")
    command = [sys.executable, "-B", str(script), "--worker", str(directory),
               "--source", str(args.source), "--resolution", "native", "--threads", str(args.threads),
               "--config", str(args.config)]
    if engine != "cpu-fp64":
        command += ["--case", "ixi002_to_ixi012", "--engine", "metal" if engine == "metal-fp32" else "cpu"]
    return command


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def retain_artifacts(directory, engine, warmup):
    names = NATIVE if engine.startswith("ants-") else OWN
    artifacts = {}
    for role, name in names.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Required output is missing: " + str(path))
        artifacts[role] = {"path": str(path), "bytes": path.stat().st_size,
                           "sha256": digest(path), "available": True}
    discarded = []
    keep = {directory / name for name in names.values()}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        volume = path.name.endswith((".nii", ".nii.gz", ".npz"))
        if volume and (warmup or path not in keep):
            discarded.append({"path": str(path), "bytes": path.stat().st_size,
                              "sha256": digest(path), "available": False,
                              "reason": "excluded warmup output" if warmup else "non-core duplicate/intermediate volume"})
            path.unlink()
    if warmup:
        for record in artifacts.values():
            record["available"] = Path(record["path"]).exists()
        # Small affine files are unnecessary after their receipt has been saved.
        for record in artifacts.values():
            if record["available"]:
                Path(record["path"]).unlink()
                record["available"] = False
    return artifacts, discarded


def integrity(args, provenance):
    return {"runtime_unchanged": runtime_hashes(args.source) == provenance["runtime_sha256"],
            "inputs_unchanged": all(digest(path) == value for path, value in provenance["input_sha256"].items()),
            "recipe_unchanged": digest(args.config) == provenance["recipe_sha256"],
            "python_binary_unchanged": digest(sys.executable) == provenance["python_binary_sha256"],
            "ants_binary_unchanged": digest(args.ants_registration) == provenance["ants_binary_sha256"],
            "tools_unchanged": all(digest(PROJECT / path) == value for path, value in provenance["tool_sha256"].items()),
            "case_manifest_unchanged": digest(PROJECT / "validation/cases.json") == provenance["case_manifest_sha256"]}


def run_one(args, config, case, report, engine, label):
    directory = args.output / engine / label
    directory.mkdir(parents=True, exist_ok=False)
    command = command_for(args, config, case, engine, directory)
    env = pipeline.thread_environment(args.threads, config["random_seed"])
    env["PYTHONUNBUFFERED"] = "1"
    stats_path = directory / "process.json"
    runner = [sys.executable, "-B", str(PROJECT / "validation/common/time_process.py"),
              "--output", str(stats_path), "--timeout", str(args.timeout), "--", *command]
    row = {"engine": engine, "label": label, "status": "running", "started_at_utc": utc(),
           "directory": str(directory), "command": command, "measurement_command": runner,
           "thread_environment": {key: env[key] for key in (*pipeline.THREAD_KEYS, "NUMBA_NUM_THREADS",
               "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS", "ANTS_RANDOM_SEED")},
           "power_before": power_observation()}
    report["execution"].append(row)
    write_json(args.output / "report.json", report)
    print(f"START {engine} {label}", flush=True)
    started = time.perf_counter()
    with (directory / "console.log").open("wb") as log:
        process = subprocess.Popen(runner, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active = {"supervisor_pid": os.getpid(), "process_group_id": process.pid, "runner_pid": process.pid,
                  "engine": engine, "label": label, "command": command, "status": "running"}
        write_json(args.output / "active-process.json", active)
        try:
            while True:
                try:
                    returncode = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.perf_counter() - started
                    active["supervisor_elapsed_seconds_diagnostic"] = elapsed
                    write_json(args.output / "active-process.json", active)
                    print(f"RUNNING {engine} {label} elapsed={elapsed:.0f}s log={directory / 'console.log'}", flush=True)
                    if elapsed > args.timeout + 30:
                        raise TimeoutError("Target exceeded its timeout and runner grace period")
            if returncode:
                stop_group(process)
                raise RuntimeError(f"{engine} {label} exited {returncode}; see {directory / 'console.log'}")
        except BaseException:
            stop_group(process)
            raise
        finally:
            active["status"] = "stopped"
            write_json(args.output / "active-process.json", active)
    measurement = json.loads(stats_path.read_text())
    if measurement["returncode"] != 0 or measurement["timed_out"]:
        raise RuntimeError("Unsuccessful target measurement: " + str(stats_path))
    row.update(measurement=measurement, whole_process_seconds=measurement["wall_seconds"],
               peak_process_rss_bytes=measurement["peak_process_rss_bytes"], power_after=power_observation(),
               completed_at_utc=utc(), integrity=integrity(args, report["provenance"]))
    if not all(row["integrity"].values()):
        row["status"] = "integrity_failed"
        write_json(directory / "receipt.json", row)
        raise RuntimeError("Provenance changed during " + engine)
    if (directory / "worker.json").exists():
        row["worker"] = json.loads((directory / "worker.json").read_text())
    artifacts, discarded = retain_artifacts(directory, engine, label == "warmup")
    row.update(status="completed", artifacts=artifacts, discarded_outputs=discarded)
    write_json(directory / "receipt.json", row)
    write_json(args.output / "report.json", report)
    print(f"DONE {engine} {label} seconds={row['whole_process_seconds']:.6f} rss_bytes={row['peak_process_rss_bytes']}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "validation/reports/current-native-20261025")
    parser.add_argument("--source", type=Path, default=PROJECT / "src")
    parser.add_argument("--config", type=Path, default=PROJECT / "src/fast_ants/presets/brain-syn.json")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ants-registration", type=Path, default=Path(sys.executable).parent / "antsRegistration")
    args = parser.parse_args(argv)
    for name in ("output", "source", "config", "ants_registration"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.threads < 1 or not args.timeout > 0:
        parser.error("threads and timeout must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text())
    pipeline.validate_recipe(config)
    case = pipeline.selected_case("ixi002_to_ixi012", "native")
    for side in ("fixed", "moving"):
        if not Path(case[side]).is_file():
            raise FileNotFoundError(case[side])
    if not args.ants_registration.is_file():
        raise FileNotFoundError(args.ants_registration)
    # Header-only inspection, outside all target measurements.
    import nibabel as nib
    headers = {side: {"shape_xyz": list(nib.load(case[side]).shape),
                      "sform_code": int(nib.load(case[side]).header["sform_code"])} for side in ("fixed", "moving")}
    version = subprocess.check_output([str(args.ants_registration), "--version"], text=True,
                                      stderr=subprocess.STDOUT, timeout=15).strip()
    scripts = [Path(__file__), PROJECT / "validation/benchmarks/measure_fp64_cpu.py",
               PROJECT / "validation/benchmarks/measure_fp32_pipeline.py", PROJECT / "validation/common/time_process.py",
               PROJECT / "validation/reference_tools/run_p00_baseline.py", PROJECT / "validation/common/runtime.py",
               PROJECT / "validation/common/paths.py", PROJECT / "validation/common/p00_common.py"]
    report = {"schema_version": 1, "status": "running", "started_at_utc": utc(),
        "scope": "One current native-resolution MRI direction; five engines, one formal run each, no old timings reused",
        "case": case, "input_headers": headers, "recipe": config, "threads": args.threads,
        "provenance": {"runtime_root": str(args.source), "runtime_sha256": runtime_hashes(args.source),
            "input_sha256": {case[side]: digest(case[side]) for side in ("fixed", "moving")},
            "recipe_path": str(args.config), "recipe_sha256": digest(args.config),
            "case_manifest_sha256": digest(PROJECT / "validation/cases.json"),
            "python_binary": sys.executable, "python_binary_sha256": digest(sys.executable),
            "ants_binary": str(args.ants_registration), "ants_binary_sha256": digest(args.ants_registration),
            "ants_version": version, "tool_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in scripts}},
        "environment": {"platform": platform.platform(), "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "nibabel", "mlx", "mlx-metal")},
            "hardware": read_observation(["/usr/sbin/sysctl", "-n", "hw.model", "hw.ncpu"])},
        "measurement_method": "Fresh common/time_process.py and RUSAGE_CHILDREN runner per target; runner startup excluded",
        "measurement_scope": "Target startup/imports, input reads, complete registration, GPU sync where applicable, built-in QA and writes included",
        "qa_timing_asymmetry": "Python backends include built-in output QA; official ANTs does not run that extra QA; no subtraction",
        "input_policy": "Python itk_scanner geometry and itk scaling, official native ITK reader; native physical-spacing shrink schedule",
        "warmup_policy": "One full excluded run immediately before each Python backend; no extra full official warmup; inputs read by preceding Python runs; --version probed before schedule",
        "logging": "Quiet Python FP32/Metal; existing sparse FP64 worker progress and native ANTs verbose log retained",
        "statistics_scope": "One measurement per engine; fixed serial order, not counterbalanced or a stable population speed estimate",
        "power_background_load": "Observed where available, not controlled; no system settings changed",
        "retention": "Warmup large products hashed then deleted; formal four core artifacts retained, no duplicate full-volume NPZ/maps",
        "planned_order": [{"engine": engine, "label": label} for engine, label in ORDER],
        "execution": [], "quality_status": "timing_only_saved_output_assessment_pending"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "source-manifest.json", {"runtime_root": str(args.source),
        "scope": "Current runtime source hashes; no legacy frozen checkpoint is used or retained",
        "file_count": len(report["provenance"]["runtime_sha256"]),
        "sha256": report["provenance"]["runtime_sha256"]})
    write_json(args.output / "report.json", report)
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Supervisor received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for engine, label in ORDER:
            run_one(args, config, case, report, engine, label)
        rows = {row["engine"]: row for row in report["execution"] if row["label"] == "repeat-1"}
        summary = {engine: {"whole_process_seconds": row["whole_process_seconds"],
                           "peak_process_rss_bytes": row["peak_process_rss_bytes"],
                           "formal_repeats": 1, "directory": row["directory"]} for engine, row in rows.items()}
        for own, reference in (("cpu-fp64", "ants-float0"), ("cpu-fp32", "ants-float1"), ("metal-fp32", "ants-float1")):
            official, actual = summary[reference]["whole_process_seconds"], summary[own]["whole_process_seconds"]
            summary[own].update(official_reference=reference, official_seconds_divided_by_backend_seconds=official / actual,
                                wall_time_reduction_percent=100 * (1 - actual / official))
        report.update(status="timing_completed_quality_assessment_pending", completed_at_utc=utc(), summary=summary,
                      final_integrity=integrity(args, report["provenance"]))
        if not all(report["final_integrity"].values()):
            raise RuntimeError("Final provenance check failed")
        write_json(args.output / "report.json", report)
        write_json(args.output / "speedups.json", {"scope": report["scope"], "statistics_scope": report["statistics_scope"],
            "qa_timing_asymmetry": report["qa_timing_asymmetry"], "summary": summary})
        table = ["| Engine | Whole process (s) | Official / backend |", "| --- | ---: | ---: |"]
        for engine in ("ants-float0", "cpu-fp64", "ants-float1", "cpu-fp32", "metal-fp32"):
            row = summary[engine];ratio = row.get("official_seconds_divided_by_backend_seconds")
            table.append(f"| {engine} | {row['whole_process_seconds']:.6f} | " + (f"{ratio:.4f}x" if ratio is not None else "reference") + " |")
        (args.output / "speedups.md").write_text("\n".join(table) + "\n")
        print("\n".join(table), flush=True)
        return 0
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed_outputs_preserved",
                      stopped_at_utc=utc(), failure={"type": type(error).__name__, "message": str(error)})
        write_json(args.output / "report.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
