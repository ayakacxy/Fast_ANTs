"""Record original ANTs FP64/float-mode numerical baselines, not speed claims."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

from validation.common.p00_common import ROOT, PROJECT, difference, sample_field_lps, sha256, write_json


def run(command, log, env, timeout):
    started = time.perf_counter()
    with log.open("w") as stream:
        stream.write(shlex.join([str(x) for x in command]) + "\n")
        stream.flush()
        with subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT) as process:
            try:
                while process.poll() is None:
                    elapsed = time.perf_counter() - started
                    if elapsed > timeout:
                        process.kill()
                        process.wait()
                        raise TimeoutError(f"Command timed out after {timeout}s; see {log}")
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        print(f"  running {log.name}: {elapsed + 15:.0f}s", flush=True)
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
            if process.returncode:
                raise RuntimeError(f"Command failed ({process.returncode}); see {log}")
    return time.perf_counter() - started


def command_for(config, case, floating, out, binaries):
    fixed, moving = [str(PROJECT / case[k]) for k in ("fixed", "moving")]
    command = [binaries["antsRegistration"], "--dimensionality", "3", "--float", str(floating),
               "--output", f"[{out / 'reg_'},{out / 'warped.nii.gz'}]",
               "--interpolation", config["interpolation"], "--use-histogram-matching", "0",
               "--collapse-output-transforms", "1", "--initialize-transforms-per-stage", "0",
               "--initial-moving-transform", f"[{fixed},{moving},{config['initialization_feature']}]",
               "--random-seed", str(config["random_seed"]), "--verbose", "1"]
    for stage in config["stages"]:
        iterations = "x".join(map(str, stage["iterations"]))
        convergence = config["convergence"]
        command += ["--transform", stage["transform"], "--metric",
                    f"{stage['metric']}[{fixed},{moving},1,{stage['bins_or_radius']},None,1,0]",
                    "--convergence", f"[{iterations},{convergence['threshold']},{convergence['window']}]",
                    "--shrink-factors", "x".join(map(str, stage["shrink_factors"])),
                    "--smoothing-sigmas", "x".join(map(str, stage["smoothing_sigmas"])) + stage["smoothing_units"]]
    return command


def evaluate(case, out):
    fixed = nib.load(PROJECT / case["fixed"])
    warped = nib.load(out / "warped.nii.gz")
    reapplied = nib.load(out / "reapplied.nii.gz")
    composite = nib.load(out / "composite-field.nii.gz")
    for image in (warped, reapplied, composite):
        if image.shape[:3] != fixed.shape or not np.allclose(image.affine, fixed.affine, atol=1e-5):
            raise RuntimeError(f"Output geometry mismatch: {out}")
        if not np.isfinite(image.get_fdata()).all():
            raise RuntimeError(f"Non-finite output: {out}")
    values = warped.get_fdata()
    fixed_values = fixed.get_fdata()
    points = np.array(case["fixed_landmarks_lps_mm"])
    expected = np.array(case["moving_landmarks_lps_mm"])
    predicted = points + sample_field_lps(out / "composite-field.nii.gz", points)
    tre = np.linalg.norm(predicted - expected, axis=1)
    return {"finite_outputs": True, "geometry_matches_fixed": True,
            "reapplication_difference": difference(values, reapplied.get_fdata()),
            "warped_vs_fixed_rmse": float(np.sqrt(np.mean((values - fixed_values) ** 2))),
            "fixed_to_moving_point_errors_mm": tre.tolist(),
            "point_error_mean_mm": float(tre.mean()),
            "predicted_moving_landmarks_lps_mm": predicted.tolist(),
            "point_metric_scope": "analytic truth" if case["kind"] == "analytic" else "four coarse non-expert landmarks; exploratory only"}


def execute(config, case, floating, out, binaries, env):
    out.mkdir(parents=True, exist_ok=True)
    registration = command_for(config, case, floating, out, binaries)
    commands = [registration]
    print(f"start {case['id']} --float {floating} -> {out.name}", flush=True)
    elapsed = {"registration_wall_seconds": run(registration, out / "registration.log", env,
                                                 config["timeout_per_command_seconds"])}
    transform_args = ["-t", str(out / "reg_1Warp.nii.gz"), "-t", str(out / "reg_0GenericAffine.mat")]
    apply = [binaries["antsApplyTransforms"], "-d", "3", "--float", str(floating),
             "-i", str(PROJECT / case["moving"]), "-r", str(PROJECT / case["fixed"]), "-n", "Linear"]
    for filename, output_arg in [("reapplied", str(out / "reapplied.nii.gz")),
                                  ("composite", f"[{out / 'composite-field.nii.gz'},1]")]:
        command = apply + ["-o", output_arg] + transform_args
        commands.append(command)
        elapsed[filename + "_wall_seconds"] = run(command, out / (filename + ".log"), env, 60)
    result = {"case": case["id"], "float_option": floating,
              "precision_description": "ANTs mixed float mode" if floating else "ANTs double mode",
              "commands": commands, "elapsed_not_benchmark": elapsed, "checks": evaluate(case, out),
              "artifacts": {p.name: sha256(p) for p in out.iterdir() if p.suffix in (".gz", ".mat")}}
    write_json(out / "report.json", result)
    print(f"done {case['id']} --float {floating}: {elapsed['registration_wall_seconds']:.2f}s", flush=True)
    return result


def compare(left, right):
    return {filename: difference(nib.load(left / filename).get_fdata(), nib.load(right / filename).get_fdata())
            for filename in ("warped.nii.gz", "composite-field.nii.gz")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Repeat to select cases; default uses frozen P00 recipe")
    parser.add_argument("--no-repeat", action="store_true", help="Omit the separate FP64 reproducibility run")
    args = parser.parse_args()
    config_path = ROOT / "configs/p00_dense.json"
    config = json.loads(config_path.read_text())
    # These fields must not silently diverge from the generated ANTs command.
    if config["sampling"] != {"strategy": "None", "fraction": 1.0, "gradient_filter": False}:
        raise ValueError("P00 implements dense, unfiltered sampling only")
    if config["histogram_matching"] or config["winsorization"] is not None or config["masks"] is not None:
        raise ValueError("P00 preprocessing options must match the frozen command")
    manifest = json.loads((ROOT / "cases.json").read_text())
    for entry in manifest["inputs"]:
        if sha256(PROJECT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Input SHA-256 mismatch: {entry['path']}")
    cases = {c["id"]: c for c in manifest["cases"]}
    selected = args.case or config["default_cases"]
    unknown = set(selected) - set(cases)
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    binary_dir = Path(sys.executable).parent
    binaries = {name: str(binary_dir / name) for name in ("antsRegistration", "antsApplyTransforms")}
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config["threads"])
    env["ANTS_RANDOM_SEED"] = str(config["random_seed"])
    env["PATH"] = str(binary_dir) + os.pathsep + env.get("PATH", "")
    base = ROOT / "baseline/p00"
    environment = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                   "logical_cpu_count": os.cpu_count(), "threads_for_numerical_reference": config["threads"],
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "nibabel", "numba")},
                   "ants_version": subprocess.check_output([binaries["antsRegistration"], "--version"], env=env, text=True).strip(),
                   "binary_sha256": {name: sha256(path) for name, path in binaries.items()},
                   "config_sha256": sha256(config_path), "case_manifest_sha256": sha256(ROOT / "cases.json")}
    report = {"schema_version": 1, "purpose": config["purpose"], "environment": environment,
              "runs": [], "precision_differences": {}, "repeatability": None,
              "acceptance_scope": "original-tool numerical baseline; no new registration implementation or final FP32 tolerance"}
    write_json(base / "environment.json", environment)
    for case_id in selected:
        for floating in (0, 1):
            report["runs"].append(execute(config, cases[case_id], floating,
                                         base / case_id / f"float{floating}", binaries, env))
        report["precision_differences"][case_id] = compare(base / case_id / "float0", base / case_id / "float1")
        write_json(base / "report.json", report)
    repeat_id = config["repeat_fp64_case"]
    if not args.no_repeat and repeat_id in selected:
        report["runs"].append(execute(config, cases[repeat_id], 0,
                                     base / repeat_id / "float0_repeat", binaries, env))
        report["repeatability"] = compare(base / repeat_id / "float0", base / repeat_id / "float0_repeat")
    write_json(base / "report.json", report)
    print(json.dumps({"report": str(base / "report.json"),
                      "precision_differences": report["precision_differences"],
                      "repeatability": report["repeatability"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
