"""Full-map CPU FP32 / Metal diagnostics against official ANTs --float 1.

Uses existing MRI/analytic fixtures, preserves failures, never overwrites a
report. --prepare-only validates the recipe and saved reference without running
registration or importing the independent numerical implementation.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, import_implementation, load_reference_image
from validation.benchmarks.measure_fp64_cpu import native_command, stages_for, write_json

ARTIFACTS = ("reg_0GenericAffine.mat", "reg_1Warp.nii.gz", "reg_1InverseWarp.nii.gz", "warped.nii.gz")


def hashes(source):
    return {str(p.relative_to(source)): digest(p)
            for p in sorted((source / "fast_ants").rglob("*"))
            if p.suffix in (".py", ".metal")}


def validate_recipe(config):
    if (config["sampling"] != {"strategy": "None", "fraction": 1., "gradient_filter": False}
            or config["winsorization"] is not None or config["histogram_matching"]
            or config["masks"] is not None or config.get("write_inverse_warped", False)
            or config["interpolation"] != "Linear"):
        raise ValueError("This verifier requires the supported dense, unmasked, linear-interpolation recipe")
    if [s["transform"].split("[")[0] for s in config["stages"]] != ["Rigid", "Affine", "SyN"]:
        raise ValueError("Expected Rigid, Affine, SyN stages")
    if [s["metric"] for s in config["stages"]] != ["MI", "MI", "CC"]:
        raise ValueError("Expected MI, MI, CC metrics")


def float1_command(config, case, directory, binary):
    command = native_command(config, case, directory, str(binary))
    command[command.index("--float") + 1] = "1"
    return command


def normalized_command(command):
    command = list(command)
    if command.count("--float") != 1 or command[command.index("--float") + 1] != "1":
        raise ValueError("Official acceptance reference must explicitly use --float 1")
    command[0] = "<verified-antsRegistration>"
    command[command.index("--output") + 1] = "<output-only>"
    return command


def verify_reference(directory, command, input_hashes, config_path, manifest_path, binary):
    """Fail closed: validate new receipts or the pinned original P00 receipts."""
    artifact_hashes = {name: digest(directory / name) for name in ARTIFACTS}
    receipt_path = directory / "reference.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["input_sha256"] != input_hashes or receipt["recipe_sha256"] != digest(config_path):
            raise ValueError("Saved reference input/recipe hashes do not match")
        if receipt["binary_sha256"] != digest(binary):
            raise ValueError("Saved reference ANTs binary does not match")
        recorded_command = receipt["command"]
        recorded_artifacts = receipt["artifacts"]
    else:
        # P00 stores recipe/manifest/binary hashes at the root, artifact hashes
        # and the complete executed argv in each case/float1/report.json.
        receipt_path = directory / "report.json"
        receipt = json.loads(receipt_path.read_text())
        environment = json.loads((directory.parent.parent / "environment.json").read_text())
        if receipt.get("float_option") != 1:
            raise ValueError("P00 receipt is not an official float1 run")
        if (environment["config_sha256"] != digest(config_path)
                or environment["case_manifest_sha256"] != digest(manifest_path)
                or environment["binary_sha256"]["antsRegistration"] != digest(binary)):
            raise ValueError("P00 environment hashes do not match current reference inputs")
        manifest = json.loads(manifest_path.read_text())
        pinned = {str((PROJECT / row["path"]).resolve()): row["sha256"] for row in manifest["inputs"]}
        # Exact argv comparison below fixes which two manifest images were used.
        for path, actual in input_hashes.items():
            if pinned.get(path) != actual:
                raise ValueError("P00 input differs from the pinned case manifest")
        recorded_command = receipt["commands"][0]
        recorded_artifacts = receipt["artifacts"]
    if normalized_command(recorded_command) != normalized_command(command):
        raise ValueError("Saved reference command differs from the requested recipe/case")
    if any(recorded_artifacts.get(k) != v for k, v in artifact_hashes.items()):
        raise ValueError("Saved reference artifacts have changed")
    return {"receipt": str(receipt_path), "receipt_sha256": digest(receipt_path),
            "command": recorded_command, "artifacts": artifact_hashes,
            "input_sha256": input_hashes, "verified_float_option": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--source", type=Path, default=PROJECT / "src")
    parser.add_argument("--config", type=Path, default=PROJECT / "validation/configs/p00_dense.json")
    parser.add_argument("--case-manifest", type=Path, default=PROJECT / "validation/cases.json")
    parser.add_argument("--case", default="ixi002_to_ixi012")
    parser.add_argument("--resolution", choices=("small", "half", "native"), default="small")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--optimizations", help="Metal flags; omitted uses current defaults")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-map-tolerance", action="store_true",
                        help="Explicit strict reproduction check; normally full-map differences are diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("source", "config", "case_manifest", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.backend == "cpu" and args.optimizations is not None:
        parser.error("--optimizations is specific to Metal")
    config = json.loads(args.config.read_text())
    validate_recipe(config)
    acceptance_path = PROJECT / "validation/configs/ordinary_fp32_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance["primary_reference"]["float_option"] != 1:
        raise ValueError("Acceptance policy must target ANTs --float 1")
    cases = json.loads(args.case_manifest.read_text())["cases"]
    case = dict(next(c for c in cases if c["id"] == args.case))
    folder = {"small": "prepared", "half": "p11-half", "native": "public"}[args.resolution]
    for side in ("fixed", "moving"):
        path = PROJECT / case[side]
        if case["kind"] != "analytic" and args.resolution != "small":
            path = PROJECT / "validation/data" / folder / path.name
        case[side] = str(path.resolve())
    if args.swap:
        case["fixed"], case["moving"] = case["moving"], case["fixed"]
    input_hashes = {case[s]: digest(case[s]) for s in ("fixed", "moving")}
    binary = Path(sys.executable).parent / "antsRegistration"
    native = args.native_dir.resolve() if args.native_dir else args.output / "native-float1"
    command = float1_command(config, case, native, binary)
    provenance = None
    if args.native_dir:
        provenance = verify_reference(native, command, input_hashes, args.config, args.case_manifest, binary)
    args.output.mkdir(parents=True)
    report = {"status": "prepared_not_executed", "backend": args.backend,
              "reference_mode": "ANTs --float 1", "fp64_is_acceptance_target": False,
              "case": case, "resolution": args.resolution, "swap": args.swap,
              "acceptance": acceptance, "acceptance_sha256": digest(acceptance_path),
              "recipe": config, "recipe_sha256": digest(args.config),
              "case_manifest": str(args.case_manifest), "case_manifest_sha256": digest(args.case_manifest),
              "input_sha256": input_hashes, "command": command,
              "source": str(args.source), "source_sha256": hashes(args.source),
              "native_provenance": provenance}
    write_json(args.output / "report.json", report)
    if args.prepare_only:
        print("Prepared; reference", "verified" if provenance else "requires execution", flush=True)
        return
    if args.native_dir is None:
        native.mkdir()
        env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",
                   ANTS_RANDOM_SEED=str(config["random_seed"]))
        print("Run official --float 1", args.case, args.resolution, flush=True)
        with (native / "console.log").open("w") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=max(600, config.get("timeout_per_command_seconds", 600)))
        receipt = {"command": command, "binary_sha256": digest(binary), "input_sha256": input_hashes,
                   "recipe_sha256": digest(args.config), "threads": 1,
                   "artifacts": {name: digest(native / name) for name in ARTIFACTS}}
        write_json(native / "reference.json", receipt)
        provenance = verify_reference(native, command, input_hashes, args.config, args.case_manifest, binary)
    api, numba = import_implementation(args.source, args.threads)
    import numpy as np
    from fast_ants.quality import field_difference
    fixed, moving = [load_reference_image(api, case[s]) for s in ("fixed", "moving")]
    linear, syn = stages_for(api, config, fixed.grid)
    options = dict(linear_stages=linear, syn_stage=syn, initialization_feature=config["initialization_feature"],
                   seed=config["random_seed"], output_directory=args.output / args.backend)
    if args.backend == "metal" and args.optimizations is not None:
        options["optimizations"] = tuple(x for x in args.optimizations.split(",") if x)
    print("Run independent", args.backend, args.case, args.resolution, flush=True)
    started = time.perf_counter()
    result = (api.register32 if args.backend == "cpu" else api.register_metal)(fixed, moving, **options)
    elapsed = time.perf_counter() - started
    saved_artifacts = {name: digest(args.output/args.backend/name) for name in
                       ("affine.mat", "warp.nii.gz", "inverse-warp.nii.gz", "warped.nii.gz")}
    affine = api.load_affine(native / ARTIFACTS[0])
    fw, iv = [api.load_field(native / name) for name in ARTIFACTS[1:3]]
    references = (api.CompositeTransform((fw, affine)), api.CompositeTransform((affine.inverse(), iv)))
    errors = {}
    for side, grid, actual, expected in (
        ("forward", fixed.grid, result.fixed_to_moving, references[0]),
        ("inverse", moving.grid, result.moving_to_fixed, references[1])):
        a, b = api.transform_to_field(grid, actual), api.transform_to_field(grid, expected)
        errors[side] = field_difference(a, b, acceptance["maximum_deformation_l2_mm"])
        errors[side]["stricter_diagnostic_passed"] = errors[side]["max_l2_mm"] <= acceptance["stricter_diagnostic_l2_mm"]
        del a, b
    finite = all(np.isfinite(a).all() for a in (result.linear_transform.matrix, result.forward_field.data,
                                               result.inverse_field.data, result.warped.data))
    geometry = {"warped_fixed": result.warped.grid.same_geometry(fixed.grid),
                "forward_fixed": result.forward_field.grid.same_geometry(fixed.grid),
                "inverse_fixed": result.inverse_field.grid.same_geometry(fixed.grid),
                "native_forward_fixed": fw.grid.same_geometry(fixed.grid),
                "native_inverse_fixed": iv.grid.same_geometry(fixed.grid)}
    unchanged = report["source_sha256"] == hashes(args.source)
    map_match = all(e["passed"] for e in errors.values())
    integrity = (finite and all(geometry.values()) and unchanged
                 and result.quality["nonpositive_jacobian_points"] == 0)
    strict = args.require_map_tolerance or acceptance.get("require_reference_map_match", True)
    passed = integrity and (map_match or not strict)
    status = ("passed" if passed else "numeric_gate_failed") if strict else (
        "integrity_passed_quality_assessment_pending" if integrity else "integrity_failed")
    report.update(status=status, native_provenance=provenance, saved_artifact_sha256=saved_artifacts,
                  complete_map_errors=errors, finite=bool(finite), geometry=geometry,
                  quality=result.quality, source_unchanged_during_run=unchanged,
                  reference_map_match=map_match, map_tolerance_enforced=bool(strict),
                  algorithm_and_registration_quality="requires separate semantic and quality assessment; integrity alone is insufficient",
                  actual_numba_threads=numba.get_num_threads(),
                  diagnostic_seconds_not_benchmark=elapsed,
                  configuration=result.configuration,
                  scope="independently fitted Rigid/Affine/SyN; all forward and inverse grid points including boundaries")
    write_json(args.output / "report.json", report)
    print(report["status"], {k: v["max_l2_mm"] for k, v in errors.items()}, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
