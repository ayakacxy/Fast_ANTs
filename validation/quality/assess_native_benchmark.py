"""Assess saved native MRI outputs whose hashes match the public benchmark.

Fitting receipts remain historical records. Assessment uses the current CPU
FP64 sampler for every engine, outside the original registration timers.
"""

import argparse
import json
from pathlib import Path

from validation.benchmarks.measure_fp64_cpu import digest, import_implementation, write_json
from validation.benchmarks.measure_fp32_pipeline import source_hashes
from validation.common.paths import PROJECT
from validation.quality.assess_saved_registration import deformation_regularity


def verified_runs(report, benchmark):
    provenance = report["provenance"]
    if provenance["recipe_sha256"] != benchmark["recipe_sha256"]:
        raise ValueError("Saved recipe differs from the public benchmark")
    if report["threads"] != benchmark["threads"] or report["recipe"] != benchmark["recipe"]:
        raise ValueError("Saved parameters differ from the public benchmark")
    expected_inputs = {Path(item["path"]).name: item["sha256"] for item in benchmark["case"]["inputs"]}
    if {Path(p).name: h for p, h in provenance["input_sha256"].items()} != expected_inputs:
        raise ValueError("Saved inputs differ from the public benchmark")
    for path, expected in provenance["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Input changed: " + path)
    if provenance["ants_binary_sha256"] != benchmark["ants_reference"]["binary_sha256"]:
        raise ValueError("Official executable differs from the published reference")
    rows = []
    for engine in (*benchmark["runs"], "ants-float0", "ants-float1"):
        selected = [r for r in report["execution"] if r["engine"] == engine and r["label"] == "repeat-1"]
        if len(selected) != 1:
            raise ValueError("Expected one saved repeat for " + engine)
        row = selected[0]
        directory = Path(row["directory"])
        if row["status"] != "completed" or not row["integrity"] or not all(row["integrity"].values()):
            raise ValueError("Incomplete fitting receipt: " + engine)
        if json.loads((directory / "receipt.json").read_text()) != row:
            raise ValueError("Run differs from its fitting receipt: " + engine)
        if json.loads((directory / "process.json").read_text()) != row["measurement"]:
            raise ValueError("Process record differs from its fitting receipt: " + engine)
        if row["measurement"]["returncode"] != 0 or row["measurement"]["timed_out"]:
            raise ValueError("Unsuccessful fitting process: " + engine)
        if engine.startswith("ants-"):
            precision = "fp64" if engine == "ants-float0" else "fp32"
            command = row["command"]
            option = "0" if precision == "fp64" else "1"
            if command.count("--float") != 1 or command[command.index("--float") + 1] != option:
                raise ValueError("Official precision differs from the published reference")
            if row["whole_process_seconds"] != benchmark["ants_reference"][precision]["whole_process_seconds"]:
                raise ValueError("Official run differs from the published reference")
        else:
            actual = {role: item["sha256"] for role, item in row["artifacts"].items()}
            if actual != benchmark["runs"][engine]["output_sha256"]:
                raise ValueError("Saved outputs differ from the public benchmark: " + engine)
        for item in row["artifacts"].values():
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Saved output changed: " + item["path"])
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Saved measure_current_native report")
    parser.add_argument("--benchmark", type=Path, default=PROJECT / "docs/assets/benchmarks/v0.1.0-native.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.output.exists() or not args.output.resolve().is_relative_to(PROJECT):
        parser.error("Use positive threads and a new output directory inside this project")
    report = json.loads(args.report.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    rows = verified_runs(report, benchmark)
    source = PROJECT / "src"
    assessment_source = source_hashes(source)
    api, _ = import_implementation(source, args.threads)
    import numpy as np

    fixed, moving = api.load_images([report["case"][side] for side in ("fixed", "moving")],
                                    geometry_policy="itk_scanner", scaling_policy="itk")
    fmin, fspan = float(fixed.data.min()), float(np.ptp(fixed.data))
    mmin, mspan = float(moving.data.min()), float(np.ptp(moving.data))
    if min(fspan, mspan) <= 0:
        raise ValueError("Nonconstant images required")
    mask = fixed.data > fmin + .05 * fspan
    target = (fixed.data[mask] - fmin) / fspan
    points, truth = [np.asarray(report["case"][side + "_landmarks_lps_mm"])
                     for side in ("fixed", "moving")]

    def similarity(transform):
        warped = api.apply_image(moving, fixed, transform).data
        values = (warped[mask] - mmin) / mspan
        x, y = target - target.mean(), values - values.mean()
        norm = np.sqrt(np.sum(x*x) * np.sum(y*y))
        error = np.linalg.norm(transform.map_points(points) - truth, axis=1)
        return {"foreground_ncc": None if norm == 0 else float(np.sum(x*y) / norm),
                "foreground_normalized_mse": float(np.mean((values-target)**2)),
                "foreground_points": int(mask.sum()), "finite": bool(np.isfinite(warped).all()),
                "landmark_error_mm": {"mean": float(error.mean()), "rms": float(np.sqrt(np.mean(error**2))),
                                      "maximum": float(error.max()), "per_point": error.tolist()}}

    assessment = {"status": "quality_metrics_running", "benchmark_sha256": digest(args.benchmark),
        "fitting_report_sha256": digest(args.report), "assessment_source_sha256": assessment_source,
        "assessment_tool_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in (
            Path(__file__), PROJECT / "validation/quality/assess_saved_registration.py")},
        "assessment_threads": args.threads, "before_identity": similarity(api.AffineTransform.identity()),
        "similarity_domain": "fixed intensity > minimum + 5% of fixed range; identical CPU FP64 sampler for every map",
        "landmark_scope": "Four coarse non-expert MRI annotations; exploratory",
        "scope": "Saved outputs verified against fitting receipts and published output hashes; assessment outside registration timers",
        "runs": []}
    args.output.mkdir(parents=True)
    for row in rows:
        products = row["artifacts"]
        affine = api.load_affine(products["affine"]["path"])
        forward = api.load_field(products["forward_warp"]["path"])
        inverse = api.load_field(products["inverse_warp"]["path"])
        saved = api.load_image(products["warped"]["path"])
        if not all(image.grid.same_geometry(fixed.grid) for image in (forward, inverse, saved)):
            raise ValueError("Saved geometry differs from the fixed grid")
        result = {"engine": row["engine"], "label": row["label"],
            "fitting_receipt_sha256": digest(Path(row["directory"]) / "receipt.json"),
            "artifacts": {role: {"name": Path(p["path"]).name, "sha256": p["sha256"]}
                          for role, p in products.items()},
            "saved_warped_finite": bool(np.isfinite(saved.data).all()),
            "saved_geometry_matches_fixed": True,
            "metrics": similarity(api.CompositeTransform((forward, affine))),
            "regularity": deformation_regularity(forward, inverse)}
        assessment["runs"].append(result)
        write_json(args.output / "report.json", assessment)
        print(json.dumps(result), flush=True)
        del saved, forward, inverse
    if source_hashes(source) != assessment_source:
        raise ValueError("Assessment source changed during evaluation")
    assessment["status"] = "quality_metrics_recorded"
    write_json(args.output / "report.json", assessment)


if __name__ == "__main__":
    main()
