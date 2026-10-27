"""Fresh-process CUDA source comparison on the pinned native IXI MRI pair.

One full excluded warmup per source precedes measured baseline/candidate/
candidate/baseline runs. Startup, reads, synchronized registration, built-in
QA and writes remain inside the shared process timer. Saved-output differences
are diagnostics; histogram reduction order need not produce identical maps.
"""

from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))

import argparse
import json
import math
from pathlib import Path
import platform
import signal
import shutil
import statistics
import sys

from validation.benchmarks import measure_cuda_native as native
from validation.common.runtime import digest, write_json


ARMS = ("baseline", "candidate")


def schedule():
    return [(arm, "warmup") for arm in ARMS] + [
        ("baseline", "repeat-1"), ("candidate", "repeat-1"),
        ("candidate", "repeat-2"), ("baseline", "repeat-2")]


def output_differences(left, right):
    """Compare saved arrays outside the registration timers with bounded scratch."""
    import nibabel as nib
    import numpy as np
    result = {}
    for role in ("warped", "forward_warp", "inverse_warp"):
        a, b = [nib.load(row["artifacts"][role]["path"]) for row in (left, right)]
        if a.shape != b.shape or not np.allclose(a.affine, b.affine, rtol=0, atol=1e-6):
            raise ValueError("Saved-output geometry differs for " + role)
        a, b = np.asarray(a.dataobj, dtype=np.float32), np.asarray(b.dataobj, dtype=np.float32)
        squared, reference, maximum, finite = 0., 0., 0., True
        for start in range(0, a.shape[0], 8):
            x, y = a[start:start+8].astype(np.float64), b[start:start+8].astype(np.float64)
            delta = y-x
            finite = finite and bool(np.isfinite(x).all() and np.isfinite(y).all())
            squared += float(np.sum(delta*delta))
            reference += float(np.sum(x*x))
            maximum = max(maximum, float(np.max(np.abs(delta))))
        if not finite:
            raise ValueError("Nonfinite saved output: " + role)
        result[role] = {"relative_l2": math.sqrt(squared/reference) if reference else None,
                        "maximum_absolute_difference": maximum, "all_finite": finite,
                        "geometry_matches": True, "units": "intensity" if role == "warped" else "mm"}
    return result


def summarize(runs):
    sources = {}
    for arm in ARMS:
        rows = [row for row in runs[arm] if row["label"].startswith("repeat-")]
        seconds = [row["whole_process_seconds"] for row in rows]
        if len(seconds) != 2 or not all(math.isfinite(value) and value > 0 for value in seconds):
            raise ValueError("Two positive finite formal process times are required per source")
        sources[arm] = {"whole_process_seconds": seconds, "median_seconds": statistics.median(seconds),
                        "maximum_peak_process_rss_bytes": max(row["peak_process_rss_bytes"] for row in rows)}
    ratio = sources["baseline"]["median_seconds"] / sources["candidate"]["median_seconds"]
    return {"sources": sources, "baseline_time_divided_by_candidate_time": ratio,
            "candidate_time_reduction_percent": 100*(1-1/ratio)}


def compare(args):
    config = json.loads(args.config.read_text())
    native.pipeline.validate_recipe(config)
    case = native.selected_case(args.data)
    files = [args.config, Path(sys.executable), Path(__file__).resolve(), Path(native.__file__).resolve(),
             native.PROJECT / "validation/cases.json", native.pipeline.POLICY_PATH,
             native.PROJECT / "validation/sources/downloaded-files.json"]
    compressor=shutil.which('pigz')
    if compressor is not None:files.append(Path(compressor))
    common = {"input_sha256": {case[side]: digest(case[side]) for side in ("fixed", "moving")},
              "file_sha256": {str(path): digest(path) for path in files},
              "tool_sha256": native.tools_hashes()}
    reports, workers = {}, {}
    for arm in ARMS:
        source = getattr(args, arm)
        hashes = native.source_hashes(source)
        if not (source / "fast_ants/__init__.py").is_file() or not hashes:
            raise ValueError("Source must contain a fast_ants package: " + str(source))
        reports[arm] = {"case": case, "provenance": {**common, "source": str(source),
                        "source_sha256": hashes}, "runs": []}
        workers[arm] = argparse.Namespace(**{**vars(args), "source": source, "output": args.output / arm})
    report = {"schema_version": 1, "status": "planned" if args.plan_only else "running",
              "started_at_utc": native.utc(), "case": case, "recipe": config,
              "threads": args.threads, "platform": platform.platform(), "python": sys.version,
              "sources": {arm: reports[arm]["provenance"] for arm in ARMS},
              "measurement_scope": "Fresh target startup/imports, input reads, CUDA initialization, registration, device synchronization, built-in host QA and final writes; timer runner startup excluded",
              "cache_policy": "Separate task-local CuPy/driver/Numba caches per source; full excluded warmups populate each cache; input page caches and system GPU state are uncontrolled",
              "quality_scope": "Built-in saved quality and paired output differences; no bitwise or map-equality acceptance gate; independent registration quality review remains separate",
              "planned_order": [{"source": arm, "label": label} for arm, label in schedule()],
              "runs": {arm: reports[arm]["runs"] for arm in ARMS}}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    write_json(path, report)
    if args.plan_only:
        print(str(path))
        return report
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Comparison received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for arm, label in schedule():
            native.run_one(workers[arm], config, reports[arm], "cuda", label)
            row = reports[arm]["runs"][-1]
            output = Path(row["output_directory"])
            if row["cli_status"].get("backend") != "cuda" or row["cli_status"].get("precision") != "fp32":
                raise RuntimeError("Expected completed CUDA FP32 CLI output")
            configuration = json.loads((output / "configuration.json").read_text())
            row["runtime_diagnostics"] = configuration["runtime"]
            row["output_quality"] = json.loads((output / "quality.json").read_text())
            for name in ("run.json", "configuration.json", "quality.json", "history.json"):
                product = output / name
                row["artifacts"][name] = {"path": str(product), "bytes": product.stat().st_size,
                                           "sha256": digest(product)}
            write_json(Path(row["directory"]) / "receipt.json", row)
            write_json(workers[arm].output / "report.json", reports[arm])
            write_json(path, report)
            if not all(all(native.integrity(reports[name]["provenance"]).values()) for name in ARMS):
                raise RuntimeError("Source, input, recipe, Python or measurement tool changed")
        report["summary"] = summarize(report["runs"])
        formal = {arm: [row for row in report["runs"][arm] if row["label"].startswith("repeat-")]
                  for arm in ARMS}
        report["paired_output_differences"] = [
            {"label": left["label"], "metrics": output_differences(left, right)}
            for left, right in zip(formal["baseline"], formal["candidate"])]
        report["final_integrity"] = {arm: native.integrity(reports[arm]["provenance"]) for arm in ARMS}
        report["saved_products_unchanged"] = all(digest(item["path"]) == item["sha256"]
            for rows in report["runs"].values() for row in rows for item in row["artifacts"].values())
        if not report["saved_products_unchanged"] or not all(all(value.values()) for value in report["final_integrity"].values()):
            raise RuntimeError("Final source or saved-product provenance check failed")
        report.update(status="timing_integrity_and_output_diagnostics_completed_quality_review_pending",
                      completed_at_utc=native.utc())
        print(json.dumps(report["summary"], indent=2), flush=True)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed_outputs_preserved",
                      failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(path, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--config", type=Path, default=native.PROJECT / "src/fast_ants/presets/brain-syn.json")
    parser.add_argument("--data", type=Path, default=native.TASK / "data")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("baseline", "candidate", "output", "config", "data"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or not args.output.is_relative_to(native.TASK):
        parser.error("Use a new task-local output directory under " + str(native.TASK))
    if args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("threads and timeout must be positive and finite")
    return args


if __name__ == "__main__":
    compare(parse_args())
