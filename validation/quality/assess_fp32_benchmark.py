"""Assess saved FP32 benchmark outputs after timing, without registration.

For a selected repeat, compare CPU/Metal outputs with the matching official
float1 repeat using assess_saved_registration.py. Receipts and artifacts are
verified before creating the adapter inputs; the benchmark is never modified.
--official-report explicitly selects an official run from a separate completed
report. The independent report may then be user-interrupted, provided selected
independent runs and their complete receipts are valid. Original reports retain
their original statuses. Its independent Python snapshot may differ from the
snapshot recorded alongside the external official run: antsRegistration loads
neither snapshot. Each recorded snapshot is still independently verified.
Cross-report quality comparisons are not paired timing repeats and never
produce a cross-report speed ratio.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from validation.benchmarks.measure_fp32_pipeline import ANTS_ARTIFACTS, OWN_ARTIFACTS, MANIFEST_PATH, POLICY_PATH, THREAD_KEYS, MEASUREMENT_SCRIPTS, float1_command, source_hashes, validate_recipe, worker_command
from validation.benchmarks.measure_fp64_cpu import PROJECT, digest, write_json

INTEGRITY_FIELDS = ("source_unchanged_during_run", "inputs_unchanged_during_run",
                    "recipe_unchanged_during_run", "binary_unchanged_during_run")
MATCH_FIELDS = ("source", "source_sha256", "input_sha256", "recipe", "recipe_sha256",
                "case", "case_manifest_sha256", "resolution", "swap", "annotations_already_directional",
                "threads", "python_binary_sha256", "quality_policy", "quality_policy_sha256",
                "scope", "measurement_method", "measurement_script_sha256", "thread_policy",
                "qa_timing_asymmetry", "platform", "python", "power_background_load")


def selected_run(report, engine, repeat):
    label = "repeat-" + str(repeat)
    rows = [row for row in report["runs"].get(engine, []) if row["label"] == label]
    if len(rows) != 1 or rows[0].get("status") != "completed" or not rows[0].get("integrity_passed"):
        raise ValueError(f"Missing completed, integrity-verified {engine} {label}")
    return rows[0]


def verified_receipt(report, row, engine):
    path = Path(row["receipt_path"])
    receipt = json.loads(path.read_text())
    if receipt.get("engine") != engine or receipt.get("command") != row["command"]:
        raise ValueError("Receipt engine/command differs from benchmark row")
    for key in ("source", "source_sha256", "input_sha256", "recipe_sha256", "case_manifest_sha256",
                "quality_policy_sha256", "python_binary_sha256", "threads"):
        if receipt[key] != report[key]:
            raise ValueError("Receipt provenance differs from benchmark: " + key)
    if receipt["binary_sha256"] != report["binary_sha256"][engine]:
        raise ValueError("Receipt binary hash differs from benchmark")
    if not all(receipt.get(key) for key in INTEGRITY_FIELDS):
        raise ValueError("Receipt does not establish fitting-time integrity")
    directory = Path(row["directory"])
    if path.resolve().parent != directory.resolve():
        raise ValueError("Receipt path differs from recorded output directory")
    names = ANTS_ARTIFACTS if engine == "ants-float1" else OWN_ARTIFACTS
    if set(receipt["artifacts"]) != set(names.values()) or row["artifact_sha256"] != receipt["artifacts"]:
        raise ValueError("Receipt does not pin the required four outputs")
    for name, expected in {**receipt["artifacts"], **receipt.get("metadata_artifact_sha256", {})}.items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError("Saved artifact changed: " + name)
    if engine == "ants-float1":
        command = receipt["command"]
        if (receipt.get("float_option") != 1 or command.count("--float") != 1
                or command[command.index("--float") + 1] != "1"):
            raise ValueError("Official reference must explicitly use --float 1")
        reference_path = directory / "reference.json"
        if json.loads(reference_path.read_text()) != receipt:
            raise ValueError("Official reference.json differs from run receipt")
    return receipt


def validate_report_status(report, *, allow_interrupted=False):
    completed = report.get("status") == "timing_completed_quality_assessment_pending"
    interrupted = (allow_interrupted and report.get("status") == "failed_outputs_preserved"
                   and report.get("failure", {}).get("type") == "KeyboardInterrupt")
    if not (completed or interrupted):
        raise ValueError("A completed report or explicitly selected user-interrupted independent report is required")
    if completed and report.get("source_unchanged_during_run") is not True:
        raise ValueError("Completed benchmark does not establish unchanged sources")
    if report.get("source_unchanged_during_run") is False:
        raise ValueError("Benchmark source changed during fitting")
    if report.get("annotations_already_directional") is not True:
        raise ValueError("Benchmark must explicitly establish directional annotations")


def match_reports(independent, official, *, allow_separate_sources=False):
    for key in MATCH_FIELDS:
        if allow_separate_sources and key in ("source", "source_sha256"):
            continue
        if key not in independent or key not in official or independent[key] != official[key]:
            raise ValueError("Independent/official report mismatch: " + key)
    official_binary = official["binary_sha256"].get("ants-float1")
    if not official_binary:
        raise ValueError("Official report must pin the executed official binary")
    # A CPU/Metal-only benchmark did not execute antsRegistration and need not
    # claim an unused binary. When it recorded one, retain that constraint.
    if ("ants-float1" in independent["binary_sha256"]
            and independent["binary_sha256"]["ants-float1"] != official_binary):
        raise ValueError("Independent/official report mismatch: official binary")
    for engine in set(independent["binary_sha256"]) & set(official["binary_sha256"]):
        if independent["binary_sha256"][engine] != official["binary_sha256"][engine]:
            raise ValueError("Independent/official report mismatch: binary " + engine)


def verify_current_provenance(report):
    if source_hashes(Path(report["source"])) != report["source_sha256"]:
        raise ValueError("Benchmark source snapshot has changed")
    for path, expected in report["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Benchmark input has changed: " + path)
    config_path = Path(report["config_path"])
    if digest(config_path) != report["recipe_sha256"] or json.loads(config_path.read_text()) != report["recipe"]:
        raise ValueError("Benchmark recipe has changed or differs from recorded recipe")
    validate_recipe(report["recipe"])
    for path, key in ((MANIFEST_PATH, "case_manifest_sha256"), (POLICY_PATH, "quality_policy_sha256")):
        if digest(path) != report[key]:
            raise ValueError("Benchmark pinned policy/manifest changed: " + key)
    if json.loads(POLICY_PATH.read_text()) != report["quality_policy"]:
        raise ValueError("Benchmark policy content differs from pinned file")
    if set(report["measurement_script_sha256"]) != set(MEASUREMENT_SCRIPTS):
        raise ValueError("Required measurement script hashes are missing")
    for name, expected in report["measurement_script_sha256"].items():
        if Path(name).name != name or digest(MEASUREMENT_SCRIPTS[name]) != expected:
            raise ValueError("Benchmark measurement script has changed: " + name)
    if digest(sys.executable) != report["python_binary_sha256"]:
        raise ValueError("Assessment interpreter differs from benchmark interpreter")


def verify_timing_receipt(report, row, receipt):
    """Match recorded process/environment receipts without merging timings."""
    directory = Path(row["directory"])
    measurement = row["measurement"]
    contract = ANTS_ARTIFACTS if row["engine"] == "ants-float1" else OWN_ARTIFACTS
    if receipt.get("output_contract") != contract or report.get("output_contract", {}).get(row["engine"]) != contract:
        raise ValueError("Output contract differs from required product roles")
    if (receipt["measurement"] != measurement or json.loads((directory / "process.json").read_text()) != measurement
            or json.loads((directory / "run.json").read_text()) != row):
        raise ValueError("Process/run receipt differs from benchmark row")
    if (measurement.get("command") != row["command"] or measurement.get("returncode") != 0
            or row.get("returncode") != 0 or measurement.get("timed_out") is not False
            or not math.isfinite(measurement.get("wall_seconds", float("nan")))
            or measurement["wall_seconds"] <= 0 or row["whole_process_seconds"] != measurement["wall_seconds"]
            or row["peak_process_rss_bytes"] != measurement["peak_process_rss_bytes"]):
        raise ValueError("Selected process did not complete with a consistent timing receipt")
    if (measurement.get("scope") != "one fresh target process; imports, input reads, registration and output writes included; runner startup excluded"
            or measurement.get("memory_method") != "fresh runner RUSAGE_CHILDREN; exactly one target child"):
        raise ValueError("Selected process scope/memory method differs from the recorded runner contract")
    expected_environment = {key: "1" for key in THREAD_KEYS}
    expected_environment.update(NUMBA_NUM_THREADS=str(report["threads"]),
                                ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(report["threads"]),
                                ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS=str(report["threads"]),
                                ANTS_RANDOM_SEED=str(report["recipe"]["random_seed"]))
    if row.get("thread_environment") != expected_environment:
        raise ValueError("Recorded timing thread/seed environment differs from recipe")
    engine = row["engine"]
    command = row["command"]
    if digest(command[0]) != report["binary_sha256"][engine]:
        raise ValueError("Executed engine binary no longer matches its receipt")
    if engine == "ants-float1":
        expected_command = float1_command(report["recipe"], report["case"], directory, Path(command[0]))
    else:
        optimizations = None
        if "--optimizations" in command:
            optimizations = command[command.index("--optimizations") + 1]
        args = argparse.Namespace(source=Path(report["source"]), case=report["case"]["id"],
                                  resolution=report["resolution"], config=Path(report["config_path"]),
                                  threads=report["threads"], swap=report["swap"], optimizations=optimizations)
        expected_command = worker_command(args, engine, directory)
        if report["binary_sha256"][engine] != report["python_binary_sha256"]:
            raise ValueError("Independent engine binary differs from pinned interpreter")
        worker = json.loads((directory / "worker.json").read_text())
        if row.get("worker") != worker or worker.get("engine") != engine:
            raise ValueError("Independent worker metadata differs from benchmark row")
        worker_env = worker["environment"]
        if (worker_env["numba_threads"] != report["threads"] or worker_env["platform"] != report["platform"]
                or worker_env["python"] != report["python"]):
            raise ValueError("Independent worker environment differs from report")
        if not {"worker.json", "history.json", "quality.json", "configuration.json"}.issubset(receipt.get("metadata_artifact_sha256", {})):
            raise ValueError("Independent metadata receipts are incomplete")
    if command != expected_command:
        raise ValueError("Executed command differs from pinned recipe/case/source/output")


def source_provenance(independent,official):
    """Keep the two snapshot identities distinct from the external ANTs binary."""
    return {"independent": {"source": independent["source"],"source_sha256": independent["source_sha256"]},
            "official_report_recorded_independent_snapshot": {
                "source": official["source"],"source_sha256": official["source_sha256"]},
            "same_source_path": independent["source"] == official["source"],
            "same_source_hashes": independent["source_sha256"] == official["source_sha256"],
            "each_snapshot_verified_independently": True,
            "external_official_imports_independent_snapshot": False,
            "official_binary_sha256": official["binary_sha256"]["ants-float1"],
            "independent_report_records_official_binary": "ants-float1" in independent["binary_sha256"],
            "interpretation": "The external antsRegistration command does not import the independent Python implementation. "
                              "Separate-report quality assessment may reuse its verified outputs with a different independently verified snapshot; "
                              "actual inputs, recipe, direction/case geometry, threads, policies, measurement scripts and saved products remain pinned. "
                              "Different snapshots are not claimed to be identical code or a paired timing benchmark."}


def assessment_input(report, report_path, own_row, own_receipt, native_row, native_receipt,
                     official_report=None, official_report_path=None):
    official_report = official_report if official_report is not None else report
    official_report_path = official_report_path if official_report_path is not None else report_path
    separate = Path(official_report_path).resolve() != Path(report_path).resolve()
    same_report_trial = not separate and own_row["label"] == native_row["label"]
    def completed_labels(engine):
        return {row["label"] for row in report["runs"].get(engine, [])
                if row.get("status") == "completed" and row.get("integrity_passed") is True
                and row.get("label", "").startswith("repeat-")}
    declared_repeats = report.get("repeats_per_engine", 0)
    repeated = (same_report_trial and isinstance(declared_repeats, int)
                and declared_repeats >= 2
                and len(completed_labels(own_row["engine"]) & completed_labels("ants-float1")) >= 2)
    native_path = Path(native_row["directory"]) / "reference.json"
    return {"status": "benchmark_adapter_prepared", "backend": own_row["engine"],
            "source": report["source"], "source_sha256": report["source_sha256"],
            "source_unchanged_during_run": own_receipt["source_unchanged_during_run"],
            "source_provenance": source_provenance(report,official_report),
            "case": report["case"], "resolution": report["resolution"], "swap": report["swap"],
            "annotations_already_directional": True,
            "input_sha256": report["input_sha256"], "recipe_sha256": report["recipe_sha256"],
            "independent_directory": own_row["directory"],
            "saved_artifact_sha256": own_receipt["artifacts"],
            "native_provenance": {"receipt": str(native_path), "receipt_sha256": digest(native_path),
                                  "artifacts": native_receipt["artifacts"], "command": native_receipt["command"],
                                  "input_sha256": native_receipt["input_sha256"], "verified_float_option": 1},
            "benchmark_provenance": {"report": str(report_path), "report_sha256": digest(report_path),
                                     "independent_report_status": report.get("status"), "independent_report_failure": report.get("failure"),
                                     "official_report": str(official_report_path), "official_report_sha256": digest(official_report_path),
                                     "official_report_status": official_report.get("status"),
                                     "independent_receipt": own_row["receipt_path"],
                                     "independent_receipt_sha256": digest(own_row["receipt_path"]),
                                     "official_run_receipt": native_row["receipt_path"],
                                     "official_run_receipt_sha256": digest(native_row["receipt_path"]),
                                     "official_reference_sha256": digest(native_path)},
            "timing_provenance": {
                "separate_reports": separate,
                "paired_same_report_trial": same_report_trial,
                "paired_repeated_benchmark": repeated,
                "cross_report_speed_ratio_computed": False,
                "interpretation": "Saved-output quality comparison. Separate reports retain separate process scopes; no cross-report time ratio is a paired benchmark.",
                "independent": {"label": own_row["label"], "measurement": own_row.get("measurement"),
                                "scope": report.get("scope"), "warmups": report.get("warmup_runs_per_engine")},
                "official": {"label": native_row["label"], "measurement": native_row.get("measurement"),
                             "scope": official_report.get("scope"), "warmups": official_report.get("warmup_runs_per_engine")}},
            "scope": "saved-output quality task after benchmark; no registration and no contribution to fitting timers"}


def prepare_assessments(report_path, official_report_path=None, repeat=1, official_repeat=None, engines=None):
    report_path = Path(report_path).resolve()
    explicit_official = official_report_path is not None
    official_path = Path(official_report_path).resolve() if explicit_official else report_path
    report = json.loads(report_path.read_text())
    official = json.loads(official_path.read_text())
    validate_report_status(report, allow_interrupted=explicit_official)
    validate_report_status(official)
    match_reports(report, official, allow_separate_sources=explicit_official)
    verify_current_provenance(report)
    if official_path != report_path:
        verify_current_provenance(official)
    engines = engines or [engine for engine in ("cpu", "metal") if report.get("runs", {}).get(engine)]
    if not engines:
        raise ValueError("No independent engines are available")
    native_row = selected_run(official, "ants-float1", official_repeat if official_repeat is not None else repeat)
    native_receipt = verified_receipt(official, native_row, "ants-float1")
    verify_timing_receipt(official, native_row, native_receipt)
    prepared = {}
    for engine in engines:
        row = selected_run(report, engine, repeat)
        receipt = verified_receipt(report, row, engine)
        verify_timing_receipt(report, row, receipt)
        if row["thread_environment"] != native_row["thread_environment"]:
            raise ValueError("Independent/official selected timing environments differ")
        prepared[engine] = assessment_input(report, report_path, row, receipt, native_row, native_receipt, official, official_path)
    return report, official, official_path, prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Independent measure_fp32_pipeline.py report.json")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--official-report", type=Path, help="Explicit completed external official report; permits a different individually verified independent snapshot and selected own runs from a user-interrupted report")
    parser.add_argument("--official-repeat", type=int, help="Official repeat index; defaults to --repeat")
    parser.add_argument("--engine", action="append", choices=("cpu", "metal"), help="Default: available independent engines")
    parser.add_argument("--output", type=Path, required=True, help="New post-timing assessment directory")
    args = parser.parse_args(argv)
    report_path, output = args.report.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    if args.repeat < 1 or (args.official_repeat is not None and args.official_repeat < 1) or args.engine and len(set(args.engine)) != len(args.engine):
        parser.error("Repeat indices must be positive and engines must be unique")
    if args.official_repeat is not None and args.official_report is None:
        parser.error("--official-repeat requires --official-report")
    report, official, official_path, prepared = prepare_assessments(report_path, args.official_report, args.repeat,
                                                                  args.official_repeat, args.engine)
    output.mkdir(parents=True)
    summary = {"status": "assessment_running", "benchmark_report": str(report_path),
               "benchmark_report_sha256": digest(report_path), "repeat": args.repeat,
               "benchmark_report_status": report["status"], "official_report": str(official_path),
               "official_report_sha256": digest(official_path), "official_report_status": official["status"],
               "source_provenance": source_provenance(report,official),
               "official_repeat": args.official_repeat if args.official_repeat is not None else args.repeat,
               "separate_report_quality_comparison": official_path != report_path,
               "cross_report_speed_ratio_computed": False,
               "scope": "post-timing common-domain quality assessment; no refitting or map-equality gate",
               "assessments": {}}
    write_json(output / "report.json", summary)
    try:
        for engine, adapter in prepared.items():
            directory = output / engine
            directory.mkdir()
            adapter_path = directory / "assessment-input.json"
            write_json(adapter_path, adapter)
            command = [sys.executable, "-B", str(PROJECT / "validation/quality/assess_saved_registration.py"),
                       "--report", str(adapter_path), "--output", str(directory / "quality")]
            print("assess saved benchmark outputs", engine, "repeat", args.repeat, flush=True)
            with (directory / "console.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            assessment_path = directory / "quality/report.json"
            summary["assessments"][engine] = {"report": str(assessment_path), "report_sha256": digest(assessment_path),
                                               "assessment_input": str(adapter_path), "assessment_input_sha256": digest(adapter_path),
                                               "receipt_provenance": adapter["benchmark_provenance"],
                                               "timing_provenance": adapter["timing_provenance"],
                                               "result": json.loads(assessment_path.read_text())}
            write_json(output / "report.json", summary)
        summary["status"] = "quality_metrics_recorded_not_full_acceptance"
        write_json(output / "report.json", summary)
        print(summary["status"], flush=True)
    except BaseException as error:
        summary.update(status="assessment_failed_outputs_preserved", failure={"type": type(error).__name__, "message": str(error)})
        write_json(output / "report.json", summary)
        raise


if __name__ == "__main__":
    main()
