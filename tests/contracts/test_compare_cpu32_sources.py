"""CPU source-comparison controls and provenance; no registrations or JIT."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import validation.benchmarks.compare_cpu32_sources as comparison


class CompareCPU32SourcesTests(unittest.TestCase):
    def arguments(self, temporary, *extra):
        root = Path(temporary)
        return ["--baseline-source", str(root / "baseline"), "--candidate-source", str(root / "candidate"),
                "--output", str(root / "new-report"), *extra]

    def test_defaults_counterbalance_and_shared_worker_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = comparison.parse_args(self.arguments(temporary))
            self.assertEqual((args.warmups, args.repeats, args.resolution), (1, 4, "half"))
            self.assertEqual([comparison.run_order(i) for i in range(4)],
                             [("baseline", "candidate"), ("candidate", "baseline")] * 2)
            self.assertEqual(sum(comparison.run_order(i)[0] == "baseline" for i in range(args.repeats)), 2)
            workers = {arm: comparison.worker_args(args, arm) for arm in comparison.ARMS}
            for arm, worker in workers.items():
                self.assertEqual(worker.source, getattr(args, arm + "_source"))
                self.assertEqual(worker.engine, ["cpu"])
                self.assertEqual(worker.output, args.output / arm)
                command = comparison.pipeline.worker_command(worker, "cpu", worker.output / "cpu/repeat-1")
                self.assertEqual(command[command.index("--engine") + 1], "cpu")
                self.assertEqual(command[command.index("--source") + 1], str(worker.source))
                self.assertEqual(Path(command[2]), Path(comparison.pipeline.__file__).resolve())

    def test_parser_rejects_existing_output_and_invalid_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stderr(io.StringIO()):
                for flag, value in (("--threads", "0"), ("--warmups", "-1"), ("--repeats", "0"),
                                    ("--timeout", "0"), ("--timeout", "nan"), ("--timeout", "inf")):
                    with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                        comparison.parse_args(self.arguments(temporary, flag, value))
                Path(temporary, "new-report").mkdir()
                with self.assertRaises(SystemExit):
                    comparison.parse_args(self.arguments(temporary))

    def test_explicit_controls_and_summary_exclude_warmups(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = comparison.parse_args(self.arguments(temporary, "--threads", "10", "--repeats", "3",
                                                        "--warmups", "0", "--swap", "--resolution", "small"))
            self.assertEqual((args.threads, args.repeats, args.warmups, args.swap), (10, 3, 0, True))
            rows = {"baseline": [{"whole_process_seconds": value, "peak_process_rss_bytes": 10} for value in (20., 22., 30.)],
                    "candidate": [{"whole_process_seconds": value, "peak_process_rss_bytes": 12} for value in (10., 11., 15.)]}
            summary = comparison.summarize(rows)
            self.assertEqual(summary["sources"]["baseline"]["median_whole_process_seconds"], 22.)
            self.assertEqual(summary["baseline_time_divided_by_candidate_time"], 2.)
            self.assertEqual(summary["candidate_time_reduction_percent"], 50.)
            self.assertEqual(summary["paired_round_baseline_time_divided_by_candidate_time"], [2., 2., 2.])

    def test_source_specific_receipts_hash_all_four_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = comparison.parse_args(self.arguments(temporary))
            for arm in comparison.ARMS:
                package = getattr(args, arm + "_source") / "fast_ants"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(arm)
            fixed, moving = root / "fixed.nii", root / "moving.nii"
            fixed.write_text("fixed"); moving.write_text("moving")
            case = {"fixed": str(fixed), "moving": str(moving)}
            sources = {arm: comparison.source_provenance(args, getattr(args, arm + "_source"), case)
                       for arm in comparison.ARMS}
            self.assertNotEqual(sources["baseline"]["source_sha256"], sources["candidate"]["source_sha256"])
            self.assertEqual(sources["baseline"]["input_sha256"], sources["candidate"]["input_sha256"])
            self.assertEqual(sources["baseline"]["recipe_sha256"], sources["candidate"]["recipe_sha256"])
            for arm in comparison.ARMS:
                directory = root / arm / "saved"
                directory.mkdir()
                for name in comparison.pipeline.OWN_ARTIFACTS.values():
                    (directory / name).write_text(arm + name)
                receipt = comparison.pipeline.make_receipt(directory, "cpu", ["python", "worker"], sources[arm], {})
                self.assertEqual(receipt["source"], str(getattr(args, arm + "_source")))
                self.assertEqual(set(receipt["artifacts"]), set(comparison.pipeline.OWN_ARTIFACTS.values()))

    def test_schedule_uses_only_shared_cpu_runner_and_preserves_each_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = comparison.parse_args(self.arguments(temporary))
            args.output.mkdir()
            provenance = {"source_sha256": {}, "input_sha256": {}, "recipe_sha256": comparison.digest(args.config)}
            calls = []

            def fake_run(worker, engine, label, config, case, source_provenance):
                calls.append((worker.source, engine, label, worker.output))
                return {"whole_process_seconds": 2. if worker.source == args.baseline_source else 1.,
                        "peak_process_rss_bytes": 100, "status": "completed", "integrity_passed": True,
                        "directory": str(worker.output / engine / label)}

            with (patch.object(comparison, "source_provenance", return_value=provenance),
                  patch.object(comparison.pipeline, "source_hashes", return_value={}),
                  patch.object(comparison.pipeline, "timed_run", side_effect=fake_run),
                  patch.object(comparison, "assess_saved", return_value={"integrity_passed": True}),
                  contextlib.redirect_stdout(io.StringIO())):
                report = comparison.run_comparison(args, {}, {})
            self.assertEqual(len(calls), 10)  # Two warmups plus four AB/BA rounds.
            self.assertTrue(all(engine == "cpu" for _, engine, _, _ in calls))
            self.assertEqual([source for source, _, label, _ in calls if label.startswith("repeat")],
                             [args.baseline_source, args.candidate_source, args.candidate_source, args.baseline_source] * 2)
            self.assertEqual(len(report["warmup_diagnostics"]["baseline"]), 1)
            self.assertEqual(len(report["runs"]["candidate"]), 4)
            self.assertTrue(report["starting_positions_exactly_balanced"])
            self.assertFalse(report["numerical_or_iteration_equality_required"])
            self.assertEqual(report["summary"]["baseline_time_divided_by_candidate_time"], 2.)
            saved = json.loads((args.output / "report.json").read_text())
            self.assertEqual(saved["status"], "timing_and_integrity_completed_registration_quality_review_pending")


if __name__ == "__main__":
    unittest.main()
