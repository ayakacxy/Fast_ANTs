"""Benchmark CLI, command and provenance contracts; no fits, JIT or GPU work."""

import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import validation.benchmarks.measure_fp32_pipeline as benchmark
import validation.quality.assess_fp32_benchmark as adapter


class PipelineBenchmarkTests(unittest.TestCase):
    def config(self):
        return json.loads((benchmark.PROJECT / "validation/configs/p00_dense.json").read_text())

    def test_parser_defaults_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = str(Path(temporary) / "new-report")
            args = benchmark.parse_args(["--output", destination])
            self.assertIsNone(args.engine)
            self.assertEqual(benchmark.ENGINES, ("cpu", "metal", "ants-float1"))
            self.assertEqual((args.warmups, args.repeats), (1, 3))
            self.assertFalse(args.verbose)
            self.assertEqual(args.log_every, 1)
            args = benchmark.parse_args(["--output", destination, "--engine", "metal", "--engine", "ants-float1",
                                         "--swap", "--resolution", "half", "--threads", "10", "--warmups", "0"])
            self.assertEqual(args.engine, ["metal", "ants-float1"])
            self.assertTrue(args.swap)
            self.assertEqual((args.threads, args.warmups, args.resolution), (10, 0, "half"))
            verbose = benchmark.parse_args(["--output", destination, "-v", "--log-every", "4"])
            self.assertTrue(verbose.verbose)
            self.assertEqual(verbose.log_every, 4)

    def test_parser_refuses_overwrite_and_invalid_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = str(Path(temporary) / "new-report")
            for options in (["--threads", "0"], ["--repeats", "0"], ["--warmups", "-1"],
                            ["--timeout", "nan"], ["--engine", "cpu", "--engine", "cpu"]):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    benchmark.parse_args(["--output", destination, *options])
            for interval in ("0", "-1", "1.5"):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    benchmark.parse_args(["--output", destination, "--log-every", interval])
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                benchmark.parse_args(["--output", temporary])

    def test_float1_command_preserves_all_stages(self):
        config = self.config()
        benchmark.validate_recipe(config)
        command = benchmark.float1_command(config, {"fixed": "/fixed", "moving": "/moving"},
                                             Path("/new/output"), Path("/antsRegistration"))
        self.assertEqual(command[command.index("--float") + 1], "1")
        self.assertEqual(command.count("--transform"), 3)
        self.assertEqual(command.count("--metric"), 3)
        self.assertIn("[8x4,1e-07,8]", command)
        self.assertEqual(command[command.index("--output") + 1],
                         "[/new/output/reg_,/new/output/warped.nii.gz]")

    def test_recipe_rejects_silently_approximated_features(self):
        for key, value in (("sampling", {"strategy": "Regular", "fraction": .25, "gradient_filter": False}),
                           ("winsorization", [.005, .995]), ("write_inverse_warped", True),
                           ("dimensionality", 2), ("interpolation", "NearestNeighbor")):
            config = self.config()
            config[key] = value
            with self.assertRaises(ValueError):
                benchmark.validate_recipe(config)
        config = copy.deepcopy(self.config())
        config["stages"][0]["iterations"].append(1)
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            benchmark.validate_recipe(config)

    def test_worker_command_forwards_case_recipe_and_backend(self):
        args = benchmark.parse_args(["--worker", "/worker", "--engine", "metal", "--swap",
                                     "--case", "synthetic_affine", "--optimizations", "parallel_cc"])
        command = benchmark.worker_command(args, "metal", Path("/worker"))
        self.assertEqual(command[command.index("--engine") + 1], "metal")
        self.assertEqual(command[command.index("--case") + 1], "synthetic_affine")
        self.assertEqual(command[command.index("--config") + 1], str(args.config))
        self.assertIn("--swap", command)
        self.assertEqual(command[command.index("--optimizations") + 1], "parallel_cc")
        cpu = benchmark.worker_command(args, "cpu", Path("/worker"))
        self.assertNotIn("--optimizations", cpu)
        self.assertNotIn("--verbose", command)
        self.assertNotIn("--log-every", command)
        legacy = SimpleNamespace(**{key: value for key, value in vars(args).items()
                                    if key not in ("verbose", "log_every")})
        self.assertEqual(benchmark.worker_command(legacy, "metal", Path("/worker")), command)
        args.verbose, args.log_every = True, 3
        live = benchmark.worker_command(args, "metal", Path("/worker"))
        self.assertIn("--verbose", live)
        self.assertEqual(live[live.index("--log-every")+1], "3")

    def test_same_explicit_thread_controls_and_seed_for_each_engine(self):
        env = benchmark.thread_environment(10, 123)
        self.assertEqual(env["NUMBA_NUM_THREADS"], "10")
        self.assertEqual(env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"], "10")
        self.assertEqual(env["ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS"], "10")
        self.assertEqual(env["ANTS_RANDOM_SEED"], "123")
        self.assertTrue(all(env[key] == "1" for key in benchmark.THREAD_KEYS))

    def test_case_resolution_and_swap_store_directional_annotations(self):
        original = benchmark.selected_case("ixi002_to_ixi012", "half")
        swapped = benchmark.selected_case("ixi002_to_ixi012", "half", True)
        self.assertIn("/p11-half/", original["fixed"])
        self.assertEqual(original["fixed"], swapped["moving"])
        self.assertEqual(original["fixed_landmarks_lps_mm"], swapped["moving_landmarks_lps_mm"])
        analytic = benchmark.selected_case("synthetic_affine", "native")
        self.assertIn("/analytic/", analytic["fixed"])

    def test_receipt_requires_and_hashes_four_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            provenance = {"source": "/source", "source_sha256": {"fast_ants/a.py": "source-hash"},
                          "input_sha256": {"/fixed": "fixed-hash", "/moving": "moving-hash"},
                          "recipe_sha256": "recipe-hash", "case_manifest_sha256": "manifest-hash",
                          "quality_policy_sha256": "policy-hash", "binary_sha256": {"ants-float1": "binary-hash"},
                          "python_binary_sha256": "python-hash", "threads": 10}
            for engine, names in (("cpu", benchmark.OWN_ARTIFACTS), ("ants-float1", benchmark.ANTS_ARTIFACTS)):
                with self.assertRaises(FileNotFoundError):
                    benchmark.make_receipt(directory, engine, ["command"], provenance, {})
                for name in names.values():
                    (directory / name).write_text("saved " + name)
                receipt = benchmark.make_receipt(directory, engine, ["command"], provenance, {"wall_seconds": 1.5})
                self.assertEqual(set(receipt["artifacts"]), set(names.values()))
                self.assertEqual(receipt["input_sha256"], provenance["input_sha256"])
                self.assertEqual(receipt["source_sha256"], provenance["source_sha256"])
                self.assertEqual(receipt["recipe_sha256"], "recipe-hash")
                for name, expected in receipt["artifacts"].items():
                    self.assertEqual(benchmark.digest(directory / name), expected)
                self.assertEqual(receipt["float_option"], 1 if engine == "ants-float1" else None)
                self.assertIn("not_assessed", receipt["quality_status"])

    def test_source_hashes_include_metal_but_ignore_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            package = source / "fast_ants"
            package.mkdir()
            for name in ("__init__.py", "kernel.metal", "cache.pyc"):
                (package / name).write_text(name)
            self.assertEqual(set(benchmark.source_hashes(source)), {"fast_ants/__init__.py", "fast_ants/kernel.metal"})

    def test_adapter_pins_matching_repeat_receipts_and_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {"source": "/source", "source_sha256": {"fast_ants/a.py": "source-hash"},
                      "input_sha256": {"/fixed": "fixed-hash", "/moving": "moving-hash"},
                      "recipe_sha256": "recipe-hash", "case_manifest_sha256": "manifest-hash",
                      "quality_policy_sha256": "policy-hash", "python_binary_sha256": "python-hash",
                      "binary_sha256": {"cpu": "python-hash", "ants-float1": "ants-hash"}, "threads": 10,
                      "case": {"id": "swapped", "fixed_landmarks_lps_mm": [[1, 2, 3]]}, "swap": True,
                      "resolution": "half", "source_unchanged_during_run": True, "runs": {}}
            for engine, names in (("cpu", benchmark.OWN_ARTIFACTS), ("ants-float1", benchmark.ANTS_ARTIFACTS)):
                directory = root / engine / "repeat-1"
                directory.mkdir(parents=True)
                for name in names.values():
                    (directory / name).write_text(name)
                command = ["antsRegistration", "--float", "1"] if engine == "ants-float1" else ["python", "worker"]
                receipt = benchmark.make_receipt(directory, engine, command, report, {"wall_seconds": 1})
                receipt.update({key: True for key in adapter.INTEGRITY_FIELDS})
                benchmark.write_json(directory / "receipt.json", receipt)
                if engine == "ants-float1":
                    benchmark.write_json(directory / "reference.json", receipt)
                row = {"engine": engine, "label": "repeat-1", "status": "completed", "integrity_passed": True,
                       "directory": str(directory), "receipt_path": str(directory / "receipt.json"),
                       "artifact_sha256": receipt["artifacts"], "command": command}
                report["runs"][engine] = [row]
            report_path = root / "benchmark.json"
            benchmark.write_json(report_path, report)
            own_row = adapter.selected_run(report, "cpu", 1)
            native_row = adapter.selected_run(report, "ants-float1", 1)
            with self.assertRaisesRegex(ValueError, "repeat-2"):
                adapter.selected_run(report, "cpu", 2)
            own = adapter.verified_receipt(report, own_row, "cpu")
            native = adapter.verified_receipt(report, native_row, "ants-float1")
            prepared = adapter.assessment_input(report, report_path, own_row, own, native_row, native)
            self.assertTrue(prepared["annotations_already_directional"])
            self.assertEqual(prepared["case"], report["case"])
            self.assertEqual(prepared["saved_artifact_sha256"], own["artifacts"])
            self.assertEqual(prepared["benchmark_provenance"]["report_sha256"], benchmark.digest(report_path))
            (Path(own_row["directory"]) / "warp.nii.gz").write_text("modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                adapter.verified_receipt(report, own_row, "cpu")


if __name__ == "__main__":
    unittest.main()
