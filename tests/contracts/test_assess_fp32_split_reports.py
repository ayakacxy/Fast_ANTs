"""Separated-report provenance checks; no fits, JIT or device work."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import validation.quality.assess_fp32_benchmark as adapter
import validation.benchmarks.measure_fp32_pipeline as benchmark


class SplitReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        (source / "fast_ants").mkdir(parents=True)
        (source / "fast_ants/__init__.py").write_text("# test snapshot\n")
        inputs = [self.root / name for name in ("fixed.nii.gz", "moving.nii.gz")]
        for path in inputs:
            path.write_text(path.name)
        recipe = json.loads((benchmark.PROJECT / "validation/configs/p00_dense.json").read_text())
        config = self.root / "config.json"
        benchmark.write_json(config, recipe)
        common = {"schema_version": 1, "source": str(source), "source_sha256": benchmark.source_hashes(source),
                  "input_sha256": {str(path): benchmark.digest(path) for path in inputs},
                  "config_path": str(config), "recipe": recipe, "recipe_sha256": benchmark.digest(config),
                  "case": {"id": "fixture", "kind": "analytic", "fixed": str(inputs[0]), "moving": str(inputs[1]),
                           "fixed_landmarks_lps_mm": [[1, 2, 3]], "moving_landmarks_lps_mm": [[2, 3, 4]]},
                  "case_manifest_sha256": benchmark.digest(benchmark.MANIFEST_PATH),
                  "resolution": "native", "swap": False, "annotations_already_directional": True, "threads": 10,
                  "python_binary_sha256": benchmark.digest(sys.executable),
                  "quality_policy": json.loads(benchmark.POLICY_PATH.read_text()),
                  "quality_policy_sha256": benchmark.digest(benchmark.POLICY_PATH),
                  "measurement_script_sha256": {name: benchmark.digest(path)
                                                for name, path in benchmark.MEASUREMENT_SCRIPTS.items()},
                  "scope": "fresh-process fitting", "measurement_method": "fresh runner", "thread_policy": "explicit matched threads",
                  "qa_timing_asymmetry": "own includes QA", "platform": "test-platform", "python": sys.version,
                  "power_background_load": "uncontrolled", "warmup_runs_per_engine": 0, "repeats_per_engine": 1}
        self.own = copy.deepcopy(common)
        self.own.update(status="failed_outputs_preserved", failure={"type": "KeyboardInterrupt", "message": ""},
                        binary_sha256={"cpu": common["python_binary_sha256"], "ants-float1": common["python_binary_sha256"]},
                        output_contract={"cpu": benchmark.OWN_ARTIFACTS}, runs={})
        self.official = copy.deepcopy(common)
        self.official.update(status="timing_completed_quality_assessment_pending", source_unchanged_during_run=True,
                             binary_sha256={"ants-float1": common["python_binary_sha256"]},
                             output_contract={"ants-float1": benchmark.ANTS_ARTIFACTS}, runs={})
        self.make_run(self.own, "cpu", 1)
        self.make_run(self.official, "ants-float1", 2)
        self.own_path = self.root / "own-report.json"
        self.official_path = self.root / "official-report.json"
        self.save_reports()

    def save_reports(self):
        benchmark.write_json(self.own_path, self.own)
        benchmark.write_json(self.official_path, self.official)

    def make_run(self, report, engine, repeat, official_binary=None):
        directory = self.root / engine / ("repeat-" + str(repeat))
        directory.mkdir(parents=True)
        names = benchmark.ANTS_ARTIFACTS if engine == "ants-float1" else benchmark.OWN_ARTIFACTS
        for name in names.values():
            (directory / name).write_text("saved " + name)
        if engine == "ants-float1":
            binary=Path(sys.executable) if official_binary is None else Path(official_binary)
            command = benchmark.float1_command(report["recipe"], report["case"], directory, binary)
            worker = None
        else:
            args = SimpleNamespace(source=Path(report["source"]), case=report["case"]["id"], resolution=report["resolution"],
                                   config=Path(report["config_path"]), threads=report["threads"], swap=report["swap"], optimizations=None)
            command = benchmark.worker_command(args, engine, directory)
            worker = {"engine": engine, "environment": {"numba_threads": report["threads"],
                                                        "platform": report["platform"], "python": report["python"]}}
            for name, value in (("worker", worker), ("history", []), ("quality", {}), ("configuration", {})):
                benchmark.write_json(directory / (name + ".json"), value)
        measurement = {"command": command, "wall_seconds": 1.5, "returncode": 0, "timed_out": False,
                       "peak_process_rss_bytes": 100,
                       "scope": "one fresh target process; imports, input reads, registration and output writes included; runner startup excluded",
                       "memory_method": "fresh runner RUSAGE_CHILDREN; exactly one target child"}
        receipt = benchmark.make_receipt(directory, engine, command, report, measurement)
        receipt.update({key: True for key in adapter.INTEGRITY_FIELDS})
        benchmark.write_json(directory / "receipt.json", receipt)
        if engine == "ants-float1":
            benchmark.write_json(directory / "reference.json", receipt)
        env = {key: "1" for key in benchmark.THREAD_KEYS}
        env.update(NUMBA_NUM_THREADS="10", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="10",
                   ITK_GLOBAL_MAXIMUM_NUMBER_OF_THREADS="10", ANTS_RANDOM_SEED=str(report["recipe"]["random_seed"]))
        row = {"engine": engine, "label": "repeat-" + str(repeat), "status": "completed", "integrity_passed": True,
               "returncode": 0, "directory": str(directory), "command": command, "measurement": measurement,
               "thread_environment": env, "whole_process_seconds": 1.5, "peak_process_rss_bytes": 100,
               "receipt_path": str(directory / "receipt.json"), "artifact_sha256": receipt["artifacts"]}
        if worker is not None:
            row["worker"] = worker
        benchmark.write_json(directory / "process.json", measurement)
        benchmark.write_json(directory / "run.json", row)
        report["runs"][engine] = [row]

    def prepare(self,repeat=1,official_repeat=2):
        return adapter.prepare_assessments(self.own_path, self.official_path, repeat, official_repeat, ["cpu"])

    def use_new_independent_source(self):
        """Construct another complete stub run with its own pinned snapshot."""
        source=self.root/'candidate-source'
        (source/'fast_ants').mkdir(parents=True)
        (source/'fast_ants/__init__.py').write_text('# different candidate snapshot\n')
        self.own.update(source=str(source),source_sha256=benchmark.source_hashes(source),
                        status='timing_completed_quality_assessment_pending',source_unchanged_during_run=True)
        self.own.pop('failure',None)
        self.make_run(self.own,'cpu',3)
        self.save_reports()

    def test_accepts_valid_explicit_split_without_rewriting_original_statuses(self):
        before = {path: path.read_bytes() for path in (self.own_path, self.official_path)}
        own, official, path, prepared = self.prepare()
        self.assertEqual(own["status"], "failed_outputs_preserved")
        self.assertEqual(official["status"], "timing_completed_quality_assessment_pending")
        self.assertEqual(path, self.official_path.resolve())
        item = prepared["cpu"]
        self.assertTrue(item["source_unchanged_during_run"])
        self.assertEqual(item["benchmark_provenance"]["report_sha256"], benchmark.digest(self.own_path))
        self.assertEqual(item["benchmark_provenance"]["official_report_sha256"], benchmark.digest(self.official_path))
        self.assertEqual(item["benchmark_provenance"]["independent_report_status"], "failed_outputs_preserved")
        self.assertTrue(item["timing_provenance"]["separate_reports"])
        self.assertFalse(item["timing_provenance"]["paired_same_report_trial"])
        self.assertFalse(item["timing_provenance"]["paired_repeated_benchmark"])
        self.assertFalse(item["timing_provenance"]["cross_report_speed_ratio_computed"])
        self.assertEqual(item["timing_provenance"]["official"]["label"], "repeat-2")
        self.assertTrue(all(path.read_bytes() == contents for path, contents in before.items()))

    def test_same_report_single_trial_does_not_claim_repeated_benchmark(self):
        combined = copy.deepcopy(self.own)
        combined.update(status="timing_completed_quality_assessment_pending", source_unchanged_during_run=True)
        combined.pop("failure")
        combined["output_contract"]["ants-float1"] = benchmark.ANTS_ARTIFACTS
        self.make_run(combined, "ants-float1", 1)
        path = self.root / "combined-report.json"

        def timing_flags():
            benchmark.write_json(path, combined)
            _, _, _, prepared = adapter.prepare_assessments(path, engines=["cpu"])
            return prepared["cpu"]["timing_provenance"]

        flags = timing_flags()
        self.assertFalse(flags["separate_reports"])
        self.assertTrue(flags["paired_same_report_trial"])
        self.assertFalse(flags["paired_repeated_benchmark"])
        # A declared repeat count alone must not turn one completed pair into
        # evidence of a repeated benchmark.
        combined["repeats_per_engine"] = 2
        self.assertFalse(timing_flags()["paired_repeated_benchmark"])
        first_cpu = combined["runs"]["cpu"][0]
        self.make_run(combined, "cpu", 2)
        combined["runs"]["cpu"].insert(0, first_cpu)
        combined["runs"]["ants-float1"].append(copy.deepcopy(self.official["runs"]["ants-float1"][0]))
        self.assertTrue(timing_flags()["paired_repeated_benchmark"])

    def test_distinct_verified_snapshots_reuse_external_official_outputs(self):
        self.use_new_independent_source()
        before={path:path.read_bytes() for path in (self.own_path,self.official_path)}
        _,_,_,prepared=self.prepare(repeat=3)
        item=prepared['cpu']
        self.assertEqual(item['source'],self.own['source'])
        self.assertEqual(item['source_sha256'],self.own['source_sha256'])
        provenance=item['source_provenance']
        self.assertEqual(provenance['independent']['source'],self.own['source'])
        self.assertEqual(provenance['official_report_recorded_independent_snapshot']['source'],self.official['source'])
        self.assertFalse(provenance['same_source_path'])
        self.assertFalse(provenance['same_source_hashes'])
        self.assertTrue(provenance['each_snapshot_verified_independently'])
        self.assertFalse(provenance['external_official_imports_independent_snapshot'])
        self.assertNotIn('--source',self.official['runs']['ants-float1'][0]['command'])
        self.assertFalse(item['timing_provenance']['paired_same_report_trial'])
        self.assertFalse(item['timing_provenance']['paired_repeated_benchmark'])
        self.assertTrue(all(path.read_bytes()==contents for path,contents in before.items()))

    def test_each_distinct_snapshot_is_still_rechecked(self):
        self.use_new_independent_source()
        for report in (self.own,self.official):
            path=Path(report['source'])/'fast_ants/__init__.py'
            original=path.read_bytes()
            with self.subTest(source=report['source']):
                try:
                    path.write_bytes(original+b'# changed after fitting\n')
                    with self.assertRaisesRegex(ValueError,'source snapshot has changed'):
                        self.prepare(repeat=3)
                finally:
                    path.write_bytes(original)

    def test_distinct_sources_do_not_relax_inputs_recipe_or_case(self):
        self.use_new_independent_source()
        original=copy.deepcopy(self.official)
        recipe=copy.deepcopy(original['recipe'])
        recipe['convergence']['window']+=1
        mutations={'input_sha256':{},'recipe':recipe,
                   'case':dict(original['case'],fixed_landmarks_lps_mm=[[0,0,0]])}
        for key,value in mutations.items():
            with self.subTest(key=key):
                self.official=copy.deepcopy(original)
                self.official[key]=value
                self.save_reports()
                with self.assertRaisesRegex(ValueError,'report mismatch'):
                    self.prepare(repeat=3)

    def test_cpu_only_report_need_not_pin_unused_official_binary(self):
        self.use_new_independent_source()
        del self.own['binary_sha256']['ants-float1']
        self.save_reports()
        _,_,_,prepared=self.prepare(repeat=3)
        provenance=prepared['cpu']['source_provenance']
        self.assertFalse(provenance['independent_report_records_official_binary'])
        self.assertEqual(provenance['official_binary_sha256'],self.official['binary_sha256']['ants-float1'])

    def test_recorded_unused_official_binary_remains_a_constraint(self):
        self.use_new_independent_source()
        self.own['binary_sha256']['ants-float1']='different-unused-official-binary'
        self.save_reports()
        with self.assertRaisesRegex(ValueError,'official binary'):
            self.prepare(repeat=3)

    def test_cpu_only_reuse_still_verifies_executed_official_binary(self):
        del self.own['binary_sha256']['ants-float1']
        binary=self.root/'official-stub'
        binary.write_text('# pinned external executable fixture\n')
        self.official['binary_sha256']['ants-float1']=benchmark.digest(binary)
        self.make_run(self.official,'ants-float1',3,official_binary=binary)
        self.save_reports()
        self.prepare(official_repeat=3)
        binary.write_text('# changed after the external run\n')
        with self.assertRaisesRegex(ValueError,'Executed engine binary no longer matches'):
            self.prepare(official_repeat=3)

    def test_interruption_requires_explicit_complete_official_report(self):
        with self.assertRaisesRegex(ValueError, "completed report"):
            adapter.prepare_assessments(self.own_path)
        self.official.update(status="failed_outputs_preserved", failure={"type": "KeyboardInterrupt"})
        self.save_reports()
        with self.assertRaisesRegex(ValueError, "completed report"):
            self.prepare()
        self.official.update(status="timing_completed_quality_assessment_pending")
        self.own["failure"]["type"] = "RuntimeError"
        self.save_reports()
        with self.assertRaisesRegex(ValueError, "completed report"):
            self.prepare()

    def test_report_mismatches_are_rejected_before_assessment(self):
        mutations = {"threads": 1, "resolution": "half", "swap": True, "platform": "other-host",
                     "annotations_already_directional": False, "recipe_sha256": "changed",
                     "source_sha256": {}, "input_sha256": {}, "quality_policy_sha256": "changed",
                     "measurement_script_sha256": {}, "case": dict(self.official["case"], fixed_landmarks_lps_mm=[[0, 0, 0]])}
        original = copy.deepcopy(self.official)
        for key, value in mutations.items():
            with self.subTest(key=key):
                self.official = copy.deepcopy(original)
                self.official[key] = value
                self.save_reports()
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_official_binary_mismatch_is_rejected(self):
        self.official["binary_sha256"]["ants-float1"] = "changed-binary"
        self.save_reports()
        with self.assertRaisesRegex(ValueError, "official binary"):
            self.prepare()

    def test_artifact_tampering_is_rejected(self):
        directory = Path(self.own["runs"]["cpu"][0]["directory"])
        (directory / "warp.nii.gz").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.prepare()

    def test_changed_process_receipt_is_rejected(self):
        directory = Path(self.official["runs"]["ants-float1"][0]["directory"])
        process = json.loads((directory / "process.json").read_text())
        process["wall_seconds"] = 99
        benchmark.write_json(directory / "process.json", process)
        with self.assertRaisesRegex(ValueError, "Process/run receipt"):
            self.prepare()

    def test_consistently_rewritten_thread_environment_still_rejected(self):
        row = self.own["runs"]["cpu"][0]
        row["thread_environment"]["NUMBA_NUM_THREADS"] = "9"
        benchmark.write_json(Path(row["directory"]) / "run.json", row)
        self.save_reports()
        with self.assertRaisesRegex(ValueError, "thread/seed environment"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
