"""User CLI contracts with mocked image/registration execution; no JIT or GPU."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from fast_ants import cli


class UserCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixed, self.moving = self.root / "fixed.nii.gz", self.root / "moving.nii.gz"
        self.fixed.write_bytes(b"fixed input fixture")
        self.moving.write_bytes(b"moving input fixture")
        self.output = self.root / "registration"

    def arguments(self, *extra):
        return ["register", str(self.fixed), str(self.moving), "--output", str(self.output), *extra]

    def recipe(self):
        return cli.load_recipe(SimpleNamespace(config=None, preset="brain-syn"))[0]

    def test_backend_defaults_and_explicit_precision(self):
        cpu = cli.parse_args(self.arguments())
        self.assertEqual((cpu.backend, cpu.precision, cpu.threads), ("cpu", "fp64", None))
        self.assertEqual((cpu.geometry_policy, cpu.scaling_policy), ("itk_scanner", "itk"))
        self.assertEqual((cpu.verbose, cpu.log_every), (True, 1))
        self.assertFalse(cli.parse_args(self.arguments("--quiet")).verbose)
        metal = cli.parse_args(self.arguments("--backend", "metal", "-v", "--log-every", "4"))
        self.assertEqual((metal.precision, metal.verbose, metal.log_every), ("fp32", True, 4))
        cuda = cli.parse_args(self.arguments("--backend", "cuda"))
        self.assertEqual(cuda.precision, "fp32")
        self.assertFalse(self.output.exists())

    def test_invalid_combinations_and_positive_controls(self):
        for options in (("--backend", "metal", "--precision", "fp64"),
                        ("--backend", "cuda", "--precision", "fp64"), ("--threads", "0"),
                        ("--log-every", "0"), ("--log-every", "1.5"),
                        ("--config", str(self.fixed), "--preset", "brain-syn")):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(self.arguments(*options))
        self.assertFalse(self.output.exists())

    def test_existing_outputs_and_missing_inputs_are_refused(self):
        self.output.mkdir()
        sentinel = self.output / "user-result"
        sentinel.write_text("preserve")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(self.arguments())
        self.assertEqual(sentinel.read_text(), "preserve")
        self.moving.unlink()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(self.arguments("--output", str(self.root / "another")))

    def test_packaged_preset_preserves_original_recipe(self):
        root = Path(__file__).resolve().parents[2]
        expected = json.loads((root / "validation/configs/fp64_brain_syn_dense.json").read_text())
        recipe, provenance = cli.load_recipe(SimpleNamespace(config=None, preset=None))
        self.assertEqual(recipe, expected)
        self.assertEqual(provenance["source"], "preset:brain-syn")
        self.assertEqual(recipe["random_seed"], 20261022)

    def test_custom_config_is_complete_replacement(self):
        config = self.root / "recipe.json"
        config.write_text(json.dumps({"schema_version": 1, "threads": 2}))
        with self.assertRaisesRegex(ValueError, "Supported recipe"):
            cli.load_recipe(SimpleNamespace(config=config, preset=None))
        recipe = self.recipe()
        recipe["random_seed"] = 42
        config.write_text(json.dumps(recipe))
        loaded, provenance = cli.load_recipe(SimpleNamespace(config=config, preset=None))
        self.assertEqual(loaded["random_seed"], 42)
        self.assertEqual(provenance["source"], str(config.resolve()))

    def test_unsupported_or_malformed_recipes_fail_before_execution(self):
        changes = (("winsorization", [.005, .995]), ("histogram_matching", True),
                   ("masks", "mask.nii.gz"), ("write_inverse_warped", True),
                   ("sampling", {"strategy": "Regular", "fraction": .25, "gradient_filter": False}),
                   ("threads", True), ("dimensionality", 2))
        for key, value in changes:
            candidate = self.recipe()
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                cli.validate_recipe(candidate)
        recipe = self.recipe()
        recipe["stages"][0]["iterations"] = [1]
        with self.assertRaisesRegex(ValueError, "matching nonzero"):
            cli.validate_recipe(recipe)
        recipe = self.recipe()
        recipe["stages"][2]["transform"] = "SyN[nan,3,0]"
        with self.assertRaisesRegex(ValueError, "finite"):
            cli.validate_recipe(recipe)

    def test_physical_spacing_schedule_and_stage_parameters(self):
        api = SimpleNamespace(LinearStage=lambda **kw: SimpleNamespace(**kw),
                              SyNStage=lambda **kw: SimpleNamespace(**kw))
        grid = SimpleNamespace(spacing_xyz=(1., 2., 3.))
        linear, syn = cli.build_stages(api, self.recipe(), grid)
        self.assertEqual(linear[0].shrink_factors, ((8, 4, 3), (4, 2, 1), (2, 1, 1), (1, 1, 1)))
        self.assertEqual((linear[0].kind, linear[1].kind), ("Rigid", "Affine"))
        self.assertEqual((syn.gradient_step, syn.update_variance, syn.total_variance, syn.radius), (.2, 3., 0., 2))
        self.assertEqual(syn.iterations, (100, 70, 50, 20))

    def test_execution_dispatch_metadata_and_live_log_without_fit(self):
        for backend, precision, expected in (("cpu", "fp64", "register"), ("cpu", "fp32", "register32"),
                                              ("metal", "fp32", "register_metal"), ("cuda", "fp32", "register_cuda")):
            with self.subTest(backend=backend, precision=precision):
                output = self.root / (backend + precision)
                args = cli.parse_args(self.arguments("--output", str(output), "--backend", backend,
                                                      "--precision", precision, "--threads", "2", "--verbose"))
                calls = []
                def implementation(name):
                    def fit(fixed, moving, **options):
                        calls.append((name, options))
                        print("MOCK registration completed")
                        return SimpleNamespace(quality={"finite": True})
                    return fit
                api = SimpleNamespace(load_images=lambda *a, **kw: [SimpleNamespace(grid=SimpleNamespace(spacing_xyz=(1., 1., 1.)))] * 2,
                    LinearStage=lambda **kw: SimpleNamespace(**kw), SyNStage=lambda **kw: SimpleNamespace(**kw),
                    **{name: implementation(name) for name in ("register", "register32", "register_metal", "register_cuda")})
                terminal = io.StringIO()
                with patch.object(cli, "_runtime", return_value=(api, SimpleNamespace(get_num_threads=lambda: 2))), redirect_stdout(terminal):
                    self.assertEqual(cli.run_registration(args), 0)
                self.assertEqual(calls[0][0], expected)
                self.assertEqual(calls[0][1]["seed"], 20261022)
                self.assertEqual(calls[0][1].get("verbose"), True if backend != "cpu" else None)
                record = json.loads((output / "run.json").read_text())
                self.assertEqual(record["threads"], 2)
                self.assertEqual(record["recipe"]["stages"][2]["transform"], "SyN[0.2,3,0]")
                self.assertEqual(record["inputs"]["fixed"]["sha256"], cli._digest(self.fixed))
                self.assertIn("not_external_acceptance", record["quality_status"])
                self.assertEqual((output / "console.log").read_text(), terminal.getvalue())

    def test_failure_keeps_english_log_without_success_record(self):
        args = cli.parse_args(self.arguments())
        terminal = io.StringIO()
        with patch.object(cli, "_execute", side_effect=RuntimeError("GPU unavailable")), redirect_stderr(terminal):
            with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                cli.run_registration(args)
        self.assertIn("RuntimeError: GPU unavailable", (self.output / "console.log").read_text())
        self.assertFalse((self.output / "run.json").exists())


if __name__ == "__main__":
    unittest.main()
