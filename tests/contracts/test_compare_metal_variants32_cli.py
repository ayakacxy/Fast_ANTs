"""CLI/history export only; no NumPy, JIT, MLX or registration calls."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import validation.benchmarks.compare_metal_variants32 as compare


class MetalVariantCLI(unittest.TestCase):
    def test_default_is_even_balanced_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = compare.parse_args(["--output", str(Path(temporary) / "new")])
            self.assertEqual(args.repeats, 4)
            self.assertEqual(args.warmups, 1)
            self.assertIsNone(args.variant)

    def test_defaults_expand_from_the_selected_source(self):
        defaults = ("parallel_cc", "gaussian_sections")
        available = defaults + ("paired_cc",)
        variants = compare.resolve_variants(None, defaults, available)
        self.assertEqual(variants["baseline"], defaults)
        self.assertEqual(variants["paired-cc"], defaults + ("paired_cc",))
        custom = compare.resolve_variants([compare.variant_spec("pair=default,paired_cc,paired_cc")], defaults, available)
        self.assertEqual(custom["pair"], defaults + ("paired_cc",))
        exact = compare.resolve_variants([compare.variant_spec("serial=")], defaults, available)
        self.assertEqual(exact["serial"], ())

    def test_unknown_or_unsafe_variants_are_rejected(self):
        for specification in ("missing-equals", "../bad=paired_cc", "baseline=paired_cc", "bad=has space"):
            with self.assertRaises(Exception):
                compare.variant_spec(specification)
        with self.assertRaisesRegex(ValueError, "Unknown Metal"):
            compare.resolve_variants([("bad", ("absent",))], ("parallel_cc",), ("parallel_cc",))

    def test_existing_output_and_invalid_controls_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = str(Path(temporary) / "new")
            for options in (["--repeats", "0"], ["--threads", "0"], ["--warmups", "-1"],
                            ["--variant", "pair=", "--variant", "pair=paired_cc"]):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    compare.parse_args(["--output", destination, *options])
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                compare.parse_args(["--output", temporary])

    def test_default_order_alternates_and_custom_starts_rotate(self):
        self.assertEqual([compare.run_order(("A", "B"), i) for i in range(4)],
                         [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]])
        starts = [compare.run_order(("A", "B", "C"), i)[0] for i in range(6)]
        self.assertEqual({label: starts.count(label) for label in ("A", "B", "C")}, {"A": 2, "B": 2, "C": 2})

    def test_saved_history_encodes_unfilled_window_without_mutation(self):
        def save_placeholder(value, path):
            path.write_text("placeholder artifact")

        api = SimpleNamespace(save_affine=save_placeholder, save_field=save_placeholder, save_image=save_placeholder)
        history = [{"kind": "Affine", "iterations": [{"convergence": float("inf"), "value": 0.25},
                                                       {"convergence": 0.01, "value": 0.2}]}]
        result = SimpleNamespace(linear_transform=None, forward_field=None, inverse_field=None, warped=None,
                                 history=history, configuration={}, quality={})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "saved"
            hashes = compare.save_outputs(api, result, directory)
            saved = json.loads((directory / "history.json").read_text())
            self.assertIsNone(saved[0]["iterations"][0]["convergence"])
            self.assertEqual(saved[0]["iterations"][1]["convergence"], 0.01)
            self.assertEqual(history[0]["iterations"][0]["convergence"], float("inf"))
            self.assertIn("history-encoding.json", hashes)
            self.assertIn("window was not yet filled", (directory / "history-encoding.json").read_text())

    def test_unexpected_nonfinite_history_values_still_fail_strict_json(self):
        for field, value in (("value", float("nan")), ("value", float("inf")),
                             ("convergence", float("nan")), ("convergence", -float("inf"))):
            history = [{"iterations": [{"convergence": float("inf"), field: value}]}]
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                json.dumps(compare.export_history(history), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
