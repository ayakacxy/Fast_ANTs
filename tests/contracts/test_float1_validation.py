"""Small receipt/recipe tests; no registration, JIT or GPU work."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
from validation.quality.verify_float1_alignment import ARTIFACTS, PROJECT, digest, float1_command, normalized_command, validate_recipe, verify_reference


class Float1ReferenceTests(unittest.TestCase):
    def test_command_uses_float1_and_keeps_recipe(self):
        config = json.loads((PROJECT / "validation/configs/p00_dense.json").read_text())
        case = {"fixed": "/fixed.nii.gz", "moving": "/moving.nii.gz"}
        validate_recipe(config)
        command = float1_command(config, case, Path("/out"), Path("/antsRegistration"))
        self.assertEqual(command[command.index("--float") + 1], "1")
        self.assertEqual(command.count("--convergence"), 3)
        self.assertIn("[8x4,1e-07,8]", command)
        wrong = command.copy()
        wrong[wrong.index("--float") + 1] = "0"
        with self.assertRaisesRegex(ValueError, "float 1"):
            normalized_command(wrong)

    def test_unsupported_recipe_is_rejected(self):
        config = json.loads((PROJECT / "validation/configs/p00_dense.json").read_text())
        for key, value in (("winsorization", [.005, .995]), ("histogram_matching", True),
                           ("interpolation", "NearestNeighbor"), ("masks", ["mask"])):
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_recipe(changed)

    def test_reuse_requires_matching_inputs_recipe_command_binary_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, manifest, binary = [root / name for name in ("config.json", "cases.json", "antsRegistration")]
            for p in (config, manifest, binary):
                p.write_text(p.name)
            for name in ARTIFACTS:
                (root / name).write_text(name)
            command = [str(binary), "--float", "1", "--output", "old", "--transform", "Rigid[0.1]"]
            inputs = {"/fixed.nii.gz": "fixed-hash", "/moving.nii.gz": "moving-hash"}
            receipt = {"command": command, "binary_sha256": digest(binary), "input_sha256": inputs,
                       "recipe_sha256": digest(config), "artifacts": {n: digest(root / n) for n in ARTIFACTS}}
            path = root / "reference.json"
            path.write_text(json.dumps(receipt))
            result = verify_reference(root, command, inputs, config, manifest, binary)
            self.assertEqual(result["verified_float_option"], 1)
            for key, value in (("input_sha256", {}), ("recipe_sha256", "wrong"),
                               ("binary_sha256", "wrong"), ("artifacts", {})):
                changed = dict(receipt, **{key: value})
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    verify_reference(root, command, inputs, config, manifest, binary)
            path.write_text(json.dumps(receipt))
            for index, value in ((2, "0"), (6, "Rigid[0.2]")):
                changed = command.copy()
                changed[index] = value
                with self.assertRaises(ValueError):
                    verify_reference(root, changed, inputs, config, manifest, binary)
            (root / ARTIFACTS[1]).write_text("modified warp")
            with self.assertRaisesRegex(ValueError, "artifacts"):
                verify_reference(root, command, inputs, config, manifest, binary)


if __name__ == "__main__":
    unittest.main()
