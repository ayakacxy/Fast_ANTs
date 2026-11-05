"""Check real child-process timing, exit handling and platform memory accounting."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validation.benchmarks.measure_cuda_native import engines, schedule


class ProcessTimingTests(unittest.TestCase):
    def measure(self, code, timeout=10):
        timer = Path(__file__).resolve().parents[2] / "validation/common/time_process.py"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "process.json"
            result = subprocess.run([sys.executable, "-B", str(timer), "--output", str(output),
                                     "--timeout", str(timeout), "--", sys.executable, "-c", code],
                                    capture_output=True, text=True, timeout=20)
            self.assertTrue(output.is_file(), result.stderr)
            return result.returncode, json.loads(output.read_text())

    def test_real_child_memory_and_exit_status(self):
        status, report = self.measure("import time; data=bytearray(32*1024*1024); time.sleep(.1)")
        self.assertEqual(status, 0)
        self.assertFalse(report["timed_out"])
        self.assertGreaterEqual(report["wall_seconds"], .1)
        self.assertGreater(report["peak_process_rss_bytes"], 32 * 1024 * 1024)
        status, report = self.measure("raise SystemExit(7)")
        self.assertEqual((status, report["returncode"]), (7, 7))

    def test_timeout_stops_target_and_retains_measurement(self):
        status, report = self.measure("import time; time.sleep(10)", timeout=.1)
        self.assertEqual((status, report["returncode"]), (124, 124))
        self.assertTrue(report["timed_out"])
        self.assertLess(report["wall_seconds"], 5)

    def test_five_engines_reverse_formal_order(self):
        rows = schedule(2, official_warmup=False, include_fp64=True)
        self.assertEqual([e for e, label in rows if label == "repeat-1"], list(engines(True)))
        self.assertEqual([e for e, label in rows if label == "repeat-2"], list(reversed(engines(True))))
        self.assertEqual([e for e, label in rows if label == "warmup"], ["cuda", "cpu", "cpu-fp64"])


if __name__ == "__main__":
    unittest.main()
