"""Run meaningful CPU unit checks and persist their exact names and logs."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from scripts.maintenance.run_tests import GROUPS, selected_suite

PROJECT = _TOOL_PROJECT
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT / "validation/.numba-cache"))


def case_names(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from case_names(case)
        else:
            yield case.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--suite", choices=tuple(GROUPS), default="cpu")
    args = parser.parse_args()
    os.chdir(PROJECT)
    if args.suite in ("metal", "all"):
        os.environ["FAST_ANTS_TEST_METAL"] = "1"
    else:
        os.environ.pop("FAST_ANTS_TEST_METAL", None)
    suite = selected_suite(args.suite, args.pattern)
    names = list(case_names(suite))
    if not names:
        raise RuntimeError("No tests selected")
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    directory = PROJECT / "validation/reports" / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "unit-tests.log").write_text(stream.getvalue())
    report = {"status": "passed" if result.wasSuccessful() else "failed", "tests": names,
              "count": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "seconds_not_benchmark": time.perf_counter() - started}
    (directory / "unit-tests.json").write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), end="")
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
