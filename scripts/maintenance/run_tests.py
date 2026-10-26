"""Run explicitly selected test groups; CPU is the default, Metal is opt-in."""

import argparse
import os
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))
GROUPS = {"core": ("core",), "contracts": ("contracts",),
          "cpu-fp64": ("cpu_fp64",), "cpu-fp32": ("cpu_fp32",),
          "metal": ("metal",), "cpu": ("core", "contracts", "cpu_fp64", "cpu_fp32"),
          "all": ("core", "contracts", "cpu_fp64", "cpu_fp32", "metal")}


def selected_suite(group, pattern="test_*.py"):
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for folder in GROUPS[group]:
        suite.addTests(loader.discover(str(PROJECT / "tests" / folder), pattern=pattern,
                                      top_level_dir=str(PROJECT)))
    if loader.errors:
        raise ImportError("Test discovery failed:\n" + "\n".join(loader.errors))
    return suite


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(GROUPS), default="cpu")
    parser.add_argument("--pattern", default="test_*.py", help="Select affected test modules, e.g. test_user_cli.py")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--list", action="store_true", help="List discovered tests without running them")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    if args.suite in ("metal", "all"):
        os.environ["FAST_ANTS_TEST_METAL"] = "1"
    else:
        os.environ.pop("FAST_ANTS_TEST_METAL", None)
    suite = selected_suite(args.suite, args.pattern)
    if args.list:
        for name in test_ids(suite):
            print(name)
        return 0
    if suite.countTestCases() == 0:
        parser.error("No tests matched the selected group and pattern")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
