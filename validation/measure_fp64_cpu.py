"""Compatibility entry point; implementation moved to validation.benchmarks.measure_fp64_cpu."""
from pathlib import Path
import importlib
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    runpy.run_module('validation.benchmarks.measure_fp64_cpu', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('validation.benchmarks.measure_fp64_cpu')
