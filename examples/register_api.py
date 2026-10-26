"""Use the Python API with the same complete bundled recipe as the CLI.

Usage: python examples/register_api.py fixed.nii.gz moving.nii.gz output-dir
The API uses explicit image-pull transform directions, in LPS millimetres.
"""

import sys
import os
from pathlib import Path
from types import SimpleNamespace

from fast_ants.cli import build_stages, load_recipe

fixed_path, moving_path, output = sys.argv[1:]
destination = Path(output)
if destination.exists():
    raise SystemExit("Output already exists: " + str(destination))
recipe, provenance = load_recipe(SimpleNamespace(config=None, preset="brain-syn"))
os.environ["NUMBA_NUM_THREADS"] = str(recipe["threads"])
for pool in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[pool] = "1"
import numba
from fast_ants import load_images, register
numba.set_num_threads(recipe["threads"])
fixed, moving = load_images((fixed_path, moving_path), geometry_policy="itk_scanner", scaling_policy="itk")
linear, syn = build_stages(__import__("fast_ants"), recipe, fixed.grid)
result = register(fixed, moving, linear_stages=linear, syn_stage=syn,
                  initialization_feature=recipe["initialization_feature"], seed=recipe["random_seed"],
                  output_directory=destination)
print("Registration complete; outputs saved to " + str(destination))
