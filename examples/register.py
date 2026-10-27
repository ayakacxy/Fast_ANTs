"""Register two local images using the bundled recipe.

Usage: python examples/register.py fixed.nii.gz moving.nii.gz output-dir
Append --backend metal --precision fp32 --verbose to select the Apple GPU.
Append --backend cuda --precision fp32 --verbose to select an NVIDIA GPU.
"""

import sys

from fast_ants.cli import main

raise SystemExit(main(["register", *sys.argv[1:3], "--output", *sys.argv[3:]]))
