<h1 align="center">Fast ANTs</h1>

<div align="center">

**An accelerated Python implementation of ANTs registration algorithms.**

[![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-5D5FEF)](docs/en/changelog.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![CPU: Numba](https://img.shields.io/badge/CPU-Numba-00A3E0)](docs/en/optimizations.md)
[![GPU: Metal](https://img.shields.io/badge/GPU-Metal-333333?logo=apple&logoColor=white)](docs/en/installation.md)
[![Precision: FP64 / FP32](https://img.shields.io/badge/Precision-FP64%20%2F%20FP32-6D4AFF)](docs/en/architecture.md)
[![CUDA: planned](https://img.shields.io/badge/CUDA-planned-76B900?logo=nvidia&logoColor=white)](docs/en/roadmap.md)

[English](README.md) · [简体中文](README.zh-CN.md) · [Documentation](docs/en/index.md) · [Algorithms](docs/en/algorithms.md) · [Optimizations](docs/en/optimizations.md) · [Performance](docs/en/performance.md)

🧠 **Rigid / affine / SyN** · ⚙️ **Multithreaded CPU** · ⚡ **Apple GPU** · 🐍 **Python API & CLI**

**15.34 s on Metal · 42.57 s on CPU FP32**\
Apple M5 · 256×256×150 MRI · [Benchmark details](docs/en/performance.md)

</div>

## 📚 Contents

- [Performance](#-performance)
- [Features](#-features)
- [Algorithm](#-algorithm)
- [Installation](#-installation)
- [Quick start](#-quick-start)
- [Python API](#-python-api)
- [Documentation](#-documentation)
- [License and acknowledgements](#-license-and-acknowledgements)

Fast ANTs follows the rigid → affine → greedy SyN workflow from ANTs/ITK, reimplementing image sampling, similarity metrics, filtering, and displacement-field operations in Python. Multithreaded Numba CPU kernels and custom Metal kernels through MLX accelerate these operations, with ANTs-compatible transform I/O.

## ⚡ Performance

Native-resolution MRI registration with the `brain-syn` recipe, using existing caches:

| Backend | Complete process / s |
| --- | ---: |
| Metal FP32 | **15.34** |
| CPU FP32 | **42.57** |
| CPU FP64 | **60.44** |

Measured on Apple M5 with a 256×256×150 MRI pair and 10 threads. Each backend ran once using existing compilation caches. Times include process startup, imports, image reading, registration, output quality checks and writing. See [benchmark settings and ANTs reference timings](docs/en/performance.md).

## 🧠 Features

- Rigid → affine → SyN registration with multiresolution image pyramids.
- Mattes mutual information for linear registration and local cross-correlation for SyN.
- CPU FP64, CPU FP32, and Apple GPU FP32 backends.
- ANTs-compatible affine and displacement-field I/O.
- Command-line and Python interfaces, English progress logs, and output quality checks.

CPU kernels use multithreading, SIMD, fused metric evaluation, and reusable workspaces. The Metal backend keeps images and fields in GPU arrays, combines sampling and histogram operations, and batches independent work to reduce synchronization. See [implementation optimizations](docs/en/optimizations.md).

An NVIDIA CUDA backend is the next development priority; see the [roadmap](docs/en/roadmap.md) for the implementation plan.

## 🧮 Algorithm

The [algorithm guide](docs/en/algorithms.md) gives the mathematical formulation of rigid and affine transforms, Mattes MI, local CC, and SyN. It also describes image gradients, multiresolution optimization, field composition, and inverse-field iteration.

## 📦 Installation

The Python distribution, import package and command-line entry point are all named `fast_ants`.

Requires Python 3.11 or later. Download the wheel from [v0.1.0](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.1.0) and install it:

```bash
python -m pip install ./fast_ants-0.1.0-py3-none-any.whl
```

For Apple Silicon macOS with Metal:

```bash
python -m pip install './fast_ants-0.1.0-py3-none-any.whl[metal]'
```

CPU execution uses NumPy, NiBabel and Numba. The Metal backend also uses MLX. See [installation](docs/en/installation.md) for source installation and conda environments.

## 🚀 Quick start

Register a moving image to a fixed image on CPU:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-cpu --backend cpu --precision fp64 \
  --preset brain-syn --threads 10 --verbose
```

Use the Apple GPU:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-metal --backend metal --precision fp32 \
  --preset brain-syn --threads 10 --verbose
```

Choose a new output directory and a thread count supported by your machine. The default input settings are `itk_scanner` geometry and `itk` intensity scaling. For standard NiBabel NIfTI handling, add `--geometry-policy nifti --scaling-policy nifti`.

The output directory contains the affine transform, forward and inverse displacement fields, warped image, iteration history, quality summaries, and `console.log`. See [usage](docs/en/usage.md) and [inputs and outputs](docs/en/inputs-outputs.md).

## 🐍 Python API

```python
from fast_ants import load_images, register_metal

fixed, moving = load_images(
    ("fixed.nii.gz", "moving.nii.gz"),
    geometry_policy="itk_scanner", scaling_policy="itk",
)
result = register_metal(
    fixed, moving, output_directory="registration-api",
    verbose=True, log_every=1,
)
warped = result.warped
```

Use `register` for CPU FP64 and `register32` for CPU FP32. The Python API defaults to a two-level registration schedule; the `brain-syn` CLI preset uses four levels. [This example](examples/register_api.py) runs the complete preset through Python.

## 📖 Documentation

- [Configuration](configs/README.md): preset parameters and custom JSON files.
- [Algorithms](docs/en/algorithms.md): transforms, similarity metrics, and SyN updates.
- [Optimizations](docs/en/optimizations.md): CPU parallelism, operator fusion, memory reuse, and GPU scheduling.
- [Architecture](docs/en/architecture.md): modules and backend implementations.
- [Testing and quality assessment](docs/en/validation.md).
- [Development](docs/en/development.md) and [contributing](.github/CONTRIBUTING.md).
- [Roadmap](docs/en/roadmap.md).

## 📄 License and acknowledgements

Fast ANTs is licensed under the [GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`).

The project follows registration methods implemented in ANTs and ITK. Their Apache-2.0 license texts are retained in [LICENSE-ANTs](licenses/LICENSE-ANTs) and [LICENSE-ITK](licenses/LICENSE-ITK), with copyright notices and implementation references in [Third-party notices](licenses/NOTICE.md).
