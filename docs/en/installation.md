# Installation

[English](installation.md) · [简体中文](../zh-CN/installation.md) · [Documentation](index.md)

## Requirements

- Python 3.11 or later.
- NumPy, NiBabel, and Numba for CPU execution.
- Apple Silicon macOS and MLX for the optional Metal backend.

The CPU and GPU registration implementations are included in this package. Official ANTs is used by the comparison tools in `validation/`.

## Install with pip

Create a virtual environment and install from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
fast-ants --help
```

For Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

To enable Metal on Apple Silicon macOS:

```bash
python -m pip install -e '.[metal]'
fast-ants register --help
```

The Metal backend uses FP32 arithmetic. CPU execution supports FP64 and FP32, and loads independently of MLX.

## Conda environments

Use the CPU environment for CPU registration:

```bash
conda env create -f environments/cpu.yml
conda activate fast-ants-cpu
python -m pip install -e .
```

Use the Metal environment on Apple Silicon macOS:

```bash
conda env create -f environments/metal.yml
conda activate fast-ants-metal
python -m pip install -e '.[metal]'
```

[CPU](../../environments/cpu.yml) and [Metal](../../environments/metal.yml) environment files specify the dependency versions. The [reference environment](../../environments/reference.yml) also installs ANTs for numerical comparisons.

## Development setup

```bash
python -m pip install -e '.[test]'
python -m pip wheel . --no-deps -w dist
```

See [development](development.md) for test commands and package builds. MRI datasets, template images, and generated registration outputs are stored separately from the installed package.
