# Installation

[English](installation.md) · [简体中文](../zh-CN/installation.md) · [Documentation](index.md)

## Requirements

- Python 3.11 or later.
- NumPy, NiBabel, and Numba for CPU execution.
- Apple Silicon macOS and MLX for the optional Metal backend.
- An NVIDIA GPU, CUDA 12.x toolkit and CuPy for the optional CUDA backend.

The CPU and GPU registration implementations are included in this package. Official ANTs is used by the comparison tools in `validation/`.

The Python distribution, import package and command-line entry point are all named `fast_ants`. The source repository is `Fast_ANTs`.

## Install a wheel

Download the wheel from the [v0.2.0 release](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.2.0), or create a virtual environment and build and install it from the source tree:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_ants-0.2.0-py3-none-any.whl
fast_ants --version
```

For Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

To enable Metal on Apple Silicon macOS:

```bash
python -m pip install './dist/fast_ants-0.2.0-py3-none-any.whl[metal]'
fast_ants register --help
```

To enable CUDA on a compatible NVIDIA system:

```bash
python -m pip install './dist/fast_ants-0.2.0-py3-none-any.whl[cuda]'
```

Metal and CUDA fitting use FP32; CPU execution supports FP64 and FP32 and loads independently of MLX and CuPy. The [v0.1.0 release](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.1.0) remains available with its CPU and Metal wheel.

## Install from source

The project source is maintained in the [Fast_ANTs repository](https://github.com/ayakacxy/Fast_ANTs). From the root of a v0.2.0 source tree, install the CPU dependencies:

```bash
python -m pip install -e .
```

For Metal, install with `python -m pip install -e '.[metal]'`; for CUDA, use `python -m pip install -e '.[cuda]'`. To build a wheel, run `python -m pip wheel . --no-deps -w dist`. The complete source tree includes examples, tests and validation tools; the Python source distribution contains the installable package and documentation.

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

## CUDA environment

The [CUDA environment](../../environments/cuda.yml) is named `ants` and uses Python 3.11. Run the following from the project root before creating the environment or installing packages. The ignored `setup/cuda` directory keeps the environment, package downloads, temporary files, and compilation caches in this project:

```bash
mkdir -p "$PWD/setup/cuda"/{tmp,envs,pkgs,cache/{xdg,pip,cupy,cuda,numba,python}}
export CONDA_ENVS_PATH="$PWD/setup/cuda/envs"
export CONDA_PKGS_DIRS="$PWD/setup/cuda/pkgs"
export CONDA_REGISTER_ENVS=false
export TMPDIR="$PWD/setup/cuda/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PWD/setup/cuda/cache/xdg"
export PIP_CACHE_DIR="$PWD/setup/cuda/cache/pip"
export CUPY_CACHE_DIR="$PWD/setup/cuda/cache/cupy"
export CUDA_CACHE_PATH="$PWD/setup/cuda/cache/cuda"
export NUMBA_CACHE_DIR="$PWD/setup/cuda/cache/numba"
export PYTHONPYCACHEPREFIX="$PWD/setup/cuda/cache/python"
conda env create -f environments/cuda.yml
conda activate ants
python -m pip install -e '.[cuda]'
```

Repeat the environment-variable exports before activating `ants` in a new shell. The optional dependency is `cupy-cuda12x>=14,<15`; CPU imports do not require CuPy. This extra uses an installed CUDA 12.x toolkit. When several toolkits are installed, set `CUDA_PATH` to the intended installation, as described in the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

Select `--backend cuda --precision fp32` for full rigid, affine, and SyN registration, or call `register_cuda` from Python. An unavailable CUDA runtime or device raises an error without switching to CPU. See [usage](usage.md) and [development](development.md).

CUDA fitting is FP32. Final field quality checks use native FP64; the implementation does not provide FP64 CUDA fitting. Current real-MRI and device coverage is described in [performance](performance.md) and [quality assessment](validation.md).

## Development setup

```bash
python -m pip install -e '.[test]'
python -m pip wheel . --no-deps -w dist
```

See [development](development.md) for test commands and package builds. MRI datasets, template images, and generated registration outputs are stored separately from the installed package.
