# Installation

[English](installation.md) · [简体中文](../zh-CN/installation.md) · [Documentation](index.md)

## Requirements

- Python 3.11 or later.
- NumPy, NiBabel, and Numba for CPU execution.
- Apple Silicon macOS and MLX for Metal.
- An NVIDIA GPU, a compatible driver, CUDA 12.x or 13.x libraries, and CuPy for CUDA.

The distribution, import package and command are named `fast_ants`. Official ANTs is used for comparison tests in `validation/`.

## Install a wheel

Build the v0.3.0 wheel from the project root. Published versions are available from [GitHub Releases](https://github.com/ayakacxy/Fast_ANTs/releases).

Select an extra for GPU execution:

| Extra | Backend | Dependency |
| --- | --- | --- |
| `metal` | Apple Metal FP32 | MLX |
| `cuda` | CUDA FP32 with CUDA 12.x | `cupy-cuda12x>=14,<15` |
| `cuda13` | CUDA FP32 with CUDA 13.x | `cupy-cuda13x>=14,<15` |

CPU execution supports FP64 and FP32; Metal and CUDA fitting use FP32. CUDA uses FP64 for final field quality checks. Install one CuPy variant matching your CUDA libraries.

### Windows PowerShell

Create an environment, build the wheel and run CPU registration:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip wheel . --no-deps -w dist
.\.venv\Scripts\python.exe -m pip install './dist/fast_ants-0.3.0-py3-none-any.whl'
.\.venv\Scripts\fast_ants.exe --version
.\.venv\Scripts\fast_ants.exe register fixed.nii.gz moving.nii.gz --output registration-cpu --backend cpu --precision fp64 --preset brain-syn --cpu-threads 16
```

For CUDA 13.x, add the optional dependency in the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install './dist/fast_ants-0.3.0-py3-none-any.whl[cuda13]'
```

Choose a CPU thread count supported by your machine. Numba and CUDA kernels compile on first use; see [Windows performance](performance.md#windows-cuda-and-cpu-16-threads) for cached and first-use timings.

For ANTs comparisons, extract the Windows archive from the [ANTs 2.6.5 release](https://github.com/ANTsX/ANTs/releases/tag/v2.6.5) and pass the path to `antsRegistration.exe` to the validation runner with `--ants-registration`.

### Linux and macOS (Bash)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_ants-0.3.0-py3-none-any.whl
fast_ants --version
```

For GPU execution, append the matching extra from the table to the wheel filename; for example, use `fast_ants-0.3.0-py3-none-any.whl[metal]` on Apple Silicon macOS.

## Install from source

From the [source tree](https://github.com/ayakacxy/Fast_ANTs):

```bash
python -m pip install -e .
```

Add `[metal]`, `[cuda]` or `[cuda13]` for the corresponding GPU dependencies, for example `python -m pip install -e '.[cuda13]'`. The complete source archive includes tests and validation tools; the Python source distribution contains the package and documentation.

## Conda environments

The [CPU](../../environments/cpu.yml) and [Metal](../../environments/metal.yml) files specify their dependencies:

```bash
conda env create -f environments/cpu.yml
conda activate fast-ants-cpu
python -m pip install -e .
```

For Metal, use `environments/metal.yml`, activate `fast-ants-metal`, and install `.[metal]`. The [reference environment](../../environments/reference.yml) includes ANTs for numerical comparisons.

## CUDA environment

The [CUDA environment](../../environments/cuda.yml) uses Python 3.11 and the CUDA 12.x CuPy package:

```bash
conda env create -f environments/cuda.yml
conda activate ants
python -m pip install -e '.[cuda]'
```

CUDA libraries can come from an installed toolkit or CuPy's `[ctk]` extra, for example `python -m pip install 'cupy-cuda12x[ctk]>=14,<15'` for CUDA 12.x. With multiple toolkits, `CUDA_PATH` selects the installation. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Development setup

```bash
python -m pip install -e '.[test,build]'
```

See [Development guide](development.md) for tests and package builds.
