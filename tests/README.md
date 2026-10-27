# Tests

[English](README.md) · [简体中文](README.zh-CN.md) · [Development guide](../docs/en/development.md)

| Directory | Tests |
| --- | --- |
| `core/` | Coordinates, transforms, image I/O, shared types, and quality calculations |
| `cpu_fp64/` | FP64 metrics, optimization, fields, and registration |
| `cpu_fp32/` | FP32 operators, numerical stability, workspaces, and host controls |
| `metal/` | Metal GPU operators and registration components |
| `cuda/` | CUDA operators, registration, retained state, and final QA |
| `contracts/` | CLI arguments, benchmark metadata, logging, and mocked orchestration |
| `fixtures/` | Small shared reference inputs and metadata |

Run from the project root:

```sh
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_stable32.py
python -B scripts/maintenance/run_tests.py --suite cpu --list
```

`cpu` is the default and includes core, interface, FP64, and FP32 tests. `--pattern` selects a filename and `--list` prints test names without running them. Numerical tests may compile Numba kernels on first use.

Metal tests require a supported Apple GPU and explicit selection:

```sh
python -B scripts/maintenance/run_tests.py --suite metal
```

CUDA tests require CuPy and an accessible NVIDIA GPU:

```sh
python -B scripts/maintenance/run_tests.py --suite cuda
```

`--suite all` includes both CPU and GPU tests. Registration quality metrics and ANTs comparisons are described in [testing and evaluation](../docs/en/validation.md).
