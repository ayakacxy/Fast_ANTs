# Documentation

[English](index.md) · [简体中文](../zh-CN/index.md) · [Fast ANTs](../../README.md)

Fast ANTs accelerates the ANTs rigid, affine, and standard greedy SyN registration workflow, with CPU FP64, CPU FP32, Metal FP32, and CUDA FP32 backends.

## Getting started

1. [Install](installation.md) the CPU package or optional Metal/CUDA dependencies.
2. Run a registration using the [command-line or Python interface](usage.md).
3. Read [inputs and outputs](inputs-outputs.md) for image coordinates, transform direction, and output files.
4. Customize the [registration preset](../../configs/README.md).

## Reference

| Topic | Contents |
| --- | --- |
| [Installation](installation.md) | Dependencies, conda environments, development setup |
| [Usage](usage.md) | Commands, Python API, parameters, logs |
| [Inputs and outputs](inputs-outputs.md) | NIfTI geometry, transform direction, files, resampling |
| [Algorithms](algorithms.md) | Linear transforms, Mattes MI, local CC, SyN |
| [Optimizations](optimizations.md) | Parallel kernels, fused operators, memory reuse, GPU scheduling |
| [Performance](performance.md) | Registration timings, test settings and measurement commands |
| [Architecture](architecture.md) | Shared modules, CPU, Metal and CUDA backends |
| [Testing and quality assessment](validation.md) | Numerical tests and registration quality metrics |
| [Development](development.md) | Tests, implementation guidelines, package builds |
| [Roadmap](roadmap.md) | Supported backends and planned improvements |
| [Changelog](changelog.md) | Version features |

See [CONTRIBUTING](../../.github/CONTRIBUTING.md) to contribute and [NOTICE](../../licenses/NOTICE.md) for implementation references and copyright notices.
