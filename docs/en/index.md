# Documentation

[English](index.md) · [简体中文](../zh-CN/index.md) · [Fast ANTs](../../README.md)

Fast ANTs accelerates the ANTs rigid, affine, and standard greedy SyN registration workflow, with CPU FP64, CPU FP32, and Metal FP32 backends.

## Getting started

1. [Install](installation.md) the CPU package or optional Metal dependencies.
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
| [Architecture](architecture.md) | Shared modules, CPU backends, Metal backend |
| [Testing and quality assessment](validation.md) | Numerical tests and registration quality metrics |
| [Development](development.md) | Tests, implementation guidelines, package builds |
| [Roadmap](roadmap.md) | CUDA support and planned improvements |

See [CONTRIBUTING](../../CONTRIBUTING.md) to contribute and [NOTICE](../../NOTICE.md) for implementation references and copyright notices.

- [Performance](performance.md)
