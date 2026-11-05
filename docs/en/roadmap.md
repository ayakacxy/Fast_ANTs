# Roadmap

[English](roadmap.md) · [简体中文](../zh-CN/roadmap.md) · [Documentation](index.md)

## Supported features

- Three-dimensional rigid, affine, and standard greedy SyN registration.
- Mattes mutual information for linear stages and local normalized cross-correlation for SyN.
- CPU FP64, CPU FP32, Apple Metal FP32, and CUDA FP32 execution.
- Image pyramids, affine and displacement-field I/O, image resampling, and quality summaries.
- Command-line and Python interfaces, JSON presets, and English progress logs.

## NVIDIA CUDA

CUDA supports FP32 rigid, affine and SyN registration with FP64 final field quality checks.

Measurements on Linux and Windows are reported in [Performance measurements](performance.md) and [Testing and quality assessment](validation.md).

Planned work includes FP64 fitting, broader MRI and geometry coverage, more NVIDIA devices, and memory and synchronization profiling.

## Other planned work

- Extend registration examples across image directions, voxel spacings, and acquisition settings.
- Add automated CPU checks on more operating systems and dependency versions.
- Provide more consistent iteration logging across CPU and Metal backends.
- Profile metric evaluation, field operations, memory allocation and GPU synchronization.
- Expand preset and API examples while keeping parameter handling consistent between the CLI and Python.
