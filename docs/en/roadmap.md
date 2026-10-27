# Roadmap

[English](roadmap.md) · [简体中文](../zh-CN/roadmap.md) · [Documentation](index.md)

## Supported features

- Three-dimensional rigid, affine, and standard greedy SyN registration.
- Mattes mutual information for linear stages and local cross-correlation for SyN.
- CPU FP64, CPU FP32, Apple Metal FP32, and CUDA FP32 execution.
- Image pyramids, affine and displacement-field I/O, image resampling, and quality summaries.
- Command-line and Python interfaces, JSON presets, and English progress logs.

## NVIDIA CUDA

The CUDA backend implements FP32 image and field operations, Mattes MI, paired CC updates, four-field SyN and native FP64 output QA through CuPy. Small optimizer and geometry controls remain on the host. CUDA dependencies are optional and an unavailable device raises an error.

Planned work includes broader MRI/geometry coverage, memory and synchronization profiling, and evaluation on more NVIDIA devices. FP64 fitting requires separate implementation and validation; final FP64 QA does not provide FP64 fitting support.

## Other planned work

- Extend registration examples across image directions, voxel spacings, and acquisition settings.
- Add automated CPU checks on more operating systems and dependency versions.
- Provide more consistent iteration logging across CPU and Metal backends.
- Profile metric evaluation, field operations, memory allocation and GPU synchronization.
- Expand preset and API examples while keeping parameter handling consistent between the CLI and Python.

Fast ANTs focuses on image registration and transform application. Segmentation, cortical thickness and other neuroimaging analysis components are outside its scope.
