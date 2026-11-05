# Roadmap

[English](roadmap.md) · [简体中文](../zh-CN/roadmap.md) · [Documentation](index.md)

## Supported features

- Three-dimensional rigid, affine, and standard greedy SyN registration.
- Mattes mutual information for linear stages and local normalized cross-correlation for SyN.
- CPU FP64, CPU FP32, Apple Metal FP32, and CUDA FP32 execution.
- Image pyramids, affine and displacement-field I/O, image resampling, and quality summaries.
- Command-line and Python interfaces, JSON presets, and English progress logs.

CUDA supports FP32 rigid, affine and SyN registration with FP64 final field quality checks.

Measurements on Linux and Windows are reported in [Performance measurements](performance.md) and [Testing and quality assessment](validation.md).

## v0.4.0: quality assessment and batch processing

Validation is limited to MRI. Planned priorities, in implementation order:

### 1. Multi-case quality assessment

Use the existing Fast ANTs algorithms for standard-dataset validation. Official ANTs comparisons remain in the published benchmarks; this work runs Fast ANTs only.

- Define a reproducible case list using manually labelled datasets such as [LPBA40](https://loni.usc.edu/research/atlases), covering different image orientations, voxel spacings and acquisition settings.
- Evaluate transformed moving-image labels against fixed-image manual labels. Report label Dice, NCC, landmark error where annotations are available, Jacobian statistics, failure rate and runtime distributions. Separate first-use and cached timings.
- Completion: the case list reproduces per-case results and an aggregate report, with input provenance, configuration and software versions recorded.

### 2. Environment diagnostics and automated tests

- Add `fast_ants doctor` to check Python, dependencies and device availability, then run a small computation on the selected CPU, CUDA or Metal backend.
- Automate ordinary wheel installation, CLI checks and small CPU registrations on Windows, Linux and macOS. Run GPU checks on the corresponding hardware.
- Completion: tests verify the installed package and runtime resources; diagnostics identify missing dependencies or runtime failures and report computation results.

### 3. Batch registration

- Add a batch command driven by a case list, with per-case outputs, continued execution after individual failures, resumable runs and a summary report.
- Evaluate process and compilation-cache reuse, using total batch time, peak memory and peak GPU memory as performance measures.
- Completion: resuming a batch skips completed cases, retains failure records and produces the same per-case results as standalone runs under the same settings, within numerical tolerances.

## Later work

- Add CUDA FP64 fitting and validate more NVIDIA devices.
- Provide more consistent iteration logging across CPU and Metal backends.
- Profile metric evaluation, field operations, memory allocation and GPU synchronization.
- Expand preset and API examples while keeping parameter handling consistent between the CLI and Python.
