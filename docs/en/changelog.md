# Changelog

[English](changelog.md) · [简体中文](../zh-CN/changelog.md)

## 0.2.0

This version adds NVIDIA CUDA execution to the CPU and Apple Metal registration workflows introduced in v0.1.0.

### Added features

- CUDA FP32 rigid, affine, and standard greedy SyN registration through optional CuPy dependencies.
- `register_cuda` Python API and `fast_ants register --backend cuda --precision fp32`.
- Native FP64 final field QA on CUDA, including inverse consistency, finite checks and exact percentile summaries; fitting remains FP32.
- Default English CLI progress logs, interactive per-level `tqdm` bars on Metal and CUDA, and plain-text diagnostic logs in `console.log`.
- `--quiet` disables progress and verbose diagnostics; `--log-every` controls GPU iteration logging.
- Project-local CUDA conda environment named `ants`; CPU imports remain independent of CuPy and MLX.

### Implementation and performance

- Device-resident images and four-field SyN state, with shared geometry and host-side optimizer controls.
- CUDA Graph replay for Mattes evaluation, grouped fixed-bin histograms, value-only evaluation and bounded pinned control transfers.
- Paired CC updates and inverse-field scheduling, coalesced field filtering and reusable geometry buffers.
- Native FP64 output QA without downloading dense QA scratch arrays.
- Streaming NIfTI output with gzip level 0 by default and optional pigz; `FAST_ANTS_GZIP_LEVEL` selects compression level.
- Native 256×256×150 MRI registration on RTX 2080 Ti and dual Xeon Gold 6230 with 32 threads: CUDA FP32 8.24 s, CPU FP32 51.79 s and CPU FP64 85.58 s, using the median of two complete fresh processes with existing caches.
- Independent saved-output checks for all measured backends and official ANTs. See [performance settings](performance.md) and [quality assessment](validation.md); v0.1.0 Apple M5 results remain recorded separately.

## 0.1.0

Fast ANTs is licensed under GPL-3.0-only, with ANTs and ITK license texts and attribution included.

### Features

- Three-dimensional rigid, affine, and standard greedy SyN registration.
- CPU FP64, CPU FP32, and Apple Metal FP32 backends.
- Mattes mutual information, local cross-correlation, multiresolution image pyramids, and transform application.
- `fast_ants register` command with the `brain-syn` preset and JSON configuration.
- NIfTI image and displacement-field I/O, affine transform I/O, and registration quality summaries.
- English progress logs and Python registration API.
- English and Simplified Chinese documentation, examples, and development tools.

### Performance

- Multithreaded Numba CPU kernels, fused sampling and metric evaluation, and reusable workspaces.
- Four-lane FP32 SIMD for dense sample mapping, Parzen histogram updates, derivative channels and unmasked local CC statistics.
- Shared interpolation indices, cached zero-background gradient support, sparse linear Jacobians and reusable CC merge coefficients.
- Device-resident Metal operations, streamed mutual-information derivatives, centred local statistics, and asynchronous submission of independent work.
- Parallel image I/O and chunked output processing.
