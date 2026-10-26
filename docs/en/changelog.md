# Changelog

[English](changelog.md) · [简体中文](../zh-CN/changelog.md)

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
