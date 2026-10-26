# Changelog

[English](changelog.md) · [简体中文](../zh-CN/changelog.md)

## 0.1.0 — 2026-10-26

Initial open-source version, licensed under GPL-3.0-only. ANTs and ITK license texts and attribution are retained.

### Added

- Three-dimensional rigid, affine, and standard greedy SyN registration.
- CPU FP64, CPU FP32, and Apple Metal FP32 backends.
- Mattes mutual information, local cross-correlation, multiresolution image pyramids, and transform application.
- `fast-ants register` command with the `brain-syn` preset and JSON configuration.
- NIfTI image and displacement-field I/O, affine transform I/O, and registration quality summaries.
- English progress logs and Python registration API.
- English and Simplified Chinese documentation, examples, and development tools.

### Implementation

- Numba parallel loops, fused sampling and metric operations, and reusable workspaces on CPU.
- Device-resident Metal operations, streamed mutual-information derivatives, centred local statistics, and asynchronous submission of independent work.
- Parallel image I/O and chunked output processing.
