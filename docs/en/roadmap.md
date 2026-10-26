# Roadmap

[English](roadmap.md) · [简体中文](../zh-CN/roadmap.md) · [Documentation](index.md)

## Supported features

- Three-dimensional rigid, affine, and standard greedy SyN registration.
- Mattes mutual information for linear stages and local cross-correlation for SyN.
- CPU FP64, CPU FP32, and Apple Metal FP32 execution.
- Image pyramids, affine and displacement-field I/O, image resampling, and quality summaries.
- Command-line and Python interfaces, JSON presets, and English progress logs.

## NVIDIA CUDA

The planned CUDA backend uses FP32 voxel calculations and shares the registration algorithms and Python interfaces.

1. **Device runtime.** Add device-array management, kernel launch support, streams/events, allocation reuse, and explicit host/device transfers. Keep CUDA dependencies optional and load them only when the backend is selected.
2. **Basic operators.** Implement trilinear and nearest-neighbour sampling, coordinate mapping, image gradients, Gaussian filtering, and reductions for counts, sums, and maximum norms.
3. **Similarity metrics.** Port fixed-bin CSR indexing, partial PDFs, streamed Mattes MI derivatives, and GPU metric finalization. Add centred local CC statistics and paired updates.
4. **Fields and SyN.** Implement field resampling, scaled composition, inverse residuals and bounded updates, then integrate the four-field SyN loop and multiresolution state adaptation. Reuse per-level workspaces and keep image/field data on the GPU.
5. **Pipeline and asynchronous execution.** Connect CUDA operators to the shared host models and optimizer. Overlap independent preparation and transfers through streams, retaining synchronization for metric values and stopping decisions. Add backend selection without changing CPU or Metal dependencies.
6. **Testing and measurement.** Compare operators and saved registration outputs with the CPU implementation and official ANTs at the same precision. Check image similarity, landmark error, Jacobians, and inverse consistency. Run complete registration benchmarks on NVIDIA hardware, recording compilation/warmup, synchronization, input/output, and device model.

FP64 support depends on target GPU capabilities and application requirements, with separate numerical and performance evaluation. CUDA runtime and dependency selection remain open.

## Other planned work

- Extend registration examples across image directions, voxel spacings, and acquisition settings.
- Add automated CPU checks on more operating systems and dependency versions.
- Provide more consistent iteration logging across CPU and Metal backends.
- Profile metric evaluation, field operations, memory allocation and GPU synchronization.
- Expand preset and API examples while keeping parameter handling consistent between the CLI and Python.

Fast ANTs focuses on image registration and transform application. Segmentation, cortical thickness and other neuroimaging analysis components are outside its scope.
