# Performance measurements

[English](performance.md) · [简体中文](../zh-CN/performance.md) · [Documentation](index.md)

## Native-resolution registration

Measured on 2026-10-25 using IXI002-Guys-0828 as the fixed image and IXI012-HH-1211 as the moving image. Both images are 256×256×150 voxels. The table compares the current implementation with ANTs 2.6.5 at matching precision.

| Backend | Time / s | ANTs precision | ANTs time / s | Speedup |
| --- | ---: | --- | ---: | ---: |
| CPU FP64 | 59.27 | `--float 0` | 505.85 | 8.53× |
| CPU FP32 | 92.01 | `--float 1` | 507.48 | 5.52× |
| Metal FP32 | 21.36 | `--float 1` | 507.48 | 23.76× |

Speedup is ANTs time divided by the backend time. Each engine has one formal measurement. CPU FP32 took longer than CPU FP64 in this run; float32 storage alone does not determine end-to-end runtime.

## Test setup

- Apple M5 (Mac17,3), 10 logical CPU cores; macOS 27.0.1 arm64.
- Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, NiBabel 5.4.2, MLX 0.32.4.
- The bundled `brain-syn` recipe, 10 threads, seed 20261022; `itk_scanner` geometry and `itk` intensity scaling.
- Each Python backend received one complete warmup immediately before its formal run. ANTs had no separate full warmup. All engines ran serially; the input images had already been read by the preceding Python runs.
- AC power, battery at 80% and not charging. Background load and temperature were not fixed.

Timing includes process startup and imports, input reading, registration, final GPU synchronization/download where applicable, output QA, and completed output writing. Python backends include their built-in QA; ANTs does not perform that additional step. The external timer's startup is excluded.

## Registration parameters

| Stage | Metric | Maximum iterations | Shrink factors | Image smoothing / vox |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI, 32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| Affine | Mattes MI, 32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| SyN | Local CC, radius 2 | 100×70×50×20 | 8×4×2×1 | 3×2×1×0 |

Sampling is dense. SyN uses step 0.2, update-field variance 3 and total-field variance 0. The convergence threshold is 1e-6 with a window of 10. Iteration counts can stop early through convergence. This recipe disables winsorization and histogram matching.

## Reproduce

Install official ANTs in the environment and prepare the input images listed in [the validation guide](../../validation/README.md). From the project root, use a new output directory:

```bash
python -B validation/benchmarks/measure_current_native.py \
  --source src --config src/fast_ants/presets/brain-syn.json \
  --threads 10 --output validation/reports/native-local
```

The runner measures CPU FP64, ANTs float0, CPU FP32, Metal FP32, and ANTs float1. It requires a working Metal device. Warmup images and redundant intermediate outputs are removed after recording their hashes; each formal run retains its four main output files.

[Machine-readable results](../assets/benchmarks/native-20261025.json) include exact times, parameters, package versions, input checksums, and runtime-source hashes. Full local logs and outputs are stored under `validation/reports/current-native-20261025/`. For implementation details, see [optimizations](optimizations.md); quality metrics are described in [testing and quality assessment](validation.md).
