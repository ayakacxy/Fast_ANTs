# Performance measurements

[English](performance.md) · [简体中文](../zh-CN/performance.md) · [Documentation](index.md)

## Native-resolution registration

Measured on 2026-10-26 using IXI002-Guys-0828 as the fixed image and IXI012-HH-1211 as the moving image. Both images are 256×256×150 voxels. Each backend ran once in a fresh process using existing caches, without another complete warmup.

| Backend | Complete process / s |
| --- | ---: |
| Metal FP32 | 15.34 |
| CPU FP32 | 42.57 |
| CPU FP64 | 60.44 |

CPU FP32 was measured after the SIMD and workspace changes described in [optimizations](optimizations.md). CPU FP64 and Metal were unchanged and retain their earlier same-day cached measurements. These are individual runs with uncontrolled background load and temperature, not stable mean runtimes.

The CPU FP32 run recorded 40 Numba disk-cache hits and one miss; that compilation is included in the 42.57 seconds. Compared with the preceding cached CPU FP32 run of 59.79 seconds, elapsed time fell by 28.8% (1.40× speedup). The affine, forward warp, inverse warp and warped-image file hashes are identical to the pre-optimization outputs. This checks preservation of the existing result; it does not establish equivalence to official ANTs.

[Cached-run measurements](../assets/benchmarks/native-cached-20261026.json) record exact times, source hashes, cache counters and output comparisons. No official ANTs run was added to this check.

## Precision-matched ANTs comparison

The v0.1.0 comparison below was measured on 2026-10-25 with the same image pair, using ANTs 2.6.5. These paired measurements are retained as recorded; the newer cached times above are not substituted into their speedup ratios.

| Backend | Time / s | ANTs precision | ANTs time / s | Speedup |
| --- | ---: | --- | ---: | ---: |
| CPU FP64 | 59.27 | `--float 0` | 505.85 | 8.53× |
| CPU FP32 | 92.01 | `--float 1` | 507.48 | 5.52× |
| Metal FP32 | 21.36 | `--float 1` | 507.48 | 23.76× |

Speedup is ANTs time divided by the backend time. Each engine has one formal measurement.

## Test setup

- Apple M5 (Mac17,3), 10 logical CPU cores; macOS 27.0.1 arm64.
- Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, NiBabel 5.4.2, MLX 0.32.4.
- The bundled `brain-syn` recipe, 10 threads, seed 20261022; `itk_scanner` geometry and `itk` intensity scaling.
- The 2026-10-26 check retained existing caches and ran each backend once, with no complete warmup. Small operator tests compiled the new SIMD kernels before the CPU FP32 measurement.
- In the 2026-10-25 ANTs comparison, each Python backend received one complete warmup immediately before its formal run; ANTs had no separate full warmup. All engines ran serially, and the input images had already been read by the preceding Python runs.
- Power observations are recorded in the reports. Background load and temperature were not fixed.

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

[ANTs comparison results](../assets/benchmarks/native-20261025.json) include exact times, parameters, package versions, input checksums, and runtime-source hashes. Full local reports are under `validation/reports/current-native-20261025/`, `validation/reports/cached-single-20261026/` and `validation/reports/cpu32-opt-20261026/`. Duplicate recheck volumes are removed after recording their hashes; the original comparison outputs remain available locally. For implementation details, see [optimizations](optimizations.md); quality metrics are described in [testing and quality assessment](validation.md).
