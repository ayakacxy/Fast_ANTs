# Performance measurements

[English](performance.md) · [简体中文](../zh-CN/performance.md) · [Documentation](index.md)

## Native-resolution registration

Fast ANTs v0.1.0 registers IXI012-HH-1211 to IXI002-Guys-0828 with the `brain-syn` preset. Both images are 256×256×150 voxels.

| Backend | Complete process / s |
| --- | ---: |
| Metal FP32 | 15.34 |
| CPU FP32 | 42.57 |
| CPU FP64 | 60.44 |

Each backend was measured once in a fresh process using existing compilation caches. Timing includes startup, imports, image reading, registration, GPU synchronization and downloads where applicable, output quality checks and completed file writes.

Numba recorded 40 disk-cache hits and one miss for CPU FP32; the compilation is included in its reported time. Runtime varies with compilation, temperature, background load and the number of iterations needed to converge.

[Benchmark data](../assets/benchmarks/v0.1.0-native.json) contains full timings, input checksums, parameters, package versions, source hashes and output checksums. See [optimizations](optimizations.md) for implementation details and [quality assessment](validation.md) for evaluating registration results.

## ANTs reference timings

ANTs 2.6.5 has the following single-run timings for the same inputs, parameters and thread count:

| Precision | ANTs option | Complete process / s |
| --- | --- | ---: |
| FP64 | `--float 0` | 505.85 |
| FP32 | `--float 1` | 507.48 |

These reference runs have no separate full warmup. Fast ANTs includes its built-in output quality checks; ANTs does not run that additional step. Timing comparisons should also account for cache state, temperature and background load.

## Test setup

- Apple M5 (Mac17,3), 10 logical CPU cores; macOS 27.0.1 arm64.
- Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, NiBabel 5.4.2, MLX 0.32.4.
- The bundled `brain-syn` recipe, 10 threads, seed 20261022; `itk_scanner` geometry and `itk` intensity scaling.
- One fresh process per Fast ANTs backend, using existing compilation caches. Small operator calls compile the SIMD kernels before timing.
- Engines run serially. Reports include power observations; background load and temperature are not fixed.

The external timing runner's startup is excluded.

## Registration parameters

| Stage | Metric | Maximum iterations | Shrink factors | Image smoothing / vox |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI, 32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| Affine | Mattes MI, 32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| SyN | Local CC, radius 2 | 100×70×50×20 | 8×4×2×1 | 3×2×1×0 |

Sampling is dense. SyN uses step 0.2, update-field variance 3 and total-field variance 0. The convergence threshold is 1e-6 with a window of 10. Iteration counts can stop early through convergence. This recipe disables winsorization and histogram matching.

## Reproduce

Prepare the inputs listed in [the validation guide](../../validation/README.md). To measure CPU FP32 once using existing caches, run from the project root with a new output directory:

```bash
python -B validation/benchmarks/measure_fp32_pipeline.py \
  --source src --config src/fast_ants/presets/brain-syn.json \
  --case ixi002_to_ixi012 --resolution native --threads 10 \
  --engine cpu --warmups 0 --repeats 1 \
  --output validation/reports/native-cpu-fp32
```

For Metal, select `--engine metal` and a different output directory. On the first use of a kernel, compilation is included in the elapsed time. GPU measurements require a working Metal device.

The [benchmark tools](../../validation/README.md) also support CPU FP64, official ANTs, repeated measurements and profiling. They save configurations, logs and output checksums under the selected directory.
