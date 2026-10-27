# Performance measurements

[English](performance.md) · [简体中文](../zh-CN/performance.md) · [Documentation](index.md)

## Linux CUDA and CPU, 32 threads

The CPU and CUDA backends register IXI012-HH-1211 to IXI002-Guys-0828 at 256×256×150 using `brain-syn` and 32 threads. The same registration algorithm and parameters are used in v0.2.0. The [benchmark data](../assets/benchmarks/linux-native-32t.json) records the test configuration and source hashes.

| Backend | Run 1 / s | Run 2 / s | Median / s | Precision-matched ANTs / backend |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 8.43 | 8.04 | 8.24 | 40.60× |
| CPU FP32 | 51.14 | 52.43 | 51.79 | 6.46× |
| CPU FP64 | 84.18 | 86.99 | 85.58 | 4.63× |
| ANTs FP32 (`--float 1`) | 333.73 | 334.90 | 334.32 | — |
| ANTs FP64 (`--float 0`) | 395.52 | 397.15 | 396.34 | — |

Each measurement starts a fresh process and uses cached inputs and compiled kernels. Each Fast ANTs backend runs once to prepare its caches before timing; official ANTs uses compiled C++ and needs no JIT warmup. Engines run serially, first in the listed order and then in reverse order. The table reports both runs and their median.

Timing includes startup, imports, image reading, rigid/affine/SyN fitting, GPU synchronization and downloads where applicable, built-in QA and completed writes. The external timer's own startup is excluded. Fast ANTs writes the affine, forward/inverse fields, warped image and diagnostic files with gzip level 0; the optional pigz writer uses two workers per file. Official ANTs retains its native writer defaults and does not run Fast ANTs' additional QA. No time is subtracted.

Hardware is two Xeon Gold 6230 CPUs, 40 physical cores / 80 logical CPUs, and an RTX 2080 Ti with 11 GiB VRAM, on Linux x86-64. Registration uses 32 threads; BLAS and OMP environment limits are 1. Python is 3.11.17, NumPy 2.4.6, Numba 0.68.0, NiBabel 5.4.2 and CuPy 14.2.0; NVIDIA driver 570.133.07. The ANTs executable is 2.6.5.dev1-g89fa5be. CPU busy samples immediately before and after formal runs range from 0.30% to 2.52%.

[Linux benchmark data](../assets/benchmarks/linux-native-32t.json) contains per-run timings, input and source hashes, parameters, hardware/software versions, output hashes, host observations and independent saved-output QA. The latter uses one CPU FP64 sampler for every engine and runs outside timing. See [quality assessment](validation.md).

## Apple M5 release measurements

### Native-resolution registration

Fast ANTs v0.1.0 registers IXI012-HH-1211 to IXI002-Guys-0828 with the `brain-syn` preset. Both images are 256×256×150 voxels.

| Backend | Complete process / s | Foreground NCC | Mean landmark error / mm |
| --- | ---: | ---: | ---: |
| Metal FP32 | 15.34 | 0.662625 | 1.4299 |
| CPU FP32 | 42.57 | 0.664474 | 1.4382 |
| CPU FP64 | 60.44 | 0.664417 | 1.4345 |

Each backend was measured once in a fresh process using existing compilation caches. Timing includes startup, imports, image reading, registration, GPU synchronization and downloads where applicable, output quality checks and completed file writes.

Numba recorded 40 disk-cache hits and one miss for CPU FP32; the compilation is included in its reported time. Runtime varies with compilation, temperature, background load and the number of iterations needed to converge.

[Benchmark data](../assets/benchmarks/v0.1.0-native.json) contains full timings, input checksums, parameters, package versions, source hashes and output checksums. See [optimizations](optimizations.md) for implementation details and [quality assessment](validation.md) for evaluating registration results.

The quality columns use saved outputs with the same SHA-256 checksums as the timed outputs. Every transform is reapplied through the same CPU FP64 sampler outside timing. Foreground NCC uses 3,236,832 voxels; mean landmark error uses four coarse corresponding points. Metal has positive forward/inverse Jacobians and inverse-consistency RMS values of 0.02297 and 0.02099 mm. See [Apple M5 quality results](validation.md#apple-m5-native-mri-results) for definitions and both composition orders.

### ANTs reference timings

ANTs 2.6.5 has the following single-run timings for the same inputs, parameters and thread count:

| Precision | ANTs option | Complete process / s | Foreground NCC | Mean landmark error / mm |
| --- | --- | ---: | ---: | ---: |
| FP64 | `--float 0` | 505.85 | 0.664417 | 1.4345 |
| FP32 | `--float 1` | 507.48 | 0.664369 | 1.4315 |

These reference runs have no separate full warmup. Fast ANTs includes its built-in output quality checks; ANTs does not run that additional step. Timing comparisons should also account for cache state, temperature and background load.

### Test setup

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
