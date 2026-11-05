# Performance measurements

[English](performance.md) · [简体中文](../zh-CN/performance.md) · [Documentation](index.md)

All tests register IXI012-HH-1211 to IXI002-Guys-0828 at 256×256×150 using the `brain-syn` preset. [Registration parameters](#registration-parameters) are shared across platforms; speedups compare matching precision on the same hardware: FP32 with ANTs `--float 1`, FP64 with `--float 0`.

Times cover process startup, imports, input reading, rigid/affine/SyN registration, GPU synchronization and downloads where applicable, and complete output writing. Fast ANTs also includes built-in quality checks. Official ANTs uses its native writer defaults without that additional step. The external timer's startup is excluded. Quality metrics are computed separately from saved transforms using a common CPU FP64 sampler; see [Testing and quality assessment](validation.md).

All measurements use fresh processes and run serially. Windows and Linux report medians of two runs: each Fast ANTs backend has one untimed full warmup, followed by two measured passes in opposite orders. macOS reports one run per backend using existing compilation caches. Official ANTs has no separate full warmup. Temperature and background load were not controlled. Per-run timings, full software versions and measurement records are linked below.

## Windows CUDA and CPU, 16 threads

Fast ANTs v0.3.0 on Windows 11: Intel Core i5-14400F (10 cores / 16 logical CPUs), 32 GB RAM and RTX 4060 Ti (8 GB VRAM). Numba and ITK use 16 threads; BLAS and OMP limits are 1. CUDA runtime 13.2, CuPy 14.2.0 and native Windows ANTs 2.6.5.

| Backend | Complete process / s | Speedup |
| --- | ---: | ---: |
| CUDA FP32 | 7.43 | 70.54× |
| CPU FP32 | 76.29 | 6.87× |
| CPU FP64 | 93.88 | 5.71× |
| ANTs FP32 | 524.23 | — |
| ANTs FP64 | 536.38 | — |

A separate single CUDA run with empty CuPy, CUDA driver and Numba compilation cache directories took **27.26 s**. Filesystem caches and GPU state were not reset. Fast ANTs writes with Python gzip level 0 and one writer thread per file.

[Windows benchmark data](../assets/benchmarks/windows-native-16t.json) · [Windows quality results](validation.md#native-windows-mri-results)

## Linux CUDA and CPU, 32 threads

Fast ANTs v0.2.0 on Linux x86-64: dual Xeon Gold 6230 (40 cores / 80 logical CPUs) and RTX 2080 Ti (11 GiB VRAM). Registration uses 32 threads; BLAS and OMP limits are 1. CuPy 14.2.0 and ANTs 2.6.5.dev1.

| Backend | Complete process / s | Speedup |
| --- | ---: | ---: |
| CUDA FP32 | 8.24 | 40.6× |
| CPU FP32 | 51.79 | 6.46× |
| CPU FP64 | 85.58 | 4.63× |
| ANTs FP32 | 334.32 | — |
| ANTs FP64 | 396.34 | — |

Input files are read before each measurement to warm the file cache. Fast ANTs writes with gzip level 0; the optional pigz writer uses two workers per file. CPU busy samples immediately before and after measured runs ranged from 0.30% to 2.52%.

[Linux benchmark data](../assets/benchmarks/linux-native-32t.json) · [Linux quality results](validation.md#native-linux-mri-results)

## macOS Metal and CPU, 10 threads

Fast ANTs v0.1.0 on macOS 27.0.1 arm64: Apple M5 (Mac17,3, 10 logical CPU cores). Registration uses 10 threads. MLX 0.32.4 and ANTs 2.6.5.

| Backend | Complete process / s | Speedup |
| --- | ---: | ---: |
| Metal FP32 | 15.34 | 33.1× |
| CPU FP32 | 42.57 | 11.9× |
| CPU FP64 | 60.44 | 8.37× |
| ANTs FP32 | 507.48 | — |
| ANTs FP64 | 505.85 | — |

Small operator calls compile SIMD kernels before timing. CPU FP32 recorded 40 Numba disk-cache hits and one miss; the resulting compilation time is included. Image loading uses `itk_scanner` geometry and `itk` intensity scaling.

[macOS benchmark data](../assets/benchmarks/v0.1.0-native.json) · [macOS quality results](validation.md#native-macos-mri-results)

## Registration parameters

| Stage | Metric | Maximum iterations | Shrink factors | Image smoothing / vox |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI, 32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| Affine | Mattes MI, 32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| SyN | Local CC, radius 2 | 100×70×50×20 | 8×4×2×1 | 3×2×1×0 |

Sampling is dense, with seed 20261022. SyN uses step 0.2, update-field variance 3 and total-field variance 0. The convergence threshold is 1e-6 with a window of 10; stages can stop early on convergence. Winsorization and histogram matching are disabled.

## Reproduce

Prepare the inputs in [the validation guide](../../validation/README.md). From the project root, measure CPU FP32 once using existing caches and a new output directory:

```bash
python -B validation/benchmarks/measure_fp32_pipeline.py \
  --source src --config src/fast_ants/presets/brain-syn.json \
  --case ixi002_to_ixi012 --resolution native --threads 10 \
  --engine cpu --warmups 0 --repeats 1 \
  --output validation/reports/native-cpu-fp32
```

For Metal, use `--engine metal` on a supported Apple GPU and select a different output directory. Compilation on first kernel use is included in elapsed time. The [benchmark tools](../../validation/README.md) also support CPU FP64, official ANTs, repeated measurements and profiling.

### Windows reproduction

Follow [Windows installation](installation.md#windows-powershell), then run from the project root. Set `--ants-registration` to your ANTs executable, choose a supported thread count and use new output directories.

```powershell
.\.venv\Scripts\python.exe -B -m validation.reference_tools.fetch_public_data
.\.venv\Scripts\python.exe -B -m validation.benchmarks.measure_cuda_native `
  --source src --data validation/data/public `
  --ants-registration .\setup\windows\reference\ants-2.6.5\bin\antsRegistration.exe `
  --threads 16 --repeats 2 --include-fp64 --skip-official-warmup `
  --timeout 3600 --output validation/reports/windows-native-16t
.\.venv\Scripts\python.exe -B -m validation.benchmarks.measure_cuda_native `
  --assess-report validation/reports/windows-native-16t/report.json `
  --threads 16 --output validation/reports/windows-native-16t-quality
```

The benchmark separates empty-cache and warmed-cache runs and retains outputs for the final saved-transform assessment.
