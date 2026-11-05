# Performance measurements

[English](performance.md) · [简体中文](../zh-CN/performance.md) · [Documentation](index.md)

## Windows CUDA and CPU, 16 threads

Fast ANTs v0.3.0 registers IXI012-HH-1211 to IXI002-Guys-0828 at 256×256×150 with the same `brain-syn` recipe used below. The native Windows comparison uses 16 CPU threads and matches official ANTs precision for each backend.

| Backend | Run 1 / s | Run 2 / s | Median / s | Precision-matched ANTs / backend |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 7.49 | 7.38 | 7.43 | 70.54× |
| CPU FP32 | 77.38 | 75.20 | 76.29 | 6.87× |
| CPU FP64 | 99.72 | 88.05 | 93.88 | 5.71× |
| ANTs FP32 | 522.11 | 526.36 | 524.23 | — |
| ANTs FP64 | 535.93 | 536.83 | 536.38 | — |

Each backend runs in a fresh process. Fast ANTs runs once to prepare its compilation caches before formal timing; the two formal passes run in opposite orders. Official ANTs uses compiled C++ and has no separate full warmup. The table reports the two measurements and their median. CUDA with empty task-owned CuPy, CUDA driver and Numba caches took **27.26 s** in a separate single run; filesystem caches and GPU state were not reset.

Timing includes startup, imports, input reading, rigid/affine/SyN fitting, GPU synchronization and downloads where applicable, and completed writes. Fast ANTs also includes built-in QA. Its outputs use Python gzip level 0 with one writer thread per file; official ANTs uses its native writer defaults and does not run the additional QA. Nothing is subtracted. The timer's startup is excluded; Windows process handles supply CPU times and peak working set.

Hardware is an Intel Core i5-14400F, 10 physical cores / 16 logical CPUs, 32 GB RAM and an RTX 4060 Ti with 8 GB VRAM, running Windows 11 Pro for Workstations, build 26200. Python is 3.11.14, NumPy 2.3.0, Numba 0.64.0, llvmlite 0.46.0, NiBabel 5.3.3, tqdm 4.70.0 and CuPy 14.2.0 (`cupy-cuda13x`), with CUDA runtime 13.2 and NVIDIA driver 610.60. Official ANTs is the Windows 2.6.5-gfdce4d2 binary. Numba and ITK use 16 threads; BLAS and OMP limits are 1. Temperature and background load were not fixed.

The [Windows benchmark data](../assets/benchmarks/windows-native-16t.json) records parameters, source/input/output hashes, cold and warmup timings, both formal runs, and independent saved-output checks. All ten formal transforms are reapplied through the same CPU FP64 sampler outside timing; see [Windows quality results](validation.md#native-windows-mri-results).

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

### Windows reproduction

Use the isolated environment and native ANTs binary described in [Windows installation](installation.md#windows-powershell). From the project root, fetch the pinned public inputs and select a new output directory:

```powershell
python -B -m validation.reference_tools.fetch_public_data
python -B -m validation.benchmarks.measure_cuda_native --source src --data validation/data/public --ants-registration .\setup\windows\reference\ants-2.6.5\bin\antsRegistration.exe --threads 16 --repeats 2 --include-fp64 --skip-official-warmup --timeout 3600 --output validation/reports/windows-native-16t
python -B -m validation.benchmarks.measure_cuda_native --assess-report validation/reports/windows-native-16t/report.json --threads 16 --output validation/reports/windows-native-16t-quality
```

The first benchmark command creates separate cold-start and shared warmup caches inside its output directory. It retains all outputs. The assessment command verifies fitting receipts and hashes, then evaluates saved images and both fields without rerunning registration. Use the actual location of your `antsRegistration.exe` and a supported thread count.
