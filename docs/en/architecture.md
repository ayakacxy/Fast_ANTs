# Architecture

[English](architecture.md) · [简体中文](../zh-CN/architecture.md) · [Documentation](index.md)

The package separates shared image geometry and optimization code from CPU, Metal, and CUDA operators. Each backend is organized by registration stage.

## Source layout

```text
src/fast_ants/
├── api.py, cli.py, progress.py
├── core/                 # Images, grids, transforms, configuration, results
├── io/                   # NIfTI and transform I/O
├── pipeline/             # Pyramids, sampling plans, linear models, optimizers
├── backends/
│   ├── cpu/
│   │   ├── common/
│   │   ├── fp64/
│   │   └── fp32/
│   ├── cuda/             # CuPy runtime, CUDA kernels, registration stages
│   └── metal/
│       ├── runtime.py, types.py, ops.py
│       ├── registration.py, linear.py, syn.py
│       ├── images.py, filtering.py, preparation.py
│       ├── metrics/mattes/
│       ├── metrics/cc/
│       └── fields/
├── quality/
└── presets/
```

| Module | Main responsibility |
| --- | --- |
| `core` | Grid geometry, image/field data structures, transforms, stage options, result types |
| `io` | Scalar images, vector fields, affine transforms, chunked NIfTI writing |
| `pipeline` | Shared pyramids, sample selection, FP64/FP32 linear models and optimization |
| `backends/cpu/fp64` | Double-precision registration stages and Numba operators |
| `backends/cpu/fp32` | Single-precision registration stages and Numba operators |
| `backends/cpu/common` | Shared CPU geometry and reduction functions |
| `backends/metal` | Single-precision device kernels and GPU scheduling |
| `backends/cuda` | CuPy device arrays, CUDA kernels, registration and final QA |
| `quality` | Image/field quality metrics and final output checks |
| `progress` | English stage and iteration logging |
| `presets` | Registration configurations distributed with the package |

## Registration pipeline

`api.py` exposes registration, I/O, and transform functions. `cli.py` loads a preset or user configuration, selects the backend, and writes run information. A registration follows this sequence:

1. Load image geometry and voxel data; initialize the linear transform.
2. Prepare image pyramids and fit rigid/affine stages.
3. Resample the SyN state between levels and update its four fields.
4. Compose the final forward/inverse maps and resample the moving image.
5. Compute output quality metrics and write the transforms, image, and logs.

The entry points are [CPU FP64 registration](../../src/fast_ants/backends/cpu/fp64/registration.py), [CPU FP32 registration](../../src/fast_ants/backends/cpu/fp32/registration.py), [Metal registration](../../src/fast_ants/backends/metal/registration.py), and [CUDA registration](../../src/fast_ants/backends/cuda/registration.py). Their data flow follows the [algorithm description](algorithms.md).

## Shared geometry and optimization

`core/types.py` stores image and grid information; `core/fields.py` and `core/state.py` define displacement fields and the SyN state. Small matrix operations and scalar arithmetic live in `core/linalg.py` and `core/arithmetic32.py` / `arithmetic64.py`.

CPU FP32, Metal, and CUDA share the FP32 linear model and optimizer under `pipeline`. Transform state, parameter updates, histogram finalization, and convergence calculations keep their respective precision settings. Image-sized operations run in the selected backend.

CPU imports do not initialize Metal or CUDA. Each public GPU entry point loads its implementation when called, allowing CPU-only installation without MLX or CuPy.

## CPU operators

Numba compiles sampling, finite differences, filtering, metric evaluation, and displacement-field operations. Operators parallelize independent rows, slices, sample blocks, or filter lines. Shared reduction code combines block results.

Each sequential registration level owns its temporary buffers. Internal updates can reuse these buffers; arrays returned through public functions or retained by callbacks remain independently usable.

Source: [CPU FP64](../../src/fast_ants/backends/cpu/fp64), [CPU FP32](../../src/fast_ants/backends/cpu/fp32), and [shared CPU functions](../../src/fast_ants/backends/cpu/common).

## Metal operators

[`MetalContext`](../../src/fast_ants/backends/metal/runtime.py) manages custom-kernel compilation, dispatch, device arrays, asynchronous submission, and host/device transfers. [Device types](../../src/fast_ants/backends/metal/types.py) associate arrays with their grids. [`MetalOps`](../../src/fast_ants/backends/metal/ops.py) combines image, CC, field, and SyN methods through static inheritance.

Mattes MI is divided into sampling, histogram construction, derivatives, and finalization. CC contains short-window, shared-memory tile, and general neighbourhood implementations. Field modules implement composition, inversion, and paired inverse scheduling.

Images and fields stay on the GPU during fitting. The host handles small geometry/coefficient calculations and optimizer control. Final arrays are downloaded for output and CPU quality checks. Independent GPU work can be submitted before a host read; later iterations still wait for the values needed to decide their next step.

See [implementation optimizations](optimizations.md) for details.

## CUDA operators and registration

[`CUDAContext`](../../src/fast_ants/backends/cuda/runtime.py) uses optional CuPy for device arrays, custom kernel compilation and dispatch, synchronization, and explicit host/device transfers. [Device types](../../src/fast_ants/backends/cuda/types.py) associate contiguous FP32 images and displacement fields with shared `Grid` geometry. [`CUDAImageOps`](../../src/fast_ants/backends/cuda/images.py) provides image upload/download, zero fields, intensity normalization, trilinear/nearest-neighbour pull sampling, and anchored physical gradients.

Images and displacement fields use FP32 on the GPU during registration. Small geometry matrices and origin differences are computed in FP64 on the CPU before forming local coordinates. Final quality checks use separate FP64 temporary arrays. Kernels in `common.cuh` follow the FP32 sampling support and boundary rules. CUDA modules are independent of Metal; CPU entry points do not import them.

[`CUDAOps`](../../src/fast_ants/backends/cuda/ops.py) combines image, field, CC, and SyN operators. Gaussian filtering, Mattes sampling/histograms/derivatives/finalization, local CC, field composition, and inversion run on the GPU. [`register_cuda`](../../src/fast_ants/backends/cuda/registration.py) connects these operators to the shared stage configuration, linear models, optimizer controls, and multiresolution schedule, retaining dense images and four-field SyN state on the device.

Small geometry matrices, filter coefficients, optimizer updates and convergence decisions are computed on the CPU. Final field quality checks run on the GPU in FP64, calculating exact percentiles and checking finite values; the program downloads the summaries, final fields and warped image for output. CUDA Mattes histograms accumulate samples in tiles and merge those tiles in order. Floating-point accumulation order can affect the last bits and later iterations, so alignment, deformation and inverse consistency are evaluated on matching inputs.

See [development](development.md) and the [roadmap](roadmap.md) for testing and planned validation and optimization work.
