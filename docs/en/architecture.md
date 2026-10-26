# Architecture

[English](architecture.md) · [简体中文](../zh-CN/architecture.md) · [Documentation](index.md)

The package separates shared image geometry and optimization code from CPU and Metal operators. Each backend is organized by registration stage.

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

The entry points are [CPU FP64 registration](../../src/fast_ants/backends/cpu/fp64/registration.py), [CPU FP32 registration](../../src/fast_ants/backends/cpu/fp32/registration.py), and [Metal registration](../../src/fast_ants/backends/metal/registration.py). Their data flow follows the [algorithm description](algorithms.md).

## Shared geometry and optimization

`core/types.py` stores image and grid information; `core/fields.py` and `core/state.py` define displacement fields and the SyN state. Small matrix operations and scalar arithmetic live in `core/linalg.py` and `core/arithmetic32.py` / `arithmetic64.py`.

CPU FP32 and Metal share the FP32 linear model and optimizer under `pipeline`. Transform state, parameter updates, histogram finalization, and convergence calculations keep their respective precision settings. Image-sized operations run in the selected backend.

CPU imports do not initialize Metal. The public GPU entry point loads its implementation when called, allowing CPU-only installation without MLX.

## CPU operators

Numba compiles sampling, finite differences, filtering, metric evaluation, and displacement-field operations. Operators parallelize independent rows, slices, sample blocks, or filter lines. Shared reduction code combines block results.

Each sequential registration level owns its temporary buffers. Internal updates can reuse these buffers; arrays returned through public functions or retained by callbacks remain independently usable.

Source: [CPU FP64](../../src/fast_ants/backends/cpu/fp64), [CPU FP32](../../src/fast_ants/backends/cpu/fp32), and [shared CPU functions](../../src/fast_ants/backends/cpu/common).

## Metal operators

[`MetalContext`](../../src/fast_ants/backends/metal/runtime.py) manages custom-kernel compilation, dispatch, device arrays, asynchronous submission, and host/device transfers. [Device types](../../src/fast_ants/backends/metal/types.py) associate arrays with their grids. [`MetalOps`](../../src/fast_ants/backends/metal/ops.py) combines image, CC, field, and SyN methods through static inheritance.

Mattes MI is divided into sampling, histogram construction, derivatives, and finalization. CC contains short-window, shared-memory tile, and general neighbourhood implementations. Field modules implement composition, inversion, and paired inverse scheduling.

Images and fields stay on the GPU during fitting. The host handles small geometry/coefficient calculations and optimizer control. Final arrays are downloaded for output and CPU quality checks. Independent GPU work can be submitted before a host read; later iterations still wait for the values needed to decide their next step.

See [implementation optimizations](optimizations.md) for details.

## Planned CUDA backend

The NVIDIA CUDA backend is planned and is not included in the package. Its design reuses the image geometry, transforms, stage configuration and result types in `core`, together with the linear models, optimizer control and pyramids in `pipeline`.

Device arrays, kernel launches, streams, events and memory transfers will have a separate runtime under `backends/cuda`. The planned operators use FP32 images and fields, reusable device workspaces and asynchronous transfers, with synchronization for metric values and stopping decisions. Optional CUDA dependencies will load only when the backend is selected.

Operator coverage and testing requirements are listed in the [roadmap](roadmap.md).

## Development resources

`configs` and `examples` provide configuration and usage examples. `tests` contains unit and integration tests; `validation` contains numerical comparison and measurement tools. These are separate from the installed runtime. Local reference sources, MRI, templates, and generated outputs are not included in the package.

Use [development](development.md) to run tests and build distributions.
