# Usage

[English](usage.md) · [简体中文](../zh-CN/usage.md) · [Documentation](index.md)

## Command-line registration

Register the moving image on the fixed image grid:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-metal \
  --backend metal --precision fp32 \
  --preset brain-syn --threads 10 --verbose
```

Use `--backend cpu --precision fp64` or `--backend cpu --precision fp32` to select CPU execution. `python -m fast_ants` runs the same command-line interface.

For NVIDIA CUDA, activate the [CUDA environment](installation.md#cuda-environment) and run:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-cuda --backend cuda --precision fp32 \
  --preset brain-syn --threads 10 --verbose
```

| Option | Default | Description |
| --- | --- | --- |
| `--backend` | `cpu` | CPU, Metal, or CUDA execution |
| `--precision` | `fp64` on CPU; `fp32` on Metal/CUDA | Numerical precision |
| `--preset` | `brain-syn` | Bundled registration parameters |
| `--config` | — | Complete custom JSON configuration |
| `--threads` | Preset value: 10 | CPU thread count |
| `--geometry-policy` | `itk_scanner` | Input image geometry handling |
| `--scaling-policy` | `itk` | Input voxel intensity scaling |
| `--verbose`, `-v` | On | Print English progress and diagnostic logs |
| `--quiet`, `-q` | Off | Disable progress and verbose diagnostics |
| `--log-every` | 1 | GPU iteration diagnostic logging interval |

The output directory must be new. Select a thread count supported by your machine. Metal and CUDA support FP32; CPU supports both precisions. The preset uses random seed 20261022.

The default scanner geometry setting requires a scanner-anatomical sform. To use standard NiBabel NIfTI geometry and intensity scaling, add:

```bash
--geometry-policy nifti --scaling-policy nifti
```

See [image geometry](inputs-outputs.md) for details.

## Registration parameters

`brain-syn` runs four resolution levels: rigid and affine stages with Mattes MI, followed by SyN with local CC. The [configuration guide](../../configs/README.md) lists its iteration counts, smoothing, sampling, and convergence parameters.

To use another configuration, copy [the preset JSON](../../src/fast_ants/presets/brain-syn.json), edit it, and pass its path:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-custom --backend cpu --precision fp64 \
  --config my-recipe.json --threads 10
```

`--config` replaces the complete preset and is mutually exclusive with `--preset`. The command-line workflow uses dense sampling, linear interpolation, and unmasked images. Histogram matching, winsorization, and inverse-warped intensity output are disabled. Unsupported configuration values produce an error.

## Python API

```python
from fast_ants import load_images, register, register32, register_metal, register_cuda

fixed, moving = load_images(
    ("fixed.nii.gz", "moving.nii.gz"),
    geometry_policy="itk_scanner", scaling_policy="itk",
)
result = register(fixed, moving, output_directory="registration-fp64")
# result = register32(fixed, moving, output_directory="registration-fp32")
# result = register_metal(fixed, moving, output_directory="registration-metal",
#                         verbose=True, log_every=1)
# result = register_cuda(fixed, moving, output_directory="registration-cuda",
#                        verbose=True, log_every=1)
warped = result.warped
pull = result.fixed_to_moving
```

The API defaults to two resolution levels. Pass `LinearStage` and `SyNStage` objects to set the registration parameters. [The complete-preset example](../../examples/register_api.py) builds the same stages as the CLI.

For CPU FP64, supplying `initial_transform` to `register` fixes the affine transform and runs the nonlinear stage. The Metal and CUDA APIs accept `verbose` and `log_every`, with `verbose=False` by default; CPU phase logging is provided by the CLI.

### Public functions and types

| Group | Names |
| --- | --- |
| Images and transforms | `Grid`, `Image`, `RegistrationResult`, `AffineTransform`, `CompositeTransform`, `DisplacementField` |
| Image I/O | `load_image`, `load_images`, `save_image` |
| Transform I/O | `load_affine`, `save_affine`, `load_field`, `save_field`, `load_ants_transform_list` |
| Sampling and resampling | `apply_image`, `sample_image`, `transform_to_field` |
| Registration | `LinearStage`, `SyNStage`, `register_linear`, `register`, `register32`, `register_metal`, `register_cuda` |

See [inputs and outputs](inputs-outputs.md) for transform direction and array layout.

## Logging

All runtime logs are English. CLI progress and detailed logging are enabled by default; `--verbose` remains available, and `--quiet` disables them in both the terminal and `console.log`. Errors are still captured. Metal and CUDA report stages, pyramid levels, iteration metrics, convergence values, elapsed time, and stop reasons. CPU CLI output reports loading and registration progress; registration includes output quality checks and writing.

For Metal and CUDA, interactive terminals show a `tqdm` progress bar for each iterative pyramid level. The bar shows `metric`, `conv`, and `valid`, plus `lr`, `gamma`, and `ls_eval` when the iteration record provides learning rate, gamma, and line-search evaluation count. The final bar includes the stop reason; early convergence retains the actual iteration count against the budget. `tqdm` is included in the package dependencies. Redirected output uses readable text without progress-bar control sequences; `console.log` retains the diagnostic records without terminal animation.

The iteration table has the following columns:

```text
DIAGNOSTIC,Iteration,metricValue,convergenceValue,ITERATION_TIME_INDEX,SINCE_LAST,validPoints,learningRate,gamma,lineSearchEvaluations
```

`ITERATION_TIME_INDEX` is elapsed time within the current level; `SINCE_LAST` is the time since the preceding iteration. SyN may report fixed and moving valid-point counts together. `--log-every N` records the first, every Nth, and final GPU iteration; the interactive bar still updates on each iteration callback.

## Gzip output

Compatible scalar/field NIfTI output defaults to lossless gzip level 0 to
reduce write time for floating-point arrays. Decoded headers and voxel bytes
are unchanged; files can be larger. Select a compression level explicitly:

```bash
FAST_ANTS_GZIP_LEVEL=1 fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-compressed --backend cuda --precision fp32 \
  --preset brain-syn --threads 10
```

`FAST_ANTS_GZIP_LEVEL` accepts integers 0 through 9, with default 0.
`FAST_ANTS_GZIP_BACKEND=auto` uses optional pigz with two workers per file when
available; `gzip` forces the standard-library path and `pigz` requires the
executable. Both streaming implementations honor the level and preserve the
same decoded bytes. Unsupported image layouts retain the NiBabel fallback.
The CLI records actual backend, version, level and worker count in
`run.json`. This setting applies to every backend using the shared writer.

## Output and errors

The CLI writes transforms, the warped image, configuration, iteration history, and quality summaries to the output directory. `run.json` records input files, configuration hashes, backend, precision, and thread settings; `console.log` captures terminal output. See [output files](inputs-outputs.md).

Invalid parameters, missing files, unsupported image geometry, or an unavailable GPU device stop the run with an error. GPU failures do not silently select CPU execution. Logs from a failed run remain in its output directory; use a new directory when retrying.
