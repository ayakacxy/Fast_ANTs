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

| Option | Default | Description |
| --- | --- | --- |
| `--backend` | `cpu` | CPU or Metal execution |
| `--precision` | `fp64` on CPU; `fp32` on Metal | Numerical precision |
| `--preset` | `brain-syn` | Bundled registration parameters |
| `--config` | — | Complete custom JSON configuration |
| `--threads` | Preset value: 10 | CPU thread count |
| `--geometry-policy` | `itk_scanner` | Input image geometry handling |
| `--scaling-policy` | `itk` | Input voxel intensity scaling |
| `--verbose`, `-v` | Off | Print English progress logs |
| `--log-every` | 1 | Metal iteration logging interval |

The output directory must be new. Select a thread count supported by your machine. Metal supports FP32; CPU supports both precisions. The preset uses random seed 20261022.

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
from fast_ants import load_images, register, register32, register_metal

fixed, moving = load_images(
    ("fixed.nii.gz", "moving.nii.gz"),
    geometry_policy="itk_scanner", scaling_policy="itk",
)
result = register(fixed, moving, output_directory="registration-fp64")
# result = register32(fixed, moving, output_directory="registration-fp32")
# result = register_metal(fixed, moving, output_directory="registration-metal",
#                         verbose=True, log_every=1)
warped = result.warped
pull = result.fixed_to_moving
```

The API defaults to two resolution levels. Pass `LinearStage` and `SyNStage` objects to set the registration parameters. [The complete-preset example](../../examples/register_api.py) builds the same stages as the CLI.

For CPU FP64, supplying `initial_transform` to `register` fixes the affine transform and runs the nonlinear stage. The Metal API accepts `verbose` and `log_every`; CPU phase logging is provided by the CLI.

### Public functions and types

| Group | Names |
| --- | --- |
| Images and transforms | `Grid`, `Image`, `RegistrationResult`, `AffineTransform`, `CompositeTransform`, `DisplacementField` |
| Image I/O | `load_image`, `load_images`, `save_image` |
| Transform I/O | `load_affine`, `save_affine`, `load_field`, `save_field`, `load_ants_transform_list` |
| Sampling and resampling | `apply_image`, `sample_image`, `transform_to_field` |
| Registration | `LinearStage`, `SyNStage`, `register_linear`, `register`, `register32`, `register_metal` |

See [inputs and outputs](inputs-outputs.md) for transform direction and array layout.

## Logging

All runtime logs are English. `--verbose` prints progress to the terminal and saves it to `console.log`. Metal reports stages, pyramid levels, iteration metrics, convergence values, elapsed time, and stop reasons. CPU CLI output reports loading and registration progress; registration includes output quality checks and writing.

The iteration table has the following columns:

```text
DIAGNOSTIC,Iteration,metricValue,convergenceValue,ITERATION_TIME_INDEX,SINCE_LAST,validPoints
```

`ITERATION_TIME_INDEX` is elapsed time within the current level; `SINCE_LAST` is the time since the preceding iteration. SyN may report fixed and moving valid-point counts together. `--log-every N` prints the first, every Nth, and final Metal iteration.

## Output and errors

The CLI writes transforms, the warped image, configuration, iteration history, and quality summaries to the output directory. `run.json` records input files, configuration hashes, backend, precision, and thread settings; `console.log` captures terminal output. See [output files](inputs-outputs.md).

Invalid parameters, missing files, unsupported image geometry, or an unavailable Metal device stop the run with an error. Logs from a failed run remain in its output directory; use a new directory when retrying.
