# Testing and registration quality

[English](validation.md) · [简体中文](../zh-CN/validation.md) · [Documentation](index.md)

## Test groups

Tests cover image geometry and I/O, numerical operators, optimization, and complete registration. CPU FP64 is compared with ANTs `--float 0`; CPU FP32 and Metal are compared with `--float 1`.

```bash
python -B scripts/maintenance/run_tests.py --suite core
python -B scripts/maintenance/run_tests.py --suite cpu-fp64
python -B scripts/maintenance/run_tests.py --suite cpu-fp32
```

Metal tests need an Apple GPU:

```bash
python -B scripts/maintenance/run_tests.py --suite metal
```

For a small change, use `--pattern` to select the affected file. [The test directory](../../tests/README.md) lists the groups.

## Numerical tests

Operator tests use small arrays and known transforms to check interpolation, physical gradients, smoothing, local statistics, histogram derivatives, field composition, and inversion. Test points should include the image boundary as well as the interior. Constant images, nearly constant regions, large coordinate origins, and anisotropic grids help expose numerical problems.

Check registration with synthetic transforms whose displacements are known. Inspect both forward and inverse maps, including their units and composition order. Floating-point reductions may produce small differences between backends; compare numerical error and final alignment rather than requiring identical optimizer trajectories.

## Image and landmark alignment

Evaluate both implementations on the same fixed-image grid and the same foreground region. The saved-output tools calculate the following measures:

| Measure | Interpretation |
| --- | --- |
| Normalized cross-correlation (NCC) | Intensity similarity after resampling; larger values indicate stronger correlation |
| Normalized mean squared error | Intensity residual after normalization; smaller values indicate closer agreement |
| Target registration error (TRE) | Distance between corresponding landmarks after applying the transform, in millimetres |

For fixed landmarks `p_i`, corresponding moving landmarks `q_i`, and the fixed-to-moving map `T`,

```math
\mathrm{TRE}_{\mathrm{RMS}}=\sqrt{\frac{1}{N}\sum_i\|T(p_i)-q_i\|^2}.
```

Image similarity and landmark error measure different aspects of alignment. Report the region used for image statistics and the source of landmark annotations.

## Deformation quality

For a displacement field `u`, the spatial Jacobian is `J=I+∇u`. Inspect the minimum determinant and the number of voxels with a nonpositive determinant. The built-in summaries evaluate the nonlinear forward and inverse fields separately.

Inverse consistency measures the residual after composing the two maps:

```math
r(x)=\phi^{-1}(\phi(x))-x.
```

Check both composition orders and report RMS, percentile, or maximum displacement in millimetres, using points where the composed samples are valid. Also check that fields, resampled images, and summary values are finite. These statistics are saved in `quality.json` or a separate evaluation report; they do not replace visual inspection of the registered images.

## Evaluation data

The evaluation examples use three IXI T1 images: IXI002-Guys-0828, IXI012-HH-1211, and IXI013-HH-1212, with dimensions 256×256×150. [Input metadata](../../validation/sources/input-inventory.json), [download checksums](../../validation/sources/downloaded-files.json), and [data sources](../../validation/sources/public-data.json) describe the files and preparation. The supplied landmarks are coarse annotations.

The original IXI data uses CC BY-SA 3.0. The Zenodo republication lists CC BY 4.0 for its added material; use the original dataset's terms for the MRI files. Data is not included in the Python package.

## Evaluation tools

The [validation directory](../../validation/README.md) contains reference generation and saved-output evaluation tools. Inspect their options before selecting inputs:

```bash
python -m validation.quality.assess_saved_registration --help
python -m validation.quality.run_float1_quality_matrix --help
```

For comparisons with ANTs, use the same images, image geometry, preprocessing, stage parameters, interpolation, sampling, and precision. Save the configuration and software versions with each result. The [quality configuration](../../validation/configs/registration_quality_policy.json) lists the checks used by the evaluation scripts.
