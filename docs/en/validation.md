# Testing and registration quality

[English](validation.md) · [简体中文](../zh-CN/validation.md) · [Documentation](index.md)

## Test groups

Tests cover image geometry and I/O, numerical operators, optimization, and complete registration. CPU FP64 is compared with ANTs `--float 0`; CPU FP32, Metal and CUDA are compared with `--float 1`.

```bash
python -B scripts/maintenance/run_tests.py --suite core
python -B scripts/maintenance/run_tests.py --suite cpu-fp64
python -B scripts/maintenance/run_tests.py --suite cpu-fp32
```

Metal tests need an Apple GPU:

```bash
python -B scripts/maintenance/run_tests.py --suite metal
```

CUDA tests need an NVIDIA GPU and CuPy:

```bash
python -B scripts/maintenance/run_tests.py --suite cuda
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

## CUDA checks

CUDA operator and complete-registration tests live in `tests/cuda`. Enable them with `FAST_ANTS_TEST_CUDA=1` on NVIDIA hardware. They cover image sampling, filtering, Mattes controls, fields, inverse stopping, retained state, saved transform reapplication and native FP64 final QA. Real MRI results are assessed after timing using a common CPU FP64 sampler, foreground similarity, exploratory landmarks, positive Jacobians in both directions and inverse consistency.

## Apple M5 native MRI results

The [Apple M5 measurements](performance.md#apple-m5-release-measurements) use the native 256×256×150 IXI pair and 10 threads. Saved outputs are verified against the published timing checksums and fitting receipts, then reapplied through one CPU FP64 sampler. The foreground mask contains 3,236,832 fixed-image voxels above 5% of the intensity range. Before registration, foreground NCC is 0.171354 and mean error across four coarse landmarks is 21.4895 mm.

| Engine | Foreground NCC | Mean landmark error / mm | Minimum forward or inverse Jacobian | Maximum inverse-consistency RMS / mm |
| --- | ---: | ---: | ---: | ---: |
| Metal FP32 | 0.662625 | 1.4299 | 0.165859 | 0.022969 |
| CPU FP32 | 0.664474 | 1.4382 | 0.160806 | 0.023699 |
| CPU FP64 | 0.664417 | 1.4345 | 0.161804 | 0.023478 |
| ANTs FP32 | 0.664369 | 1.4315 | 0.160959 | 0.023638 |
| ANTs FP64 | 0.664417 | 1.4345 | 0.161804 | 0.023478 |

Metal landmark RMS and maximum errors are 1.8645 and 3.3925 mm; ANTs FP32 values are 1.8650 and 3.3914 mm. Normalized foreground MSE is 0.013558 for Metal and 0.013509 for ANTs FP32. The four landmarks are coarse non-expert annotations; these values describe this image pair's alignment.

Both saved fields and all resampled images are finite and match the fixed grid. Forward and inverse Jacobians are finite and positive for all five engines, with zero nonpositive voxels. The last column takes the larger RMS of the two nonlinear composition orders. For Metal, the forward and inverse Jacobian minima are 0.165859 and 0.179500. Inverse consistency uses all 9,830,400 grid points in each direction:

| Composition order | Metal RMS / mm | ANTs FP32 RMS / mm | Metal P99 / mm | ANTs FP32 P99 / mm | Metal maximum / mm | ANTs FP32 maximum / mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward after inverse | 0.022969 | 0.023638 | 0.083221 | 0.085717 | 0.392295 | 0.409191 |
| Inverse after forward | 0.020985 | 0.021935 | 0.077979 | 0.081582 | 0.478370 | 0.489421 |

Inverse-consistency residuals measure how well a map and its inverse compose to identity; landmark errors measure alignment to marked points. The [Apple M5 data](../assets/benchmarks/v0.1.0-native.json) records both, including per-point errors, output hashes and assessment provenance. This saved-output assessment is outside the original timers; the performance figures remain single v0.1.0 measurements.

## Native Linux MRI results

The [32-thread Linux measurements](performance.md#linux-cuda-and-cpu-32-threads) retain ten formal saved outputs, two per engine. All were reapplied through the same CPU FP64 sampler after timing. The foreground mask contains 3,236,832 fixed-image voxels above 5% of the intensity range. Before registration, foreground NCC is 0.171354 and mean error across four coarse landmarks is 21.4895 mm.

| Engine | Foreground NCC range | Mean landmark error range / mm | Minimum forward or inverse Jacobian | Maximum inverse-consistency RMS / mm |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 0.662979–0.664397 | 1.4310–1.4337 | 0.161487 | 0.023656 |
| CPU FP32 | 0.664355–0.664355 | 1.4309–1.4309 | 0.160964 | 0.023635 |
| CPU FP64 | 0.664417–0.664417 | 1.4345–1.4345 | 0.161804 | 0.023478 |
| ANTs FP32 | 0.662904–0.664416 | 1.4253–1.4333 | 0.161125 | 0.023659 |
| ANTs FP64 | 0.663937–0.663937 | 1.4336–1.4336 | 0.163237 | 0.023333 |

All saved warped images and fields are finite and retain the fixed grid geometry. Every forward and inverse Jacobian is finite and positive. The last column takes the maximum RMS over both composition directions and both repeats. The four landmarks are coarse, non-expert annotations used to assess alignment. Source, input and output hashes and per-run metrics are in the [Linux summary](../assets/benchmarks/linux-native-32t.json).
