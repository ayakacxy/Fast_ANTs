# Inputs and outputs

[English](inputs-outputs.md) · [简体中文](../zh-CN/inputs-outputs.md) · [Documentation](index.md)

## Image format and coordinates

Input images are three-dimensional scalar NIfTI files with finite values. Fast ANTs uses the following array and coordinate conventions:

| Data | Layout or units |
| --- | --- |
| Scalar image array | `[z,y,x]` |
| Index and physical point vectors | `[x,y,z]` |
| Physical coordinates | LPS, millimetres |
| Displacement array | `[3,z,y,x]`, LPS components in millimetres |

NIfTI RAS+ coordinates are converted to LPS on loading. Spatial units in metres or microns are converted to millimetres; missing units are treated as millimetres and recorded in the image metadata. Image grids may be oblique and anisotropic, with orthogonal direction axes. Sheared grids are unsupported.

For index vector `i`, origin `o`, direction matrix `D`, and voxel-spacing matrix `S`, physical coordinates are

```math
x=o+DSi.
```

## Loading options

`load_image` and `load_images` default to `geometry_policy="nifti"` and `scaling_policy="nifti"`. The CLI defaults to `itk_scanner` and `itk`.

| Setting | Behavior |
| --- | --- |
| `geometry_policy="nifti"` | Uses NiBabel's selected affine: coded sform, qform, then fallback |
| `geometry_policy="itk_scanner"` | Requires scanner-anatomical sform; normalizes direction columns and uses header voxel spacing |
| `scaling_policy="nifti"` | Uses standard NiBabel intensity conversion |
| `scaling_policy="itk"` | Uses ITK-compatible scalar conversion, including its FP32 intermediate for supported scaled integer inputs |

Choose the same options when comparing runs. To preserve integer labels, pass the appropriate integer `dtype` to the image loader.

## Transform direction

`result.fixed_to_moving` maps fixed-grid physical points to sampling positions in the moving image:

```math
W(x)=M(T(x)),\qquad T:\Omega_F\rightarrow\Omega_M.
```

`warped.nii.gz` is therefore the moving image resampled on the fixed grid. To map moving-space points into fixed space, use the inverse transform, `result.moving_to_fixed`.

With nonlinear map `phi` and affine transform `A`, the complete sampling map is `T = A ◦ phi`. `CompositeTransform((first, second))` applies `first`, then `second`. Inversion reverses this order and inverts both transforms. The inverse displacement field is computed iteratively.

## Output files

| File | Contents |
| --- | --- |
| `affine.mat` | Affine matrix and centre parameters |
| `warp.nii.gz` | Forward nonlinear displacement field on the fixed-domain grid |
| `inverse-warp.nii.gz` | Inverse nonlinear displacement field |
| `warped.nii.gz` | Moving image resampled on the fixed grid |
| `configuration.json` | Registration parameters, backend, precision, and coordinate settings |
| `history.json` | Stages, levels, iteration values, and stop reasons |
| `quality.json` | Output quality measurements |
| `console.log` | CLI output and enabled progress logs |
| `run.json` | CLI input files, hashes, and run settings |

`console.log` and `run.json` are written by the CLI. Python API calls write image, transform, and JSON results when `output_directory` is supplied; otherwise results remain in memory.

Compatible scalar and field NIfTI output uses lossless gzip level 0 by default. Set `FAST_ANTS_GZIP_LEVEL` to a value from 0 to 9 to select compression; decoded headers and arrays are unchanged. See [usage](usage.md) for the optional pigz writer and fallback behavior.

Affine I/O supports the ANTs MAT-v4 and ITK text formats implemented by the affine reader. Displacement-field NIfTI storage uses `[x,y,z,1,3]`; the reader also accepts `[x,y,z,3]`. Vector components remain LPS millimetres even though the NIfTI affine uses RAS coordinates.

## Apply saved transforms

```python
from fast_ants import load_image, load_ants_transform_list, apply_image, save_image

moving = load_image("moving.nii.gz")
fixed = load_image("fixed.nii.gz")
pull = load_ants_transform_list(["registration/warp.nii.gz", "registration/affine.mat"])
warped = apply_image(moving, fixed.grid, pull, interpolation="Linear")
save_image(warped, "reapplied.nii.gz")
```

Use `Linear` for scalar intensities and `NearestNeighbor` for integer labels. Keep each displacement field with its grid information when saving or resampling. See [algorithms](algorithms.md) for transform composition.
