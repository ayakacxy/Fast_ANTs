# Registration configuration

[English](README.md) · [简体中文](README.zh-CN.md) · [Usage](../docs/en/usage.md)

The bundled [`brain-syn`](../src/fast_ants/presets/brain-syn.json) preset uses dense sampling and four resolution levels.

| Stage | Metric | Iteration limits | Shrink factors | Smoothing sigma / voxels |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI, 32 bins | 1000, 500, 250, 100 | 8, 4, 2, 1 | 3, 2, 1, 0 |
| Affine | Mattes MI, 32 bins | 1000, 500, 250, 100 | 8, 4, 2, 1 | 3, 2, 1, 0 |
| SyN | Local CC, radius 2 | 100, 70, 50, 20 | 8, 4, 2, 1 | 3, 2, 1, 0 |

The rigid and affine step sizes are 0.1. SyN uses `SyN[0.2,3,0]`: step size 0.2, update-field variance 3, and total-field variance 0. The convergence threshold is 1e-6 with a window of 10; random seed is 20261022. Per-level iterations stop early when convergence is reached. Shrink factors are adjusted for physical voxel spacing.

Run the preset:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration --preset brain-syn
```

## Custom parameters

Copy the preset JSON to another file and edit its supported values:

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-custom --config my-recipe.json
```

`--config` replaces the entire preset and is mutually exclusive with `--preset`. Set `iterations`, `shrink_factors`, and `smoothing_sigmas` with equal lengths for each stage. Image smoothing uses standard deviations; the SyN update and total-field smoothing values are variances.

The CLI supports Rigid/MI → Affine/MI → SyN/CC with dense sampling, linear interpolation, unmasked images, and disabled histogram matching and winsorization. The Python interfaces provide stage objects for direct parameter configuration. See [algorithms](../docs/en/algorithms.md) for parameter definitions.
