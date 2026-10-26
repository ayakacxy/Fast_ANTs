# Development

[English](development.md) · [简体中文](../zh-CN/development.md) · [Documentation](index.md)

## Setup

```bash
python -m pip install -e '.[test,build]'
# Optional Metal dependencies:
python -m pip install -e '.[metal]'
```

The package uses NumPy for arrays, NiBabel for image I/O, Numba for CPU kernels, and MLX for custom Metal kernels. See [installation](installation.md) for conda environments and [architecture](architecture.md) for the source layout.

CUDA is the next planned GPU backend. Its runtime and device operators will be implemented separately, reusing image geometry, stage settings and host-side optimization. See the [CUDA roadmap](roadmap.md).

## Adding or modifying an operator

Place the implementation in the relevant backend and stage. Keep shared image types, configuration, and host-side calculations in `core` or `pipeline`. Use explicit array dtypes and document the array layout, units, interpolation rules, and treatment of values outside the image.

CPU kernels should compile in Numba's nopython mode. Parallelize independent work and merge reductions in a numerically stable order. For Metal, keep intermediate arrays on the device, combine related operations, and reuse buffers where possible. GPU-to-CPU reads should be limited to control values and final outputs. [Optimizations](optimizations.md) describes the existing implementations.

Workspaces belong to a registration call or pyramid level. Functions returning arrays must keep them independent of buffers that later iterations overwrite. Callback data needs the same treatment. CPU modules must remain importable without MLX; new logging should use already computed values.

## Testing

The test runner selects groups and individual files:

```bash
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp64 --pattern test_fields.py
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_gaussian_sections32.py
python -B scripts/maintenance/run_tests.py --suite cpu --list
```

`cpu` is the default and includes the core, interface, FP64, and FP32 tests. Run GPU tests explicitly on a machine with Metal:

```bash
python -B scripts/maintenance/run_tests.py --suite metal --pattern test_metal_cc_window32.py
```

New numerical tests should include relevant image boundaries, constant images, invalid values, anisotropic spacing, oblique grids, and large origins with small displacements. Compare operators with small reference calculations before evaluating complete registrations. Use [registration quality metrics](validation.md) when changing an optimizer, reduction, or field update.

## Profiling

The [validation tools](../../validation/README.md) support operator profiling, full-pipeline timing, and saved-output evaluation. Keep inputs, parameters, precision, and thread counts fixed during comparisons. Warm Numba and Metal kernels before measuring repeated execution, and synchronize GPU work before stopping a timer. Measure image reading and writing separately when profiling individual stages.

Run only the tests needed for a change. A documentation edit needs Markdown lint and link checks, while changes to interpolation or field composition need numerical tests.

## Documentation and packaging

Maintain English and Chinese pages together. Keep commands, defaults, filenames, and mathematical notation consistent. Runtime logs and CLI messages are English.

Use fenced `math` blocks for displayed equations and `$...$` for inline equations. Separate inline equations from adjacent Chinese text or punctuation with a space. Upright labels and function names use `\mathrm{...}`.

```bash
npx --yes markdownlint-cli2@0.23.2
python -B scripts/maintenance/check_markdown_links.py
python -m build
```

Markdown lint requires Node.js. `.markdownlint-cli2.jsonc` defines the editor and command-line rules for the project documentation. It excludes reference sources and build artifacts, disables the line-length limit, and allows `h1` and `div` for README centering.

Check that the wheel includes `common.metal` and the preset JSON files. MRI data, generated results, reference checkouts, and caches are excluded from distribution. Preserve the applicable notices in [NOTICE](../../licenses/NOTICE.md) and the upstream license files.
