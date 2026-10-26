# Validation tools

[English](README.md) · [简体中文](README.zh-CN.md) · [Testing and quality assessment](../docs/en/validation.md) · [Optimizations](../docs/en/optimizations.md)

This directory contains tools for checking numerical operators, generating ANTs reference results, evaluating saved transforms, and measuring execution time. Normal image registration uses `fast-ants register` and does not require these tools.

## Directory layout

| Directory | Contents |
| --- | --- |
| [benchmarks/](benchmarks/) | Current five-engine timing, source/flag comparisons, and focused operator timing |
| [quality/](quality/) | Saved-image/field evaluation and reference comparisons |
| [operators/](operators/) | Independent prefix, reduction, and serial-watchdog checks; routine operator tests are in `tests/` |
| [reference_tools/](reference_tools/) | Dataset preparation and official reference generation |
| [maintenance/](maintenance/) | Dependency and implementation checks |
| [common/](common/) | Shared paths, process helpers, and tool index |
| [configs/](configs/) | Current reference recipes, quality policy, and small-case matrix |
| [sources/](sources/) | Input provenance and checksum records |
| `reports/` | Local measured outputs; excluded warmup volumes are discarded |

A [tool index](common/tool-index.json) maps script names to modules. Use the module path for new commands. The root `measure_fp32_pipeline.py`, `measure_fp64_cpu.py`, and `assess_saved_registration.py` also provide command-line entry points.

## Inspect tool options

Run from the project root:

```bash
python -m validation.benchmarks.measure_current_native --help
python -m validation.benchmarks.measure_fp32_pipeline --help
python -m validation.benchmarks.measure_fp64_cpu --help
python -m validation.quality.assess_saved_registration --help
```

The current native runner executes CPU FP64, ANTs float0, CPU FP32, Metal FP32, and ANTs float1 serially from the current `src` tree. Reports include source hashes, parameters, timings and output files. The FP32 pipeline tool compares CPU FP32, Metal, and ANTs float1. The FP64 tool compares CPU FP64 with ANTs float0. Saved-output assessment evaluates transforms and images without repeating the fit.

A script can also be selected by name:

```bash
python -B validation/legacy.py --list
python -B validation/legacy.py verify_float1_alignment --help
```

The index lists maintained commands only. Source-comparison tools accept caller-supplied source trees; this directory does not retain old experimental copies. Use the grouped tests for routine operator development.

## Quality evaluation

Compare images on the same physical grid and evaluate forward/inverse fields separately. Common measurements include NCC, normalized mean squared error, landmark TRE, Jacobian determinants, and inverse consistency. Details and equations are in [testing and quality assessment](../docs/en/validation.md).

For ANTs comparisons, select `--float 0` for CPU FP64 and `--float 1` for CPU FP32 or Metal, with matching images, stage parameters, interpolation, sampling, and input geometry. [Quality settings](configs/registration_quality_policy.json) describe the evaluation options.

## Timing

Record image dimensions, registration parameters, backend, precision, thread count, software versions, warmups, and repeated runs. Synchronize GPU work before recording completion and wait for all output files to be written. Report separately whether timing includes process startup, compilation, image I/O, and final quality calculations.

Numba and Metal compile kernels on first use. Keep compilation separate from repeated-execution measurements. Run comparisons sequentially under similar system load and power conditions.

## Data

[cases.json](cases.json), [input metadata](sources/input-inventory.json), [download checksums](sources/downloaded-files.json), and [data sources](sources/public-data.json) describe the three IXI images used in development. Different resolutions are stored under `data/public`, `data/p11-half`, and `data/prepared`. Image data and generated results are not included in the Python package.

Template and atlas resources are kept locally under `templates/` and follow their original source licenses.

## Run tests

```bash
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp64 --pattern test_fields.py
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_gaussian_sections32.py
```

Use `--list` to inspect selected tests. GPU tests require an Apple GPU and explicit `--suite metal` selection. The [test directory](../tests/README.md) describes all groups.
