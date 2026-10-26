# Contributing

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md) · [Development guide](../docs/en/development.md)

Contributions to registration algorithms, CPU and Metal kernels, the planned CUDA backend, image I/O, tests, and documentation are welcome. Before a substantial change, describe the problem and the affected backend or registration stage. The [roadmap](../docs/en/roadmap.md) lists the CUDA implementation steps.

## Development setup

```bash
python -m pip install -e '.[test,build]'
# For Metal development on Apple Silicon:
python -m pip install -e '.[metal]'
```

See [architecture](../docs/en/architecture.md) for the module layout. Shared image types, configuration, and host-side optimization belong in `core` or `pipeline`; numerical execution belongs in the appropriate backend.

## Code changes

- Preserve LPS coordinates, millimetre units, transform order, array layout, and supported dtypes.
- Implement CPU loops with Numba and GPU operators with backend-specific kernels. Keep scalar setup and array handling separate from voxel loops.
- Reuse temporary buffers where safe. Returned arrays and arrays retained by callbacks must not be overwritten by later iterations.
- Keep CPU imports independent of optional GPU libraries. Avoid extra GPU synchronization or metric evaluations for logging.
- Add a small test that exercises the changed behavior, including relevant boundary cases. Document new arguments and update both language versions.

## Tests

Run the relevant test groups from the project root:

```bash
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp64 --pattern test_fields.py
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_gaussian_sections32.py
python -B scripts/maintenance/check_markdown_links.py
npx --yes markdownlint-cli2@0.23.2
```

Use `--list` to inspect the selection. Metal tests require an Apple GPU and `--suite metal`; see [tests](../tests/README.md). For numerical changes, check registration quality as described in [testing and evaluation](../docs/en/validation.md). Documentation changes do not require registration runs.

The Markdown check requires Node.js. The project configuration covers both language versions and current measurement reports. It allows the `div` and `h1` elements used for README centering; all other HTML remains checked. Line-length checking follows the VS Code extension default and is disabled for prose, formulas, and wide tables.

## Pull requests

Explain the problem, the resulting behavior, and the tests run. Include a reproducible example for bug fixes. For performance changes, include the input dimensions, parameters, precision, thread count, warmup procedure, and timing method. Describe the relevant implementation changes rather than relying on a speedup figure alone.

Keep MRI data, generated outputs, caches, credentials, and local configuration out of the package. Preserve applicable copyright and license notices; see [NOTICE](../licenses/NOTICE.md). Runtime logs and CLI messages are English.
