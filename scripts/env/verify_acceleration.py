"""Verify CPU JIT, MLX GPU execution and a custom Metal kernel, without timing claims."""

import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numba
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def cpu_kernel(values):
    output = np.empty_like(values)
    for i in prange(values.size):
        output[i] = (values[i] + 1.0) ** 2
    return output


def main():
    numba.set_num_threads(2)
    values = np.linspace(-2, 2, 32768, dtype=np.float32)
    expected = (values + np.float32(1)) ** 2
    cpu_output = cpu_kernel(values)
    np.testing.assert_allclose(cpu_output, expected, rtol=2e-6, atol=2e-6)
    if not cpu_kernel.nopython_signatures:
        raise RuntimeError("Numba did not compile a native implementation")

    mx.set_default_device(mx.gpu)
    gpu_values = mx.array(values)
    gpu_output = (gpu_values + 1) ** 2
    mx.eval(gpu_output)
    np.testing.assert_allclose(np.asarray(gpu_output), expected, rtol=2e-6, atol=2e-6)
    kernel = mx.fast.metal_kernel(
        name="verify_polynomial",
        input_names=["inp"], output_names=["out"],
        source="""
            uint i = thread_position_in_grid.x;
            float v = inp[i] + 1.0f;
            out[i] = v * v;
        """,
    )
    metal_output = kernel(
        inputs=[gpu_values], grid=(values.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[gpu_values.shape], output_dtypes=[mx.float32],
    )[0]
    mx.eval(metal_output)
    mx.synchronize()
    np.testing.assert_allclose(np.asarray(metal_output), expected, rtol=2e-6, atol=2e-6)
    report = {
        "status": "passed", "purpose": "runtime check; not a performance benchmark",
        "numba": version("numba"), "llvmlite": version("llvmlite"),
        "mlx": version("mlx"), "mlx_metal": version("mlx-metal"),
        "numba_threading_layer": numba.threading_layer(),
        "numba_nopython_signatures": [str(s) for s in cpu_kernel.nopython_signatures],
        "mlx_device": str(mx.default_device()), "numba_native_execution": True,
        "mlx_gpu_array_execution": True, "custom_metal_kernel_execution": True,
        "numba_max_abs_error": float(np.max(np.abs(cpu_output - expected))),
        "mlx_max_abs_error": float(np.max(np.abs(np.asarray(gpu_output) - expected))),
        "metal_max_abs_error": float(np.max(np.abs(np.asarray(metal_output) - expected))),
    }
    target = Path(__file__).resolve().parents[2] / "setup/acceleration-report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
