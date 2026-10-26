"""Run a small native ANTs registration to verify the installation.

This synthetic 32^3 case is an installation check, not a performance benchmark.
Run with the ants environment's Python. Outputs stay in setup/smoke-test/.
"""

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    out = Path(__file__).resolve().parents[2] / "setup/smoke-test"
    out.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    # A bounded, explicit thread count for this tiny check only.
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    env["ANTS_RANDOM_SEED"] = "20261021"
    binaries = {}
    for name in ("antsRegistration", "antsApplyTransforms", "N4BiasFieldCorrection"):
        binaries[name] = shutil.which(name, path=env["PATH"])
        if binaries[name] is None:
            raise RuntimeError(f"Missing executable: {name}")

    version = subprocess.check_output(
        [binaries["antsRegistration"], "--version"], env=env, text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    grid = np.indices((32, 32, 32), dtype=np.float32)
    fixed_data = np.zeros((32, 32, 32), dtype=np.float32)
    for center, width, amplitude in (
        ((14, 16, 15), (5, 7, 6), 1.0),
        ((21, 12, 18), (2, 3, 2), 0.8),
        ((10, 21, 11), (2, 2, 3), 0.6),
    ):
        distance = sum(((grid[i] - center[i]) / width[i]) ** 2 for i in range(3))
        fixed_data += amplitude * np.exp(-0.5 * distance)
    # Known integer translation (2, -1, 1), with zero outside the image.
    moving_data = np.zeros_like(fixed_data)
    moving_data[2:, :-1, 1:] = fixed_data[:-2, 1:, :-1]
    fixed = out / "fixed.nii.gz"
    moving = out / "moving.nii.gz"
    for path, data in ((fixed, fixed_data), (moving, moving_data)):
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    prefix = out / "reg_"
    warped = out / "warped.nii.gz"
    command = [
        binaries["antsRegistration"], "--dimensionality", "3", "--float", "1",
        "--output", f"[{prefix},{warped}]", "--interpolation", "Linear",
        "--initial-moving-transform", f"[{fixed},{moving},1]",
    ]
    for transform in ("Rigid[0.1]", "Affine[0.1]"):
        command += [
            "--transform", transform,
            "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
            "--convergence", "[30x15,1e-6,5]",
            "--shrink-factors", "2x1", "--smoothing-sigmas", "1x0vox",
        ]
    command += [
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{fixed},{moving},1,2]",
        "--convergence", "[10x5,1e-6,5]",
        "--shrink-factors", "2x1", "--smoothing-sigmas", "1x0vox",
        "--verbose", "1",
    ]
    started = time.perf_counter()
    with (out / "registration.log").open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=120)
    registration_seconds = time.perf_counter() - started
    affine = out / "reg_0GenericAffine.mat"
    warp = out / "reg_1Warp.nii.gz"
    inverse_warp = out / "reg_1InverseWarp.nii.gz"
    for path in (affine, warp, inverse_warp, warped):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing registration output: {path}")

    reapplied = out / "reapplied.nii.gz"
    apply_command = [
        binaries["antsApplyTransforms"], "-d", "3", "--float", "1",
        "-i", str(moving), "-r", str(fixed), "-o", str(reapplied),
        "-n", "Linear", "-t", str(warp), "-t", str(affine),
    ]
    with (out / "apply-transforms.log").open("w") as log:
        log.write(shlex.join(apply_command) + "\n")
        log.flush()
        subprocess.run(apply_command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=60)

    warped_image = nib.load(warped)
    warped_data = warped_image.get_fdata(dtype=np.float32)
    reapplied_data = nib.load(reapplied).get_fdata(dtype=np.float32)
    for data in (warped_data, reapplied_data):
        if data.shape != fixed_data.shape or not np.isfinite(data).all():
            raise RuntimeError("Invalid image shape or non-finite voxels")
    if not np.allclose(warped_image.affine, nib.load(fixed).affine):
        raise RuntimeError("Warped output does not match the fixed image geometry")
    for path in (warp, inverse_warp):
        field = nib.load(path).get_fdata(dtype=np.float32)
        if field.shape[:3] != fixed_data.shape or field.shape[-1] != 3:
            raise RuntimeError(f"Invalid displacement-field shape: {path}")
        if not np.isfinite(field).all():
            raise RuntimeError(f"Non-finite displacement field: {path}")
    mse_before = float(np.mean((moving_data - fixed_data) ** 2))
    mse_after = float(np.mean((warped_data - fixed_data) ** 2))
    reapply_difference = float(np.max(np.abs(reapplied_data - warped_data)))
    if not mse_after < mse_before:
        raise RuntimeError(f"Alignment did not improve: {mse_before} -> {mse_after}")
    if not np.allclose(reapplied_data, warped_data, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"Transform reapplication differs by {reapply_difference}")

    report = {
        "status": "passed",
        "purpose": "installation smoke check; not a speed or clinical accuracy benchmark",
        "python": sys.version.split()[0], "architecture": platform.machine(),
        "ants_version": version, "numpy": np.__version__,
        "nibabel": nib.__version__,
        "executables": binaries, "image_shape": list(fixed_data.shape),
        "stages": ["Rigid", "Affine", "SyN"], "threads": 2,
        "registration_wall_seconds": registration_seconds,
        "mse_before": mse_before, "mse_after": mse_after,
        "transform_reapplication_max_abs_difference": reapply_difference,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
