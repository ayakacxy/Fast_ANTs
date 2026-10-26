"""User-facing registration command, separate from benchmark orchestration.

Argument and recipe handling use the standard library. Numerical imports are
deferred until thread controls are installed and a run has been validated.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
from importlib import metadata, resources
import json
import math
import os
from pathlib import Path
import re
import sys
import time


PRESETS = ("brain-syn",)
THREAD_POOLS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def version():
    try:
        return metadata.version("fast_ants")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def parser():
    result = argparse.ArgumentParser(prog="fast-ants", description="Independent 3-D MRI registration")
    result.add_argument("--version", action="version", version="%(prog)s " + version())
    commands = result.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="Register a moving image to a fixed image")
    register.add_argument("fixed", type=Path, help="Fixed scalar 3-D NIfTI")
    register.add_argument("moving", type=Path, help="Moving scalar 3-D NIfTI")
    register.add_argument("--output", "-o", type=Path, required=True, help="New output directory; existing paths are refused")
    register.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    register.add_argument("--precision", choices=("fp64", "fp32"), help="Default: fp64 for CPU, fp32 for Metal")
    recipe = register.add_mutually_exclusive_group()
    recipe.add_argument("--preset", choices=PRESETS, help="Bundled recipe (default: brain-syn)")
    recipe.add_argument("--config", type=Path, help="Complete recipe JSON replacing the preset; no implicit merge")
    register.add_argument("--threads", type=positive_integer, help="Numba thread budget (default: recipe value)")
    register.add_argument("--geometry-policy", choices=("itk_scanner", "nifti"), default="itk_scanner",
                          help="Input geometry policy (default: itk_scanner; requires scanner sform)")
    register.add_argument("--scaling-policy", choices=("itk", "nifti"), default="itk",
                          help="Input voxel scaling policy (default: itk)")
    register.add_argument("--verbose", "-v", action="store_true", help="Print English phase logs; Metal also prints iteration diagnostics")
    register.add_argument("--log-every", type=positive_integer, default=1, help="Metal iteration log interval (default: 1; first/final retained)")
    return result


def parse_args(argv=None):
    command = parser()
    args = command.parse_args(argv)
    args.precision = args.precision or ("fp32" if args.backend == "metal" else "fp64")
    if args.backend == "metal" and args.precision != "fp32":
        command.error("Metal supports fp32 only; no CPU fallback is performed")
    if args.output.exists():
        command.error("Output already exists: " + str(args.output))
    for side in ("fixed", "moving"):
        path = getattr(args, side)
        if not path.is_file():
            command.error("Input file does not exist: " + str(path))
    if args.config is not None and not args.config.is_file():
        command.error("Configuration file does not exist: " + str(args.config))
    return args


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(name + " must be finite and >= " + str(minimum))
    return value


def validate_recipe(config):
    """Reject unsupported behavior instead of silently dropping JSON options."""
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Expected a schema_version 1 recipe object")
    if (config.get("dimensionality") != 3 or config.get("interpolation") != "Linear"
            or config.get("histogram_matching") is not False
            or config.get("winsorization") is not None or config.get("masks") is not None
            or config.get("write_inverse_warped", False)):
        raise ValueError("Supported recipe is 3D, Linear, unmasked, without histogram matching, winsorization or inverse-warped output")
    if config.get("sampling") != {"strategy": "None", "fraction": 1.0, "gradient_filter": False}:
        raise ValueError("The CLI currently supports dense sampling without gradient filtering")
    _integer(config.get("threads"), "threads", 1)
    _integer(config.get("random_seed"), "random_seed")
    if type(config.get("initialization_feature")) is not int or config["initialization_feature"] not in (0, 1, 2):
        raise ValueError("initialization_feature must be 0, 1 or 2")
    convergence = config.get("convergence", {})
    if not isinstance(convergence, dict):
        raise ValueError("convergence must be an object with threshold and window")
    _finite(convergence.get("threshold"), "convergence threshold")
    _integer(convergence.get("window"), "convergence window", 2)
    policy = config.get("shrink_factor_policy", "explicit")
    if policy not in ("explicit", "ants_physical_spacing"):
        raise ValueError("Unsupported shrink factor policy")
    stages = config.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("Expected three stages: Rigid, Affine, SyN")
    for stage, kind, metric in zip(stages, ("Rigid", "Affine", "SyN"), ("MI", "MI", "CC")):
        if not isinstance(stage, dict) or stage.get("metric") != metric:
            raise ValueError("Expected MI, MI, CC stage metrics")
        transform = stage.get("transform", "")
        match = re.fullmatch(r"(Rigid|Affine|SyN)\[([^\[\]]+)\]", transform) if isinstance(transform, str) else None
        if match is None or match.group(1) != kind:
            raise ValueError("Expected Rigid, Affine, SyN transforms")
        parameters = tuple(float(x) for x in match.group(2).split(","))
        if len(parameters) != (3 if kind == "SyN" else 1):
            raise ValueError("Unexpected transform parameter count for " + kind)
        for parameter in parameters:
            _finite(parameter, kind + " parameter")
        if parameters[0] <= 0:
            raise ValueError("Gradient step must be positive")
        _integer(stage.get("bins_or_radius"), kind + " bins/radius", 0 if kind == "SyN" else 5)
        schedules = [stage.get(key) for key in ("iterations", "shrink_factors", "smoothing_sigmas")]
        if any(not isinstance(values, list) or not values for values in schedules) or len({len(x) for x in schedules}) != 1:
            raise ValueError("Stage iteration, shrink and smoothing schedules must have matching nonzero lengths")
        for value in schedules[0]:
            _integer(value, kind + " iterations")
        for value in schedules[1]:
            _integer(value, kind + " shrink factor", 1)
        for value in schedules[2]:
            _finite(value, kind + " smoothing sigma")
        if stage.get("smoothing_units") not in ("vox", "mm"):
            raise ValueError("Smoothing units must be vox or mm")
    return config


def load_recipe(args):
    """Read one complete recipe, retaining its original provenance fields."""
    if args.config is not None:
        payload = args.config.read_bytes()
        source = str(args.config.resolve())
    else:
        name = args.preset or "brain-syn"
        payload = resources.files("fast_ants.presets").joinpath(name + ".json").read_bytes()
        source = "preset:" + name
    config = validate_recipe(json.loads(payload.decode("utf-8")))
    return config, {"source": source, "sha256": hashlib.sha256(payload).hexdigest()}


def _shrink_schedule(stage, recipe, grid):
    schedule = tuple(stage["shrink_factors"])
    if recipe.get("shrink_factor_policy", "explicit") == "explicit":
        return schedule
    spacing = grid.spacing_xyz
    minimum = min(range(3), key=lambda d: spacing[d])
    output = []
    # Preserve the pinned ANTs per-axis search, including its <= tie rule.
    for factor in schedule:
        factors = [factor] * 3
        target = spacing[minimum] * factor
        for d in range(3):
            difference = abs(spacing[d] * factor - target)
            if d == minimum:
                continue
            for candidate in range(factor - 1, 0, -1):
                delta = abs(spacing[d] * candidate - target)
                if delta <= difference:
                    difference, factors[d] = delta, candidate
        output.append(tuple(factors))
    return tuple(output)


def build_stages(api, recipe, grid):
    common = recipe["convergence"]
    linear = []
    for stage in recipe["stages"][:2]:
        kind, parameter = stage["transform"].split("[")
        linear.append(api.LinearStage(kind=kind, iterations=tuple(stage["iterations"]),
            shrink_factors=_shrink_schedule(stage, recipe, grid), smoothing_sigmas=tuple(stage["smoothing_sigmas"]),
            smoothing_units=stage["smoothing_units"], gradient_step=float(parameter[:-1]),
            metric=stage["metric"], bins=stage["bins_or_radius"],
            convergence_threshold=common["threshold"], convergence_window=common["window"],
            sampling_strategy=recipe["sampling"]["strategy"], sampling_fraction=recipe["sampling"]["fraction"]))
    stage = recipe["stages"][2]
    step, update, total = map(float, stage["transform"][4:-1].split(","))
    syn = api.SyNStage(iterations=tuple(stage["iterations"]), shrink_factors=_shrink_schedule(stage, recipe, grid),
        smoothing_sigmas=tuple(stage["smoothing_sigmas"]), smoothing_units=stage["smoothing_units"],
        gradient_step=step, radius=stage["bins_or_radius"], update_variance=update, total_variance=total,
        convergence_threshold=common["threshold"], convergence_window=common["window"])
    return tuple(linear), syn


def _runtime(threads):
    # Entry-point imports remain lightweight; set pools before numerical imports.
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    for key in THREAD_POOLS:
        os.environ[key] = "1"
    api = importlib.import_module("fast_ants")
    numba = importlib.import_module("numba")
    numba.set_num_threads(threads)
    return api, numba


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def run_registration(args):
    recipe, provenance = load_recipe(args)
    threads = args.threads or recipe["threads"]
    # Atomically claim a fresh directory. Existing results are never replaced.
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "console.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                return _execute(args, recipe, provenance, threads)
            except Exception as error:
                print(f"[fast-ants][error] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                raise


def _execute(args, recipe, provenance, threads):
    from .progress import create_progress
    progress = create_progress(args.verbose, args.log_every)
    started = time.perf_counter()
    if progress is not None:
        progress.message("setup", f"backend={args.backend} precision={args.precision} threads={threads} recipe={provenance['source']}")
    api, numba = _runtime(threads)
    read = progress.start("read", "Loading fixed and moving images") if progress is not None else None
    fixed, moving = api.load_images((args.fixed, args.moving), geometry_policy=args.geometry_policy,
                                   scaling_policy=args.scaling_policy)
    if read is not None:
        read.finish("loaded")
    linear, syn = build_stages(api, recipe, fixed.grid)
    options = dict(linear_stages=linear, syn_stage=syn, initialization_feature=recipe["initialization_feature"],
                   seed=recipe["random_seed"], output_directory=args.output)
    if args.backend == "metal":
        implementation = api.register_metal
        options.update(verbose=args.verbose, log_every=args.log_every)
    else:
        implementation = api.register if args.precision == "fp64" else api.register32
    stage = progress.start("registration", "Running registration, output QA and writes") if progress is not None else None
    result = implementation(fixed, moving, **options)
    if stage is not None:
        stage.finish("completed")
    elapsed = time.perf_counter() - started
    record = {"schema_version": 1, "status": "registration_completed_quality_metrics_recorded",
        "package_version": version(), "backend": args.backend, "precision": args.precision,
        "threads": numba.get_num_threads(), "random_seed": recipe["random_seed"],
        "recipe_provenance": provenance, "recipe": recipe,
        "geometry_policy": args.geometry_policy, "scaling_policy": args.scaling_policy,
        "inputs": {side: {"path": str(getattr(args, side).resolve()), "sha256": _digest(getattr(args, side))}
                   for side in ("fixed", "moving")},
        "verbose": args.verbose, "log_every": args.log_every,
        "logging_scope": "stage and iteration diagnostics" if args.backend == "metal" else "CLI phases; no CPU iteration diagnostics",
        "load_register_qa_write_seconds_diagnostic_not_benchmark": elapsed,
        "timing_scope": "Runtime import, input reads, registration, built-in QA and core writes; excludes CLI parsing and final input hashing/run metadata",
        "quality_status": "built-in_metrics_only_not_external_acceptance",
        "quality": result.quality,
        "outputs": {"affine": "affine.mat", "forward_warp": "warp.nii.gz", "inverse_warp": "inverse-warp.nii.gz",
                    "warped": "warped.nii.gz", "history": "history.json", "configuration": "configuration.json",
                    "quality": "quality.json", "console": "console.log"}}
    temporary = args.output / "run.json.part"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output / "run.json")
    if progress is not None:
        progress.complete(f"Outputs saved to {args.output}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_registration(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"fast-ants: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("fast-ants: Interrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
