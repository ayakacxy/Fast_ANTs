from fast_ants.core.results import SyNRegistrationResult
from fast_ants.core.config import SyNStage
"""Independent CPU Rigid -> Affine -> standard multi-level greedy SyN."""

from dataclasses import asdict,dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from fast_ants.backends.cpu.fp64.fields import inverse_consistency
from fast_ants.backends.cpu.fp64.fields import jacobian_determinant
from fast_ants.backends.cpu.fp64.fields import resample_field
from fast_ants.core.types import RegistrationResult
from fast_ants.core.config import LinearStage
from fast_ants.backends.cpu.fp64.linear import register_linear
from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.backends.cpu.fp64.preprocessing import prepare_level
from fast_ants.backends.cpu.fp64.preprocessing import preprocess_image
from fast_ants.backends.cpu.fp64.sampling import apply_image
from fast_ants.backends.cpu.fp64.sampling import transform_to_field
from fast_ants.core.state import SyNState
from fast_ants.backends.cpu.fp64.syn import syn_step
from fast_ants.backends.cpu.fp64.syn import _SyNWorkspace
from fast_ants.core.transforms import CompositeTransform
from fast_ants.io.transforms import save_affine
from fast_ants.io.transforms import save_field
from fast_ants.io.images import save_image






def register(fixed,moving,*,linear_stages=None,syn_stage=None,initial_transform=None,
             initialization_feature=1,seed=20261022,fixed_mask=None,moving_mask=None,
             output_directory=None,callback=None,_progress_callback=None):
    """Run CPU FP64 registration with explicit image-pull transform directions.

    A supplied initial_transform freezes the linear map and skips linear fitting.
    Fields describe the nonlinear map on fixed-space grids. The complete pull is
    linear(forward_field(x)); its inverse applies inverse linear before inverse
    field. The original moving image is used for final output resampling.
    """
    stage=SyNStage() if syn_stage is None else syn_stage
    stages=(LinearStage('Rigid'),LinearStage('Affine')) if linear_stages is None else tuple(linear_stages)
    out=None if output_directory is None else Path(output_directory)
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
    history=[]; started=time.perf_counter()
    if initial_transform is None:
        linear_result=register_linear(fixed,moving,stages,initialization_feature,seed,fixed_mask,moving_mask,
                                      None if out is None else out/'linear')
        linear=linear_result.fixed_to_moving
        history.extend(linear_result.history)
    else:
        linear=initial_transform
    linear_seconds=time.perf_counter()-started
    normalized_fixed,normalized_moving=preprocess_image(fixed),preprocess_image(moving)
    state=None
    for level,(iterations,shrink,sigma) in enumerate(zip(stage.iterations,stage.shrink_factors,stage.smoothing_sigmas)):
        level_started=time.perf_counter()
        images=prepare_level(normalized_fixed,normalized_moving,shrink,sigma,stage.smoothing_units)
        state=SyNState.identity(images.virtual_grid) if state is None else state.adapt(images.virtual_grid)
        workspace=_SyNWorkspace(images.virtual_grid) if callback is None else None
        convergence=WindowConvergence(stage.convergence_window)
        records=[]; stop_reason='max_iterations'
        for iteration in range(iterations):
            result=syn_step(images.fixed,images.moving,state,linear_transform=linear,radius=stage.radius,
                            gradient_step=stage.gradient_step,update_variance=stage.update_variance,
                            total_variance=stage.total_variance,fixed_mask=fixed_mask,moving_mask=moving_mask,_workspace=workspace)
            state=result.state
            profile=convergence.add(result.value)
            records.append({'iteration':iteration+1,'value':result.value,'convergence':profile,**result.details})
            if callback is not None:
                callback(level,iteration+1,result)
            if _progress_callback is not None:
                _progress_callback(level,iteration+1,float(result.value))
            if profile<stage.convergence_threshold:
                stop_reason='window_convergence'
                break
        history.append({'kind':'SyN','level':level,'shrink_factor':shrink,'sigma':sigma,
                        'sigma_units':stage.smoothing_units,'stop_reason':stop_reason,
                        'seconds_diagnostic_not_benchmark':time.perf_counter()-level_started,'iterations':records})
    forward,inverse=state.combined()
    # For public outputs use the original fixed grid, regardless of final shrink.
    forward=resample_field(forward,fixed.grid); inverse=resample_field(inverse,fixed.grid)
    forward.inverse_field=inverse; inverse.inverse_field=forward
    fixed_to_moving=CompositeTransform((forward,linear))
    moving_to_fixed=CompositeTransform((linear.inverse(),inverse))
    warped=apply_image(moving,fixed,fixed_to_moving)
    jacobian=jacobian_determinant(forward)
    quality={'nonlinear_inverse_consistency':inverse_consistency(forward,inverse),
             'minimum_nonlinear_jacobian':float(jacobian.min()),'nonpositive_jacobian_points':int((jacobian<=0).sum()),
             'linear_seconds_diagnostic_not_benchmark':linear_seconds,
             'total_seconds_diagnostic_not_benchmark':time.perf_counter()-started,
             'peak_retained_state_field_bytes':int(sum(f.data.nbytes for f in state.fields().values())),
             'memory_scope':'retained four fields only; not peak process RSS'}
    configuration={'precision':'CPU FP64 reference (ITK float intermediate image smoothing retained)',
                   'linear_stages':[asdict(s) for s in stages] if initial_transform is None else [],
                   'initial_transform_supplied':initial_transform is not None,'syn_stage':asdict(stage),
                   'initialization_feature':initialization_feature,'seed':seed,
                   'histogram_matching':False,'winsorization':None,'interpolation':'Linear',
                   'parameter_semantics':'ANTs CLI float-parsed step/field variances, including in FP64 mode',
                   'fp64_execution':'private-block Mattes PDFs with compensated merge; same-grid direct field pulls; local-coordinate inverse residuals with pairwise norm reduction',
                   'reference_policy':'ANTs float0 algorithm and registration quality; trajectory/bitwise equality diagnostic only',
                   'preprocessing_profile':'ANTs 256-bin endpoint histogram, marginal scale 10, float endpoints and window factor/offset',
                   'coordinate_system':'LPS mm','fixed_grid_shape_zyx':fixed.grid.shape_zyx,
                   'moving_grid_shape_zyx':moving.grid.shape_zyx,
                   'fixed_geometry_policy':fixed.metadata.get('geometry_policy','explicit Grid'),
                   'moving_geometry_policy':moving.metadata.get('geometry_policy','explicit Grid'),
                   'fixed_mask_supplied':fixed_mask is not None,'moving_mask_supplied':moving_mask is not None}
    result=SyNRegistrationResult(fixed.grid,moving.grid,fixed_to_moving,moving_to_fixed,warped,history,
                                  linear,forward,inverse,state,configuration,quality)
    if out is not None:
        save_affine(linear,out/'affine.mat')
        # Each writer owns a different NIfTI and gzip stream. zlib releases the
        # GIL; numerical arrays and the returned result remain read-only here.
        with ThreadPoolExecutor(max_workers=3) as writers:
            futures=[writers.submit(save_field,forward,out/'warp.nii.gz'),
                     writers.submit(save_field,inverse,out/'inverse-warp.nii.gz'),
                     writers.submit(save_image,warped,out/'warped.nii.gz')]
            for future in futures:
                future.result()
        for filename,item in [('configuration.json',configuration),('history.json',history),('quality.json',quality)]:
            (out/filename).write_text(json.dumps(item,indent=2)+'\n')
    return result
