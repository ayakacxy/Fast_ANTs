"""FP32 registration assembly, with its precision boundaries recorded."""

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from fast_ants.backends.cpu.fp64.fields import inverse_consistency
from fast_ants.backends.cpu.fp64.fields import jacobian_determinant
from fast_ants.pipeline.convergence import WindowConvergence
from fast_ants.backends.cpu.fp32.images import image32
from fast_ants.backends.cpu.fp32.filtering import smooth_image32
from fast_ants.backends.cpu.fp32.syn import identity_state32
from fast_ants.backends.cpu.fp32.syn import adapt_state32
from fast_ants.backends.cpu.fp32.syn import syn_step32
from fast_ants.backends.cpu.fp32.syn import combine_state32
from fast_ants.backends.cpu.fp32.fields import resample_field32
from fast_ants.backends.cpu.fp32.images import pull_image32
from fast_ants.backends.cpu.fp32.syn import _SyNWorkspace32
from fast_ants.pipeline.pyramid import shrink_grid
from fast_ants.core.config import SyNStage
from fast_ants.core.results import SyNRegistrationResult
from fast_ants.core.transforms import CompositeTransform
from fast_ants.io.transforms import save_affine
from fast_ants.io.transforms import save_field
from fast_ants.io.images import save_image
from fast_ants.backends.cpu.fp32.linear import register_linear32
from fast_ants.core.config import LinearStage


def register_nonlinear32(fixed,moving,initial_transform,*,syn_stage=None,fixed_mask=None,
                          moving_mask=None,output_directory=None,callback=None):
    """Validate FP32 SyN independently of linear-optimizer trajectory differences.

    The linear map is an explicit frozen input. This function alone is not a
    validation of FP32 Mattes/Rigid/Affine. Dense images, gradients, CC, Gaussian
    recurrences, composition, inverse fields and reductions all use FP32 kernels.
    """
    stage=SyNStage() if syn_stage is None else syn_stage
    started=time.perf_counter();history=[]
    nf,nm=image32(fixed,normalize=True),image32(moving,normalize=True)
    state=None
    for level,(iterations,shrink,sigma) in enumerate(zip(stage.iterations,stage.shrink_factors,stage.smoothing_sigmas)):
        level_started=time.perf_counter()
        fs,ms=smooth_image32(nf,sigma,stage.smoothing_units),smooth_image32(nm,sigma,stage.smoothing_units)
        grid=shrink_grid(fixed.grid,shrink)
        state=identity_state32(grid) if state is None else adapt_state32(state,grid)
        # Callback results must retain independent storage. Without an observer,
        # the sequential level can reuse its private next-state buffers.
        workspace=_SyNWorkspace32(grid) if callback is None else None
        monitor=WindowConvergence(stage.convergence_window);records=[];stop_reason='max_iterations'
        for i in range(iterations):
            result=syn_step32(fs,ms,state,linear_transform=initial_transform,radius=stage.radius,
                              gradient_step=stage.gradient_step,update_variance=stage.update_variance,
                              total_variance=stage.total_variance,fixed_mask=fixed_mask,moving_mask=moving_mask,
                              _workspace=workspace)
            state=result.state
            convergence=monitor.add(result.value)
            records.append({'iteration':i+1,'value':result.value,'convergence':convergence,**result.details})
            if callback is not None:callback(level,i+1,result)
            if convergence<stage.convergence_threshold:stop_reason='window_convergence';break
        history.append({'kind':'SyN','level':level,'shrink_factor':shrink,'sigma':sigma,
                        'sigma_units':stage.smoothing_units,'stop_reason':stop_reason,'iterations':records,
                        'seconds_diagnostic_not_benchmark':time.perf_counter()-level_started})
    forward,inverse=combine_state32(state)
    forward,inverse=resample_field32(forward,fixed.grid),resample_field32(inverse,fixed.grid)
    forward.inverse_field=inverse;inverse.inverse_field=forward
    pull=CompositeTransform((forward,initial_transform))
    inverse_pull=CompositeTransform((initial_transform.inverse(),inverse))
    warped=pull_image32(image32(moving),fixed.grid,forward,initial_transform)
    # QA is separate from the production kernels and evaluates stored FP32 data in FP64.
    jacobian=jacobian_determinant(forward)
    quality={'nonlinear_inverse_consistency':inverse_consistency(forward,inverse),
             'minimum_nonlinear_jacobian':float(jacobian.min()),'nonpositive_jacobian_points':int((jacobian<=0).sum()),
             'total_seconds_diagnostic_not_benchmark':time.perf_counter()-started,
             'peak_retained_state_field_bytes':int(sum(f.data.nbytes for f in state.fields().values()))}
    configuration={'precision':'FP32 dense nonlinear kernels; frozen linear input; CPU FP64 small geometry/coefficient/control setup',
                   'linear_transform_scope':'supplied frozen map; FP32 linear optimization not asserted by this function',
                   'syn_stage':asdict(stage),'coordinate_system':'LPS mm',
                   'input_quantization':'explicit FP32 arrays before normalization',
                   'geometry_policy':fixed.metadata.get('geometry_policy','explicit Grid'),
                   'cc_stability':'plane-local two-pass statistics, Chan merging and compensated FP32 sums; ANTs float1 CC epsilon',
                   'field_arithmetic':'local indices, no subtraction of global world positions',
                   'field_execution':'private level smoothing and alternating state buffers; inline paired CC gradients; fused inverse residual/norm/update scratch',
                   'step_normalization':'one maximum norm and one scale multiplication, without post-scale ULP adjustment',
                   'inverse_final_residual':'per-inversion post-update diagnostic skipped on private no-callback path; final output QA retained',
                   'host_fp64':'small 3x3/4x4 matrices, filter coefficient construction, scalar window convergence and QA only'}
    configuration['preprocessing_profile']='ANTs 256-bin endpoint histogram, marginal scale 10 and float endpoints'
    configuration['parameter_semantics']='ANTs CLI float-parsed steps/field variances'
    result=SyNRegistrationResult(fixed.grid,moving.grid,pull,inverse_pull,warped,history,
                                  initial_transform,forward,inverse,state,configuration,quality)
    if output_directory is not None:
        out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
        save_affine(initial_transform,out/'affine.mat')
        # Separate gzip streams can write concurrently; wait for all products
        # before returning so whole-process timing includes every output.
        with ThreadPoolExecutor(max_workers=3) as writers:
            futures=[writers.submit(save_field,forward,out/'warp.nii.gz'),
                     writers.submit(save_field,inverse,out/'inverse-warp.nii.gz'),
                     writers.submit(save_image,warped,out/'warped.nii.gz')]
            for future in futures:
                future.result()
        for name,item in [('configuration',configuration),('history',history),('quality',quality)]:
            (out/(name+'.json')).write_text(json.dumps(item,indent=2)+'\n')
    return result


def register32(fixed,moving,*,linear_stages=None,syn_stage=None,initialization_feature=1,
               seed=20261022,fixed_mask=None,moving_mask=None,output_directory=None,callback=None,linear_precision='single'):
    """Run ordinary FP32 Rigid/Affine/SyN targeting official float1 semantics.

    No high/low floating-point emulation is used in this public path.
    """
    started=time.perf_counter()
    out=None if output_directory is None else Path(output_directory)
    linear_stages=(LinearStage('Rigid'),LinearStage('Affine')) if linear_stages is None else tuple(linear_stages)
    linear=register_linear32(fixed,moving,linear_stages,initialization_feature,seed,fixed_mask,moving_mask,
                              None if out is None else out/'linear',precision_policy=linear_precision)
    result=register_nonlinear32(fixed,moving,linear.fixed_to_moving,syn_stage=syn_stage,fixed_mask=fixed_mask,
                                 moving_mask=moving_mask,output_directory=out,callback=callback)
    result.history=linear.history+result.history
    result.configuration.update({'precision':'ordinary FP32 volume kernels and float1 transform/CG state; float1 histogram finalization and native CPU FP64 geometry/scalar setup',
                                  'reference_target':'ANTs --float 1 behavior reference; algorithm and registration quality under validation',
                                  'linear_intensity_policy':'normalized images also at zero smoothing',
        'image_gaussian':'DC-preserving FP32 two-section recurrence',
                                  'linear_transform_scope':'independent '+linear_precision+' FP32 dense Mattes, Rigid/Affine optimization',
                                  'linear_precision':linear_precision,
                                  'linear_execution':'streamed value and derivative MI with bounded reusable block PDFs; dense integer sampling specialization',
                                  'full_pipeline':True,'initialization_feature':initialization_feature,'seed':seed,
                                  'linear_stages':[asdict(s) for s in linear_stages],
                                  'software_expanded_precision':False,
                                  'host_fp64':'3x3/4x4 geometry/filter setup; joint-PDF total, convergence energy, scalar norm setup and QA'})
    result.quality['full_pipeline_seconds_diagnostic_not_benchmark']=time.perf_counter()-started
    if out is not None:
        (out/'history.json').write_text(json.dumps(result.history,indent=2)+'\n')
        (out/'configuration.json').write_text(json.dumps(result.configuration,indent=2)+'\n')
        (out/'quality.json').write_text(json.dumps(result.quality,indent=2)+'\n')
    return result
