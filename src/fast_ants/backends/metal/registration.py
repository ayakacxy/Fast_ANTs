"""Stage implementation extracted without changing numerical definitions."""
from dataclasses import asdict

from concurrent.futures import ThreadPoolExecutor

import json

from pathlib import Path

import time

import numpy as np

from fast_ants.backends.cpu.fp64.fields import inverse_consistency
from fast_ants.backends.cpu.fp64.fields import jacobian_determinant

from fast_ants.pipeline.models32 import ComposedLinearModel32 as ComposedLinearModel
from fast_ants.pipeline.models32 import AffineTransform32 as AffineTransform

from fast_ants.core.config import LinearStage

from fast_ants.pipeline.convergence import WindowConvergence

from fast_ants.pipeline.optimizer32 import conjugate_gradient32 as conjugate_gradient

from fast_ants.pipeline.pyramid import shrink_grid

from fast_ants.core.config import SyNStage
from fast_ants.core.results import SyNRegistrationResult

from fast_ants.pipeline.sampling_plan import sample_virtual_domain

from fast_ants.core.transforms import CompositeTransform

from fast_ants.io.transforms import save_affine
from fast_ants.io.transforms import save_field

from fast_ants.io.images import save_image

from fast_ants.progress import create_progress

from fast_ants.backends.metal.preparation import _PreparationCache
from fast_ants.backends.metal.linear import _fit_linear
from fast_ants.backends.metal.syn import _syn_level

def register_metal(fixed,moving,*,linear_stages=None,syn_stage=None,initial_transform=None,
                   initialization_feature=1,seed=20261022,fixed_mask=None,moving_mask=None,
                   output_directory=None,callback=None,optimizations=None,verbose=False,log_every=1):
    """Run GPU Rigid/Affine/SyN without software-expanded floating point.

    Input/output and explicit QA are host arrays. All iterative image/field
    work uses custom Metal kernels; small native CPU FP64 controls are logged.
    Supplying initial_transform skips linear fitting and records that choice.
    ``verbose`` enables scoped stage/iteration output, independently of the
    numerical callback. ``log_every`` is a positive iteration print interval;
    the first and final iterations are always retained.
    """
    progress=create_progress(verbose,log_every)
    setup=progress.start('setup','backend=Metal precision=FP32') if progress is not None else None
    from fast_ants.backends.metal.ops import MetalOps
    from fast_ants.backends.metal.runtime import MetalContext
    started=time.perf_counter();ops=MetalOps(MetalContext(optimizations));ctx=ops.context
    stage=SyNStage() if syn_stage is None else syn_stage
    stages=(LinearStage('Rigid'),LinearStage('Affine')) if linear_stages is None else tuple(linear_stages)
    if initial_transform is None and not stages:raise ValueError('At least one linear stage or an initial transform is required')
    gpu_fixed,gpu_moving=ops.image(fixed),ops.image(moving)
    def mask_input(mask):
        if mask is None:return None
        if not np.all((mask.data==0)|(mask.data==1)):raise ValueError('Masks must be binary')
        return ops.image(mask)
    fm,mm=mask_input(fixed_mask),mask_input(moving_mask)
    preparation=_PreparationCache() if 'cached_preparation' in ctx.optimizations else None
    normalized=(ops.image(gpu_fixed,True),ops.image(gpu_moving,True)) if preparation is not None else None
    if setup is not None:setup.finish('ready')
    if initial_transform is None:
        linear,history=_fit_linear(ops,gpu_fixed,gpu_moving,stages,initialization_feature,seed,fm,mm,
                                   _normalized=normalized,_preparation=preparation,_progress=progress)
    else:
        linear,history=initial_transform,[]
        if progress is not None:progress.message('initialization','Initial transform supplied; linear fitting skipped')
    linear_seconds=time.perf_counter()-started
    nf,nm=(ops.image(gpu_fixed,True),ops.image(gpu_moving,True)) if normalized is None else normalized
    state=None
    for level,(iterations,shrink,sigma) in enumerate(zip(stage.iterations,stage.shrink_factors,stage.smoothing_sigmas)):
        level_started=time.perf_counter()
        smooth=ops.smooth_image if preparation is None else lambda image,sigma,units:preparation.smooth(ops,image,sigma,units)
        fs,ms=smooth(nf,sigma,stage.smoothing_units),smooth(nm,sigma,stage.smoothing_units)
        grid=shrink_grid(fixed.grid,shrink)
        state=ops.identity(grid) if state is None else ops.adapt(state,grid)
        level_progress=(progress.start('syn',
            f'level={level+1}/{len(stage.iterations)} shrink={shrink} sigma={sigma} {stage.smoothing_units} '
            f'samples={int(np.prod(grid.shape_zyx))} budget={iterations}',iterations=True,started=level_started)
            if progress is not None else None)
        state,records,reason,discarded=_syn_level(ops,fs,ms,state,stage,linear,fm,mm,iterations,level,callback,
                                                _progress=level_progress)
        ctx.finish(*(f.data for f in state.fields().values()))
        if level_progress is not None:level_progress.finish(reason)
        history.append({'kind':'SyN','level':level,'shrink_factor':shrink,'sigma':sigma,
            'sigma_units':stage.smoothing_units,'stop_reason':reason,'iterations':records,'cc_scalar_guard_discarded_candidate_iterations':discarded,
            'seconds_diagnostic_not_benchmark':time.perf_counter()-level_started})
    combined=progress.start('combine','Compose fields and warp moving image') if progress is not None else None
    forward,inverse=ops.combine(state)
    forward,inverse=ops.resample(forward,fixed.grid),ops.resample(inverse,fixed.grid)
    warped=ops.pull(gpu_moving,fixed.grid,forward,linear)
    ctx.finish(forward.data,inverse.data,warped.data)
    if combined is not None:combined.finish('combined')
    gpu_seconds=time.perf_counter()-started
    download_started=time.perf_counter()
    download=progress.start('download','Transfer final fields and warped image') if progress is not None else None
    forward,inverse,warped=ops.download_field(forward),ops.download_field(inverse),ops.download_image(warped)
    download_seconds=time.perf_counter()-download_started
    if download is not None:download.finish('downloaded')
    forward.inverse_field=inverse;inverse.inverse_field=forward
    pull=CompositeTransform((forward,linear));inverse_pull=CompositeTransform((linear.inverse(),inverse))
    qa_started=time.perf_counter()
    qa=progress.start('qa','Final inverse consistency and Jacobian summaries') if progress is not None else None
    if 'streamed_output_qa' in ctx.optimizations:
        from fast_ants.quality.output_kernels import output_quality
        quality=output_quality(forward,inverse)
    else:
        jacobian=jacobian_determinant(forward)
        quality={'nonlinear_inverse_consistency':inverse_consistency(forward,inverse),
                 'minimum_nonlinear_jacobian':float(jacobian.min()),'nonpositive_jacobian_points':int((jacobian<=0).sum())}
    if qa is not None:qa.finish('checked')
    quality.update({
        'output_download_seconds_diagnostic_not_benchmark':download_seconds,
        'output_qa_seconds_diagnostic_not_benchmark':time.perf_counter()-qa_started,
        'linear_seconds_diagnostic_not_benchmark':linear_seconds,
        'synchronized_pipeline_seconds_diagnostic_not_benchmark':gpu_seconds,
        'total_with_download_and_qa_seconds_diagnostic_not_benchmark':time.perf_counter()-started,
        'retained_state_field_bytes':sum(f.data.nbytes for f in state.fields().values()),
        'memory_scope':'four retained state fields; not peak process or device memory'})
    configuration={'backend':'MLX custom Metal','precision':'ordinary FP32 volume work; native CPU FP64 small controls',
        'software_expanded_precision':False,'optimizations':sorted(ctx.optimizations),'linear_stages':[asdict(s) for s in stages] if initial_transform is None else [],
        'initial_transform_supplied':initial_transform is not None,'syn_stage':asdict(stage),
        'initialization_feature':initialization_feature,'seed':seed,'coordinate_system':'LPS mm',
        'linear_intensity_policy':'ANTs float1 target: normalized images also at zero smoothing',
        'image_gaussian':('DC-preserving FP32 two-section recurrence' if 'gaussian_sections' in ctx.optimizations else 'legacy FP32 fourth-order recurrence'),
        'stability':'integer voxel origins/local FP32 offsets; rounded-pixel float1 bins; FMA; fixed-order compensated histograms',
        'step_normalization':'one maximum norm and one scale multiplication, without post-scale ULP adjustment',
        'initial_moments':('parallel centered FP32 tile/tree reduction' if 'parallel_moments' in ctx.optimizations else 'legacy compensated serial reduction'),
        'inverse_control':('independent sides, fused residual/norm tiles and shared control reads' if 'batched_inverse' in ctx.optimizations else 'separate per-side inverse controls'),
        'mi_finalization':('GPU FP32 PDF finalization for supported private streamed metrics; legacy host fallback retained' if 'device_mi_control' in ctx.optimizations else 'bounded host float1 PDF finalization'),
        'mi_search_batching':('uncached independent point/middle pairs; adaptive dependent trials remain sequential' if {'device_mi_control','batch_mi_search'} <= ctx.optimizations else 'sequential trial controls'),
        'preprocessing_cache':({'maximum_bytes':preparation.maximum_bytes,'retained_bytes':preparation.bytes,
                                'hits':preparation.hits,'misses':preparation.misses} if preparation is not None else None),
        'output_qa':('streamed CPU FP64 compositions, exact percentile and Jacobian summary' if 'streamed_output_qa' in ctx.optimizations else 'legacy CPU FP64 output QA'),
        'scaled_composition':('paired maximum controls and endpoint-scaled interpolation; capture retains separate intermediates' if 'fused_scaled_compose' in ctx.optimizations else 'separate scaling and composition'),
        'cc_statistics':('three-axis centered short-window/Chan statistics, radius 0..4; dense fallback' if 'cc_window_stats' in ctx.optimizations else 'centered two-pass stencil statistics'),
        'mi_device_finalizer':('one group with parallel shared log preparation and ordered channel FMA' if 'fused_mi_finalize' in ctx.optimizations else 'two-stage device finalization when supported'),
        'async_control_scheduling':{'independent_mi_trials':'async_control_prefetch' in ctx.optimizations,
                                    'deferred_cc_score':'deferred_syn_controls' in ctx.optimizations,
                                    'device_max_inverse_update':'overlapped_inverse' in ctx.optimizations,
                                    'scope':'explicit async graph submission; actual overlap depends on runtime; dependent line-search/inverse stop decisions retained'},
        'inverse_final_residual':('evaluated for callback observations' if callback is not None else
                                  'per-inversion post-update diagnostic skipped during fitting; final output QA retained'),
        'reference_target':'ANTs --float 1 behavior reference; algorithm and registration quality under validation',
        'host_fp64':'3x3/4x4 geometry/filter coefficients; joint-PDF total in legacy/fallback path; convergence energy and scalar setup; output QA',
        'cpu_fallbacks':[],'device_state_retained':True,'runtime':ctx.report()}
    result=SyNRegistrationResult(fixed.grid,moving.grid,pull,inverse_pull,warped,history,linear,
                                 forward,inverse,state,configuration,quality)
    if output_directory is not None:
        writing=progress.start('write',f'Output directory: {output_directory}') if progress is not None else None
        write_started=time.perf_counter()
        out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
        save_affine(linear,out/'affine.mat')
        # Independent gzip streams write concurrently. Join all products
        # before returning so synchronized whole-process timing includes IO.
        with ThreadPoolExecutor(max_workers=3) as writers:
            futures=[writers.submit(save_field,forward,out/'warp.nii.gz'),
                     writers.submit(save_field,inverse,out/'inverse-warp.nii.gz'),
                     writers.submit(save_image,warped,out/'warped.nii.gz')]
            for future in futures:
                future.result()
        quality['output_data_write_seconds_diagnostic_not_benchmark']=time.perf_counter()-write_started
        for name,item in [('history',history),('configuration',configuration),('quality',quality)]:
            (out/(name+'.json')).write_text(json.dumps(item,indent=2)+'\n')
        if writing is not None:writing.finish('written')
    if progress is not None:progress.complete()
    return result

