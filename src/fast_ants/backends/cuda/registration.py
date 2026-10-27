"""CUDA FP32 registration assembly with explicit precision boundaries."""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from fast_ants.core.config import LinearStage, SyNStage
from fast_ants.core.results import SyNRegistrationResult
from fast_ants.core.transforms import CompositeTransform
from fast_ants.pipeline.pyramid import shrink_grid
from fast_ants.io.transforms import save_affine, save_field
from fast_ants.io.images import save_image
from fast_ants.progress import create_progress
from fast_ants.backends.cuda.output_quality import output_quality
from fast_ants.backends.cuda.preparation import _PreparationCache
from fast_ants.backends.cuda.linear import _fit_linear
from fast_ants.backends.cuda.syn import _syn_level


def register_cuda(fixed,moving,*,linear_stages=None,syn_stage=None,initial_transform=None,
                  initialization_feature=1,seed=20261022,fixed_mask=None,moving_mask=None,
                  output_directory=None,callback=None,verbose=False,log_every=1):
    """Run Rigid/Affine/SyN with ordinary FP32 CUDA volume calculations.

    Geometry, optimizer controls, final QA and I/O use the existing precision
    contracts. Image-sized fitting arrays remain on the device. Supplying an
    initial transform explicitly skips linear fitting. Callbacks retain their
    own field arrays; inverse stopping diagnostics are evaluated when observed.
    """
    from fast_ants.backends.cuda.ops import CUDAOps
    progress=create_progress(verbose,log_every)
    setup=progress.start('setup','backend=CUDA precision=FP32') if progress is not None else None
    started=time.perf_counter();ops=CUDAOps();ctx=ops.context
    stage=SyNStage() if syn_stage is None else syn_stage
    stages=(LinearStage('Rigid'),LinearStage('Affine')) if linear_stages is None else tuple(linear_stages)
    if initial_transform is None and not stages:
        raise ValueError('At least one linear stage or an initial transform is required')
    gpu_fixed,gpu_moving=ops.image(fixed),ops.image(moving)
    def mask_input(mask):
        if mask is None:return None
        if not np.all((mask.data==0)|(mask.data==1)):raise ValueError('Masks must be binary')
        return ops.image(mask)
    fm,mm=mask_input(fixed_mask),mask_input(moving_mask)
    preparation=_PreparationCache()
    normalized=ops.image(gpu_fixed,True),ops.image(gpu_moving,True)
    if setup is not None:setup.finish('ready')
    if initial_transform is None:
        linear,history=_fit_linear(ops,gpu_fixed,gpu_moving,stages,initialization_feature,seed,fm,mm,
                                   _normalized=normalized,_preparation=preparation,_progress=progress)
    else:
        linear,history=initial_transform,[]
        if progress is not None:progress.message('initialization','Initial transform supplied; linear fitting skipped')
    ctx.finish()
    linear_seconds=time.perf_counter()-started
    nf,nm=normalized;state=None
    for level,(iterations,shrink,sigma) in enumerate(zip(stage.iterations,stage.shrink_factors,stage.smoothing_sigmas)):
        level_started=time.perf_counter()
        fs=preparation.smooth(ops,nf,sigma,stage.smoothing_units)
        ms=preparation.smooth(ops,nm,sigma,stage.smoothing_units)
        grid=shrink_grid(fixed.grid,shrink)
        state=ops.identity(grid) if state is None else ops.adapt(state,grid)
        level_progress=(progress.start('syn',
            f'level={level+1}/{len(stage.iterations)} shrink={shrink} sigma={sigma} {stage.smoothing_units} '
            f'samples={int(np.prod(grid.shape_zyx))} budget={iterations}',iterations=True,total=iterations,started=level_started)
            if progress is not None else None)
        state,records,reason=_syn_level(ops,fs,ms,state,stage,linear,fm,mm,iterations,level,callback,
                                        _progress=level_progress)
        ctx.finish(*(field.data for field in state.fields().values()))
        if level_progress is not None:level_progress.finish(reason)
        history.append({'kind':'SyN','level':level,'shrink_factor':shrink,'sigma':sigma,
            'sigma_units':stage.smoothing_units,'stop_reason':reason,'iterations':records,
            'seconds_diagnostic_not_benchmark':time.perf_counter()-level_started})
    combined=progress.start('combine','Compose fields and warp moving image') if progress is not None else None
    forward,inverse=ops.combine(state)
    forward,inverse=ops.resample(forward,fixed.grid),ops.resample(inverse,fixed.grid)
    warped=ops.pull(gpu_moving,fixed.grid,forward,linear)
    ctx.finish(forward.data,inverse.data,warped.data)
    if combined is not None:combined.finish('combined')
    gpu_seconds=time.perf_counter()-started
    qa_started=time.perf_counter()
    qa=progress.start('qa','Final inverse consistency and Jacobian summaries') if progress is not None else None
    quality=output_quality(ctx,forward,inverse)
    qa_seconds=time.perf_counter()-qa_started
    if qa is not None:qa.finish('checked')
    download_started=time.perf_counter()
    download=progress.start('download','Transfer final fields and warped image') if progress is not None else None
    forward,inverse,warped=ops.download_field(forward),ops.download_field(inverse),ops.download_image(warped)
    download_seconds=time.perf_counter()-download_started
    if download is not None:download.finish('downloaded')
    forward.inverse_field=inverse;inverse.inverse_field=forward
    pull=CompositeTransform((forward,linear));inverse_pull=CompositeTransform((linear.inverse(),inverse))
    quality.update({
        'output_download_seconds_diagnostic_not_benchmark':download_seconds,
        'output_qa_seconds_diagnostic_not_benchmark':qa_seconds,
        'linear_seconds_diagnostic_not_benchmark':linear_seconds,
        'synchronized_pipeline_seconds_diagnostic_not_benchmark':gpu_seconds,
        'total_with_download_and_qa_seconds_diagnostic_not_benchmark':time.perf_counter()-started,
        'retained_state_field_bytes':sum(field.data.nbytes for field in state.fields().values()),
        'memory_scope':'four retained state fields; not peak process or device memory'})
    configuration={'backend':'CuPy custom CUDA','precision':'ordinary FP32 fitting; native FP64 small controls and output QA',
        'software_expanded_precision':False,'linear_stages':[asdict(item) for item in stages] if initial_transform is None else [],
        'initial_transform_supplied':initial_transform is not None,'syn_stage':asdict(stage),
        'initialization_feature':initialization_feature,'seed':seed,'coordinate_system':'LPS mm',
        'linear_intensity_policy':'ANTs float1 target: normalized images also at zero smoothing',
        'image_gaussian':'DC-preserving FP32 two-section recurrence',
        'step_normalization':'one maximum norm and one scale multiplication, without post-scale ULP adjustment',
        'preprocessing_cache':{'maximum_bytes':preparation.maximum_bytes,'retained_bytes':preparation.bytes,
                               'hits':preparation.hits,'misses':preparation.misses},
        'output_qa':'native CUDA FP64 composition, exact percentile and Jacobian summary',
        'inverse_final_residual':('evaluated for callback observations' if callback is not None else
                                 'per-inversion post-update diagnostic skipped during fitting; final output QA retained'),
        'reference_target':'ANTs --float 1; matching algorithms and final registration quality',
        'host_fp64':'small geometry/filter coefficients, optimizer and convergence controls',
        'cpu_fallbacks':[],'device_state_retained':True,'runtime':ctx.report()}
    result=SyNRegistrationResult(fixed.grid,moving.grid,pull,inverse_pull,warped,history,linear,
                                 forward,inverse,state,configuration,quality)
    if output_directory is not None:
        writing=progress.start('write',f'Output directory: {output_directory}') if progress is not None else None
        write_started=time.perf_counter()
        out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
        save_affine(linear,out/'affine.mat')
        # Join all independent output streams before returning so the external
        # whole-process measurement includes completed image/field writes.
        with ThreadPoolExecutor(max_workers=3) as writers:
            futures=[writers.submit(save_field,forward,out/'warp.nii.gz'),
                     writers.submit(save_field,inverse,out/'inverse-warp.nii.gz'),
                     writers.submit(save_image,warped,out/'warped.nii.gz')]
            for future in futures:future.result()
        quality['output_data_write_seconds_diagnostic_not_benchmark']=time.perf_counter()-write_started
        for name,item in [('history',history),('configuration',configuration),('quality',quality)]:
            (out/(name+'.json')).write_text(json.dumps(item,indent=2)+'\n')
        if writing is not None:writing.finish('written')
    if progress is not None:progress.complete()
    return result
