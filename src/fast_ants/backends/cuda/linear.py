"""Multi-level CUDA rigid/affine fitting with shared float1 host control."""

import time
import numpy as np

from fast_ants.pipeline.models32 import ComposedLinearModel32 as ComposedLinearModel
from fast_ants.pipeline.models32 import AffineTransform32 as AffineTransform
from fast_ants.pipeline.optimizer32 import conjugate_gradient32 as conjugate_gradient
from fast_ants.pipeline.pyramid import shrink_grid
from fast_ants.pipeline.sampling_plan import sample_virtual_domain


def _fit_linear(ops,fixed,moving,stages,initialization_feature,seed,fixed_mask,moving_mask,
                _normalized=None,_preparation=None,_progress=None):
    from fast_ants.backends.cuda.metrics.mattes.metric import CUDAMattesMetric, moments
    def center(image):
        if initialization_feature==0:return image.grid.center
        if initialization_feature==2:return image.grid.origin
        if initialization_feature!=1:raise ValueError('Initialization feature must be 0, 1 or 2')
        totals=moments(ops.context,image.data).astype(np.float64)
        if abs(totals[0])<=np.finfo(np.float32).eps:raise ValueError('Nonzero image mass required')
        return image.grid.index_to_physical(totals[1:]/totals[0]+(np.asarray(image.grid.shape_xyz)-1)*.5)
    initialization=(_progress.start('initialization',f'feature={initialization_feature}')
                    if _progress is not None else None)
    origin,target=center(fixed),center(moving)
    current=AffineTransform.centered(np.eye(3),target-origin,origin)
    if initialization is not None:initialization.finish('initialized')
    nf,nm=(ops.image(fixed,True),ops.image(moving,True)) if _normalized is None else _normalized
    history=[]
    for stage_index,stage in enumerate(stages):
        if stage.metric!='MI':raise ValueError('CUDA linear fitting currently supports Mattes MI')
        model=ComposedLinearModel(stage.kind,origin,current);parameters=model.identity_parameters()
        for level,(iterations,shrink,sigma) in enumerate(zip(stage.iterations,stage.shrink_factors,stage.smoothing_sigmas)):
            started=time.perf_counter()
            smooth=(ops.smooth_image if _preparation is None else
                    lambda image,sigma,units:_preparation.smooth(ops,image,sigma,units))
            fs,ms=smooth(nf,sigma,stage.smoothing_units),smooth(nm,sigma,stage.smoothing_units)
            grid=shrink_grid(fixed.grid,shrink);points=None
            if stage.sampling_strategy!='None':
                points=sample_virtual_domain(grid,stage.sampling_strategy,stage.sampling_fraction,
                                              seed+stage_index*100+level).points_lps
            metric=CUDAMattesMetric(ops,fs,ms,grid,stage.bins,points,fixed_mask,moving_mask)
            metric._device_control_mode=True
            level_progress=(_progress.start(stage.kind.lower(),
                f'stage={stage_index+1}/{len(stages)} level={level+1}/{len(stage.iterations)} '
                f'shrink={shrink} sigma={sigma} {stage.smoothing_units} '
                f'sampling={stage.sampling_strategy} samples={metric.indices.shape[0]} budget={iterations}',
                iterations=True,total=iterations,started=started) if _progress is not None else None)
            observer={'progress':level_progress} if level_progress is not None else {}
            optimized=conjugate_gradient(metric,model,parameters,grid,iterations,stage.gradient_step,
                                         stage.convergence_threshold,stage.convergence_window,**observer)
            parameters=optimized.parameters;current=model.transform(parameters)
            final=metric.evaluate(model,parameters,False);ops.context.finish()
            if level_progress is not None:level_progress.finish(optimized.stop_reason)
            history.append({'stage':stage_index,'kind':stage.kind,'level':level,'shrink_factor':shrink,
                'sigma':sigma,'sigma_units':stage.smoothing_units,'sampling':stage.sampling_strategy,
                'sample_count':metric.indices.shape[0],'stop_reason':optimized.stop_reason,
                'scales':optimized.scales.tolist(),'final_value_reevaluated':final.value,
                'iterations':optimized.history,'seconds_diagnostic_not_benchmark':time.perf_counter()-started,
                'precision':'ordinary FP32 CUDA volume work and float1 transform/CG state; GPU PDF finalization with native FP64 joint total',
                'host_control':{'evaluations':metric.control_evaluations,
                    'derivative_evaluations':metric.control_derivative_evaluations,'histogram_bytes':metric.control_bytes,
                    'transfer_scope':'successful evaluations only; small final packets; physical reads recorded in runtime.transfers'}})
    return current,history
