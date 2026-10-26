"""FP32 dense linear registration targeting ANTs float1 control semantics."""

from pathlib import Path
import time
import numpy as np

from fast_ants.backends.cpu.fp32.metrics.mattes_kernels import moments32
from fast_ants.core.types import RegistrationResult
from fast_ants.pipeline.models32 import ComposedLinearModel32 as ComposedLinearModel
from fast_ants.pipeline.models32 import AffineTransform32 as AffineTransform
from fast_ants.core.config import LinearStage
from fast_ants.backends.cpu.fp32.metrics.mattes import MattesMetric32
from fast_ants.pipeline.optimizer32 import conjugate_gradient32 as conjugate_gradient
from fast_ants.backends.cpu.fp32.images import image32
from fast_ants.backends.cpu.fp32.filtering import smooth_image32
from fast_ants.backends.cpu.fp32.images import pull_image32
from fast_ants.pipeline.pyramid import shrink_grid
from fast_ants.pipeline.sampling_plan import sample_virtual_domain


def initialization_point32(image,feature,precision_policy='single'):
    if feature==0:return image.grid.center
    if feature==2:return image.grid.origin
    if feature!=1:raise ValueError("Initialization feature must be 0,1,2")
    if precision_policy!='single':raise ValueError('Software-expanded FP32 is retired from the registration path')
    totals=moments32(image32(image).data).astype(np.float64)
    if abs(totals[0])<=np.finfo(np.float32).eps:raise ValueError("Nonzero image mass required")
    return image.grid.index_to_physical(totals[1:]/totals[0]+(np.asarray(image.grid.shape_xyz)-1)*.5)


def register_linear32(fixed,moving,stages=None,initialization_feature=1,seed=20261022,
                        fixed_mask=None,moving_mask=None,output_directory=None,precision_policy='single'):
    stages=(LinearStage('Rigid'),LinearStage('Affine')) if stages is None else tuple(stages)
    if not stages:raise ValueError("At least one linear stage is required")
    if any(s.metric!='MI' for s in stages):raise ValueError("FP32 linear fitting currently supports Mattes MI")
    if precision_policy!='single':raise ValueError('Registration uses ordinary FP32; software-expanded precision is retired')
    center,moving_center=initialization_point32(fixed,initialization_feature,precision_policy),initialization_point32(moving,initialization_feature,precision_policy)
    prepare=lambda image:image32(image,normalize=True)
    smooth=smooth_image32
    metric_type=MattesMetric32
    nf,nm=prepare(fixed),prepare(moving)
    current=AffineTransform.centered(np.eye(3),moving_center-center,center)
    history=[]
    out=None if output_directory is None else Path(output_directory)
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    for stage_index,stage in enumerate(stages):
        model=ComposedLinearModel(stage.kind,center,current)
        parameters=model.identity_parameters()
        for level,(iterations,shrink,sigma) in enumerate(zip(stage.iterations,stage.shrink_factors,stage.smoothing_sigmas)):
            started=time.perf_counter()
            # ANTs preprocesses every stage even when a level has zero
            # smoothing. Real-valued MI scale invariance does not preserve
            # float1 quantization and Parzen-bin decisions on raw intensities.
            fs,ms=smooth(nf,sigma,stage.smoothing_units),smooth(nm,sigma,stage.smoothing_units)
            grid=shrink_grid(fixed.grid,shrink)
            points=None
            if stage.sampling_strategy!='None':
                plan=sample_virtual_domain(grid,stage.sampling_strategy,stage.sampling_fraction,seed+stage_index*100+level)
                points=plan.points_lps
                if out is not None:plan.save(out/f'samples-stage{stage_index}-level{level}.npz')
            metric=metric_type(fs,ms,grid,stage.bins,points,fixed_mask,moving_mask)
            optimized=conjugate_gradient(metric,model,parameters,grid,iterations,stage.gradient_step,
                                         stage.convergence_threshold,stage.convergence_window)
            parameters=optimized.parameters;current=model.transform(parameters)
            final=metric.evaluate(model,parameters,False)
            history.append({'kind':stage.kind,'stage':stage_index,'level':level,'shrink_factor':shrink,
                            'sigma':sigma,'sigma_units':stage.smoothing_units,'sampling':stage.sampling_strategy,
                            'sample_count':int(metric.indices.shape[0]),'stop_reason':optimized.stop_reason,
                            'scales':optimized.scales.tolist(),'final_value_reevaluated':final.value,
                            'iterations':optimized.history,'precision':'ordinary FP32 dense samples/histograms/derivatives and float1 transform/CG state; float1 FP32 histogram finalization with native FP64 joint total',
                            'host_control_estimate':{
                                'histogram_bytes_per_derivative':stage.bins*stage.bins*(1+model.parameter_count)*4+stage.bins*8,
                                'histogram_bytes_per_value_only':stage.bins*stage.bins*4+stage.bins*8,
                                'evaluations':getattr(metric,'control_evaluations',0),
                                'derivative_evaluations':getattr(metric,'control_derivative_evaluations',0),
                                'total_histogram_bytes':getattr(metric,'control_bytes',0),
                                'scope':'hypothetical device-to-host boundary; this CPU run measures no GPU synchronization latency'},
                            'seconds_diagnostic_not_benchmark':time.perf_counter()-started})
    return RegistrationResult(fixed.grid,moving.grid,current,current.inverse(),pull_image32(image32(moving),fixed.grid,linear=current),history)
