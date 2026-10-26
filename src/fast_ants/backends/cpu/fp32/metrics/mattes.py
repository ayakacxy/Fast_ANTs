"""FP32 Mattes kernels with float1 mixed-precision histogram controls."""

import numpy as np
from fast_ants.backends.cpu.fp32.metrics import mattes_kernels as k
from fast_ants.backends.cpu.fp32.metrics.value import fixed_bins32
from fast_ants.backends.cpu.fp32.metrics.value import value_histogram32,ValueWorkspace32
from fast_ants.backends.cpu.fp32.metrics.derivative import DerivativeWorkspace32
from fast_ants.backends.cpu.fp32.metrics.derivative import derivative_histogram32
from fast_ants.core.results import MetricEvaluation
from fast_ants.core.results import NoValidSamplesError
from fast_ants.backends.cpu.fp32.images import array32
from fast_ants.backends.cpu.fp32.images import require32
from fast_ants.backends.cpu.fp32.images import image32
from fast_ants.core.transforms import AffineTransform
from fast_ants.pipeline.mattes32 import finalize_mattes_float1


def evaluate_mattes_samples32(fixed,moving,intensity_derivatives=None,bins=32,fixed_range=None,moving_range=None):
    fixed,moving=np.asarray(fixed),np.asarray(moving)
    require32(fixed);require32(moving)
    if fixed.ndim!=1 or fixed.shape!=moving.shape or not fixed.size:raise ValueError("Mattes needs matching nonempty 1D samples")
    if not isinstance(bins,int) or bins<5 or not np.isfinite(fixed).all() or not np.isfinite(moving).all():raise ValueError("Invalid MI samples/bins")
    fr=(fixed.min(),fixed.max()) if fixed_range is None else fixed_range
    mr=(moving.min(),moving.max()) if moving_range is None else moving_range
    if not np.isfinite([*fr,*mr]).all() or fr[1]<=fr[0] or mr[1]<=mr[0]:raise ValueError("MI ranges must be finite and nonconstant")
    derivatives=np.empty((fixed.size,0),np.float32) if intensity_derivatives is None else require32(np.asarray(intensity_derivatives))
    if derivatives.ndim!=2 or derivatives.shape[0]!=fixed.size or not np.isfinite(derivatives).all():raise ValueError("Invalid MI intensity derivatives")
    value,descent,joint,fixed_pdf,moving_pdf,pdf_derivative,valid=k.mattes32(
        array32(fixed),array32(moving),array32(derivatives),bins,*(np.float32(v) for v in (*fr,*mr)))
    if not valid:raise NoValidSamplesError("No valid FP32 MI samples")
    return MetricEvaluation(float(value),descent,int(valid),{'joint_pdf':joint,'fixed_pdf':fixed_pdf,
                             'moving_pdf':moving_pdf,'pdf_derivative':pdf_derivative})


class MattesMetric32:
    """Cache fixed sampling and moving support for unchanged images and grids."""
    def __init__(self,fixed,moving,virtual_grid=None,bins=32,points_lps=None,fixed_mask=None,moving_mask=None):
        require32(fixed.data);require32(moving.data)
        self.fixed,self.moving,self.bins=fixed,moving,bins
        self.control_evaluations=0;self.control_derivative_evaluations=0;self.control_bytes=0
        self.grid=fixed.grid if virtual_grid is None else virtual_grid
        self.moving_mask=moving_mask
        self.points_are_integer=points_lps is None
        # Saved sparse points are input data; the import conversion is explicit.
        self.indices=(k.index_points32(self.grid.shape_zyx) if points_lps is None else
                       array32(self.grid.physical_to_index(points_lps)))
        values,valid,_=self.sample(fixed,AffineTransform.identity())
        if fixed_mask is not None:valid&=self.mask(fixed_mask,AffineTransform.identity())
        self.indices=np.ascontiguousarray(self.indices[valid]);self.fixed_values=np.ascontiguousarray(values[valid])
        if not len(self.fixed_values):raise NoValidSamplesError("No fixed FP32 MI samples")
        self.fixed_range=self.image_range(fixed,fixed_mask)
        self.moving_range=self.image_range(moving,moving_mask)
        width=np.float32(np.float32(self.moving_range[1]-self.moving_range[0])/np.float32(bins-4))
        if width<=0:raise ValueError('MI ranges must be nonconstant')
        self.bin_width_control=width
        self.bin_normalized_min=np.float32(np.float32(self.moving_range[0]/width)-np.float32(2))
        if points_lps is not None:
            nearest,inside,_=self.sample(fixed,AffineTransform.identity(),nearest=True)
            self.fixed_range=(nearest[inside].min(),nearest[inside].max())
        self.fixed_bins=fixed_bins32(self.fixed_values,bins,
                                    np.float32(self.fixed_range[0]),np.float32(self.fixed_range[1]))
        self._derivative_workspace=None
        self._value_workspace=None
        self._zero_gradient_support=None
        self.derivative_workspace_bytes=0
        self._gradient_geometry=(array32(self.moving.grid.lps_to_index[:3,:3]@
            np.diag(self.moving.grid.spacing_xyz*.5)),array32(self.moving.grid.spacing_xyz))
        self._local_derivative_linear=array32(self.grid.index_to_lps[:3,:3])
        self._local_derivative_center=array32((np.asarray(self.grid.shape_xyz)-1)*.5)

    def mapping_parameters(self,image,transform):
        inverse=image.grid.lps_to_index[:3,:3]
        linear=inverse@transform.matrix[:3,:3]@self.grid.index_to_lps[:3,:3]
        center=np.floor((np.asarray(self.grid.shape_xyz)-1)*.5)
        offset=inverse@(transform.map_points(self.grid.index_to_physical(center))-image.grid.origin)
        integer_linear=np.rint(linear).astype(np.int32);integer_offset=np.rint(offset).astype(np.int32)
        return integer_linear,array32(linear-integer_linear),integer_offset,array32(offset-integer_offset),array32(center)

    def mapping(self,image,transform):
        function=k.local_affine_integer32 if self.points_are_integer else k.local_affine32
        return function(self.indices,*self.mapping_parameters(image,transform))

    def sample(self,image,transform,nearest=False):
        indices=self.mapping(image,transform)
        values,valid=k.local_samples32(require32(image.data),*indices,bool(nearest))
        return values,valid,indices

    def mask(self,mask,transform):
        if not np.all((mask.data==0)|(mask.data==1)):raise ValueError("Mask must be binary")
        values,valid,_=self.sample(image32(mask),transform,nearest=True)
        return valid&(values!=0)

    def image_range(self,image,mask):
        if mask is None:return image.data.min(),image.data.max()
        saved_grid,saved_indices=self.grid,self.indices
        try:
            self.grid=image.grid;self.indices=k.index_points32(image.grid.shape_zyx)
            valid=self.mask(mask,AffineTransform.identity())
        finally:self.grid,self.indices=saved_grid,saved_indices
        values=image.data.ravel()[valid]
        if not values.size:raise NoValidSamplesError("Mask has no image samples")
        return values.min(),values.max()

    def evaluate(self,model,parameters,derivative=True):
        transform=model.transform(parameters)
        if not derivative and self.moving_mask is None:
            workspace=self._value_workspace
            if workspace is None or (workspace.count,workspace.bins)!=(len(self.indices),self.bins):
                workspace=ValueWorkspace32(len(self.indices),self.bins)
                self._value_workspace=workspace
            hist,counts,count,geometric=value_histogram32(require32(self.moving.data),self.indices,
                self.fixed_bins,self.mapping_parameters(self.moving,transform),self.bins,
                self.moving_range,self.bin_width_control,self.bin_normalized_min,
                points_are_integer=self.points_are_integer,workspace=workspace)
            if not geometric:raise NoValidSamplesError("All FP32 samples map outside moving")
            return self._finish(hist,counts,np.empty((self.bins,self.bins,0),np.float32),count,False)
        if derivative and self.moving_mask is None:
            linear,translation=model.derivatives(parameters)
            parameters_count=linear.shape[0]
            if 1<=parameters_count<=12:
                workspace=self._derivative_workspace
                if (workspace is None or
                    (workspace.count,workspace.bins,workspace.parameters)!=(len(self.indices),self.bins,parameters_count)):
                    workspace=DerivativeWorkspace32(len(self.indices),self.bins,parameters_count)
                    self._derivative_workspace=workspace
                    self.derivative_workspace_bytes=sum(a.nbytes for a in vars(workspace).values()
                                                        if isinstance(a,np.ndarray))
                zero_support=self._zero_gradient_support
                if (zero_support is None and len(self.indices)>=1048576
                        and np.isfinite(self._gradient_geometry[0]).all()
                        and np.all(np.abs(self._gradient_geometry[0])<=1)):
                    from fast_ants.backends.cpu.fp32.metrics.background import zero_support32
                    zero_support=zero_support32(self.moving.data)
                    self._zero_gradient_support=zero_support
                hist,counts,pdf,count,geometric,bad=derivative_histogram32(
                    require32(self.moving.data),self.indices,self.fixed_bins,
                    self.mapping_parameters(self.moving,transform),self.bins,self.moving_range,
                    self.bin_width_control,self.bin_normalized_min,self._gradient_geometry,
                    (self._local_derivative_linear,array32(self.grid.center-model.center),
                     self._local_derivative_center),array32(linear),array32(translation),
                    points_are_integer=self.points_are_integer,workspace=workspace,zero_support=zero_support)
                if not geometric:raise NoValidSamplesError("All FP32 samples map outside moving")
                if bad:raise ValueError('Intensity derivatives must be finite')
                # Finalization copies both raw PDFs. Borrowed workspace arrays
                # never escape in the public MetricEvaluation result.
                result=self._finish(hist,counts,pdf,count,True)
                result.details.update(fused_derivative=True,
                                      cached_zero_gradient_support=zero_support is not None,
                                      derivative_workspace_bytes=self.derivative_workspace_bytes)
                return result
        moving,valid,indices=self.sample(self.moving,transform)
        if self.moving_mask is not None:valid&=self.mask(self.moving_mask,transform)
        if not valid.any():raise NoValidSamplesError("All FP32 samples map outside moving")
        intensity=None
        if derivative:
            inverse=self.moving.grid.lps_to_index[:3,:3]
            shift=inverse@np.diag(self.moving.grid.spacing_xyz*.5)
            gradient=k.local_gradient32(self.moving.data,np.ascontiguousarray(indices[0][valid]),np.ascontiguousarray(indices[1][valid]),array32(shift),array32(self.moving.grid.spacing_xyz))
            local=k.centered_affine_indices32(np.ascontiguousarray(self.indices[valid]),array32(self.grid.index_to_lps[:3,:3]),
                       array32(self.grid.center-model.center),array32((np.asarray(self.grid.shape_xyz)-1)*.5))
            linear,translation=model.derivatives(parameters)
            intensity=k.intensity_derivatives32(local,gradient,array32(linear),array32(translation))
        selected_fixed=np.ascontiguousarray(self.fixed_values[valid]);selected_moving=np.ascontiguousarray(moving[valid])
        bi,bf=k.sample_bin_fractions32(selected_moving,self.bin_width_control,self.bin_normalized_min,self.bins)
        if intensity is None:intensity=np.empty((selected_fixed.size,0),np.float32)
        hist,counts,derivatives,count,width=k.histograms32(selected_fixed,selected_moving,intensity,self.bins,
                           *(np.float32(v) for v in (*self.fixed_range,*self.moving_range)),bi,bf)
        return self._finish(hist,counts,derivatives,count,derivative)

    def _finish(self,hist,counts,derivatives,count,derivative):
        if not count:raise NoValidSamplesError('No ordinary FP32 MI samples after range checks')
        value,descent,joint,fp,mp,pdf=finalize_mattes_float1(hist,counts,derivatives,count,self.bin_width_control)
        transfer_bytes=int(hist.nbytes+counts.nbytes+derivatives.nbytes)
        self.control_evaluations+=1;self.control_derivative_evaluations+=int(derivative);self.control_bytes+=transfer_bytes
        return MetricEvaluation(float(value),descent,int(count),{'joint_pdf':joint,'fixed_pdf':fp,'moving_pdf':mp,
                     'pdf_derivative':pdf,'control_transfer_bytes':transfer_bytes,'software_expanded_precision':False})
