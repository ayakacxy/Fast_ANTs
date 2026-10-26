import unittest
import numpy as np

from fast_ants import AffineTransform, DisplacementField, Grid, Image, LinearStage, SyNStage
from fast_ants.backends.cpu.fp64.metrics.cc import neighborhood_cc
from fast_ants.backends.cpu.fp64.fields import compose_fields, invert_field
from fast_ants.backends.cpu.fp64.filtering import smooth_field, smooth_image
from fast_ants.backends.cpu.fp32.images import image32, zero32, pull_image32
from fast_ants.backends.cpu.fp32.metrics.cc import neighborhood_cc32
from fast_ants.backends.cpu.fp32.filtering import smooth_image32, smooth_field32
from fast_ants.backends.cpu.fp32.fields import compose_fields32, invert_field32, scale_update32
from fast_ants.backends.cpu.fp32.syn import identity_state32, syn_step32
from fast_ants.quality.metrics import field_difference
from fast_ants.core.state import SyNState
from fast_ants.backends.cpu.fp64.syn import syn_step
from fast_ants.backends.cpu.fp64.metrics.mattes import evaluate_mattes_samples
from fast_ants.backends.cpu.fp32.metrics.mattes import evaluate_mattes_samples32


class Precision32Tests(unittest.TestCase):
    def test_shared_inverse_scratch_preserves_returned_fields_and_initial_input(self):
        from fast_ants.backends.cpu.fp32.inverse import InverseWorkspace32
        fixed,_=self.fixture()
        data=np.zeros((3,*fixed.grid.shape_zyx),np.float32);data[0]=.02
        field=DisplacementField(data,fixed.grid)
        workspace=InverseWorkspace32(fixed.grid.shape_zyx)
        first=invert_field32(field,_workspace=workspace)
        saved=first.field.data.copy()
        second=invert_field32(first.field,field,_workspace=workspace)
        np.testing.assert_array_equal(first.field.data,saved)
        np.testing.assert_array_equal(field.data,data)
        self.assertFalse(np.shares_memory(first.field.data,second.field.data))
        for value in (first.field.data,second.field.data):
            self.assertFalse(np.shares_memory(value,workspace.residual))
        zero_iteration=invert_field32(field,first.field,max_iterations=0,_workspace=workspace)
        np.testing.assert_array_equal(zero_iteration.field.data,saved)
        self.assertFalse(np.shares_memory(zero_iteration.field.data,first.field.data))

    def test_cli_step_precision_and_normalization_threshold(self):
        effective=float(np.float32(.1))
        self.assertEqual(LinearStage('Rigid').gradient_step,effective)
        self.assertEqual(SyNStage().gradient_step,effective)
        self.assertGreater(effective,.1)
        f,_=self.fixture()
        rng=np.random.default_rng(97)
        force=DisplacementField(rng.normal(size=(3,*f.grid.shape_zyx)).astype(np.float32),f.grid)
        before=force.data.copy()
        scaled,details=scale_update32(force,.1)
        factor=np.float32(effective/details['maximum_spacing_scaled_norm'])
        self.assertEqual(details['scale'],float(factor))
        np.testing.assert_array_equal(scaled.data,before*factor)
        np.testing.assert_array_equal(force.data,before)
        result=invert_field32(scaled)
        reference=invert_field(scaled)
        self.assertIn(result.stop_reason,('mean_error_threshold','maximum_error_threshold','max_iterations'))
        self.assertTrue(np.isfinite(result.field.data).all())
        self.assertLess(field_difference(result.field,reference.field)['max_l2_mm'],1e-5)

    def test_mattes_compensated_fp32_and_large_dc_bins(self):
        rng=np.random.default_rng(104)
        # A 257-step range avoids internal exact bin boundaries; those are
        # checked separately by the accepted paired-precision metric.
        for offset,scale in ((0.,1.),(1e6,16.0625)):
            fixed=(offset+scale*rng.uniform(size=5000)).astype(np.float32)
            moving=(offset+scale*rng.uniform(size=5000)).astype(np.float32)
            d=rng.normal(size=(5000,4)).astype(np.float32)
            actual=evaluate_mattes_samples32(fixed,moving,d)
            # Compare the mathematical MI at the declared float1 bin
            # coordinates. Recentring raw values in double changes those bins
            # at large DC offsets and is a different precision policy.
            bins=32
            fw=np.float32((fixed.max()-fixed.min())/np.float32(bins-4))
            mw=np.float32((moving.max()-moving.min())/np.float32(bins-4))
            ft=fixed/fw-np.float32(fixed.min()/fw-np.float32(2))
            mt=moving/mw-np.float32(moving.min()/mw-np.float32(2))
            reference=evaluate_mattes_samples(ft.astype(np.float64),mt.astype(np.float64),
                d.astype(np.float64)/float(mw),bins,fixed_range=(2.,bins-2.),moving_range=(2.,bins-2.))
            self.assertLess(abs(actual.value-reference.value),4e-7)
            np.testing.assert_allclose(actual.details['joint_pdf'],reference.details['joint_pdf'],atol=5e-9,rtol=0)
            np.testing.assert_allclose(actual.derivative,reference.derivative,atol=3e-7,rtol=3e-4)
            self.assertEqual(actual.derivative.dtype,np.float32)

    def fixture(self,origin=None):
        a=.22
        matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@np.diag([.8,1.4,2.])
        matrix[:3,3]=[12,-21,8] if origin is None else origin
        grid=Grid((12,14,16),matrix)
        z,y,x=np.indices(grid.shape_zyx)
        f=np.exp(-((x-7.5)**2+(y-6.5)**2+(z-5.5)**2)/12.).astype(np.float32)
        m=np.exp(-((x-7.7)**2+(y-6.4)**2+(z-5.5)**2)/12.).astype(np.float32)
        return Image(f,grid),Image(m,grid)

    def test_fp32_smoothing_and_exact_dc(self):
        f,_=self.fixture()
        out=smooth_image32(f,1.)
        self.assertEqual(out.data.dtype,np.float32)
        np.testing.assert_allclose(out.data,smooth_image(f,1.).data,atol=8e-7,rtol=0)
        dc=Image(np.full(f.data.shape,1e6,np.float32),f.grid)
        np.testing.assert_array_equal(smooth_image32(dc,1.).data,dc.data)
        field=DisplacementField(np.stack([f.data,f.data*.2,f.data*.3]),f.grid)
        self.assertLess(field_difference(smooth_field32(field,3),smooth_field(field,3))['max_l2_mm'],1e-6)

    def test_local_coordinates_preserve_subvoxel_field_at_large_origins(self):
        f,m=self.fixture(origin=[1e8,-2e8,3e8])
        field=zero32(f.grid);field.data[0]=.0001
        result=compose_fields32(field,field)
        np.testing.assert_allclose(result.data[0,1:-1,1:-1,1:-1],.0002,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(pull_image32(f,f.grid).data,f.data)
        inverse=invert_field32(field,max_iterations=100,mean_tolerance=1e-10,max_tolerance=1e-10,enforce_boundary=False)
        np.testing.assert_allclose(inverse.field.data[0],-.0001,atol=2e-10,rtol=0)

    def test_centered_cc_and_single_syn_update_against_fp64(self):
        f,m=self.fixture()
        # Float1 and float0 deliberately use different degeneracy epsilons.
        # Check their common CC/update formulas away from those cutoffs.
        f,m=Image(f.data*np.float32(16),f.grid),Image(m.data*np.float32(16),m.grid)
        a,b=neighborhood_cc32(f,m),neighborhood_cc(f,m)
        self.assertLess(abs(a.value-b.value),2e-7)
        self.assertLess(field_difference(a.update,b.update)['max_l2_mm'],2e-6)
        actual=syn_step32(f,m,identity_state32(f.grid),gradient_step=.08)
        reference=syn_step(f,m,SyNState.identity(f.grid),gradient_step=.08)
        for name,field in actual.state.fields().items():
            self.assertEqual(field.data.dtype,np.float32)
            self.assertLess(field_difference(field,reference.state.fields()[name])['max_l2_mm'],1e-4)

    def test_fp64_arrays_cannot_silently_enter_dense_fp32_wrappers(self):
        f,m=self.fixture()
        with self.assertRaises(ValueError):neighborhood_cc32(Image(f.data.astype(np.float64),f.grid),m)
        constant=Image(np.full(f.data.shape,3,np.float32),f.grid)
        result=neighborhood_cc32(constant,constant)
        self.assertEqual(result.value,-1.)
        self.assertEqual(np.max(np.abs(result.update.data)),0.)
