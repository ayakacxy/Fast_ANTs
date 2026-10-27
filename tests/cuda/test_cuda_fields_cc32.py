"""Critical CUDA filtering, field and paired-CC checks against CPU FP32."""
import os
import unittest
import numpy as np

from fast_ants.core.types import Grid, Image
from fast_ants.core.fields import DisplacementField


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA')=='1','CUDA device tests require explicit opt-in')
class CUDAFieldsCC32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.cuda.images import CUDAImageOps
        from fast_ants.backends.cuda.fields.operators import CUDAFieldOps
        from fast_ants.backends.cuda.metrics.cc.operators import CUDACCOps
        from fast_ants.backends.cuda.syn import CUDASyNOps
        class TestOps(CUDAImageOps,CUDAFieldOps,CUDACCOps,CUDASyNOps):
            pass
        cls.ops_type=TestOps

    def setUp(self):
        self.ops=self.ops_type();self.ctx=self.ops.context
        angle=.21;geometry=np.eye(4)
        geometry[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.7,1.3,2.1])
        geometry[:3,3]=[32.125,-17.25,8.5]
        self.grid=Grid((7,9,11),geometry);self.rng=np.random.default_rng(74)

    def same(self,actual,expected):
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

    def test_gaussian_sections_recursive_and_field_boundaries(self):
        from fast_ants.backends.cuda.filtering import smooth_image
        from fast_ants.backends.cpu.fp32.filtering import smooth_image32, smooth_field32
        source=Image((4096+self.rng.normal(size=self.grid.shape_zyx)).astype(np.float32),self.grid)
        gpu=self.ops.image(source)
        for method in ('sections','recursive'):
            with self.subTest(method=method):
                expected=smooth_image32(source,[.7,1.1,1.6],method=method)
                actual=smooth_image(self.ctx,gpu,[.7,1.1,1.6],method=method)
                self.same(self.ops.download_image(actual).data,expected.data)
        field=DisplacementField(self.rng.normal(size=(3,*self.grid.shape_zyx)).astype(np.float32),self.grid)
        gpu_field=self.ops.field(field)
        for variance in (.1,3.):
            self.same(self.ops.download_field(self.ops.smooth_field(gpu_field,variance)).data,
                      smooth_field32(field,variance).data)

    def test_compose_scale_inverse_match_cpu_local_arithmetic(self):
        from fast_ants.backends.cpu.fp32.fields import compose_fields32, scale_update32, invert_field32
        values=self.rng.uniform(-.12,.12,size=(3,*self.grid.shape_zyx)).astype(np.float32)
        values[:,0]=values[:,-1]=0;values[:,:,0]=values[:,:,-1]=0;values[:,:,:,0]=values[:,:,:,-1]=0
        first=DisplacementField(values,self.grid)
        second=DisplacementField(values*np.float32(.3),self.grid)
        gf,gs=self.ops.field(first),self.ops.field(second)
        self.same(self.ops.download_field(self.ops.compose(gf,gs)).data,compose_fields32(first,second).data)
        actual,control=self.ops.scale(gf,.1);expected,reference=scale_update32(first,.1)
        self.same(self.ops.download_field(actual).data,expected.data)
        self.assertEqual(control,reference)
        kwargs={'max_iterations':4,'mean_tolerance':0.,'max_tolerance':0.}
        actual=self.ops.invert(gf,**kwargs);expected=invert_field32(first,**kwargs)
        self.same(self.ops.download_field(actual.field).data,expected.field.data)
        self.assertEqual(actual.history,expected.history)
        self.assertEqual(actual.final_residual,expected.final_residual)

    def test_inverse_row_reduction_pairwise_order_and_overflow(self):
        from fast_ants.backends.cpu.fp32.inverse import InverseWorkspace32,residual32_into
        from fast_ants.backends.cuda.reductions import row_norms
        shapes=((1,1,3),(1,3,7),(17,15,11),(16,16,13),(1,257,5),(256,256,3))
        for shape in shapes:
            with self.subTest(shape=shape):
                field=np.zeros((3,*shape),np.float32)
                field[0]=np.exp(self.rng.uniform(-30,30,size=shape)).astype(np.float32)
                workspace=InverseWorkspace32(shape)
                _,norms,mean,maximum=residual32_into(field,np.zeros_like(field),
                    np.eye(3,dtype=np.float32),np.ones(3,np.float32),workspace)
                actual=row_norms(self.ctx,self.ctx.upload(norms,'test_norms'))
                self.same(self.ctx.download(actual),np.asarray([mean,maximum],np.float32))
        # Finite field components can overflow the squared norm. The row
        # Kahan state must retain +inf, including an odd final row.
        field=np.zeros((3,1,257,3),np.float32);field[0,0,-1,-1]=np.finfo(np.float32).max
        workspace=InverseWorkspace32(field.shape[1:])
        _,norms,mean,maximum=residual32_into(field,np.zeros_like(field),
            np.eye(3,dtype=np.float32),np.ones(3,np.float32),workspace)
        actual=row_norms(self.ctx,self.ctx.upload(norms,'test_norms'))
        self.same(self.ctx.download(actual),np.asarray([mean,maximum],np.float32))

    def test_maximum_vector_preserves_earliest_ties_and_zero(self):
        from fast_ants.backends.cuda.reductions import maximum_vector
        for count in (1,257,65537):
            values=np.zeros((3,1,1,count),np.float32);values[:,0,0,0]=[-0.,0.,-0.]
            for tied in (False,True):
                if tied and count>1:
                    values[:,0,0,1]=[.75,1.,0.];values[:,0,0,-1]=[-.75,-1.,0.]
                expected=values[:,0,0,1 if tied and count>1 else 0]
                actual=maximum_vector(self.ctx,self.ctx.upload(values,'test_argmax'),np.ones(3))
                self.same(self.ctx.download(actual),expected)
        values[:,0,0,1]=[np.finfo(np.float32).max,0.,0.]
        values[:,0,0,-1]=[-np.finfo(np.float32).max,0.,0.]
        actual=maximum_vector(self.ctx,self.ctx.upload(values,'test_argmax'),np.ones(3))
        self.same(self.ctx.download(actual),values[:,0,0,1])

    def test_paired_inverse_preserves_guards_independent_stops_and_inputs(self):
        from fast_ants.backends.cuda.fields.inverse import invert_pair
        from fast_ants.backends.cuda.fields.kernels import residual_norms
        values=self.rng.uniform(-.12,.12,size=(3,*self.grid.shape_zyx)).astype(np.float32)
        values[:,0]=values[:,-1]=0;values[:,:,0]=values[:,:,-1]=0;values[:,:,:,0]=values[:,:,:,-1]=0
        fields=(self.ops.zero(self.grid),self.ops.field(DisplacementField(values,self.grid)))
        initials=(self.ops.zero(self.grid),self.ops.zero(self.grid))
        snapshots=[self.ctx.download(field.data) for field in (*fields,*initials)]
        _,_,mean,maximum=residual_norms(self.ctx,fields[1].data,initials[1].data,
            self.grid.lps_to_index[:3,:3],self.grid.spacing_xyz)
        for iterations,mean_limit,max_limit,evaluate in ((4,0.,0.,True),(4,mean,0.,False),
                (4,0.,maximum,True),(0,0.,0.,True)):
            options={'max_iterations':iterations,'mean_tolerance':mean_limit,'max_tolerance':max_limit,
                     '_evaluate_final_residual':evaluate}
            expected=invert_pair(self.ops,fields,initials,_reference=True,**options)
            actual=invert_pair(self.ops,fields,initials,**options)
            for got,wanted in zip(actual,expected):
                self.same(self.ctx.download(got.field.data),self.ctx.download(wanted.field.data))
                self.assertEqual(got.history,wanted.history)
                self.assertEqual(got.stop_reason,wanted.stop_reason)
                self.assertEqual(got.final_residual,wanted.final_residual)
            if iterations==4 and mean_limit==max_limit==0.:
                self.assertEqual([len(result.history) for result in actual],[1,4])
        for field,snapshot in zip((*fields,*initials),snapshots):
            self.same(self.ctx.download(field.data),snapshot)

    def test_paired_syn_step_matches_reference_and_retains_prior_state(self):
        fixed=self.ops.image(Image(self.rng.normal(size=self.grid.shape_zyx).astype(np.float32),self.grid))
        moving=self.ops.image(Image(self.rng.normal(size=self.grid.shape_zyx).astype(np.float32),self.grid))
        initial=self.ops.identity(self.grid);actual_state=expected_state=initial
        retained=[]
        for _ in range(2):
            expected=self.ops.syn_step(fixed,moving,expected_state,_reference_inversions=True,
                _evaluate_final_residual=False)
            actual=self.ops.syn_step(fixed,moving,actual_state,_evaluate_final_residual=False)
            self.assertEqual(actual.value,expected.value);self.assertEqual(actual.details,expected.details)
            for got,wanted in zip(actual.state.fields().values(),expected.state.fields().values()):
                self.same(self.ctx.download(got.data),self.ctx.download(wanted.data))
            for field,snapshot in retained:self.same(self.ctx.download(field.data),snapshot)
            retained=[(field,self.ctx.download(field.data)) for field in actual.state.fields().values()]
            actual_state,expected_state=actual.state,expected.state
        for field in initial.fields().values():self.same(self.ctx.download(field.data),np.zeros(field.data.shape,np.float32))

    def test_tiled_cc_masked_radius_four_and_deferred_score(self):
        from fast_ants.backends.cpu.fp32.images import gradient_image32
        from fast_ants.backends.cpu.fp32.metrics.cc_pair import cc_pair32
        from fast_ants.core.results import NoValidSamplesError
        fixed=Image((4096+self.rng.normal(size=self.grid.shape_zyx)).astype(np.float32),self.grid)
        moving=Image((-8192+self.rng.normal(size=self.grid.shape_zyx)).astype(np.float32),self.grid)
        valid=self.rng.uniform(size=self.grid.shape_zyx)>.2
        f,m=self.ops.image(fixed),self.ops.image(moving)
        device_valid=self.ctx.upload(valid,'test_mask',np.bool_)
        for radius in (0,2,4,5):
            with self.subTest(radius=radius):
                value,count,forward,reverse=cc_pair32(fixed.data,moving.data,
                    gradient_image32(fixed),gradient_image32(moving),valid,radius)
                actual=self.ops.cc_pair(f,m,radius,device_valid,_defer_control=radius==2)
                self.same(self.ops.download_field(actual[0].update).data,forward)
                self.same(self.ops.download_field(actual[1].update).data,reverse)
                self.assertEqual(actual[0].valid_points,count)
                self.assertEqual(actual[1].valid_points,count)
                self.assertAlmostEqual(actual[0].value,value,places=6)
                self.assertEqual(actual[0].value,actual[1].value)
        with self.assertRaises(NoValidSamplesError):
            self.ops.cc_pair(f,m,2,self.ctx.upload(np.zeros(self.grid.shape_zyx,np.bool_),'test_mask',np.bool_))


if __name__=='__main__':
    unittest.main()
