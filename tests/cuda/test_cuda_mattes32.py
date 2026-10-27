"""Opt-in CUDA Mattes contracts against independent CPU float1 controls."""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA')=='1',
                     'CUDA device tests require explicit opt-in')
class CUDAMattes32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.cuda.runtime import CUDAContext
        from fast_ants.backends.cuda.images import CUDAImageOps
        from fast_ants.backends.cuda.metrics.mattes import metric
        cls.context_type,cls.ops_type,cls.mattes=CUDAContext,CUDAImageOps,metric

    def setUp(self):
        self.ctx=self.context_type();self.ops=self.ops_type(self.ctx)

    def fixture(self):
        from fast_ants.core.types import Grid,Image
        angle=.21;matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
                               [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.8,1.3,2.1])
        matrix[:3,3]=[17,-33,9];grid=Grid((15,13,11),matrix)
        z,y,x=np.indices(grid.shape_zyx)
        fixed=.5+.17*np.sin(.6*x)+.11*np.cos(.5*y)+.03*z
        moving=.4+.15*np.sin(.6*x+.09)+.13*np.cos(.5*y-.12)+.025*z+.004*x*y
        mask=np.ones(grid.shape_zyx,np.float32);mask[1:3,2:4,3:5]=0
        return [Image(a.astype(np.float32),grid) for a in (fixed,moving,mask)]

    def test_float1_bins_at_rounded_boundaries(self):
        for low,high in ((0,1),(-3,8),(4096,4128)):
            params=self.mattes.float1_bin_parameters((low,high),32)
            edges=np.float32(low)+np.arange(30,dtype=np.float32)*params[0]
            values=np.concatenate((edges,np.nextafter(edges,np.float32(-np.inf)),
                                    np.nextafter(edges,np.float32(np.inf))))
            term=np.asarray(values/params[0]-params[1],np.float32)
            expected=np.clip(np.trunc(term),2,29).astype(np.int32)
            indices,fractions=self.mattes.float1_bin_fractions(self.ctx,self.ctx.upload(values),32,
                                                             self.ctx.upload(params))
            np.testing.assert_array_equal(self.ctx.download(indices,'test_output'),expected)
            np.testing.assert_array_equal(self.ctx.download(fractions,'test_output').view(np.uint32),
                (term-expected.astype(np.float32)).view(np.uint32))

    def test_fixed_state_mi_derivatives_masks_and_device_controls(self):
        from fast_ants.backends.cpu.fp32.metrics.mattes import MattesMetric32
        from fast_ants.pipeline.models32 import LinearModel32
        from fast_ants.core.results import NoValidSamplesError
        fixed,moving,mask=self.fixture()
        for kind,masked in (('Rigid',False),('Affine',True)):
            kwargs={'fixed_mask':mask,'moving_mask':mask} if masked else {}
            expected=MattesMetric32(fixed,moving,bins=16,**kwargs)
            actual=self.mattes.CUDAMattesMetric(self.ops,self.ops.image(fixed),self.ops.image(moving),bins=16,**kwargs)
            model=LinearModel32(kind,fixed.grid.center);parameters=model.identity_parameters()
            parameters[-3:]=[.09,-.07,.08]
            if kind=='Rigid':parameters[:3]=[.015,-.012,.008]
            else:parameters[:2]+=[.01,.005]
            for derivative in (False,True):
                with self.subTest(kind=kind,derivative=derivative):
                    cpu=expected.evaluate(model,parameters,derivative)
                    gpu=actual.evaluate(model,parameters,derivative)
                    self.assertEqual(gpu.valid_points,cpu.valid_points)
                    np.testing.assert_allclose(gpu.value,cpu.value,atol=5e-6,rtol=5e-6)
                    np.testing.assert_allclose(gpu.derivative,cpu.derivative,atol=5e-6,rtol=1e-4)
                    for key in ('joint_pdf','fixed_pdf','moving_pdf','pdf_derivative'):
                        np.testing.assert_allclose(gpu.details[key],cpu.details[key],atol=5e-6,rtol=1e-4)
            actual._device_control_mode=True
            result=actual.evaluate(model,parameters,True)
            self.assertTrue(result.details['pdf_details_omitted'])
            self.assertEqual(result.details['control_transfer_bytes'],(model.parameter_count+3)*4)
            batch=actual.evaluate_values_many(model,(parameters,parameters))
            np.testing.assert_allclose(batch,[result.value,result.value],atol=5e-6,rtol=5e-6)
            parameters[-3:]=[1e6,-1e6,1e6]
            with self.assertRaises(NoValidSamplesError):actual.evaluate(model,parameters)

    def test_device_finalization_matches_float1_host_oracle(self):
        from fast_ants.backends.cuda.metrics.mattes.finalize import finalize_device,finalize_registration,decode_control
        from fast_ants.pipeline.mattes32 import finalize_mattes_float1
        rng=np.random.default_rng(17);bins,parameters=16,12
        joint=rng.uniform(0,5,(bins,bins)).astype(np.float32)
        derivative=rng.normal(0,.3,(bins,bins,parameters)).astype(np.float32)
        counts=rng.integers(30,80,bins,dtype=np.uint32);width=np.float32(.03125)
        packed=np.concatenate((joint[:,:,None],derivative),axis=2)
        packet,pdfs=finalize_device(self.ctx,self.ctx.upload(packed),self.ctx.upload(counts,dtype=np.uint32),width)
        fused,_=finalize_registration(self.ctx,self.ctx.upload(packed),
            self.ctx.upload(counts,dtype=np.uint32),self.ctx.upload([width]))
        np.testing.assert_array_equal(self.ctx.download(fused,'test_output'),
                                      self.ctx.download(packet,'test_output'))
        value,descent,count=decode_control(self.ctx.download(packet,'test_output'))
        oracle=finalize_mattes_float1(joint,counts,derivative,int(counts.sum()),width)
        self.assertEqual(count,int(counts.sum()))
        np.testing.assert_allclose(value,oracle[0],atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(descent,oracle[1],atol=2e-6,rtol=2e-6)
        for gpu,cpu in zip(pdfs,oracle[2:]):
            np.testing.assert_array_equal(self.ctx.download(gpu,'test_output'),cpu)

    def test_fused_sampling_retains_fp32_boundaries_and_one_control_upload(self):
        from fast_ants.pipeline.models32 import LinearModel32
        from fast_ants.backends.cuda.metrics.mattes import sampling
        fixed,moving,_=self.fixture()
        actual=self.mattes.CUDAMattesMetric(self.ops,self.ops.image(fixed),self.ops.image(moving))
        model=LinearModel32('Affine',fixed.grid.center);parameters=model.identity_parameters()
        parameters[:2]+=[.01,.005];parameters[-3:]=[.09,-.07,.08]
        transform=model.transform(parameters);static=sampling.constants(actual,model)
        packet=sampling.controls(self.ctx,actual.mapping_parameters(actual.moving,transform),
                                 model.derivatives(parameters))
        fused=sampling.sample(self.ctx,actual.moving.data,actual.indices,packet,static,32,True)
        values,valid,indices=actual.sample(actual.moving,transform)
        bi,bf=self.mattes.float1_bin_fractions(self.ctx,values,32,actual.moving_bin_parameters)
        inverse=moving.grid.lps_to_index[:3,:3]
        grad=self.mattes.gradient(self.ctx,actual.moving.data,indices,
            inverse@np.diag(moving.grid.spacing_xyz*.5),moving.grid.spacing_xyz)
        for candidate,reference in zip(fused,(values,valid,bi,bf,grad)):
            np.testing.assert_array_equal(self.ctx.download(candidate,'test_output'),
                                          self.ctx.download(reference,'test_output'))
        reference=actual.evaluate(model,parameters)
        actual._device_control_mode=True
        candidate=actual.evaluate(model,parameters)
        np.testing.assert_allclose(candidate.value,reference.value,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(candidate.derivative,reference.derivative,rtol=2e-5,atol=2e-6)
        before=sum(entry['calls'] for key,entry in self.ctx.transfers.items() if key.startswith('host_to_device:'))
        actual.evaluate(model,parameters)
        after=sum(entry['calls'] for key,entry in self.ctx.transfers.items() if key.startswith('host_to_device:'))
        self.assertEqual(after-before,1)
        actual._optimized=False
        unfused=actual.evaluate(model,parameters)
        np.testing.assert_allclose(candidate.value,unfused.value,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(candidate.derivative,unfused.derivative,rtol=2e-5,atol=2e-6)

    def test_parallel_centered_moments_match_double_reference(self):
        from fast_ants.backends.cuda.moments import moments
        rng=np.random.default_rng(18);data=rng.normal(size=(17,19,23)).astype(np.float32)
        z,y,x=np.indices(data.shape)
        expected=[data.sum(dtype=np.float64)]
        expected += [(data*(axis-(size-1)*.5)).sum(dtype=np.float64)
                     for axis,size in zip((x,y,z),data.shape[::-1])]
        actual=self.ctx.download(moments(self.ctx,self.ctx.upload(data)),'test_output')
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=5e-5)

    def test_graph_control_replay_and_private_value_histogram(self):
        from fast_ants.core.types import Image
        from fast_ants.pipeline.models32 import LinearModel32
        fixed,moving,_=self.fixture()
        images=[]
        for image in (fixed,moving):
            data=image.data.copy();data[:,:6,:8]=0
            images.append(Image(data,image.grid))
        fixed,moving=images
        metric=self.mattes.CUDAMattesMetric(self.ops,self.ops.image(fixed),self.ops.image(moving))
        metric._device_control_mode=True
        model=LinearModel32('Affine',fixed.grid.center)
        first=model.identity_parameters();first[-3:]=[.09,-.07,.08]
        second=first.copy();second[-1]+=.11
        metric._graph_replay=False;metric._aggregate_histogram=False;metric._private_values=False
        reference={(i,derivative):metric.evaluate(model,trial,derivative)
                   for i,trial in enumerate((first,second)) for derivative in (False,True)}
        metric._graph_replay=True;metric._aggregate_histogram=True;metric._private_values=True
        retained=None
        for i in (0,1,0):
            for derivative in (False,True):
                actual=metric.evaluate(model,(first,second)[i],derivative)
                expected=reference[i,derivative]
                self.assertEqual(actual.valid_points,expected.valid_points)
                np.testing.assert_allclose(actual.value,expected.value,rtol=5e-6,atol=5e-6)
                np.testing.assert_allclose(actual.derivative,expected.derivative,rtol=1e-4,atol=5e-6)
                if retained is None and derivative:retained=(actual,actual.derivative.copy())
        for order in ((0,1),(1,0)):
            pair=metric.evaluate_values_many(model,tuple((first,second)[i] for i in order))
            np.testing.assert_allclose(pair,[reference[i,False].value for i in order],rtol=5e-6,atol=5e-6)
        np.testing.assert_array_equal(retained[0].derivative,retained[1])
        self.assertEqual(len(metric._graph_plans),3)


if __name__=='__main__':unittest.main()
