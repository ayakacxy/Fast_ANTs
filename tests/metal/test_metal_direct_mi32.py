"""Fixed-state checks for original-index MI validity without compaction."""
import os
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalDirectMI32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.metrics.mattes import metric as mattes
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import DEFAULT_OPTIMIZATIONS, MetalContext, mx
        cls.mattes,cls.ops_type,cls.context_type,cls.mx=mattes,MetalOps,MetalContext,mx
        cls.baseline_options=tuple(x for x in DEFAULT_OPTIMIZATIONS
                                   if x not in ('direct_mi_validity','simd_mi_histograms',
                                                'streamed_mi_derivative','streamed_mi_fixed_plan',
                                                'packed_mi_controls'))

    def test_histogram_chunking_keeps_ranges_counts_and_derivatives(self):
        rng=np.random.default_rng(631)
        bins,n=16,20003
        fixed=rng.uniform(0,1,n).astype(np.float32)
        moving=rng.uniform(-.1,1.1,n).astype(np.float32)
        moving[:4]=[0.,1.,np.float32(-1e-5),
                    np.nextafter(np.float32(1),np.float32(np.inf))]
        valid=rng.random(n)>.27
        for parameters in (0,6):
            ctx=self.context_type(self.baseline_options)
            simd_ctx=self.context_type((*self.baseline_options,'simd_mi_histograms'))
            f,m=ctx.upload(fixed),ctx.upload(moving)
            intensity=ctx.upload(rng.normal(0,.2,(n,parameters)).astype(np.float32))
            bi,bf=self.mattes.float1_bin_fractions(ctx,m,bins,
                ctx.upload(self.mattes.float1_bin_parameters((0.,1.),bins)))
            cached_bins=self.mattes.fixed_bins(ctx,f,bins,(0.,1.))
            weights=self.mattes.cubic_tables(ctx,bf,parameters>0)
            for mask in (valid,np.zeros(n,bool)):
                with self.subTest(parameters=parameters,valid=int(mask.sum())):
                    device_mask=ctx.upload(mask,dtype=np.bool_)
                    selection,count=self.mattes.ordered_selection(ctx,device_mask)
                    args=(ctx,f,m,intensity,bins,(0.,1.),(0.,1.),bi,bf)
                    expected=self.mattes.histograms(*args,selection=selection,sample_count=count,
                                                    cached_bins=cached_bins,weights=weights)
                    candidates=[self.mattes.histograms(*args,valid=device_mask,cached_bins=cached_bins,weights=weights)]
                    simd_args=(simd_ctx,*args[1:])
                    candidates.append(self.mattes.histograms(*simd_args,valid=device_mask,cached_bins=cached_bins,weights=weights))
                    candidates.append(self.mattes.histograms(*simd_args,selection=selection,sample_count=count,
                                                            cached_bins=cached_bins,weights=weights))
                    for actual in candidates:
                        self.assertEqual(actual[3:],expected[3:])
                        self.assertEqual(actual[3],int(np.count_nonzero(mask&(moving>=0)&(moving<=1))))
                        np.testing.assert_array_equal(actual[1],expected[1])
                        for index in (0,2):
                            self.assertEqual(actual[index].dtype,np.float32)
                            self.assertTrue(np.isfinite(actual[index]).all())
                            np.testing.assert_allclose(actual[index],expected[index],atol=5e-5,rtol=5e-6)
            self.assertTrue(any('simd32' in name for name in simd_ctx.kernels))
            before=simd_ctx.kernels.copy()
            self.mattes.histograms(simd_ctx,f[:513],m[:513],intensity[:513],bins,(0.,1.),(0.,1.),
                bi[:513],bf[:513],valid=ctx.upload(valid[:513],dtype=np.bool_),
                cached_bins=cached_bins[:513],weights=weights[:513])
            self.assertFalse(any('simd32' in name for name in simd_ctx.kernels-before))

    def test_metric_masks_empty_domain_and_compaction_dispatches(self):
        from fast_ants.core.types import Grid, Image
        from fast_ants.pipeline.models32 import LinearModel32
        from fast_ants.core.results import NoValidSamplesError
        matrix=np.diag([.8,1.3,2.1,1.]);matrix[:3,3]=[17,-33,9]
        grid=Grid((17,29,37),matrix)
        z,y,x=np.indices(grid.shape_zyx)
        fixed=(.5+.17*np.sin(.6*x)+.11*np.cos(.5*y)+.03*z).astype(np.float32)
        moving=(.4+.15*np.sin(.6*x+.09)+.13*np.cos(.5*y-.12)+.025*z+.004*x*y).astype(np.float32)
        mask=np.ones(grid.shape_zyx,np.float32);mask[1:5,2:6,3:7]=0
        mask_matrix=matrix.copy();mask_matrix[:3,3]+=[.3,-.2,.1]
        mask_grid=Grid(grid.shape_zyx,mask_matrix)
        for masked in (False,True):
            metrics=[]
            for flags in (self.baseline_options,(*self.baseline_options,'direct_mi_validity','simd_mi_histograms')):
                ctx=self.context_type(flags);ops=self.ops_type(ctx)
                kwargs={'moving_mask':ops.image(Image(mask,mask_grid))} if masked else {}
                metric=self.mattes.MetalMattesMetric(ops,ops.image(Image(fixed,grid)),
                                                     ops.image(Image(moving,grid)),bins=16,**kwargs)
                metrics.append((ctx,metric))
            model=LinearModel32('Affine',grid.center)
            p=model.identity_parameters();p[1]=np.float32(.007);p[-3:]=[.3,-.21,.17]
            for derivative in (False,True):
                base_ctx,baseline=metrics[0];ctx,candidate=metrics[1]
                before=ctx.kernels.copy();base_before=base_ctx.kernels.copy()
                expected=baseline.evaluate(model,p,derivative)
                actual=candidate.evaluate(model,p,derivative)
                self.assertEqual(actual.valid_points,expected.valid_points)
                np.testing.assert_allclose(actual.value,expected.value,atol=5e-6,rtol=5e-6)
                np.testing.assert_allclose(actual.derivative,expected.derivative,atol=5e-6,rtol=1e-5)
                for key in ('joint_pdf','fixed_pdf','moving_pdf','pdf_derivative'):
                    np.testing.assert_allclose(actual.details[key],expected.details[key],atol=5e-6,rtol=1e-5)
                calls=ctx.kernels-before;reference_calls=base_ctx.kernels-base_before
                for name in ('compact_prefix_tiles','compact_prefix_block_offsets','mi_selection_scatter'):
                    self.assertEqual(calls[name],0)
                    self.assertEqual(reference_calls[name],1)
                self.assertTrue(any('direct_validity' in name for name in calls))
                self.assertTrue(any('simd32' in name for name in calls))
                self.assertTrue(all(not key.startswith('device_to_host:') or 'control_' in key
                                    for key in ctx.transfers))
            p[-3:]=[1e6,-1e6,1e6]
            for _,metric in metrics:
                for derivative in (False,True):
                    with self.assertRaises(NoValidSamplesError):metric.evaluate(model,p,derivative)


if __name__=='__main__':
    unittest.main()
