"""Fixed-state checks for fused CPU FP32 value-only Mattes evaluation."""
import unittest
import numpy as np
from numba import get_num_threads,set_num_threads

from fast_ants import Grid, Image
from fast_ants.backends.cpu.fp32.metrics import mattes_kernels as k
from fast_ants.backends.cpu.fp32.metrics.value import fixed_bins32, value_histogram32, ValueWorkspace32
from fast_ants.pipeline.models32 import LinearModel32
from fast_ants.pipeline.mattes32 import finalize_mattes_float1
from fast_ants.core.results import NoValidSamplesError
from fast_ants.backends.cpu.fp32.metrics.mattes import MattesMetric32


class FusedValueMI32Tests(unittest.TestCase):
    def test_workspace_reuse_keeps_counts_and_public_metric_results_owned(self):
        shape=(7,11,13)
        z,y,x=np.indices(shape,dtype=np.float32)
        data=np.asarray(.2*x+.1*y+.03*z,np.float32)
        points=k.index_points32(shape)
        bins=16
        fixed=fixed_bins32(data.ravel(),bins,data.min(),data.max())
        mapping=(np.eye(3,dtype=np.int32),np.zeros((3,3),np.float32),
                 np.zeros(3,np.int32),np.zeros(3,np.float32),np.zeros(3,np.float32))
        width=np.float32((data.max()-data.min())/np.float32(bins-4))
        normalized=np.float32(data.min()/width-np.float32(2))
        workspace=ValueWorkspace32(len(points),bins)
        args=(data,points,fixed,mapping,bins,(data.min(),data.max()),width,normalized)
        expected=value_histogram32(*args,points_are_integer=True)
        actual=value_histogram32(*args,points_are_integer=True,workspace=workspace)
        self.assertIs(actual[0],workspace.joint)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        first=actual[0].copy()
        workspace.partial.fill(np.nan);workspace.partial_correction.fill(np.nan)
        repeated=value_histogram32(*args,points_are_integer=True,workspace=workspace)
        np.testing.assert_array_equal(repeated[0],first)

    def test_raw_histograms_keep_sampling_float1_bins_and_valid_counts(self):
        rng=np.random.default_rng(318)
        shape=(17,29,37)
        points=k.index_points32(shape)
        points=np.vstack((points,np.array([[-.5,.2,.3],[-.5001,.2,.3],
                            [36.5,.2,.3],[36.4999,.2,.3]],np.float32)))
        mapping=(np.eye(3,dtype=np.int32),np.zeros((3,3),np.float32),
                 np.zeros(3,np.int32),np.zeros(3,np.float32),np.zeros(3,np.float32))
        threads=get_num_threads()
        try:
            for offset in (0.,4096.):
                data=(offset+rng.uniform(0,1,shape)).astype(np.float32)
                fixed=(offset+rng.uniform(0,1,points.shape[0])).astype(np.float32)
                # Include stored pixel/bin endpoints and range-rejected samples.
                low,high=np.float32(offset+.1),np.float32(offset+.9)
                for bins in (5,32):
                    width=np.float32((high-low)/np.float32(bins-4))
                    normalized=np.float32(low/width-np.float32(2))
                    fixed_bins=fixed_bins32(fixed,bins,np.float32(offset),np.float32(offset+1))
                    values,valid=k.local_samples32(data,*k.local_affine32(points,*mapping))
                    bi,bf=k.sample_bin_fractions32(values[valid],width,normalized,bins)
                    reference=k.histograms32(fixed[valid],values[valid],np.empty((int(valid.sum()),0),np.float32),
                        bins,np.float32(offset),np.float32(offset+1),low,high,bi,bf)
                    for workers in (1,2):
                        set_num_threads(workers)
                        hist,counts,count,geometric=value_histogram32(data,points,fixed_bins,
                            mapping,bins,(low,high),width,normalized)
                        with self.subTest(offset=offset,bins=bins,workers=workers):
                            self.assertEqual(geometric,int(valid.sum()))
                            self.assertEqual(count,reference[3])
                            np.testing.assert_array_equal(counts,reference[1])
                            np.testing.assert_allclose(hist,reference[0],rtol=8e-7,atol=2e-5)
                            actual=finalize_mattes_float1(hist,counts,np.empty((bins,bins,0),np.float32),count,width)
                            expected=finalize_mattes_float1(*reference[:3],reference[3],width)
                            self.assertLess(abs(float(actual[0])-float(expected[0])),3e-6)
        finally:set_num_threads(threads)

    def test_metric_matches_masked_old_path_and_preserves_derivatives(self):
        matrix=np.diag([.7,1.3,2.1,1.]);matrix[:3,3]=[17,-33,9]
        grid=Grid((9,11,13),matrix)
        z,y,x=np.indices(grid.shape_zyx)
        fixed=Image((np.sin(.3*x)+np.cos(.4*y)+.12*z).astype(np.float32),grid)
        moving=Image((np.sin(.3*x+.09)+np.cos(.4*y-.12)+.1*z+.02*x*y).astype(np.float32),grid)
        mask=Image(np.ones(grid.shape_zyx,np.float32),grid)
        model=LinearModel32('Affine',grid.center)
        parameters=model.identity_parameters();parameters[-3:]=[.13,-.07,.03]
        indices=np.stack((x,y,z),axis=-1)[2:-2,2:-2,2:-2].reshape(-1,3)
        for points in (None,grid.index_to_physical(indices)):
            fused=MattesMetric32(fixed,moving,points_lps=points)
            old=MattesMetric32(fixed,moving,points_lps=points,moving_mask=mask)
            self.assertEqual(fused.points_are_integer,points is None)
            transform=model.transform(parameters)
            mapping=fused.mapping_parameters(moving,transform)
            reference_coordinates=k.local_affine32(fused.indices,*mapping)
            for actual,expected in zip(fused.mapping(moving,transform),reference_coordinates):
                np.testing.assert_array_equal(actual,expected)
            if points is None:
                # A direct general-kernel oracle remains necessary: the
                # all-ones-mask path also uses integer mapping on dense plans.
                args=(moving.data,fused.indices,fused.fixed_bins,mapping,fused.bins,
                      fused.moving_range,fused.bin_width_control,fused.bin_normalized_min)
                general=value_histogram32(*args)
                specialized=value_histogram32(*args,points_are_integer=True)
                for actual,expected in zip(specialized,general):
                    np.testing.assert_array_equal(actual,expected)
            for derivative in (False,True):
                actual=fused.evaluate(model,parameters,derivative)
                expected=old.evaluate(model,parameters,derivative)
                self.assertEqual(actual.valid_points,expected.valid_points)
                self.assertLess(abs(actual.value-expected.value),3e-6)
                np.testing.assert_allclose(actual.derivative,expected.derivative,rtol=0,atol=1e-7)
                np.testing.assert_allclose(actual.details['joint_pdf'],expected.details['joint_pdf'],rtol=1e-6,atol=1e-7)
            self.assertEqual(fused.control_evaluations,2)
            self.assertEqual(fused.control_derivative_evaluations,1)
            value_result=fused.evaluate(model,parameters,False)
            retained_value=value_result.details['joint_pdf'].copy()
            value_workspace=fused._value_workspace
            changed=parameters.copy();changed[-3]+=np.float32(.01)
            fused.evaluate(model,changed,False)
            self.assertIs(fused._value_workspace,value_workspace)
            np.testing.assert_array_equal(value_result.details['joint_pdf'],retained_value)
            # Reusing raw derivative scratch must not overwrite an earlier
            # public result retained by an optimizer or caller.
            self.assertTrue(actual.details['fused_derivative'])
            workspace=fused._derivative_workspace
            retained=(actual.derivative.copy(),actual.details['joint_pdf'].copy(),
                      actual.details['pdf_derivative'].copy())
            shifted=parameters.copy();shifted[-3]+=np.float32(.01)
            fused.evaluate(model,shifted,True)
            self.assertIs(fused._derivative_workspace,workspace)
            for got,wanted in zip((actual.derivative,actual.details['joint_pdf'],
                                  actual.details['pdf_derivative']),retained):
                np.testing.assert_array_equal(got,wanted)
            outside=parameters.copy();outside[-3:]=1e4
            with self.assertRaises(NoValidSamplesError):fused.evaluate(model,outside,False)


if __name__=='__main__':
    unittest.main()
