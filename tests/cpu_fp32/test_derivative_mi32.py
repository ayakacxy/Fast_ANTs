"""Independent existing-kernel oracle for streaming FP32 Mattes derivatives."""
import unittest
import numpy as np
from numba import get_num_threads,set_num_threads

from fast_ants.backends.cpu.fp32.metrics import mattes_kernels as k
from fast_ants.backends.cpu.fp32.metrics.derivative import DerivativeWorkspace32, derivative_histogram32, _jacobian_plan32
from fast_ants.backends.cpu.fp32.metrics.value import fixed_bins32
from fast_ants.pipeline.mattes32 import finalize_mattes_float1


class StreamingDerivativeMI32Tests(unittest.TestCase):
    def test_dense_sparse_gradients_parameters_and_reusable_parallel_blocks(self):
        shape=(16,29,38);z,y,x=np.indices(shape,dtype=np.float32)
        rng=np.random.default_rng(866)
        center=(np.asarray(shape[::-1],np.float32)-np.float32(1))*np.float32(.5)
        floor_center=np.floor(center).astype(np.float32)
        integer=np.eye(3,dtype=np.int32)
        residual=np.array([[.013,.004,-.002],[-.006,-.009,.003],[.001,-.002,.007]],np.float32)
        mapping=(integer,residual,floor_center.astype(np.int32),
                 np.array([.13,-.21,.09],np.float32),floor_center)
        angle=.23
        direction=np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],[0,0,1]])
        spacing=np.array([.7,1.3,2.1],np.float32)
        shift=np.asarray(np.diag(1/spacing)@direction.T@np.diag(spacing*.5),np.float32)
        local_geometry=(np.asarray(direction@np.diag(spacing),np.float32),
                        np.array([.12,-.07,.11],np.float32),center)
        old_threads=get_num_threads()
        try:
            for parameters,dc,dense in ((6,0.,True),(12,4096.,False)):
                data=(dc+.5+.16*np.sin(.3*x)+.11*np.cos(.2*y)+.004*z).astype(np.float32)
                points=k.index_points32(shape)
                if not dense:
                    points=np.vstack((points,np.array([[-.5,.2,.3],[-.5001,.2,.3],
                        [37.5,.2,.3],[37.4999,.2,.3]],np.float32)))
                    points[::11]+=[.17,-.09,.13]
                fixed=rng.uniform(0,1,len(points)).astype(np.float32);bins=32
                fixed_bins=fixed_bins32(fixed,bins,np.float32(0),np.float32(1))
                low,high=np.float32(dc+.42),np.float32(dc+.70)
                width=np.float32((high-low)/np.float32(bins-4))
                normalized=np.float32(low/width-np.float32(2))
                linear=rng.normal(0,.1,(parameters,3,3)).astype(np.float32)
                translation=rng.normal(0,.1,(parameters,3)).astype(np.float32)
                if parameters==12:
                    # Interleave exact affine bases, unit translations and a
                    # generic derivative; parameter order is not assumed.
                    for p in range(parameters-1):
                        linear[p].fill(0);translation[p].fill(0)
                        if p in (1,6,9):translation[p,(p+1)%3]=np.float32(1)
                        else:linear[p,(2*p)%3,(p//3)%3]=np.float32(1)
                map_function=k.local_affine_integer32 if dense else k.local_affine32
                bases,offsets=map_function(points,*mapping)
                moving,valid=k.local_samples32(data,bases,offsets)
                gradient=k.local_gradient32(data,bases[valid],offsets[valid],shift,spacing)
                local=k.centered_affine_indices32(points[valid],*local_geometry)
                intensity=k.intensity_derivatives32(local,gradient,linear,translation)
                bi,bf=k.sample_bin_fractions32(moving[valid],width,normalized,bins)
                reference=k.histograms32(fixed[valid],moving[valid],intensity,bins,
                    np.float32(0),np.float32(1),low,high,bi,bf)
                finalized=finalize_mattes_float1(*reference[:3],reference[3],width)
                workspace=DerivativeWorkspace32(len(points),bins,parameters)
                identities=tuple(id(a) for a in (workspace.partial_joint,workspace.partial_derivative,
                    workspace.joint,workspace.derivative,workspace.counts))
                data.setflags(write=False);points.setflags(write=False)
                previous=None
                for workers in (1,2):
                    set_num_threads(workers)
                    result=derivative_histogram32(data,points,fixed_bins,mapping,bins,(low,high),
                        width,normalized,(shift,spacing),local_geometry,linear,translation,
                        points_are_integer=dense,workspace=workspace)
                    with self.subTest(parameters=parameters,workers=workers,dense=dense):
                        self.assertEqual(result[3:],(reference[3],int(valid.sum()),False))
                        self.assertIs(result[0],workspace.joint);self.assertIs(result[2],workspace.derivative)
                        np.testing.assert_array_equal(result[1],reference[1])
                        np.testing.assert_allclose(result[0],reference[0],atol=1e-4,rtol=2e-6)
                        np.testing.assert_allclose(result[2],reference[2],atol=3e-4,rtol=2e-5)
                        actual=finalize_mattes_float1(*result[:3],result[3],width)
                        self.assertLess(abs(float(actual[0])-float(finalized[0])),3e-6)
                        np.testing.assert_allclose(actual[1],finalized[1],atol=3e-6,rtol=2e-5)
                        self.assertEqual(result[0].dtype,np.float32)
                        self.assertEqual(result[2].dtype,np.float32)
                        if previous is not None:
                            for a,b in zip(result[:3],previous):np.testing.assert_array_equal(a,b)
                        previous=tuple(a.copy() for a in result[:3])
                self.assertEqual(identities,tuple(id(a) for a in (workspace.partial_joint,
                    workspace.partial_derivative,workspace.joint,workspace.derivative,workspace.counts)))
        finally:set_num_threads(old_threads)

    def test_zero_background_extreme_derivatives_range_rejection_and_empty_support(self):
        data=np.full((5,7,9),5,np.float32);points=k.index_points32(data.shape);bins=16;parameters=6
        fixed_bins=np.full(len(points),2,np.int32)
        mapping=(np.eye(3,dtype=np.int32),np.zeros((3,3),np.float32),
                 np.zeros(3,np.int32),np.zeros(3,np.float32),np.zeros(3,np.float32))
        gradient=(np.eye(3,dtype=np.float32)*np.float32(.5),np.ones(3,np.float32))
        local=(np.eye(3,dtype=np.float32),np.zeros(3,np.float32),np.zeros(3,np.float32))
        linear=np.ones((parameters,3,3),np.float32);translation=np.zeros((parameters,3),np.float32)
        workspace=DerivativeWorkspace32(len(points),bins,parameters)
        def run(value_range,local_geometry=local,jacobian=linear,map_parameters=mapping):
            low,high=np.asarray(value_range,np.float32)
            width=np.float32((high-low)/np.float32(bins-4));normalized=np.float32(low/width-np.float32(2))
            return derivative_histogram32(data,points,fixed_bins,map_parameters,bins,(low,high),width,
                normalized,gradient,local_geometry,jacobian,translation,points_are_integer=True,workspace=workspace)
        ordinary=run((4,6))
        self.assertEqual(ordinary[3:],(len(points),len(points),False))
        np.testing.assert_array_equal(ordinary[2],0)
        # Safe-zero must not suppress 0*inf for extreme but finite inputs.
        extreme_local=(local[0],np.full(3,1e30,np.float32),local[2])
        extreme_linear=np.full(linear.shape,1e30,np.float32)
        extreme=run((4,6),extreme_local,extreme_linear)
        self.assertEqual(extreme[3:],(len(points),len(points),True))
        # Existing FP32 semantics reject out-of-range pixels before PDF work.
        rejected=run((0,1),extreme_local,extreme_linear)
        self.assertEqual(rejected[3:],(0,len(points),False))
        np.testing.assert_array_equal(rejected[0],0);np.testing.assert_array_equal(rejected[2],0)
        outside=(*mapping[:2],np.full(3,1000,np.int32),*mapping[3:])
        empty=run((4,6),map_parameters=outside)
        self.assertEqual(empty[3:],(0,0,False))
        np.testing.assert_array_equal(empty[0],0);np.testing.assert_array_equal(empty[2],0)


    def test_parameter_plan_structure_and_unsafe_unit_translation_fallback(self):
        linear=np.zeros((5,3,3),np.float32)
        translation=np.zeros((5,3),np.float32)
        linear[0,2,1]=np.float32(1)
        translation[1,0]=np.float32(1)
        linear[2,1,0]=np.float32(2)  # Nonunit coefficients are generic.
        linear[3,0,2]=np.float32(1);translation[3,1]=np.float32(.1)
        linear[4,0,1]=np.float32(1);linear[4,2,0]=np.float32(1)
        np.testing.assert_array_equal(_jacobian_plan32(linear,translation),
            np.array([[1,2,1],[2,0,0],[0,0,0],[0,0,0],[0,0,0]],np.int32))
        shape=(3,4,5);z,y,x=np.indices(shape,dtype=np.float32)
        data=z.copy();points=k.index_points32(shape);bins=16
        fixed_bins=np.full(len(points),2,np.int32)
        mapping=(np.eye(3,dtype=np.int32),np.zeros((3,3),np.float32),
                 np.zeros(3,np.int32),np.zeros(3,np.float32),np.zeros(3,np.float32))
        shift=np.eye(3,dtype=np.float32)*np.float32(.5)
        local_linear=np.eye(3,dtype=np.float32)
        local_center=np.zeros(3,np.float32)
        # This derivative uses only x. An infinite unselected z gradient
        # must still expose the full expression's 0*inf, not return gradient_x.
        unit_linear=np.zeros((1,3,3),np.float32)
        unit_translation=np.array([[1,0,0]],np.float32)
        width=np.float32(3/(bins-4));normalized=np.float32(-2)
        def run(spacing,offset):
            return derivative_histogram32(data,points,fixed_bins,mapping,bins,
                (np.float32(0),np.float32(3)),width,normalized,(shift,spacing),
                (local_linear,offset,local_center),unit_linear,unit_translation,
                points_are_integer=True)
        safe=run(np.ones(3,np.float32),np.zeros(3,np.float32))
        self.assertFalse(safe[5])
        np.testing.assert_array_equal(safe[2],0)
        unselected_infinite=run(np.array([1,1,1e-39],np.float32),np.zeros(3,np.float32))
        self.assertTrue(unselected_infinite[5])
        # Even a unit translation must not ignore nonfinite local coordinates:
        # zero linear coefficients multiply them in the original expression.
        infinite_local=run(np.ones(3,np.float32),np.full(3,np.inf,np.float32))
        self.assertTrue(infinite_local[5])


if __name__=='__main__':unittest.main()
