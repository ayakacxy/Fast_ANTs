"""Paired maxima preserve scale controls and scale each vertex before lerps."""

import os
import unittest

import numpy as np

from fast_ants.core.types import Grid
from fast_ants.core.fields import DisplacementField


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_METAL')=='1','Explicit GPU test opt-in')
class MetalScaledCompose32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fast_ants.backends.metal.ops import MetalOps
        from fast_ants.backends.metal.runtime import MetalContext
        from fast_ants.backends.metal.fields.composition import scaled_compose_pair
        cls.ops_type,cls.context_type=MetalOps,MetalContext
        cls.pair=staticmethod(scaled_compose_pair)

    def compare(self,ops,fields,inners,rate):
        before=ops.context.kernels.copy()
        actual=self.pair(ops,fields,inners,rate)
        delta=ops.context.kernels-before
        if 'parallel_max' in ops.context.optimizations:
            self.assertEqual(delta['multiply'],0)
            self.assertEqual(delta['compose'],0)
            self.assertEqual(delta['scaled_compose32'],2)
        expected=[]
        for field,inner in zip(fields,inners):
            scaled,details=ops.scale(field,rate)
            expected.append((ops.compose(scaled,inner),details))
        self.assertEqual(actual[2:],tuple(value[1] for value in expected))
        for value,(reference,_) in zip(actual[:2],expected):
            self.assertIs(value.grid,reference.grid)
            np.testing.assert_array_equal(ops.context.download(value.data),
                                          ops.context.download(reference.data))
        return actual

    def test_pair_one_read_and_scale_before_lerps_on_different_and_near_grids(self):
        angle=.23
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],
                           [np.sin(angle),np.cos(angle),0],[0,0,1]])
        matrix=np.eye(4);matrix[:3,:3]=rotation@np.diag([.7,1.3,2.1])
        matrix[:3,3]=[1e7,-2e7,3e7]
        outer_grid=Grid((5,7,9),matrix)
        changed=matrix.copy();changed[:3,:3]=rotation@np.diag([.66,1.28,2.1])
        changed[:3,3]+=matrix[:3,:3]@[-.49,.23,-.7]
        inner_grid=Grid((4,6,8),changed)
        near_matrix=matrix.copy();near_matrix[:3,3]+=[2e-7,-3e-7,1e-7]
        near_grid=Grid(outer_grid.shape_zyx,near_matrix)
        self.assertTrue(outer_grid.same_geometry(near_grid))
        rng=np.random.default_rng(507)
        update=rng.normal(0,.2,(3,*outer_grid.shape_zyx)).astype(np.float32)
        # Nonuniform magnitudes exercise multiply-rounding before differences.
        update[...,::2]*=np.float32(2**18)
        update[...,1::2]*=np.float32(2**-17)
        old=rng.uniform(-1.5,1.5,(3,*inner_grid.shape_zyx)).astype(np.float32)
        near_old=rng.uniform(-.3,.3,update.shape).astype(np.float32)
        ops=self.ops_type(self.context_type(('parallel_max',)))
        fields=tuple(ops.field(DisplacementField(value,grid)) for value,grid in
                     ((update,outer_grid),(-update,near_grid)))
        inners=tuple(ops.field(DisplacementField(value,grid)) for value,grid in
                     ((old,inner_grid),(near_old,outer_grid)))
        self.compare(ops,fields,inners,.10000000001)
        transfer=ops.context.transfers['device_to_host:control_scaled_compose_maxima']
        self.assertEqual((transfer['calls'],transfer['bytes']),(1,24))
        for field,original in zip(fields,(update,-update)):
            np.testing.assert_array_equal(ops.context.download(field.data),original)
        for inner,original in zip(inners,(old,near_old)):
            np.testing.assert_array_equal(ops.context.download(inner.data),original)

    def test_zero_update_zero_rate_earliest_argmax_and_serial_fallback(self):
        grid=Grid((2,3,4),np.eye(4))
        tied=np.zeros((3,*grid.shape_zyx),np.float32)
        flat=tied.reshape(3,-1)
        # Both squared norms round to 1 in FP32. Their FP64 norms differ;
        # maximum selection must keep the first voxel, as the old reduction.
        flat[0,:2]=1;flat[1,:2]=[1e-4,2e-4]
        old=np.full_like(tied,np.float32(.125))
        for flags in (('parallel_max',),()):
            ops=self.ops_type(self.context_type(flags))
            fields=tuple(ops.field(DisplacementField(value,grid)) for value in
                         (np.zeros_like(tied),tied))
            inners=tuple(ops.field(DisplacementField(old.copy(),grid)) for _ in range(2))
            for rate in (.1,0.):
                with self.subTest(flags=flags,rate=rate):
                    result=self.compare(ops,fields,inners,rate)
                    self.assertEqual(result[2]['maximum_spacing_scaled_norm'],0.)
                    self.assertEqual(result[2]['scale'],float(np.float32(rate)))
                    chosen=np.asarray([1.,np.float32(1e-4),0.],np.float64)
                    self.assertEqual(result[3]['maximum_spacing_scaled_norm'],
                                     float(np.sqrt(chosen@chosen)))
            if not flags:
                self.assertNotIn('device_to_host:control_scaled_compose_maxima',ops.context.transfers)


if __name__=='__main__':unittest.main()
