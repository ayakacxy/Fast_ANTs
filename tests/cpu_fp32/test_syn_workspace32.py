"""Ownership and execution equivalence of private sequential FP32 SyN scratch."""
import unittest
import numpy as np

from fast_ants import DisplacementField, Grid, Image
from fast_ants.backends.cpu.fp32.syn import _SyNWorkspace32, identity_state32, syn_step32
from fast_ants.backends.cpu.fp32.fields import compose_fields32, invert_field32, scale_update32
from fast_ants.backends.cpu.fp32.filtering import smooth_field32
from fast_ants.backends.cpu.fp32.inverse import InverseWorkspace32


class SyNWorkspace32Tests(unittest.TestCase):
    def fixture(self):
        angle=.19
        matrix=np.eye(4)
        matrix[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])@np.diag([.8,1.3,1.9])
        matrix[:3,3]=[13.,-21.,7.]
        grid=Grid((7,9,11),matrix)
        z,y,x=np.indices(grid.shape_zyx,dtype=np.float32)
        fixed=(16*np.exp(-((x-5)**2+(y-4)**2+(z-3)**2)/10)).astype(np.float32)
        moving=(16*np.exp(-((x-5.2)**2+(y-3.9)**2+(z-3)**2)/10)).astype(np.float32)
        fixed.setflags(write=False);moving.setflags(write=False)
        return Image(fixed,grid),Image(moving,grid)

    def test_private_operator_outputs_and_inverse_zero_iteration_ownership(self):
        fixed,_=self.fixture();grid=fixed.grid
        rng=np.random.default_rng(918)
        values=rng.normal(0,.01,(3,*grid.shape_zyx)).astype(np.float32)
        field=DisplacementField(values,grid);before=values.copy()
        buffers=[np.empty_like(values),np.empty_like(values)]
        for variance in (0.,.2,3.):
            expected=smooth_field32(field,variance)
            actual=smooth_field32(field,variance,_buffers=buffers)
            self.assertIs(actual.data,buffers[0])
            np.testing.assert_array_equal(actual.data,expected.data)
            np.testing.assert_array_equal(field.data,before)
        scaled,details=scale_update32(field,.1)
        inplace=DisplacementField(values.copy(),grid)
        direct,direct_details=scale_update32(inplace,.1,_in_place=True)
        self.assertIs(direct.data,inplace.data)
        self.assertEqual(details,direct_details)
        np.testing.assert_array_equal(direct.data,scaled.data)
        out=np.empty_like(values)
        composed=compose_fields32(field,field,_output=out)
        self.assertIs(composed.data,out)
        np.testing.assert_array_equal(composed.data,compose_fields32(field,field).data)
        with self.assertRaises(ValueError):compose_fields32(field,field,_output=field.data)
        scratch=InverseWorkspace32(grid.shape_zyx)
        inverse=invert_field32(field,field,max_iterations=0,_workspace=scratch,_output=out)
        self.assertIs(inverse.field.data,out)
        np.testing.assert_array_equal(out,before)
        np.testing.assert_array_equal(field.data,before)
        with self.assertRaises(ValueError):
            invert_field32(field,field,max_iterations=0,_workspace=scratch,_output=field.data)

    def test_sequential_workspace_matches_owned_results_and_preserves_current_state(self):
        fixed,moving=self.fixture();workspace=_SyNWorkspace32(fixed.grid)
        owned=identity_state32(fixed.grid);borrowed=identity_state32(fixed.grid)
        scratch_ids=[id(value) for batch in workspace.outputs for value in batch]+[id(v) for v in workspace.smoothing]
        for step in range(3):
            current={name:field.data.copy() for name,field in borrowed.fields().items()}
            expected=syn_step32(fixed,moving,owned,gradient_step=.08)
            actual=syn_step32(fixed,moving,borrowed,gradient_step=.08,_workspace=workspace)
            self.assertEqual(actual.value,expected.value)
            for name in actual.details:
                if name!='inversions':self.assertEqual(actual.details[name],expected.details[name])
            for name,details in actual.details['inversions'].items():
                self.assertEqual(details['history'],expected.details['inversions'][name]['history'])
                self.assertEqual(details['stop_reason'],expected.details['inversions'][name]['stop_reason'])
                self.assertEqual(details['final_residual'],{'evaluated':False})
                self.assertIn('mean_spacing_scaled_norm',expected.details['inversions'][name]['final_residual'])
            for name,field in actual.state.fields().items():
                np.testing.assert_array_equal(field.data,expected.state.fields()[name].data)
                np.testing.assert_array_equal(borrowed.fields()[name].data,current[name])
                for old in borrowed.fields().values():
                    self.assertFalse(np.shares_memory(field.data,old.data))
            owned,borrowed=expected.state,actual.state
        self.assertEqual(scratch_ids,[id(value) for batch in workspace.outputs for value in batch]+[id(v) for v in workspace.smoothing])

    def test_capture_ignores_borrowed_output_path_and_remains_independent(self):
        fixed,moving=self.fixture();workspace=_SyNWorkspace32(fixed.grid)
        initial=identity_state32(fixed.grid)
        captured=syn_step32(fixed,moving,initial,gradient_step=.08,capture=True,_workspace=workspace)
        saved={name:value.data.copy() for name,value in captured.intermediates.items()}
        state=initial
        for _ in range(3):state=syn_step32(fixed,moving,state,gradient_step=.08,_workspace=workspace).state
        for name,value in captured.intermediates.items():
            np.testing.assert_array_equal(value.data,saved[name])
        for field in captured.state.fields().values():
            self.assertFalse(any(np.shares_memory(field.data,value) for batch in workspace.outputs for value in batch))


if __name__=='__main__':unittest.main()
