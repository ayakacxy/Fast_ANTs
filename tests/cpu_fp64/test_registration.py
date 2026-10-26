import tempfile
from pathlib import Path
import unittest
import numpy as np

from fast_ants import AffineTransform, Grid, Image, SyNStage, register, apply_image, load_ants_transform_list


class RegistrationTests(unittest.TestCase):
    def test_reused_syn_buffers_match_callback_path_and_preserve_retained_states(self):
        grid=Grid((9,11,13),np.diag([.8,1.3,2.,1.]))
        z,y,x=np.indices(grid.shape_zyx)
        fixed=Image(np.exp(-((x-6.)**2+(y-5.)**2+(z-4.)**2)/9.),grid)
        moving=Image(np.exp(-((x-6.3)**2+(y-4.8)**2+(z-4.)**2)/9.),grid)
        stage=SyNStage(iterations=(4,),shrink_factors=(1,),smoothing_sigmas=(0,),convergence_window=8)
        retained=[]

        def keep(level,iteration,result):
            retained.append((result.state,{name:field.data.copy() for name,field in result.state.fields().items()}))

        observed=register(fixed,moving,initial_transform=AffineTransform.identity(),syn_stage=stage,callback=keep)
        reused=register(fixed,moving,initial_transform=AffineTransform.identity(),syn_stage=stage)
        np.testing.assert_array_equal(observed.forward_field.data,reused.forward_field.data)
        np.testing.assert_array_equal(observed.inverse_field.data,reused.inverse_field.data)
        np.testing.assert_array_equal(observed.warped.data,reused.warped.data)
        self.assertEqual(observed.history[0]['iterations'],reused.history[0]['iterations'])
        for state,snapshot in retained:
            for name,field in state.fields().items():
                np.testing.assert_array_equal(field.data,snapshot[name])
        for older,newer in zip(retained,retained[1:]):
            for name in older[0].fields():
                self.assertFalse(np.shares_memory(older[0].fields()[name].data,newer[0].fields()[name].data))

    def test_multilevel_identity_and_output_chain_reapplication(self):
        grid=Grid((12,14,16),np.diag([.8,1.3,2.,1.]))
        z,y,x=np.indices(grid.shape_zyx)
        fixed=Image(np.exp(-((x-7.5)**2+(y-6.5)**2+(z-5.5)**2)/12.),grid)
        stage=SyNStage(iterations=(2,2),shrink_factors=(2,1),smoothing_sigmas=(1,0))
        with tempfile.TemporaryDirectory() as temporary:
            result=register(fixed,fixed,initial_transform=AffineTransform.identity(),syn_stage=stage,output_directory=temporary)
            self.assertEqual(len(result.history),2)
            np.testing.assert_allclose(result.forward_field.data,0,atol=1e-14)
            paths=[Path(temporary)/'warp.nii.gz',Path(temporary)/'affine.mat']
            reapplied=apply_image(fixed,fixed,load_ants_transform_list(paths))
            np.testing.assert_array_equal(reapplied.data,result.warped.data)
            self.assertEqual(result.quality['nonpositive_jacobian_points'],0)

    def test_window_stopping_and_bad_configuration(self):
        grid=Grid((8,8,8),np.eye(4))
        image=Image(np.ones(grid.shape_zyx),grid)
        stage=SyNStage(iterations=(12,),shrink_factors=(1,),smoothing_sigmas=(0,),convergence_window=8,convergence_threshold=1e-5)
        result=register(image,image,initial_transform=AffineTransform.identity(),syn_stage=stage)
        self.assertEqual(result.history[0]['stop_reason'],'window_convergence')
        self.assertEqual(len(result.history[0]['iterations']),8)
        with self.assertRaises(ValueError):
            SyNStage(iterations=(2,),shrink_factors=(1,2))
