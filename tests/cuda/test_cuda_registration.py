"""Opt-in complete CUDA registration and retained-output contracts."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fast_ants import AffineTransform, Grid, Image, LinearStage, SyNStage


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA')=='1',
                     'CUDA device tests require explicit opt-in')
class CUDARegistrationTests(unittest.TestCase):
    def pair(self):
        grid=Grid((17,19,21),np.diag([.8,1.3,2.,1.]))
        z,y,x=np.indices(grid.shape_zyx)
        def phantom(dx,dy,dz):
            return (np.exp(-((x-9-dx)**2/18+(y-8-dy)**2/12+(z-8-dz)**2/9))
                    +.7*np.exp(-((x-14-dx)**2/5+(y-13-dy)**2/8+(z-6-dz)**2/7))
                    +.4*np.exp(-((x-6-dx)**2/7+(y-12-dy)**2/5+(z-11-dz)**2/4))).astype(np.float32)
        return Image(phantom(0,0,0),grid),Image(phantom(.8,-.4,.3),grid)

    def test_rigid_affine_syn_known_translation_and_saved_outputs(self):
        from fast_ants import register_cuda, load_ants_transform_list, apply_image
        fixed,moving=self.pair()
        linear=[LinearStage(kind,iterations=(4,3),shrink_factors=(2,1),smoothing_sigmas=(1,0))
                for kind in ('Rigid','Affine')]
        stage=SyNStage(iterations=(2,2),shrink_factors=(2,1),smoothing_sigmas=(1,0))
        before=np.corrcoef(fixed.data.ravel(),moving.data.ravel())[0,1]
        with tempfile.TemporaryDirectory() as temporary:
            result=register_cuda(fixed,moving,linear_stages=linear,syn_stage=stage,output_directory=temporary)
            self.assertEqual([entry['kind'] for entry in result.history],['Rigid','Rigid','Affine','Affine','SyN','SyN'])
            self.assertTrue(np.isfinite(result.warped.data).all())
            self.assertEqual(result.warped.data.dtype,np.float32)
            self.assertEqual(result.quality['nonpositive_jacobian_points'],0)
            self.assertGreater(np.corrcoef(fixed.data.ravel(),result.warped.data.ravel())[0,1],before)
            landmarks=fixed.grid.index_to_physical([[9,8,8],[14,13,6],[6,12,11]])
            truth=landmarks+np.array([.8*.8,-.4*1.3,.3*2.])
            error=np.linalg.norm(result.fixed_to_moving.map_points(landmarks)-truth,axis=1)
            self.assertLess(float(error.max()),1.0)
            out=Path(temporary)
            for name in ('affine.mat','warp.nii.gz','inverse-warp.nii.gz','warped.nii.gz',
                         'history.json','quality.json','configuration.json'):
                self.assertGreater((out/name).stat().st_size,0)
            chain=load_ants_transform_list([out/'warp.nii.gz',out/'affine.mat'])
            reapplied=apply_image(moving,fixed,chain)
            np.testing.assert_allclose(reapplied.data,result.warped.data,rtol=2e-5,atol=3e-6)
            self.assertEqual(result.configuration['cpu_fallbacks'],[])

    def test_syn_matches_cpu_and_callback_fields_remain_owned(self):
        from fast_ants import register_cuda
        from fast_ants.backends.cpu.fp32.registration import register_nonlinear32
        from fast_ants.backends.cuda.runtime import cp
        fixed,moving=self.pair();retained=[]
        def observe(level,iteration,step):
            retained.append((step.state,{name:cp.asnumpy(field.data) for name,field in step.state.fields().items()}))
        stage=SyNStage(iterations=(2,),shrink_factors=(1,),smoothing_sigmas=(0,))
        linear=AffineTransform.identity()
        gpu=register_cuda(fixed,moving,initial_transform=linear,syn_stage=stage,callback=observe)
        cpu=register_nonlinear32(fixed,moving,linear,syn_stage=stage)
        np.testing.assert_allclose(gpu.forward_field.data,cpu.forward_field.data,rtol=4e-4,atol=3e-5)
        np.testing.assert_allclose(gpu.warped.data,cpu.warped.data,rtol=4e-4,atol=3e-5)
        self.assertEqual(len(retained),2)
        for state,snapshot in retained:
            for name,field in state.fields().items():
                np.testing.assert_array_equal(cp.asnumpy(field.data),snapshot[name])


if __name__=='__main__':
    unittest.main()
