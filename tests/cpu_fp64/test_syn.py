import unittest
import numpy as np

from fast_ants import Grid, Image
from fast_ants.core.state import SyNState
from fast_ants.backends.cpu.fp64.syn import syn_step, run_syn_level


class SyNTests(unittest.TestCase):
    def fixture(self):
        grid=Grid((9,10,11),np.diag([1.,1.2,1.7,1.]))
        z,y,x=np.indices(grid.shape_zyx)
        f=np.exp(-((x-5)**2+(y-4.5)**2+(z-4)**2)/9.)
        m=np.exp(-((x-5.2)**2+(y-4.4)**2+(z-4)**2)/9.)
        return Image(f,grid),Image(m,grid)

    def test_identity_and_constant_images_keep_zero_state(self):
        f,m=self.fixture()
        for image in (f,Image(np.ones(f.data.shape),f.grid)):
            result=syn_step(image,image,SyNState.identity(f.grid))
            self.assertAlmostEqual(result.value,-1.,places=12)
            for field in result.state.fields().values():
                np.testing.assert_allclose(field.data,0,atol=1e-14)
            self.assertEqual(len(result.details['inversions']),4)

    def test_resume_fixed_iterations_and_input_state_not_mutated(self):
        f,m=self.fixture()
        start=SyNState.identity(f.grid)
        one=syn_step(f,m,start,capture=True)
        continued,_=run_syn_level(f,m,one.state,2)
        direct,history=run_syn_level(f,m,start,3)
        self.assertEqual(len(history),3)
        for name,field in start.fields().items():
            self.assertEqual(np.max(np.abs(field.data)),0)
            np.testing.assert_array_equal(continued.fields()[name].data,direct.fields()[name].data)
        self.assertIn('fixed_raw_update',one.intermediates)
        forward,inverse=direct.combined()
        self.assertIs(forward.inverse(),inverse)
