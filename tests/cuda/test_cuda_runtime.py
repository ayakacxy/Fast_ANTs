"""Opt-in CUDA runtime ownership and explicit transfer contracts."""

import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('FAST_ANTS_TEST_CUDA')=='1',
                     'CUDA device tests require explicit opt-in')
class CUDARuntimeTests(unittest.TestCase):
    def test_stream_transfers_and_owned_outputs(self):
        from fast_ants.backends.cuda.runtime import CUDAContext, cp
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            ctx=CUDAContext()
        source=np.arange(513,dtype=np.float32)
        values=ctx.upload(source,'test_input')
        first=ctx.call('double','out0[i]=data[i]*2.0f;',{'data':values},[values.shape],values.size)
        second=ctx.call('double','out0[i]=data[i]*2.0f;',{'data':first},[values.shape],values.size)
        ctx.finish(first,second)
        np.testing.assert_array_equal(ctx.download(first,'test_output'),source*2)
        np.testing.assert_array_equal(ctx.download(second,'test_output'),source*4)
        with self.assertRaisesRegex(ValueError,'declared bound'):
            ctx.control(first,'oversized',32)
        with self.assertRaises(ValueError):ctx.upload(source,dtype=np.float64)
        with self.assertRaisesRegex(ValueError,'contiguous'):
            ctx.call('copy','out0[i]=data[i];',{'data':values[::2]},[(257,)],257)
        self.assertIs(ctx.meta(3,4),ctx.meta(3,4))
        geometry=np.array([.25,1.,2.],np.float32)
        original=ctx.geometry(geometry)
        self.assertIs(ctx.geometry(geometry.copy()),original)
        geometry[0]=.5
        changed=ctx.geometry(geometry)
        self.assertIsNot(changed,original)
        np.testing.assert_array_equal(ctx.download(original,'test_geometry'),[.25,1.,2.])
        report=ctx.report()
        self.assertEqual(report['kernel_dispatches'],{'double':2})
        self.assertEqual(report['transfers']['device_to_host:test_output']['bytes'],source.nbytes*2)
        self.assertEqual(report['cpu_fallbacks'],[])
        self.assertEqual(report['device']['id'],values.device.id)


if __name__=='__main__':
    unittest.main()
