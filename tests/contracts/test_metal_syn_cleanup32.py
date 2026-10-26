"""CPU-only orchestration checks for Metal control-pass cleanup.

Compile the actual methods with mocked device boundaries, so importing this
test neither imports MLX nor initializes a GPU. Kernel numerics are covered
by the existing device tests; here the contract is dispatch count, diagnostic
ownership and callback selection rather than a second kernel implementation.
"""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT=Path(__file__).resolve().parents[2]


def source_function(relative,name,namespace,owner=None):
    path=ROOT/relative
    body=ast.parse(path.read_text()).body
    if owner is not None:
        body=next(node.body for node in body if isinstance(node,ast.ClassDef) and node.name==owner)
    function=next(node for node in body if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def field(data,grid):
    return SimpleNamespace(data=data,grid=grid)


def inversion_result(value,history,reason,final_residual):
    return SimpleNamespace(field=value,history=history,stop_reason=reason,final_residual=final_residual)


class MetalSyNCleanupTests(unittest.TestCase):
    def test_scale_dispatches_one_norm_and_one_source_factor_multiply(self):
        dense=SimpleNamespace(maximum_control=Mock(return_value=7.),
                              multiply=Mock(return_value='scaled-device-array'))
        namespace={'np':np,'dense':dense,'MetalField':field}
        scale=source_function('src/fast_ants/backends/metal/fields/operators.py','scale',namespace,'MetalFieldOps')
        grid=SimpleNamespace(spacing_xyz=np.array([.7,1.3,2.1]))
        value=field('unscaled-device-array',grid)
        ops=SimpleNamespace(context=object())
        actual,details=scale(ops,value,.1)
        factor=np.float32(float(np.float32(.1))/7.)
        dense.maximum_control.assert_called_once_with(ops.context,value.data,grid.spacing_xyz)
        dense.multiply.assert_called_once_with(ops.context,value.data,factor)
        self.assertEqual(actual.data,'scaled-device-array')
        self.assertEqual(details,{'maximum_spacing_scaled_norm':7.,'scale':float(factor),
                                  'effective_target':float(np.float32(.1))})
        for rate in (-1.,float('nan'),float('inf')):
            dense.maximum_control.reset_mock();dense.multiply.reset_mock()
            with self.assertRaises(ValueError):scale(ops,value,rate)
            dense.maximum_control.assert_not_called();dense.multiply.assert_not_called()

    def test_final_residual_skip_keeps_inverse_updates_and_history(self):
        grid=SimpleNamespace(spacing_xyz=np.ones(3),same_geometry=lambda other:True)
        value=field('forward-device-array',grid)
        for iterations in (0,2):
            results=[]
            for diagnostic in (True,False):
                dense=SimpleNamespace(
                    norms=Mock(return_value=('norm-device-array',.3,.5)),
                    inverse_step=Mock(side_effect=lambda ctx,current,*args:('updated',current)),
                    maximum_control=Mock())
                namespace={'np':np,'dense':dense,'MetalField':field,'InversionResult':inversion_result}
                invert=source_function('src/fast_ants/backends/metal/fields/operators.py','invert',namespace,'MetalFieldOps')
                ops=SimpleNamespace(context=SimpleNamespace(optimizations=frozenset()),
                                    zero=lambda shape:field('zero-device-array',shape),
                                    compose=Mock(side_effect=lambda outer,current:field(('residual',current.data),grid)))
                kwargs={} if diagnostic else {'_evaluate_final_residual':False}
                result=invert(ops,value,max_iterations=iterations,mean_tolerance=.01,max_tolerance=.02,**kwargs)
                self.assertEqual(dense.norms.call_count,iterations+int(diagnostic))
                self.assertEqual(ops.compose.call_count,iterations+int(diagnostic))
                self.assertEqual(dense.inverse_step.call_count,iterations)
                dense.maximum_control.assert_not_called()
                if diagnostic:
                    self.assertEqual(result.final_residual,{'mean_spacing_scaled_norm':.3,'max_spacing_scaled_norm':.5})
                else:self.assertEqual(result.final_residual,{'evaluated':False})
                results.append(result)
            self.assertEqual(results[0].field.data,results[1].field.data)
            self.assertEqual(results[0].history,results[1].history)
            self.assertEqual(results[0].stop_reason,results[1].stop_reason)

    def test_registration_preserves_callback_diagnostics_and_skips_internal_ones(self):
        class Monitor:
            def __init__(self,window):pass
            def add(self,value):return 1.
        namespace={'np':np,'WindowConvergence':Monitor}
        run=source_function('src/fast_ants/backends/metal/syn.py','_syn_level',namespace)
        stage=SimpleNamespace(radius=2,gradient_step=.1,update_variance=3.,total_variance=0.,
                              convergence_window=10,convergence_threshold=1e-6)
        for callback in (None,Mock()):
            context=SimpleNamespace(serial_cc_guard=False,optimizations=frozenset(),control_guards=Counter())
            step=SimpleNamespace(state='next-state',value=.5,details={})
            ops=SimpleNamespace(context=context,syn_step=Mock(return_value=step))
            state,records,reason,discarded=run(ops,'fixed','moving','initial',stage,'linear',None,None,1,0,callback)
            self.assertEqual(state,'next-state');self.assertEqual(len(records),1)
            self.assertEqual(reason,'max_iterations');self.assertEqual(discarded,0)
            self.assertIs(ops.syn_step.call_args.kwargs['_evaluate_final_residual'],callback is not None)
            if callback is not None:callback.assert_called_once_with(0,1,step)


if __name__=='__main__':unittest.main()
