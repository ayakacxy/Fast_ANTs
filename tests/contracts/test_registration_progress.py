"""CPU mocks for registration logging; no GPU, registration, or JIT execution."""
import ast
from collections import Counter
from contextlib import nullcontext
import importlib.util
from io import StringIO
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import warnings


ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('fast_ants_progress_contract',ROOT/'src/fast_ants/progress.py')
PROGRESS=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROGRESS)


class FakeClock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


class FlushedStream(StringIO):
    def __init__(self):super().__init__();self.flushes=0
    def flush(self):self.flushes+=1;super().flush()


class TerminalStream(FlushedStream):
    def isatty(self):return True


class ProgressTee:
    def __init__(self):self.progress_terminal=TerminalStream();self.progress_log=FlushedStream()
    def write(self,text):
        self.progress_terminal.write(text);self.progress_log.write(text)
        return len(text)
    def flush(self):self.progress_terminal.flush();self.progress_log.flush()


class FakeBar:
    def __init__(self,**kwargs):
        self.options=kwargs;self.n=0;self.updates=[];self.postfixes=[];self.closed=0
        kwargs['file'].write('\rprogress')
    def set_postfix(self,postfix,refresh):
        self.postfixes.append(dict(postfix));self.refresh=refresh
    def update(self,amount):self.n+=amount;self.updates.append(amount)
    def close(self):self.closed+=1
    def write(self,message,*,file):print(message,file=file,flush=True)


def source_function(relative,name,namespace):
    path=ROOT/relative
    function=next(node for node in ast.parse(path.read_text()).body
                  if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class FakeVector:
    size=1
    shape=(1,)
    def copy(self):return FakeVector()
    def __le__(self,other):return False
    def tolist(self):return [0.]


class RegistrationProgressTests(unittest.TestCase):
    def test_format_throttle_real_intervals_final_iteration_reason_and_flush(self):
        clock=FakeClock();stream=FlushedStream()
        progress=PROGRESS.RegistrationProgress(3,stream=stream,clock=clock,wall_clock=lambda:0.)
        stage=progress.start('rigid','level=1 shrink=2 sigma=1 vox samples=17 budget=5',iterations=True)
        entries=[]
        for iteration in range(1,6):
            clock.now=iteration*.25
            entry={'iteration':iteration,'value':-.1*iteration,
                   'convergence':float('inf') if iteration==1 else 1e-5/iteration,'valid_points':17,
                   'learning_rate':.1,'gamma':0.,'line_search_evaluations':6}
            entries.append(entry.copy());stage(entry)
            self.assertEqual(entry,entries[-1])
        clock.now=2.
        stage.finish('maximum_iterations');stage.finish('must_not_duplicate')
        progress.complete()
        lines=stream.getvalue().splitlines()
        for line in lines:
            self.assertRegex(line,r'^\[fast_ants\]\[[a-z]+\] \d{2}:\d{2}:\d{2} \| INFO \| ')
        self.assertIn('DIAGNOSTIC,Iteration,metricValue,convergenceValue,ITERATION_TIME_INDEX,SINCE_LAST,validPoints',stream.getvalue())
        rows=[line.split('DIAGNOSTIC,',1)[1].split(',') for line in lines
              if 'DIAGNOSTIC,' in line and 'DIAGNOSTIC,Iteration,' not in line]
        self.assertEqual([int(row[0]) for row in rows],[1,3,5])
        self.assertEqual([float(row[3]) for row in rows],[.25,.75,1.25])
        self.assertEqual([float(row[4]) for row in rows],[.25,.25,.25])
        self.assertEqual(rows[0][2],'inf')
        self.assertEqual(rows[-1][5],'17')
        self.assertEqual(rows[-1][6:],[f'{.1:.12e}',f'{0.:.12e}','6'])
        self.assertIn('END reason=maximum_iterations; iterations=5; elapsed=2.000000s',stream.getvalue())
        self.assertNotIn('must_not_duplicate',stream.getvalue())
        self.assertIn('[fast_ants][complete]',stream.getvalue())
        self.assertNotIn('\x1b',stream.getvalue())
        self.assertEqual(stream.flushes,len(lines))

    def test_default_every_iteration_silence_validation_and_no_global_logging_changes(self):
        root=logging.getLogger();handlers=list(root.handlers);level=root.level
        warning_formatter=warnings.formatwarning;warning_hook=warnings.showwarning
        stream=StringIO();clock=FakeClock()
        self.assertIsNone(PROGRESS.create_progress(False,1,stream=stream,clock=clock))
        self.assertEqual(stream.getvalue(),'')
        for bad in (0,-1,True,1.5,'2',None):
            with self.subTest(interval=bad),self.assertRaisesRegex(ValueError,'positive integer'):
                PROGRESS.create_progress(False,bad)
        progress=PROGRESS.create_progress(True,stream=stream,clock=clock,wall_clock=lambda:0.)
        stage=progress.start('syn',iterations=True)
        for iteration in range(1,4):
            clock.now=iteration
            stage({'iteration':iteration,'value':-.7,'convergence':1e-9,
                   'fixed_valid_centres':21,'moving_valid_centres':20})
        stage.finish('window_convergence')
        rows=[line for line in stream.getvalue().splitlines()
              if 'DIAGNOSTIC,' in line and 'DIAGNOSTIC,Iteration,' not in line]
        self.assertEqual(len(rows),3)
        self.assertTrue(all(line.endswith(',21/20,n/a,n/a,n/a') for line in rows))
        progress.start('affine','budget=0',iterations=True).finish('maximum_iterations')
        self.assertIn('iterations=0',stream.getvalue())
        self.assertEqual(root.handlers,handlers);self.assertEqual(root.level,level)
        self.assertIs(warnings.formatwarning,warning_formatter);self.assertIs(warnings.showwarning,warning_hook)
        imports=[node for node in ast.parse((ROOT/'src/fast_ants/progress.py').read_text()).body
                 if isinstance(node,(ast.Import,ast.ImportFrom))]
        modules=[node.module if isinstance(node,ast.ImportFrom) else alias.name
                 for node in imports for alias in (node.names if isinstance(node,ast.Import) else [None])]
        self.assertEqual(set(modules),{'numbers','sys','time'})

    def test_terminal_bar_updates_existing_records_and_keeps_file_diagnostics_clean(self):
        clock=FakeClock();stream=ProgressTee();bars=[]
        def make_bar(**options):
            bar=FakeBar(**options);bars.append(bar)
            return bar
        factory=Mock(side_effect=make_bar)
        with patch.dict('sys.modules',{'tqdm':SimpleNamespace(tqdm=factory)}):
            progress=PROGRESS.create_progress(True,3,stream=stream,clock=clock,wall_clock=lambda:0.)
            stage=progress.start('rigid','stage=1/2 level=1 budget=9',iterations=True,total=9)
            for iteration in range(1,6):
                clock.now=iteration*.25
                entry={'iteration':iteration,'value':-.1*iteration,'convergence':1e-5,
                       'valid_points':17,'learning_rate':.1*iteration,'gamma':.2,
                       'line_search_evaluations':6}
                original=entry.copy();stage(entry)
                self.assertEqual(entry,original)
            stage.message('checkpoint')
            stage.finish('window_converged');stage.finish('must_not_duplicate')
        factory.assert_called_once()
        bar=bars[0]
        self.assertIs(bar.options['file'],stream.progress_terminal)
        self.assertEqual(bar.options['total'],9)
        self.assertEqual(bar.options['desc'],'[fast_ants][rigid] level=1')
        self.assertEqual(bar.n,5);self.assertEqual(bar.updates,[1]*5)
        self.assertEqual(bar.closed,1)
        self.assertEqual(bar.postfixes[-1],{'metric':f'{-.5:.3e}','conv':f'{1e-5:.3e}',
            'valid':17,'lr':.5,'gamma':.2,'ls_eval':6,'stop':'window_converged'})
        terminal=stream.progress_terminal.getvalue();log=stream.progress_log.getvalue()
        self.assertNotIn('DIAGNOSTIC',terminal)
        self.assertIn('checkpoint',terminal);self.assertIn('checkpoint',log)
        self.assertIn('START stage=1/2 level=1 budget=9',log)
        self.assertIn('END reason=window_converged; iterations=5',log)
        self.assertNotIn('must_not_duplicate',log)
        self.assertNotIn('\r',log);self.assertNotIn('\x1b',log)
        rows=[line for line in log.splitlines()
              if 'DIAGNOSTIC,' in line and 'DIAGNOSTIC,Iteration,' not in line]
        self.assertEqual([int(line.split('DIAGNOSTIC,')[1].split(',')[0]) for line in rows],[1,3,5])
        self.assertIsNone(progress.bar)

    def test_nonterminal_and_quiet_modes_do_not_load_tqdm(self):
        factory=Mock(side_effect=AssertionError('tqdm must not be loaded'))
        with patch.dict('sys.modules',{'tqdm':SimpleNamespace(tqdm=factory)}):
            stream=FlushedStream()
            progress=PROGRESS.create_progress(True,stream=stream,wall_clock=lambda:0.)
            stage=progress.start('syn',iterations=True,total=2)
            stage({'iteration':1,'value':-.7,'fixed_valid_centres':21,'moving_valid_centres':20})
            stage.finish('window_converged')
            self.assertIn('DIAGNOSTIC,1,',stream.getvalue())
            self.assertIsNone(PROGRESS.create_progress(False,stream=ProgressTee()))
        factory.assert_not_called()

    def cg(self,observer,monitor):
        vector=FakeVector()
        monitor_values=iter((1.,0.))
        metric=SimpleNamespace(evaluate=Mock(return_value=SimpleNamespace(value=-.4,derivative=vector,valid_points=17)))
        namespace={
            'np':SimpleNamespace(float32=float,inf=float('inf'),any=lambda value:False,
                divide=Mock(return_value=vector),multiply=Mock(return_value=vector),
                zeros_like=lambda value:FakeVector(),errstate=lambda **kwargs:nullcontext()),
            'FLOAT1_EPSILON':1e-7,'FLOAT1_MAX':1e30,
            '_finite_scalar32':lambda value,name:float(value),
            '_vector32':lambda value,name:value,
            'parameter_scales32':Mock(return_value=vector),
            'WindowConvergence32':lambda window:SimpleNamespace(add=lambda value:next(monitor_values)),
            '_inner_product32':Mock(return_value=0.),'step_scale32':Mock(return_value=1.),
            'initial_learning_rate32':Mock(return_value=.1),
            'polak_ribiere_gamma32':Mock(return_value=0.),
            'conjugate_direction32':Mock(return_value=vector),
            'golden_section_search32':Mock(return_value=(.1,2)),
            'apply_parameter_step32':Mock(return_value=vector),
            'OptimizationResult':lambda parameters,history,reason,scales:
                SimpleNamespace(parameters=parameters,history=history,stop_reason=reason,scales=scales),
            'NoValidSamplesError':ValueError}
        function=source_function('src/fast_ants/pipeline/optimizer32.py','conjugate_gradient32',namespace)
        result=function(metric,SimpleNamespace(),vector,object(),iterations=2,
                        convergence_threshold=.01,monitor=monitor,scales=vector,progress=observer)
        return result,metric.evaluate.call_count

    def test_cg_hooks_observe_existing_completed_records_on_both_append_paths(self):
        for monitor,reason in ((False,'maximum_iterations'),(True,'window_converged')):
            with self.subTest(monitor=monitor):
                seen=[]
                observed,calls=self.cg(seen.append,monitor)
                quiet,quiet_calls=self.cg(None,monitor)
                self.assertEqual(calls,quiet_calls)
                self.assertEqual(calls,2)
                self.assertEqual(observed.history,quiet.history)
                self.assertEqual(observed.stop_reason,reason)
                self.assertEqual(len(seen),2)
                for entry,record in zip(seen,observed.history):self.assertIs(entry,record)
                self.assertIn('parameters_after_step',seen[0])
                if monitor:self.assertNotIn('parameters_after_step',seen[-1])

    def test_syn_observer_preserves_callback_and_diagnostic_selection_and_finally(self):
        class Window:
            def __init__(self,size):pass
            def add(self,value):return 1.
        function=source_function('src/fast_ants/backends/metal/syn.py','_syn_level',{'WindowConvergence':Window})
        stage=SimpleNamespace(convergence_window=3,convergence_threshold=.01,radius=2,
                              gradient_step=.1,update_variance=3.,total_variance=0.)
        for callback in (None,Mock()):
            records=[]
            ctx=SimpleNamespace(optimizations=frozenset(),serial_cc_guard=False,control_guards=Counter())
            step=SimpleNamespace(state='next',value=-.5,details={'fixed_valid_centres':21,'moving_valid_centres':20})
            ops=SimpleNamespace(context=ctx,syn_step=Mock(return_value=step))
            result=function(ops,'fixed','moving','initial',stage,'linear',None,None,2,4,callback,
                            _progress=records.append)
            self.assertEqual(ops.syn_step.call_count,2)
            self.assertEqual(len(records),2)
            self.assertEqual(records,result[1])
            self.assertIs(ops.syn_step.call_args.kwargs['_evaluate_final_residual'],callback is not None)
            self.assertFalse(ctx.serial_cc_guard)
            if callback is not None:
                self.assertEqual(callback.call_args_list[0].args,(4,1,step))
                self.assertEqual(callback.call_args_list[1].args,(4,2,step))
        ctx.serial_cc_guard=True
        with self.assertRaisesRegex(RuntimeError,'stream failure'):
            function(ops,'fixed','moving','initial',stage,'linear',None,None,1,0,None,
                     _progress=Mock(side_effect=RuntimeError('stream failure')))
        self.assertTrue(ctx.serial_cc_guard)


if __name__=='__main__':unittest.main()
