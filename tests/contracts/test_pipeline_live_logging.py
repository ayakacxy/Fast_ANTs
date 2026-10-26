"""CPU-only fake subprocess logs; no implementation imports, JIT or MRI fits."""

import contextlib
import io
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'validation'))
import validation.benchmarks.measure_fp32_pipeline as benchmark


class PipelineLiveLoggingTests(unittest.TestCase):
    def arguments(self, root, timeout=5.):
        return benchmark.parse_args(['--output',str(root/'report'),'--engine','cpu',
                                     '--verbose','--log-every','3','--timeout',str(timeout)])

    def test_live_utf8_arrives_before_exit_and_failure_log_is_exact_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);ack=root/'seen-early'
            script=root/'fake.py'
            script.write_text('''import os,sys,time
from pathlib import Path
raw='early 你好'.encode('utf-8')
os.write(1,raw[:-1]);time.sleep(.07);os.write(1,raw[-1:]+b'\\n')
deadline=time.monotonic()+3
while not Path(sys.argv[1]).exists() and time.monotonic()<deadline:time.sleep(.01)
if not Path(sys.argv[1]).exists():raise SystemExit(9)
os.write(2,'stderr 完整\\n'.encode('utf-8'))
os.write(1,b'tail\\n')
raise SystemExit(7)
''')
            class Terminal(io.StringIO):
                def write(self, text):
                    result=super().write(text)
                    if 'early 你好\n' in self.getvalue():ack.touch()
                    return result
            terminal=Terminal();args=self.arguments(root)
            with patch.object(benchmark,'worker_command',return_value=[sys.executable,'-B',str(script),str(ack)]), \
                    contextlib.redirect_stdout(terminal):
                row=benchmark.timed_run(args,'cpu','fake',{'random_seed':1},{},{})
            expected='early 你好\nstderr 完整\ntail\n'
            self.assertTrue(ack.is_file(),'First output was not visible while the target was alive')
            self.assertEqual(row['returncode'],7)
            self.assertEqual(row['status'],'failed')
            self.assertEqual((Path(row['directory'])/'console.log').read_bytes(),expected.encode('utf-8'))
            self.assertIn(expected,terminal.getvalue())
            for line in expected.splitlines():self.assertEqual(terminal.getvalue().count(line),1)
            self.assertEqual((row['verbose'],row['log_every']),(True,3))

    def test_timeout_and_crashed_runner_are_reaped_without_pipe_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            commands=(('timeout',[sys.executable,'-B','-c',
                        "import time; print('timeout output',flush=True); time.sleep(5)"],.2,124),
                      ('runner-crash',[sys.executable,'-B','-c',
                        "import os,signal,time; print('runner output',flush=True); os.kill(os.getppid(),signal.SIGKILL); time.sleep(5)"],5.,-signal.SIGKILL))
            for label,command,timeout,code in commands:
                with self.subTest(label=label):
                    args=self.arguments(root/label,timeout)
                    with patch.object(benchmark,'worker_command',return_value=command), \
                            patch.object(benchmark.os,'killpg',wraps=os.killpg) as killpg, \
                            contextlib.redirect_stdout(io.StringIO()):
                        row=benchmark.timed_run(args,'cpu','fake',{'random_seed':1},{},{})
                    self.assertEqual(row['returncode'],code)
                    self.assertEqual(row['status'],'failed')
                    self.assertTrue(killpg.called,'Failed runner process group was not cleaned up')
                    self.assertTrue((Path(row['directory'])/'console.log').read_text())

    def test_signature_compatibility_and_bounded_live_drain_without_backend_imports(self):
        module=ModuleType('fast_ants.registration_metal')
        def old_register(fixed,moving,*,linear_stages=None):pass
        def new_register(fixed,moving,*,verbose=False,log_every=1):pass
        with patch.dict(sys.modules,{'fast_ants.registration_metal':module}):
            module.register_metal=old_register
            self.assertEqual(benchmark.metal_logging_options(True,3),{})
            module.register_metal=new_register
            self.assertEqual(benchmark.metal_logging_options(True,3),{'verbose':True,'log_every':3})
            self.assertEqual(benchmark.metal_logging_options(False,3),{})
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'console.log';path.write_bytes(b'')
            terminal=io.StringIO();live=benchmark._LiveLog(path,terminal)
            try:
                with patch.object(live,'reader') as reader:
                    # A permanently productive reader must return control to
                    # timeout/process-exit checks after four finite reads.
                    reader.read.return_value=b'piece'
                    live.drain()
                    self.assertEqual(reader.read.call_count,4)
                    self.assertEqual(terminal.getvalue(),'piece'*4)
                    reader.read.reset_mock()
                    reader.read.side_effect=[b'\xe4',b'\xbd\xa0\n',b'']
                    live.drain(final=True)
                    self.assertEqual(reader.read.call_count,3)
                    self.assertTrue(terminal.getvalue().endswith('你\n'))
            finally:
                live.close()


if __name__=='__main__':unittest.main()
