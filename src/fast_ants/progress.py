"""Scoped, dependency-free terminal progress for one registration run.

The presentation follows FlashCHARM's stage prefix and ANTs' diagnostic
columns. Records identify fast-ants explicitly. No global logger, warning
handler, numerical dependency, or background thread is configured here.
"""

from numbers import Integral
import sys
import time


def validate_log_every(value):
    """Return a positive integer interval; booleans are not intervals."""
    if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
        raise ValueError('log_every must be a positive integer')
    return int(value)


def create_progress(verbose=False,log_every=1,**options):
    """Return no hook in quiet mode, while still validating the interval."""
    interval=validate_log_every(log_every)
    return RegistrationProgress(interval,**options) if verbose else None


class RegistrationProgress:
    """A per-run stream and clocks, without process-wide logging state."""
    def __init__(self,log_every=1,*,stream=None,clock=None,wall_clock=None):
        self.log_every=validate_log_every(log_every)
        self.stream=sys.stdout if stream is None else stream
        self.clock=time.perf_counter if clock is None else clock
        self.wall_clock=time.time if wall_clock is None else wall_clock
        self.started=self.clock()

    def message(self,stage,message):
        stamp=time.strftime('%H:%M:%S',time.localtime(self.wall_clock()))
        text=str(message).replace('\n','\\n').replace('\r','\\r')
        print(f'[fast-ants][{stage}] {stamp} | INFO | {text}',file=self.stream,flush=True)

    def start(self,stage,message='',*,iterations=False,started=None):
        return _ProgressStage(self,stage,message,iterations,started)

    def complete(self,message='Registration complete'):
        self.message('complete',f'{message}; elapsed={self.clock()-self.started:.6f}s')


class _ProgressStage:
    def __init__(self,owner,stage,message,iterations,started):
        self.owner,self.stage=owner,stage
        self.started=owner.clock() if started is None else float(started)
        self.previous_iteration=self.started
        self.sequence=0;self.printed_sequence=0;self.latest=None;self.finished=False
        owner.message(stage,'START'+(' '+message if message else ''))
        if iterations:
            owner.message(stage,'DIAGNOSTIC,Iteration,metricValue,convergenceValue,'
                          'ITERATION_TIME_INDEX,SINCE_LAST,validPoints')

    def message(self,message):
        self.owner.message(self.stage,message)

    @staticmethod
    def _number(value):
        return 'n/a' if value is None else f'{float(value):.12e}'

    def __call__(self,entry):
        """Observe an existing history record without mutating or reevaluating it."""
        now=self.owner.clock();self.sequence+=1
        interval=now-self.previous_iteration;self.previous_iteration=now
        self.latest=(self.sequence,entry,now-self.started,interval)
        iteration=int(entry['iteration'])
        if iteration==1 or iteration%self.owner.log_every==0:self._emit(self.latest)

    def _emit(self,sample):
        sequence,entry,elapsed,interval=sample
        if sequence==self.printed_sequence:return
        valid=entry.get('valid_points')
        if valid is None:
            fixed,moving=entry.get('fixed_valid_centres'),entry.get('moving_valid_centres')
            valid=f'{fixed}/{moving}' if fixed is not None and moving is not None else 'n/a'
        self.owner.message(self.stage,
            f'DIAGNOSTIC,{int(entry["iteration"])},{self._number(entry.get("value"))},'
            f'{self._number(entry.get("convergence"))},{elapsed:.6f},{interval:.6f},{valid}')
        self.printed_sequence=sequence

    def finish(self,reason):
        """Flush a throttled final iteration and record the actual stop reason."""
        if self.finished:return
        if self.latest is not None:self._emit(self.latest)
        iterations=0 if self.latest is None else int(self.latest[1]['iteration'])
        self.owner.message(self.stage,
            f'END reason={reason}; iterations={iterations}; elapsed={self.owner.clock()-self.started:.6f}s')
        self.finished=True
