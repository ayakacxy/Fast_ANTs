"""Measure one fresh target process, excluding the timer's own startup.

Unix uses RUSAGE_CHILDREN; Windows queries the retained target process handle.
"""

# Support both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time


def windows_usage(process):
    """Read Windows CPU times and peak working set from the retained process handle."""
    import ctypes
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    kernel.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(int(process._handle), created, exited, system, user):
        raise ctypes.WinError(ctypes.get_last_error())
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not kernel.K32GetProcessMemoryInfo(int(process._handle), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    def seconds(value):
        return ((value.dwHighDateTime << 32) + value.dwLowDateTime) / 10_000_000
    return {"user_seconds": seconds(user), "system_seconds": seconds(system),
            "peak_process_rss_bytes": int(counters.PeakWorkingSetSize),
            "memory_method": "Windows GetProcessMemoryInfo PeakWorkingSetSize; one target child"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("a command and positive timeout are required")
    started = time.perf_counter()
    timed_out = False
    with subprocess.Popen(command) as process:
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            timed_out, returncode = True, 124
        elapsed = time.perf_counter()-started
        if os.name == "nt":
            usage = windows_usage(process)
        else:
            import resource
            record = resource.getrusage(resource.RUSAGE_CHILDREN)
            usage = {"user_seconds": record.ru_utime, "system_seconds": record.ru_stime,
                     "peak_process_rss_bytes": int(record.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
                     "memory_method": "fresh runner RUSAGE_CHILDREN; exactly one target child"}
    args.output.write_text(json.dumps({
        "command": command, "wall_seconds": elapsed, "returncode": returncode,
        "timed_out": timed_out, **usage,
        "scope": "one fresh target process; imports, input reads, registration and output writes included; runner startup excluded",
    }, indent=2)+"\n")
    raise SystemExit(returncode if returncode >= 0 else 128-returncode)


if __name__ == "__main__":
    main()
