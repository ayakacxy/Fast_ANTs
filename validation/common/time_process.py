"""Measure one fresh child without macOS time(1)'s restricted sysctl query.

Only the target child's launch, execution and wait enter wall_seconds. This
runner's Python startup is excluded. A fresh runner has exactly one child, so
RUSAGE_CHILDREN describes that run rather than the maximum over prior repeats.
"""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


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
    try:
        result = subprocess.run(command, timeout=args.timeout)
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        timed_out, returncode = True, 124
    elapsed = time.perf_counter()-started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    args.output.write_text(json.dumps({
        "command": command, "wall_seconds": elapsed, "returncode": returncode,
        "timed_out": timed_out, "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "peak_process_rss_bytes": int(usage.ru_maxrss*(1 if sys.platform == "darwin" else 1024)),
        "scope": "one fresh target process; imports, input reads, registration and output writes included; runner startup excluded",
        "memory_method": "fresh runner RUSAGE_CHILDREN; exactly one target child"
    }, indent=2)+"\n")
    raise SystemExit(returncode if returncode >= 0 else 128-returncode)


if __name__ == "__main__":
    main()
