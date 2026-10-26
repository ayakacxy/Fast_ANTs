"""Dispatch a maintained validation tool by its short command name."""

import argparse
import json
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", nargs="?", help="Maintained tool basename, with or without .py")
    parser.add_argument("--list", action="store_true", help="List maintained names and modules without executing them")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded unchanged to the selected tool")
    args = parser.parse_args(argv)
    index = json.loads((PROJECT / "validation/common/tool-index.json").read_text())
    if args.list:
        for name, module in sorted(index.items()):
            print(name + " -> " + module)
        return 0
    if args.tool is None:
        parser.error("Choose a tool, or use --list")
    name = args.tool.removesuffix(".py")
    if name not in index:
        parser.error("Unknown validation tool: " + args.tool)
    forwarded = args.arguments
    sys.argv = [index[name], *(forwarded[1:] if forwarded[:1] == ["--"] else forwarded)]
    runpy.run_module(index[name], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
