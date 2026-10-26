"""Download the versioned P00 MRI subset; no registration libraries required."""

# Permit both direct script execution and python -m invocation.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from validation.common.paths import PROJECT as _TOOL_PROJECT, VALIDATION as _TOOL_VALIDATION


import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = _TOOL_VALIDATION


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry):
    name = entry["name"]
    if Path(name).name != name:
        raise ValueError(f"Unsafe filename: {name}")
    destination = ROOT / "data" / "public" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm, expected = entry["publisher_checksum"].split(":", 1)
    if destination.exists():
        if digest(destination, algorithm) != expected:
            raise RuntimeError(f"Existing file checksum mismatch: {destination}")
        print(f"cached {name}", flush=True)
    else:
        temporary = destination.with_name(destination.name + ".part")
        print(f"download {name} ({entry['bytes'] / 2**20:.1f} MiB)", flush=True)
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--retry-delay", "2",
             "--connect-timeout", "20", "--max-time", "240", "--silent",
             "--show-error", entry["url"], "--output", str(temporary)],
            check=True,
        )
        if temporary.stat().st_size != entry["bytes"] or digest(temporary, algorithm) != expected:
            raise RuntimeError(f"Downloaded file checksum/size mismatch: {name}")
        temporary.replace(destination)
        print(f"verified {name}", flush=True)
    return {**entry, "path": str(destination.relative_to(ROOT.parent)),
            "sha256": digest(destination, "sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    manifest = json.loads((ROOT / "sources" / "public-data.json").read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verified = list(pool.map(fetch, manifest["files"]))
    target = ROOT / "sources" / "downloaded-files.json"
    target.write_text(json.dumps({"schema_version": 1, "files": verified}, indent=2) + "\n")
    print(f"verified {len(verified)} public files; manifest: {target}", flush=True)


if __name__ == "__main__":
    main()
