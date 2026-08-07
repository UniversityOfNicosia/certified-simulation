#!/usr/bin/env python3
"""Write a SHA-256 manifest of every tracked file at the current commit.

The manifest is the integrity anchor for a release: it lists the commit,
then one line per tracked file in `sha256sum` format, sorted by path.
Attach the output to the release; verify a checkout with

    sha256sum -c MANIFEST.sha256

Run from anywhere inside the repository:

    python tooling/make_manifest.py [--output MANIFEST.sha256]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


def repository_root() -> Path:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(top)


def tracked_files(root: Path) -> list[str]:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return sorted(entry.decode("utf-8") for entry in listing.split(b"\0") if entry)


def head_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("MANIFEST.sha256"))
    args = parser.parse_args()

    root = repository_root()
    lines = [f"# commit {head_commit(root)}"]
    for name in tracked_files(root):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    payload = "\n".join(lines) + "\n"
    args.output.write_bytes(payload.encode("utf-8"))
    print(f"{len(lines) - 1} files -> {args.output}")


if __name__ == "__main__":
    main()
