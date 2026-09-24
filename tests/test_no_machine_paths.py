#!/usr/bin/env python3
"""No machine-specific absolute paths in tracked files.

The repository is public. A path like /nas2/... or /work/<user>/... names one
workstation's disk layout: a reader cannot resolve it, and it discloses local
account and mount names for nothing. Scripts read such locations from the
git-ignored scripts/local_paths.sh; recorded evidence stores them relative to
KF_DATA / KF_VOCAB (scripts/weights/collect_evidence.py:portable_paths).

The one exemption is the rewrite table that performs that relativisation,
which has to name the prefixes it removes.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"/nas2/|/home/user/|/home/ymj1123|/work/ymj1123|/media/user/|192\.168\.")
EXEMPT = {"scripts/weights/collect_evidence.py", "tests/test_no_machine_paths.py"}


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    for rel in out.stdout.splitlines():
        path = REPO / rel
        if rel in EXEMPT or not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            yield rel, path.read_text()
        except UnicodeDecodeError:
            continue


def test_no_tracked_file_contains_a_machine_path():
    hits = [f"{rel}:{i}: {line.strip()[:100]}"
            for rel, text in tracked_text_files()
            for i, line in enumerate(text.splitlines(), 1) if PATTERN.search(line)]
    assert not hits, "machine-specific paths in tracked files:\n" + "\n".join(hits[:20])


def test_local_paths_file_is_ignored():
    r = subprocess.run(["git", "check-ignore", "scripts/local_paths.sh"], cwd=REPO)
    assert r.returncode == 0, "scripts/local_paths.sh must stay git-ignored"
