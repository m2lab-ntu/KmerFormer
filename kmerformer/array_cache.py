"""Validated, atomic caches for arrays derived from a source file."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _stat_identity(path, stat):
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino}


def file_identity(path, *, content_hash=False):
    path = Path(path)
    stat = path.stat()
    identity = _stat_identity(path, stat)
    # Vocabulary line order defines the model's token IDs. Verify its content
    # even on filesystems with coarse timestamps. Small FASTA fixtures get the
    # same protection; large training FASTAs use their filesystem identity.
    if content_hash or stat.st_size <= 1024 * 1024:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        identity["sha256"] = digest.hexdigest()
    return identity


def load_array_cache(path, identity):
    """Return None for legacy, stale, incomplete or corrupt cache entries."""
    path = Path(path)
    try:
        metadata = json.loads(Path(str(path) + ".meta.json").read_text())
        if not isinstance(metadata, dict):
            return None
        if metadata.get("source") != identity:
            return None
        before = file_identity(path)
        if metadata.get("array") != before:
            return None
        array = np.load(path, allow_pickle=False)
        if file_identity(path) != before:
            return None
        return array
    except (OSError, ValueError, EOFError):
        return None


def save_array_cache(path, array, identity):
    """Readers only accept a complete array with its matching metadata."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.replace(temporary, path)
            temporary = None
            # Keep the identity of OUR inode. Another writer may already have
            # replaced the destination; its array must never get our source ID.
            stat = os.fstat(f.fileno())
            own_array = _stat_identity(path, stat)
            if stat.st_size <= 1024 * 1024:
                f.seek(0)
                own_array["sha256"] = hashlib.sha256(f.read()).hexdigest()
        metadata = {"source": identity, "array": own_array}
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         delete=False) as f:
            temporary = Path(f.name)
            json.dump(metadata, f)
        os.replace(temporary, Path(str(path) + ".meta.json"))
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
