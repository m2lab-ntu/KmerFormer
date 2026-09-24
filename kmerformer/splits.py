"""Explicit dataset partitions shared by single-process and DDP training."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from sklearn.model_selection import train_test_split


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_space(values):
    """Preserve raw nonnegative class IDs, including gaps in species catalogues."""
    values = np.asarray(values)
    if values.size == 0 or not np.issubdtype(values.dtype, np.integer) or values.min() < 0:
        raise ValueError("Class IDs must be a nonempty array of nonnegative integers")
    return int(values.max()) + 1


def split_options(cfg):
    data = cfg["data"]
    # Older lazy configurations used a different algorithm. Explicitly retain
    # it while ensuring the selected strategy is independent of process count.
    strategy = data.get("split_strategy", "random" if data.get("lazy", False) else "stratified")
    manifest = data.get("split_manifest")
    if manifest is None:
        manifest = str(Path(cfg["output"]["dir"]) / "split.npz")
    return {"split_strategy": strategy, "split_manifest": manifest}


def split_indices(labels, val_ratio, seed, *, strategy="stratified", manifest=None,
                  fasta_path=None, labels_path=None, task="genus"):
    labels = np.asarray(labels)
    if not 0 <= val_ratio < 1 or len(labels) == 0:
        raise ValueError("Split requires nonempty data and val_ratio in [0, 1)")
    if strategy not in ("stratified", "random"):
        raise ValueError("split_strategy must be stratified or random")
    metadata = {"schema_version": 1, "strategy": strategy, "seed": int(seed),
                "val_ratio": float(val_ratio), "task": task, "n_reads": len(labels),
                "ordered_labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest()}
    if manifest:
        # Full content hashes bind the partition to both source files, not to
        # absolute paths, timestamps or a sampled prefix of the input.
        metadata["fasta_sha256"] = sha256_file(fasta_path)
        metadata["labels_sha256"] = sha256_file(labels_path)
        path = Path(manifest)
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                if json.loads(str(stored["metadata"])) != metadata:
                    raise ValueError(f"Split manifest {path} does not match this dataset/protocol; choose a new run directory")
                train, val = stored["train"], stored["val"]
            merged = np.concatenate((train, val))
            if (merged.dtype.kind not in "iu" or len(merged) != len(labels) or
                    not np.array_equal(np.sort(merged), np.arange(len(labels)))):
                raise ValueError(f"Split manifest {path} is not a complete disjoint partition")
            return train, val
    if val_ratio == 0:
        train, val = np.arange(len(labels)), np.empty(0, dtype=np.int64)
    elif strategy == "stratified":
        train, val = train_test_split(np.arange(len(labels)), test_size=val_ratio,
                                     random_state=seed, stratify=labels)
    else:
        perm = np.random.default_rng(seed).permutation(len(labels))
        n_val = int(len(labels) * val_ratio)
        train, val = perm[n_val:], perm[:n_val]
    if manifest:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as dest:
                np.savez(dest, train=train, val=val, metadata=json.dumps(metadata, sort_keys=True))
            # Concurrent ranks generate identical arrays; publish a complete file.
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return train, val
