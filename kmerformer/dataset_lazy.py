#!/usr/bin/env python3
"""
Lazy-loading FASTA dataset for large-scale DDP training.

Stores only file byte-offsets in RAM; reads sequences on demand via seek().
Suitable for 258M-scale datasets where loading all sequences into RAM is infeasible.

Memory usage:
  Index array:  N × 16 bytes  (int64 offset + int32 genus + int32 species)
  For 258M reads: ~4.1 GB per DDP rank

Usage:
  dataset = LazyFASTADataset(
      fasta_path, labels_path, tokenizer, max_length, split="train", val_ratio=0.1, seed=42
  )

The dataset builds (or loads a cached) binary index file `<fasta_path>.idx.npy`
on first instantiation.  Subsequent runs reuse the cache (build time: ~3-5 min).
"""

import hashlib
import os
import threading
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
import pandas as pd

from .array_cache import file_identity, load_array_cache, save_array_cache
from .splits import class_space, split_indices


INDEX_DTYPE = np.dtype([("offset", np.int64), ("genus", np.int32),
                        ("species", np.int32)])


class LazyFASTADataset(Dataset):
    """
    Random-access FASTA dataset backed by byte-offset index.

    Each worker thread maintains its own open file handle to avoid contention.
    """

    def __init__(
        self,
        fasta_path: str,
        labels_path: str,
        tokenizer,
        max_length: int = 32,
        split: str = "train",          # "train" or "val"
        val_ratio: float = 0.1,
        seed: int = 42,
        rc_augment: bool = False,
        kmer_preprocess=None,
        split_strategy="random",
        split_manifest=None,
        task="genus",
    ):
        if task not in ("genus", "species"):
            raise ValueError("task must be genus or species")
        self.task = task
        self.fasta_path = str(fasta_path)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.rc_augment = rc_augment
        self.kmer_preprocess = kmer_preprocess
        self._local = threading.local()   # per-thread file handle
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")

        # ── 1. Load or build index ──────────────────────────────────────────
        index_path = Path(str(fasta_path) + ".idx.npy")
        identity = {"fasta": file_identity(fasta_path), "version": 1}
        index = load_array_cache(index_path, identity)
        if index is not None and (index.dtype != INDEX_DTYPE or index.ndim != 1):
            index = None
        if index is not None:
            print(f"  Loading cached index: {index_path}")
        else:
            print(f"  Building FASTA index (one-time, ~3-5 min): {fasta_path}")
            index = self._build_index(fasta_path, index_path, identity)

        # dtype: offset (int64), genus_class (int32), species_class (int32)
        self._index = index  # shape (N, 3)

        # ── 2. Load labels for class-name lookup + consistency check ────────
        # Only the two class columns are needed for output widths. Reading just
        # these keeps a 50M-row labels TSV at ~0.4 GB instead of ~12 GB per
        # rank — matters because every DDP rank builds train+val datasets.
        df = pd.read_csv(labels_path, sep="\t",
                         usecols=["species_class", "genus_class"],
                         dtype={"species_class": "int32", "genus_class": "int32"})
        self.num_genera = class_space(df["genus_class"].to_numpy())
        self.num_species = class_space(df["species_class"].to_numpy())
        # The indexed path reads labels from headers; require their row order
        # and IDs to agree with the supplied training label table.
        if len(df) != len(index) or not np.array_equal(df["genus_class"], index["genus"]) or not np.array_equal(df["species_class"], index["species"]):
            raise ValueError("FASTA header labels and TSV rows disagree; align the label table with FASTA order")
        train, val = split_indices(
            index[task], val_ratio, seed, strategy=split_strategy, manifest=split_manifest,
            fasta_path=fasta_path, labels_path=labels_path, task=task)
        self._idx = val if split == "val" else train

        print(f"  Split={split}: {len(self._idx):,} reads "
              f"(genera={self.num_genera}, species={self.num_species})")

    # ── Index building ──────────────────────────────────────────────────────

    @staticmethod
    def _build_index(fasta_path: str, index_path: Path, identity=None) -> np.ndarray:
        """Single-pass scan: record (offset, genus_class, species_class) per read."""
        records = []

        with open(fasta_path, "rb") as f:
            offset = 0
            n = 0
            for raw_line in f:
                if raw_line[0:1] == b">":
                    header = raw_line.decode().rstrip()[1:]
                    parts  = header.split("|")
                    try:
                        sp  = int(parts[1])
                        gn  = int(parts[3])
                        if gn < 0:
                            raise ValueError("negative genus class ID")
                    except (IndexError, ValueError) as exc:
                        raise ValueError(
                            f"{fasta_path}: invalid labelled FASTA header {header!r}; "
                            "expected >lbl|species_class|genome|genus_class|..."
                        ) from exc
                    records.append((offset, gn, sp))
                    n += 1
                    if n % 5_000_000 == 0:
                        print(f"    indexed {n:,} reads...", flush=True)
                offset += len(raw_line)

        if not records:
            raise ValueError(f"{fasta_path}: FASTA contains no reads")
        arr = np.array(records, dtype=INDEX_DTYPE)
        identity = identity or {"fasta": file_identity(fasta_path), "version": 1}
        try:
            save_array_cache(index_path, arr, identity)
            print(f"  Index saved to {index_path}  ({len(arr):,} reads)")
        except OSError as exc:
            print(f"  Could not cache FASTA index ({exc}); using in-memory index")
        return arr

    # ── Dataset interface ───────────────────────────────────────────────────

    def __len__(self):
        return len(self._idx)

    def _get_file(self):
        # A handle opened for a parent-process sanity check must not share its
        # seek position with forked DataLoader workers.
        if getattr(self._local, "pid", None) != os.getpid():
            if hasattr(self._local, "fh"):
                self._local.fh.close()
            self._local = threading.local()
        if not hasattr(self._local, "fh") or self._local.fh.closed:
            self._local.fh = open(self.fasta_path, "rb")
            self._local.pid = os.getpid()
        return self._local.fh

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def _read_seq(self, global_idx: int) -> str:
        rec = self._index[global_idx]
        fh  = self._get_file()
        fh.seek(int(rec["offset"]))
        fh.readline()           # skip header line
        parts = []
        for line in fh:
            if line.startswith(b">"):
                break
            parts.append(line.strip())
        return b"".join(parts).decode()

    def _reverse_complement(self, seq: str) -> str:
        comp = {"A": "T", "T": "A", "C": "G", "G": "C",
                "a": "t", "t": "a", "c": "g", "g": "c"}
        return "".join(comp.get(b, "N") for b in reversed(seq))

    def _kmer_split(self, seq: str, k: int, stride: int) -> str:
        kmers = [seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)]
        return " ".join(kmers)

    def __getitem__(self, local_idx: int):
        global_idx  = int(self._idx[local_idx])
        rec         = self._index[global_idx]
        genus_class = int(rec[self.task])
        seq         = self._read_seq(global_idx)

        # Optional RC augmentation
        if self.rc_augment and np.random.random() < 0.5:
            seq = self._reverse_complement(seq)

        # Optional k-mer preprocessing
        if self.kmer_preprocess is not None:
            seq = self._kmer_split(seq, self.kmer_preprocess["k"],
                                   self.kmer_preprocess.get("stride", 1))

        enc = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label          = torch.tensor(genus_class, dtype=torch.long)

        return input_ids, attention_mask, label

    def get_genus_labels(self) -> np.ndarray:
        """Return genus labels for all samples in this split (for class-weight computation)."""
        return self._index[self._idx]["genus"].astype(np.int32)

    def get_labels(self):
        return self._index[self._idx][self.task].astype(np.int32)
