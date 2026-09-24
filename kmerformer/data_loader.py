#!/usr/bin/env python3
"""
Data loader for Token-level GFM Classifier training.

Loads paired FASTA + labels TSV, performs train/val split,
and returns a dict consumed by train.py.

Labels TSV columns: idx, seq_id, species_class, genus_class, genus_name, species_name
FASTA headers:      >lbl|species_idx|genome_name|genus_idx|species_name/read_num
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .splits import class_space, split_indices


def _parse_fasta(fasta_path: str):
    """Return parallel lists (seq_ids, sequences) from a FASTA file."""
    seq_ids, sequences = [], []
    current_id, parts = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seq_ids.append(current_id)
                    sequences.append("".join(parts))
                current_id = line[1:]
                parts = []
            elif line:
                parts.append(line)
    if current_id is not None:
        seq_ids.append(current_id)
        sequences.append("".join(parts))
    return seq_ids, sequences


def load_test_data(
    fasta_path: str,
    labels_path: str,
    task: str = "genus",
) -> dict:
    """
    Load a test/eval FASTA + labels TSV without any train/val split.
    All reads are returned as the 'val' split for use with evaluate.py.
    """
    print(f"Loading test labels: {labels_path}")
    df = pd.read_csv(labels_path, sep="\t", dtype={"species_class": int, "genus_class": int})
    required = {"seq_id", "genus_class", "genus_name"}
    if task != "genus":
        required |= {"species_class", "species_name"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"test labels missing columns: {sorted(missing_columns)}")
    # Track A/B genus-only manifests do not assign training species classes to
    # unseen genomes. Preserve that distinction; never invent species class 0.
    has_species = "species_class" in df and "species_name" in df
    if not has_species:
        df["species_class"] = -1
        df["species_name"] = "unassigned"

    id2genus   = dict(zip(df["genus_class"],   df["genus_name"]))
    id2species = dict(zip(df["species_class"], df["species_name"])) if has_species else {}
    num_genera  = int(df["genus_class"].nunique())
    num_species = int(df["species_class"].nunique()) if has_species else 0
    print(f"  {num_genera} genera, {num_species} species")

    seq_id_to_genus   = dict(zip(df["seq_id"], df["genus_class"]))
    seq_id_to_species = dict(zip(df["seq_id"], df["species_class"]))

    print(f"Loading test FASTA: {fasta_path}")
    seq_ids, sequences = _parse_fasta(fasta_path)
    print(f"  Parsed {len(sequences):,} reads")

    matched_seqs, genus_labels, species_labels = [], [], []
    missing = 0
    for sid, seq in zip(seq_ids, sequences):
        g = seq_id_to_genus.get(sid)
        s = seq_id_to_species.get(sid)
        if g is None or s is None:
            missing += 1
            continue
        matched_seqs.append(seq)
        genus_labels.append(g)
        species_labels.append(s)
    if missing:
        print(f"  Dropped {missing:,} reads with no label match")
    print(f"  Matched {len(matched_seqs):,} reads")

    seqs_arr    = np.array(matched_seqs,  dtype=object)
    genus_arr   = np.array(genus_labels,  dtype=np.int64)
    species_arr = np.array(species_labels, dtype=np.int64)

    return {
        "train_sequences":      seqs_arr[:0],
        "val_sequences":        seqs_arr,
        "train_genus_labels":   genus_arr[:0],
        "val_genus_labels":     genus_arr,
        "train_species_labels": species_arr[:0],
        "val_species_labels":   species_arr,
        "num_genera":           num_genera,
        "num_species":          num_species,
        "id2genus":             id2genus,
        "id2species":           id2species,
    }


def load_data(
    fasta_path: str,
    labels_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    task: str = "genus",
    split_strategy: str = "stratified",
    split_manifest=None,
) -> dict:
    """
    Load FASTA + labels TSV, split train/val, return dict for train.py.

    Args:
        fasta_path:  Path to balanced FASTA (reads_50M.fa or similar).
        labels_path: Path to labels TSV with columns:
                     idx, seq_id, species_class, genus_class, genus_name, species_name
        val_ratio:   Fraction of data to use as validation set.
        seed:        Random seed for train/val split.
        task:        "genus" or "species" (used only for console reporting).

    Returns dict with keys:
        train_sequences, val_sequences         (numpy str arrays)
        train_genus_labels, val_genus_labels   (numpy int arrays)
        train_species_labels, val_species_labels
        num_genera, num_species                (int)
        id2genus, id2species                   (int→str dicts)
    """
    print(f"Loading labels: {labels_path}")
    df = pd.read_csv(labels_path, sep="\t", dtype={"species_class": int, "genus_class": int})

    # Build label mappings
    id2genus   = dict(zip(df["genus_class"],   df["genus_name"]))
    id2species = dict(zip(df["species_class"], df["species_name"]))
    num_genera  = class_space(df["genus_class"].to_numpy())
    num_species = class_space(df["species_class"].to_numpy())
    print(f"  {num_genera} genera, {num_species} species")

    # Index by seq_id for fast lookup
    seq_id_to_genus   = dict(zip(df["seq_id"], df["genus_class"]))
    seq_id_to_species = dict(zip(df["seq_id"], df["species_class"]))

    print(f"Loading FASTA: {fasta_path}")
    seq_ids, sequences = _parse_fasta(fasta_path)
    print(f"  Parsed {len(sequences):,} reads")

    # Match sequences to labels (drop unmatched)
    matched_seqs, genus_labels, species_labels = [], [], []
    missing = 0
    for sid, seq in zip(seq_ids, sequences):
        g = seq_id_to_genus.get(sid)
        s = seq_id_to_species.get(sid)
        if g is None or s is None:
            missing += 1
            continue
        matched_seqs.append(seq)
        genus_labels.append(g)
        species_labels.append(s)
    if missing:
        print(f"  Dropped {missing:,} reads with no label match")
    print(f"  Matched {len(matched_seqs):,} reads")

    seqs_arr     = np.array(matched_seqs,   dtype=object)
    genus_arr    = np.array(genus_labels,   dtype=np.int64)
    species_arr  = np.array(species_labels, dtype=np.int64)

    # Stratify on the task label to preserve class balance
    strat_labels = genus_arr if task == "genus" else species_arr

    train_idx, val_idx = split_indices(
        strat_labels, val_ratio, seed, strategy=split_strategy, manifest=split_manifest,
        fasta_path=fasta_path, labels_path=labels_path, task=task,
    )

    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    return {
        "train_sequences":      seqs_arr[train_idx],
        "val_sequences":        seqs_arr[val_idx],
        "train_genus_labels":   genus_arr[train_idx],
        "val_genus_labels":     genus_arr[val_idx],
        "train_species_labels": species_arr[train_idx],
        "val_species_labels":   species_arr[val_idx],
        "num_genera":           num_genera,
        "num_species":          num_species,
        "id2genus":             id2genus,
        "id2species":           id2species,
    }
