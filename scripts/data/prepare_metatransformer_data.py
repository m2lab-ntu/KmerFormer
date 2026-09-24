#!/usr/bin/env python3
"""
prepare_metatransformer_data.py

Split our 50M balanced FASTA into MetaTransformer-compatible train/val directories.
Uses 90/10 stratified split (seed=42, stratified by genus) matching our GFM classifier.

MetaTransformer expects:
  - Directory of .fa files for train and val
  - FASTA headers with class IDs at configurable positions (split by "|")
  - Multiple .fa chunks for multi-worker data loading

Our FASTA header format:
  >lbl|{species_class}|{species_name}|{genus_class}|{genus_name}-{read_id}/{pair}
  MetaTransformer uses class_indices to pick label position:
    species: class_indices=1 → species_class
    genus:   class_indices=3 → genus_class

Usage:
  python3 prepare_metatransformer_data.py \
      --fasta $KF_DATA/balanced_50M/reads_50M.fa \
      --labels $KF_DATA/balanced_50M/labels_50M.tsv \
      --output $KF_DATA/balanced_50M/metatransformer_format \
      --val_ratio 0.1 --seed 42 \
      --train_chunks 16 --val_chunks 4
"""

import argparse
import os
import sys
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Prepare 50M data for MetaTransformer")
    p.add_argument("--fasta", required=True, help="Path to reads_50M.fa")
    p.add_argument("--labels", required=True, help="Path to labels_50M.tsv")
    p.add_argument("--output", required=True, help="Output base directory")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_chunks", type=int, default=16)
    p.add_argument("--val_chunks", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Prepare MetaTransformer Data from 50M Balanced Dataset")
    print("=" * 60)

    # ---- 1. Read labels TSV to get genus labels for stratified split ----
    print("\n[1/5] Reading labels TSV...")
    genus_labels = []
    species_labels = []
    genus_names_map = {}
    species_names_map = {}

    with open(args.labels, "r") as f:
        header = f.readline()
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            s_class = int(parts[2])
            g_class = int(parts[3])
            g_name = parts[4]
            s_name = parts[5]
            species_labels.append(s_class)
            genus_labels.append(g_class)
            if g_class not in genus_names_map:
                genus_names_map[g_class] = g_name
            if s_class not in species_names_map:
                species_names_map[s_class] = s_name

            if (i + 1) % 10_000_000 == 0:
                print(f"    ...read {i + 1:,} label rows")

    n_total = len(genus_labels)
    genus_labels = np.array(genus_labels, dtype=np.int32)
    species_labels = np.array(species_labels, dtype=np.int32)
    n_genera = len(np.unique(genus_labels))
    n_species = len(np.unique(species_labels))
    print(f"    Total reads: {n_total:,}")
    print(f"    Unique genera: {n_genera}")
    print(f"    Unique species: {n_species}")

    # ---- 2. Stratified train/val split ----
    print(f"\n[2/5] Stratified split (val_ratio={args.val_ratio}, seed={args.seed})...")
    from sklearn.model_selection import train_test_split

    indices = np.arange(n_total)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        stratify=genus_labels,
        random_state=args.seed,
    )
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    print(f"    Train: {len(train_idx):,}, Val: {len(val_idx):,}")

    # ---- 3. Create output directories ----
    print(f"\n[3/5] Creating output directories at {args.output}")
    train_dir = os.path.join(args.output, "train")
    val_dir = os.path.join(args.output, "val")
    mapping_dir = os.path.join(args.output, "sequence_metadata")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(mapping_dir, exist_ok=True)

    # ---- 4. Read FASTA and write chunked train/val files ----
    print(f"\n[4/5] Splitting FASTA → {args.train_chunks} train chunks + {args.val_chunks} val chunks...")

    train_handles = [
        open(os.path.join(train_dir, f"reads_{i:03d}.fa"), "w")
        for i in range(args.train_chunks)
    ]
    val_handles = [
        open(os.path.join(val_dir, f"reads_{i:03d}.fa"), "w")
        for i in range(args.val_chunks)
    ]

    train_counter = 0
    val_counter = 0
    read_idx = 0
    current_header = None
    current_seq_lines = []

    def flush_read(hdr, seq_lines, idx):
        nonlocal train_counter, val_counter
        seq_text = "".join(seq_lines)
        if idx in train_set:
            chunk_id = train_counter % args.train_chunks
            train_handles[chunk_id].write(hdr + "\n" + seq_text + "\n")
            train_counter += 1
        elif idx in val_set:
            chunk_id = val_counter % args.val_chunks
            val_handles[chunk_id].write(hdr + "\n" + seq_text + "\n")
            val_counter += 1

    with open(args.fasta, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_header is not None:
                    flush_read(current_header, current_seq_lines, read_idx)
                    read_idx += 1
                    if read_idx % 5_000_000 == 0:
                        print(f"    Processed {read_idx:,} / {n_total:,} reads "
                              f"(train: {train_counter:,}, val: {val_counter:,})")
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

    if current_header is not None:
        flush_read(current_header, current_seq_lines, read_idx)
        read_idx += 1

    for fh in train_handles:
        fh.close()
    for fh in val_handles:
        fh.close()

    print(f"    Train reads written: {train_counter:,} ({args.train_chunks} chunks)")
    print(f"    Val reads written:   {val_counter:,} ({args.val_chunks} chunks)")

    # ---- 5. Create mapping files ----
    print(f"\n[5/5] Creating mapping files...")

    with open(os.path.join(mapping_dir, "genus_mapping.tab"), "w") as f:
        f.write("Class\tGenus\n")
        for cls_id in sorted(genus_names_map.keys()):
            f.write(f"{cls_id}\t{genus_names_map[cls_id]}\n")

    with open(os.path.join(mapping_dir, "species_mapping.tab"), "w") as f:
        f.write("Class\tSpecies\n")
        for cls_id in sorted(species_names_map.keys()):
            f.write(f"{cls_id}\t{species_names_map[cls_id]}\n")

    print(f"    Genus mapping:   {len(genus_names_map)} classes")
    print(f"    Species mapping: {len(species_names_map)} classes")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Done! Output structure:")
    print(f"  {args.output}/")
    print(f"    train/              ({args.train_chunks} .fa files, {train_counter:,} reads)")
    print(f"    val/                ({args.val_chunks} .fa files, {val_counter:,} reads)")
    print(f"    sequence_metadata/")
    print(f"      genus_mapping.tab   ({len(genus_names_map)} genera)")
    print(f"      species_mapping.tab ({len(species_names_map)} species)")
    print()
    print("Next steps:")
    print("  1. Copy this directory to Taiwania-2 if not already there")
    print("  2. Update MetaTransformer config:")
    print(f"     - Genus:   num_classes=381, class_indices=3")
    print(f"     - Species: num_classes=2505, class_indices=1")
    print("  3. Train with:")
    print("     python3 train.py experiment_name=genus_50M \\")
    print(f"         experiment_base_dir=$KF_OUT/metatransformer_experiments \\")
    print(f"         cfg_path=config/config_genus_50M.yaml \\")
    print(f"         data_path_root={args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
