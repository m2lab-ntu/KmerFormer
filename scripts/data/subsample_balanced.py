#!/usr/bin/env python3
"""
Balanced subsampling from a large labeled FASTA file.

FASTA headers encode labels:  >lbl|species_class|species_name|genus_class|genus_name-readid/pair

Usage:
    python subsample_balanced.py \
        --input /path/to/reads.fa \
        --output_fasta /path/to/reads_5M.fa \
        --output_labels /path/to/labels_5M.tsv \
        --total_reads 5000000 \
        --balance_by species \
        --seed 42
"""

import argparse
import random
import sys
from collections import defaultdict

import numpy as np


def parse_header(header: str):
    """Parse >lbl|species_class|species_name|genus_class|genus_name-readid/pair"""
    clean = header.lstrip(">")
    parts = clean.split("|")
    species_class = int(parts[1])
    species_name = parts[2]
    genus_class = int(parts[3])
    genus_and_read = parts[4]
    genus_name = genus_and_read.split("-")[0]
    return species_class, species_name, genus_class, genus_name


def main():
    parser = argparse.ArgumentParser(description="Balanced subsampling from labeled FASTA")
    parser.add_argument("--input", required=True, help="Input FASTA path")
    parser.add_argument("--output_fasta", required=True, help="Output FASTA path")
    parser.add_argument("--output_labels", required=True, help="Output labels TSV path")
    parser.add_argument("--total_reads", type=int, default=5_000_000)
    parser.add_argument("--balance_by", choices=["species", "genus"], default="species")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Pass 1: Count reads per class ---
    print(f"Pass 1: Counting reads per {args.balance_by} in {args.input} ...")
    class_counts = defaultdict(int)
    total_reads = 0

    with open(args.input, "r") as f:
        for line in f:
            if line.startswith(">"):
                sp_cls, sp_name, gen_cls, gen_name = parse_header(line.strip())
                key = sp_cls if args.balance_by == "species" else gen_cls
                class_counts[key] += 1
                total_reads += 1
                if total_reads % 10_000_000 == 0:
                    print(f"  ... scanned {total_reads / 1e6:.0f}M reads", flush=True)

    n_classes = len(class_counts)
    print(f"  Total reads: {total_reads:,}")
    print(f"  Unique {args.balance_by} classes: {n_classes}")

    # --- Compute per-class quota ---
    per_class_target = args.total_reads // n_classes
    remainder = args.total_reads - per_class_target * n_classes

    quota = {}
    for cls_id in sorted(class_counts.keys()):
        available = class_counts[cls_id]
        quota[cls_id] = min(per_class_target, available)

    allocated = sum(quota.values())
    shortfall = args.total_reads - allocated

    if shortfall > 0:
        classes_with_surplus = [
            c for c in quota if class_counts[c] > quota[c]
        ]
        random.shuffle(classes_with_surplus)
        for c in classes_with_surplus:
            if shortfall <= 0:
                break
            extra = min(shortfall, class_counts[c] - quota[c])
            quota[c] += extra
            shortfall -= extra

    final_total = sum(quota.values())
    min_q = min(quota.values())
    max_q = max(quota.values())
    print(f"\n  Target: {args.total_reads:,}, Allocated: {final_total:,}")
    print(f"  Per-class: min={min_q:,}, max={max_q:,}, target={per_class_target:,}")

    # --- Pre-compute per-class keep masks (memory-light) ---
    # Reservoir-in-RAM blows up for large outputs (250M reads ≈ 100+ GB in
    # Python objects → OOM). Since the train/val split and DataLoader both
    # shuffle downstream, the output FASTA does NOT need a global shuffle.
    # Instead we pre-decide which of each class's occurrences to keep as a
    # boolean mask (sum of all masks = total reads ≈ 258M bools ≈ 260 MB),
    # then stream-read the FASTA and write selected reads immediately.
    print(f"\nBuilding per-class keep masks ...")
    np_rng = np.random.default_rng(args.seed)
    keep_mask = {}
    for cls_id in sorted(class_counts.keys()):
        avail = class_counts[cls_id]
        q = quota[cls_id]
        if q >= avail:
            keep_mask[cls_id] = np.ones(avail, dtype=bool)
        else:
            m = np.zeros(avail, dtype=bool)
            sel = np_rng.choice(avail, size=q, replace=False)
            m[sel] = True
            keep_mask[cls_id] = m

    # --- Pass 2: stream-read, write selected reads directly ---
    print(f"\nPass 2: streaming write ...")
    from collections import Counter
    class_seen = defaultdict(int)
    genus_dist = Counter()
    species_dist = Counter()
    out_idx = 0

    def emit(header, seq, f_fa, f_tsv):
        nonlocal out_idx
        sp_cls, sp_name, gen_cls, gen_name = parse_header(header)
        key = sp_cls if args.balance_by == "species" else gen_cls
        i = class_seen[key]
        class_seen[key] += 1
        if not keep_mask[key][i]:
            return
        clean = header.lstrip(">")
        f_fa.write(f">{clean}\n{seq}\n")
        f_tsv.write(f"{out_idx}\t{clean}\t{sp_cls}\t{gen_cls}\t{gen_name}\t{sp_name}\n")
        genus_dist[gen_name] += 1
        species_dist[sp_cls] += 1
        out_idx += 1

    with open(args.input, "r") as f, \
         open(args.output_fasta, "w") as f_fa, \
         open(args.output_labels, "w") as f_tsv:
        f_tsv.write("idx\tseq_id\tspecies_class\tgenus_class\tgenus_name\tspecies_name\n")
        header = None
        seq_lines = []
        reads_processed = 0
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    emit(header, "".join(seq_lines), f_fa, f_tsv)
                    reads_processed += 1
                    if reads_processed % 10_000_000 == 0:
                        print(f"  ... processed {reads_processed / 1e6:.0f}M reads "
                              f"(written {out_idx / 1e6:.1f}M)", flush=True)
                header = line.strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        # last record
        if header is not None:
            emit(header, "".join(seq_lines), f_fa, f_tsv)

    # --- Summary stats ---
    print(f"\n=== Summary ===")
    print(f"  Total reads: {out_idx:,}")
    print(f"  Unique genera: {len(genus_dist)}")
    print(f"  Unique species: {len(species_dist)}")
    print(f"  Reads per species: min={min(species_dist.values()):,}, "
          f"max={max(species_dist.values()):,}, "
          f"mean={sum(species_dist.values())/len(species_dist):.0f}")
    print(f"\n  Top 10 genera:")
    for gen_name, count in genus_dist.most_common(10):
        print(f"    {gen_name}: {count:,}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
