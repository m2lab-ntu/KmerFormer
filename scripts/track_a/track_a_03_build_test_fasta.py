#!/usr/bin/env python3
"""
Track A – Step 3: Merge per-genus per-genome FASATAs into one test FASTA.

Concatenates all reads/<genus>/*.fa into a single:
    test_data/newgenome_test.fa   (single-line sequences, required by MT reader)
and writes the accompanying:
    test_data/newgenome_test_labels.tsv   (seq_id, genus_class, genus_name)

The MT inference script (`extract_mt_predictions.py`) reads headers directly, so
the per-genus FASTAs from step 2 are already in the right format.  This script
just merges them and produces the NT-v2 labels TSV.

Usage
─────
  python track_a_03_build_test_fasta.py \
      --reads_dir  reads \
      --out_dir    test_data \
      [--max_reads 0]   # 0 = no limit; set e.g. 200000 to cap
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reads_dir", default="reads")
    p.add_argument("--out_dir", default="test_data")
    p.add_argument("--max_reads", type=int, default=0,
                   help="Cap total reads (0 = unlimited)")
    p.add_argument("--max_per_genus", type=int, default=0,
                   help="Cap reads per genus (0 = unlimited; recommended 5000)")
    return p.parse_args()


def main():
    args = parse_args()
    reads_dir = Path(args.reads_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fa = out_dir / "newgenome_test.fa"
    out_tsv = out_dir / "newgenome_test_labels.tsv"

    # Also create a directory with only the test FASTA so MT --val_dir globs it
    val_dir = out_dir / "val_dir"
    val_dir.mkdir(exist_ok=True)
    val_fa = val_dir / "newgenome_test.fa"

    tsv_rows = []
    total = 0
    capped = False

    with open(out_fa, "w") as fa_out:
        for genus_dir in sorted(reads_dir.iterdir()):
            if not genus_dir.is_dir():
                continue
            genus_count = 0  # reads written for this genus so far
            for fa_file in sorted(genus_dir.glob("*.fa")):
                if capped:
                    break
                with open(fa_file) as f:
                    hdr = None
                    seq_parts = []
                    for line in f:
                        line = line.rstrip()
                        if line.startswith(">"):
                            if hdr is not None:
                                seq = "".join(seq_parts)
                                fields = hdr[1:].split("|")
                                genus_class = int(fields[3])
                                genus_name = fields[4].rsplit("-", 1)[0]
                                seq_id = hdr[1:]
                                fa_out.write(f">{seq_id}\n{seq}\n")
                                tsv_rows.append({
                                    "seq_id": seq_id,
                                    "genus_class": genus_class,
                                    "genus_name": genus_name,
                                })
                                total += 1
                                genus_count += 1
                                if args.max_reads > 0 and total >= args.max_reads:
                                    capped = True
                                    break
                                if args.max_per_genus > 0 and genus_count >= args.max_per_genus:
                                    capped = True
                                    break
                            hdr = line
                            seq_parts = []
                        else:
                            seq_parts.append(line)
                    if not capped and hdr is not None:
                        seq = "".join(seq_parts)
                        fields = hdr[1:].split("|")
                        genus_class = int(fields[3])
                        genus_name = fields[4].rsplit("-", 1)[0]
                        seq_id = hdr[1:]
                        fa_out.write(f">{seq_id}\n{seq}\n")
                        tsv_rows.append({
                            "seq_id": seq_id,
                            "genus_class": genus_class,
                            "genus_name": genus_name,
                        })
                        total += 1
                        genus_count += 1
            # reset per-genus cap for next genus
            capped = False if (args.max_per_genus > 0 and not
                               (args.max_reads > 0 and total >= args.max_reads)) else capped
            if args.max_reads > 0 and total >= args.max_reads:
                break

    df = pd.DataFrame(tsv_rows, columns=["seq_id", "genus_class", "genus_name"])
    df.to_csv(out_tsv, sep="\t", index=False)
    if df.empty:
        print("WARNING: no reads found — check that step 2 produced output")
        return

    # Symlink or copy to val_dir so MT --val_dir only sees this one .fa
    if val_fa.exists() or val_fa.is_symlink():
        val_fa.unlink()
    val_fa.symlink_to(out_fa.resolve())

    genera_covered = df["genus_name"].nunique()
    print(f"Merged {total:,} reads from {genera_covered} genera")
    print(f"  FASTA:      {out_fa}")
    print(f"  Labels TSV: {out_tsv}")
    print(f"  MT val_dir: {val_dir}  (symlink to same FASTA)")
    if capped:
        print(f"  (capped at {args.max_reads})")

    # Quick per-genus count
    print(f"\nPer-genus read counts (top 10):")
    print(df.groupby("genus_name").size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
