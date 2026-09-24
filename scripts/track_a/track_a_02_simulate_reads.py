#!/usr/bin/env python3
"""
Track A – Step 2: Simulate 150bp Illumina reads with ART from downloaded genomes.

For each genome FASTA under genomes/<genus>/
  → run art_illumina (HiSeq 2500, paired-end, 150bp, insert 400±50)
  → convert paired-end FASTQ output to single-line FASTA
  → write to  reads/<genus>/<accession>.fa  with labelled headers:
        >lbl|0|<accession>|<genus_class>|<genus_name>-<read_id>
     Field 3 (0-indexed) = genus_class  ← what extract_mt_predictions uses.
     Species class (field 1) is a placeholder 0; genus eval only.

Usage
─────
  python track_a_02_simulate_reads.py \
      --labels   assets/genus_map.tsv \
      --genome_dir  genomes \
      --out_dir  reads \
      [--coverage 10] \
      [--seed 42] \
      [--skip_existing]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True,
                   help="TSV with genus_name column (use assets/genus_map.tsv, 120 rows)")
    p.add_argument("--genome_dir", default="genomes")
    p.add_argument("--out_dir", default="reads")
    p.add_argument("--coverage", type=float, default=10,
                   help="Fold coverage for ART (default 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def build_genus_map(labels_tsv: str) -> dict[str, int]:
    """Return {genus_name: genus_class} exactly as the training pipeline builds it."""
    df = pd.read_csv(labels_tsv, sep="\t")
    # Replicate run_genus_rctta.py: sorted unique genus names → 0-indexed
    sorted_genera = sorted(df["genus_name"].unique())
    genus2id = {g: i for i, g in enumerate(sorted_genera)}
    print(f"Genus map: {len(genus2id)} genera (0–{len(genus2id)-1})")
    return genus2id


def fastq_to_singleline_fasta(fastq_path: Path, out_fa: Path,
                               genus_name: str, genus_class: int,
                               accession: str, read_offset: int) -> int:
    """Convert ART FASTQ to single-line labelled FASTA.  Returns number of reads written."""
    n = 0
    with open(fastq_path) as fq, open(out_fa, "a") as fa:
        while True:
            hdr = fq.readline()
            if not hdr:
                break
            seq = fq.readline().strip()
            fq.readline()   # +
            fq.readline()   # qual
            read_id = n + read_offset
            # >lbl|species_class|species_name|genus_class|genus_name-readid
            fa.write(f">lbl|0|{accession}|{genus_class}|{genus_name}-{read_id}\n")
            fa.write(seq + "\n")
            n += 1
    return n


def simulate_genome(fna_path: Path, genus_name: str, genus_class: int,
                    out_fa: Path, coverage: float, seed: int) -> int:
    """Run ART on one genome; append results to out_fa.  Returns total reads written."""
    accession = fna_path.stem
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "art_out")
        cmd = [
            "art_illumina",
            "-ss", "HS25",
            "-i", str(fna_path),
            "-p",               # paired-end
            "-l", "150",
            "-m", "400",
            "-s", "50",
            "-f", str(coverage),
            "-o", prefix,
            "-rs", str(seed),
            "-na",              # skip alignment file (saves time)
            "-q",               # quiet
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ART FAILED for {fna_path.name}:\n{result.stderr[:500]}")
            return 0

        r1 = Path(f"{prefix}1.fq")
        r2 = Path(f"{prefix}2.fq")
        n = 0
        for fq in [r1, r2]:
            if fq.exists():
                n += fastq_to_singleline_fasta(fq, out_fa, genus_name,
                                               genus_class, accession, n)
    return n


def main():
    args = parse_args()
    genome_dir = Path(args.genome_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genus2id = build_genus_map(args.labels)

    total_reads = 0
    missing_in_map = []

    for genus_dir in sorted(genome_dir.iterdir()):
        if not genus_dir.is_dir() or genus_dir.name == "__pycache__":
            continue
        genus_name = genus_dir.name.replace("_", " ")
        if genus_name not in genus2id:
            missing_in_map.append(genus_name)
            continue
        genus_class = genus2id[genus_name]

        genus_out_dir = out_dir / genus_dir.name
        genus_out_dir.mkdir(exist_ok=True)

        fnas = sorted(genus_dir.glob("*.fna"))
        if not fnas:
            print(f"  SKIP {genus_name}: no .fna files")
            continue

        for fna in fnas:
            out_fa = genus_out_dir / f"{fna.stem}.fa"
            if args.skip_existing and out_fa.exists():
                print(f"  SKIP {genus_name}/{fna.stem} (exists)")
                continue
            print(f"  Simulating {genus_name}/{fna.stem} (class {genus_class}) …",
                  flush=True)
            if out_fa.exists():
                out_fa.unlink()
            n = simulate_genome(fna, genus_name, genus_class, out_fa,
                                args.coverage, args.seed)
            print(f"    → {n:,} reads")
            total_reads += n

    if missing_in_map:
        print(f"\nWARNING: {len(missing_in_map)} genome dirs not found in genus map:",
              missing_in_map)
    print(f"\nDone.  Total reads written: {total_reads:,}")
    print(f"Reads directory: {out_dir}")


if __name__ == "__main__":
    main()
