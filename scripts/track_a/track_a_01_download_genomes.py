#!/usr/bin/env python3
"""
Track A – Step 1: Download new genomes for each of the 120 training genera.

For each genus in the label map, fetch 1–3 COMPLETE (or chromosome/scaffold)
assemblies from NCBI that are NOT among the 1,535 training accessions.  Uses
ncbi-datasets-cli (`datasets download genome taxon`).

Output layout
─────────────
  genomes/<genus_name>/
      <accession_1>.fna
      <accession_2>.fna   (if available)
      ...
  genomes/download_log.tsv   ← per-genus summary (genus, accession, status)

Usage
─────
  python track_a_01_download_genomes.py \
      --labels   assets/genus_map.tsv \
      --out_dir  genomes \
      [--max_per_genus 2] \
      [--assembly_level complete,chromosome,scaffold] \
      [--skip_existing]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True,
                   help="TSV with genus_name column (use assets/genus_map.tsv, 120 rows)")
    p.add_argument("--out_dir", default="genomes",
                   help="Directory to place downloaded genomes")
    p.add_argument("--max_per_genus", type=int, default=2,
                   help="Max assemblies to keep per genus (default 2)")
    p.add_argument("--assembly_level",
                   default="complete,chromosome,scaffold",
                   help="Comma-separated NCBI assembly levels, best-first")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip genera with ≥1 genome already downloaded")
    p.add_argument(
        "--exclude_accessions",
        help="Optional file of training GC[AF] accessions to exclude",
    )
    return p.parse_args()


def get_genus_names(labels_tsv: str) -> list[str]:
    df = pd.read_csv(labels_tsv, sep="\t")
    if "genus_name" not in df.columns:
        sys.exit("ERROR: labels TSV must have a 'genus_name' column.")
    genera = sorted(df["genus_name"].unique().tolist())
    print(f"Found {len(genera)} unique genera in label map.")
    return genera


def accession_base(value: str) -> str:
    match = re.match(r"(GC[AF]_\d+)", value)
    return match.group(1) if match else value.split(".", 1)[0]


def load_exclusions(path: str | None) -> set[str]:
    if not path:
        return set()
    with open(path) as handle:
        return {
            accession_base(line.strip())
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def download_genus(genus: str, out_dir: Path, max_n: int,
                   assembly_levels: list[str],
                   excluded_accessions: set[str]) -> list[str]:
    """Try each assembly level until we get ≥1 genome; return list of .fna paths."""
    genus_dir = out_dir / genus.replace(" ", "_")
    genus_dir.mkdir(parents=True, exist_ok=True)

    for level in assembly_levels:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                "datasets", "download", "genome", "taxon", genus,
                "--assembly-level", level,
                "--assembly-source", "RefSeq",
                "--filename", f"{tmp}/dl.zip",
                "--no-progressbar",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not Path(f"{tmp}/dl.zip").exists():
                continue

            # Unzip
            subprocess.run(["unzip", "-q", "-o", f"{tmp}/dl.zip", "-d", tmp],
                           check=True)

            # Find all .fna files
            fnas = sorted(Path(tmp).rglob("*.fna"))
            if not fnas:
                fnas = sorted(Path(tmp).rglob("*.fa"))
            if not fnas:
                continue

            # Copy up to max_n genomes
            copied = []
            for fna in fnas:
                acc = fna.stem.split("_genomic")[0]
                if accession_base(acc) in excluded_accessions:
                    continue
                dest = genus_dir / f"{acc}.fna"
                if not dest.exists():
                    shutil.copy(fna, dest)
                copied.append(str(dest))
                if len(copied) >= max_n:
                    break
            if copied:
                return copied

    return []


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genera = get_genus_names(args.labels)
    levels = [l.strip() for l in args.assembly_level.split(",")]
    exclusions = load_exclusions(args.exclude_accessions)
    print(f"Excluding {len(exclusions)} training accessions.")

    log_rows = []
    for i, genus in enumerate(genera):
        genus_dir = out_dir / genus.replace(" ", "_")
        existing = list(genus_dir.glob("*.fna")) if genus_dir.exists() else []

        if args.skip_existing and existing:
            blocked = [
                path for path in existing
                if accession_base(path.stem) in exclusions
            ]
            if blocked:
                names = ", ".join(path.name for path in blocked)
                sys.exit(
                    f"ERROR: {genus_dir} contains excluded training "
                    f"accession(s): {names}. Remove or replace them before "
                    "using --skip_existing."
                )
            print(f"[{i+1:3d}/{len(genera)}] SKIP  {genus}  ({len(existing)} existing)")
            for p in existing:
                log_rows.append({"genus": genus, "accession": p.stem, "status": "existing"})
            continue

        print(f"[{i+1:3d}/{len(genera)}] Downloading  {genus} …", flush=True)
        paths = download_genus(
            genus, out_dir, args.max_per_genus, levels, exclusions
        )

        if paths:
            print(f"           → {len(paths)} genome(s) saved")
            for p in paths:
                log_rows.append({"genus": genus, "accession": Path(p).stem, "status": "downloaded"})
        else:
            print(f"           → FAILED (no assembly found at any level)")
            log_rows.append({"genus": genus, "accession": "", "status": "FAILED"})

    log_df = pd.DataFrame(log_rows)
    log_path = out_dir / "download_log.tsv"
    log_df.to_csv(log_path, sep="\t", index=False)

    n_ok = (log_df["status"] != "FAILED").sum()
    n_fail = (log_df["status"] == "FAILED").sum()
    print(f"\nDone.  {n_ok} genomes downloaded/existing,  {n_fail} genera failed.")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
