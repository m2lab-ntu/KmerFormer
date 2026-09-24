#!/usr/bin/env python3
"""
Generate labels_258M.tsv from full HGR-UMGS reads.fa.

FASTA header format:
  >lbl|species_idx|genome_name|genus_idx|species_name/read_num

Usage:
  python scripts/generate_labels_258M.py \
      --fasta $KF_DATA/balanced_258M/reads.fa \
      --out   $KF_DATA/balanced_258M/labels_258M.tsv

Runtime: ~5-10 min for 258M reads (streaming, no RAM for sequences)
Output TSV columns: idx, seq_id, species_class, genus_class, genus_name, species_name
  (same schema as labels_50M.tsv)
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading headers from: {fasta_path}")
    print(f"Writing labels to:    {out_path}")

    # Collect unique name mappings first pass? No — just write inline.
    # Header: >lbl|species_idx|genome_name|genus_idx|species_name/read_num
    # Fields after split on '|': [0]=lbl [1]=species_idx [2]=genome_name [3]=genus_idx [4]=species_name/read_num

    header_open = open(out_path, "w")
    header_open.write("idx\tseq_id\tspecies_class\tgenus_class\tgenus_name\tspecies_name\n")

    genus_names  = {}   # genus_class → genus_name
    species_names = {}  # species_class → species_name

    n = 0
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            seq_id = line[1:].rstrip()
            parts  = seq_id.split("|")
            if len(parts) < 5:
                print(f"  [warn] unexpected header at read {n}: {seq_id[:80]}", file=sys.stderr)
                continue

            try:
                species_class = int(parts[1])
                genus_class   = int(parts[3])
            except ValueError:
                print(f"  [warn] non-int idx at read {n}: {seq_id[:80]}", file=sys.stderr)
                continue

            # Extract names (species_name/read_num → take before '/')
            raw_sp  = parts[4].split("/")[0] if "/" in parts[4] else parts[4]
            sp_name = raw_sp.strip()
            gn_name = parts[2].strip()          # genome_name used as genus proxy

            species_names.setdefault(species_class, sp_name)
            genus_names.setdefault(genus_class,   gn_name)

            header_open.write(f"{n}\t{seq_id}\t{species_class}\t{genus_class}\t{gn_name}\t{sp_name}\n")
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n:,} reads processed...", flush=True)

    header_open.close()
    print(f"\nDone. Wrote {n:,} rows to {out_path}")
    print(f"  Unique genera:  {len(genus_names)}")
    print(f"  Unique species: {len(species_names)}")


if __name__ == "__main__":
    main()
