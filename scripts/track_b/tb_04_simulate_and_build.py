#!/usr/bin/env python3
"""
Track B – Step 4: simulate reads and build one test pool per ladder level.

ART parameters are identical to Track A (HS25, paired-end, 150 bp, insert
400+-50, seed 42) so the two tracks stay directly comparable; the only change
is that coverage is derived per genome so every genome can supply the same
number of reads, and the pool is then balanced by genome and by genus.

FASTA headers keep Track A's exact format so the existing MT / NT inference
scripts run unchanged:

    >lbl|0|<accession>|<genus_class>|<genus_name>-<read_id>

Per-read ANI is *not* stored in the header -- it is recoverable from the
accession via genome_metadata.tsv, which is what tb_07 uses to build the
ANI-stratified curve.

Output (per level, e.g. L2_species)
-----------------------------------
  reads/<level>/<genus>/<accession>.fa   per-genome simulated reads
  test_data/<level>.fa                   balanced pool
  test_data/<level>_labels.tsv           seq_id, genus_class, genus_name, accession
  test_data/<level>/val_dir/<level>.fa   copy for MT's --val_dir glob
  test_data/genome_metadata.tsv          accession -> level, ANI, read counts
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ART_READ_LEN = 150
COVERAGE_SAFETY = 3.0
MIN_COVERAGE = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ani_summary", required=True)
    p.add_argument("--genomes_dir", required=True)
    p.add_argument("--genus_map", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--levels", default="L1_strain,L2_species,L2_far")
    p.add_argument("--reads_per_genome", type=int, default=5000)
    p.add_argument("--max_reads_per_genus", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def read_tsv(path: str) -> list[dict]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def resolve_class(genus: str, row: dict, genus2class: dict[str, int]) -> int:
    """Class id for a genus, falling back to the row's own value.

    Novel-genus (L3) pools have no valid class -- the correct answer is "none of
    the 120" -- so tb_13 writes a -1 sentinel and it is carried through instead
    of failing the lookup.  For in-label-space pools the two must agree.
    """
    row_cls = int(row["genus_class"])
    if genus in genus2class:
        if row_cls != genus2class[genus]:
            sys.exit(f"{genus}: summary says class {row_cls}, "
                     f"genus_map says {genus2class[genus]}")
        return genus2class[genus]
    if row_cls >= 0:
        sys.exit(f"{genus} is absent from the genus map but claims class {row_cls}")
    return row_cls


def load_genus_map(path: str) -> dict[str, int]:
    rows = read_tsv(path)
    mapping = {r["genus_name"]: int(r["genus_class"]) for r in rows}
    # The training pipeline derives class ids from sorted genus names; verify
    # the shipped column agrees so a silent off-by-one cannot corrupt labels.
    expected = {g: i for i, g in enumerate(sorted(mapping))}
    if mapping != expected:
        bad = [g for g in mapping if mapping[g] != expected[g]]
        sys.exit(f"genus_map class ids are not sorted-order: {bad[:5]}")
    return mapping


def simulate_genome(fna: Path, out_fa: Path, accession: str, genus_name: str,
                    genus_class: int, genome_len: int, reads_needed: int,
                    seed: int) -> int:
    """Run ART on one genome and write labelled single-line FASTA."""
    if genome_len > 0:
        cov = reads_needed * ART_READ_LEN * COVERAGE_SAFETY / genome_len
    else:
        cov = 5.0
    cov = max(MIN_COVERAGE, cov)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "art_out")
        cmd = [
            "art_illumina", "-ss", "HS25", "-i", str(fna), "-p",
            "-l", str(ART_READ_LEN), "-m", "400", "-s", "50",
            "-f", f"{cov:.3f}", "-o", prefix, "-rs", str(seed), "-na", "-q",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    ART FAILED {accession}: {proc.stderr.strip()[:300]}")
            return 0

        n = 0
        out_fa.parent.mkdir(parents=True, exist_ok=True)
        with open(out_fa, "w") as fa:
            for mate in (1, 2):
                fq = Path(f"{prefix}{mate}.fq")
                if not fq.exists():
                    continue
                with open(fq) as fh:
                    while True:
                        hdr = fh.readline()
                        if not hdr:
                            break
                        seq = fh.readline().strip()
                        fh.readline()
                        fh.readline()
                        if "N" in seq or len(seq) != ART_READ_LEN:
                            continue
                        fa.write(f">lbl|0|{accession}|{genus_class}|"
                                 f"{genus_name}-{n}\n{seq}\n")
                        n += 1
        return n


def read_fasta_pairs(path: Path):
    with open(path) as fh:
        hdr = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                hdr = line[1:]
            elif hdr is not None:
                yield hdr, line
                hdr = None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    reads_root = out_dir / "reads"
    test_root = out_dir / "test_data"
    test_root.mkdir(parents=True, exist_ok=True)

    genus2class = load_genus_map(args.genus_map)
    wanted_levels = [lv for lv in args.levels.split(",") if lv]
    rng = random.Random(args.seed)

    rows = [r for r in read_tsv(args.ani_summary)
            if r["keep"] == "1" and r["level"] in wanted_levels]
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    print("Genomes to simulate: "
          + ", ".join(f"{lv}={len(by_level[lv])}" for lv in wanted_levels))

    meta_rows = []
    for level in wanted_levels:
        level_rows = sorted(by_level[level], key=lambda r: (r["genus"], r["accession"]))
        if not level_rows:
            print(f"\n== {level}: no genomes, skipped")
            continue
        print(f"\n== {level}: {len(level_rows)} genomes")

        # 1) simulate
        for i, r in enumerate(level_rows, 1):
            genus, acc = r["genus"], r["accession"]
            fna = Path(args.genomes_dir) / genus / f"{acc}.fna"
            out_fa = reads_root / level / genus / f"{acc}.fa"
            if args.skip_existing and out_fa.exists() and out_fa.stat().st_size > 0:
                n = sum(1 for _ in read_fasta_pairs(out_fa))
            else:
                n = simulate_genome(
                    fna, out_fa, acc, genus, resolve_class(genus, r, genus2class),
                    int(r["total_length"] or 0), args.reads_per_genome, args.seed)
            r["_n_simulated"] = n
            if i % 25 == 0 or i == len(level_rows):
                print(f"  [{i}/{len(level_rows)}] simulated", flush=True)

        # 2) balanced pool: cap per genome, then cap per genus
        pool_fa = test_root / f"{level}.fa"
        pool_tsv = test_root / f"{level}_labels.tsv"
        val_dir = test_root / level / "val_dir"
        val_dir.mkdir(parents=True, exist_ok=True)

        by_genus = defaultdict(list)
        for r in level_rows:
            by_genus[r["genus"]].append(r)

        n_written_total = 0
        with open(pool_fa, "w") as fa_out, open(pool_tsv, "w") as tsv_out:
            tsv_out.write("seq_id\tgenus_class\tgenus_name\taccession\n")
            for genus in sorted(by_genus):
                genome_rows = by_genus[genus]
                # Split the per-genus budget evenly across that genus' genomes
                # so a genus with 5 genomes does not outweigh one with 1.
                per_genome_cap = min(
                    args.reads_per_genome,
                    max(1, args.max_reads_per_genus // len(genome_rows)))
                genus_written = 0
                for r in genome_rows:
                    src = reads_root / level / genus / f"{r['accession']}.fa"
                    if not src.exists():
                        r["_n_kept"] = 0
                        continue
                    records = list(read_fasta_pairs(src))
                    if len(records) > per_genome_cap:
                        records = rng.sample(records, per_genome_cap)
                    gcls = resolve_class(genus, r, genus2class)
                    for hdr, seq in records:
                        fa_out.write(f">{hdr}\n{seq}\n")
                        tsv_out.write(f"{hdr}\t{gcls}\t{genus}\t"
                                      f"{r['accession']}\n")
                    r["_n_kept"] = len(records)
                    genus_written += len(records)
                n_written_total += genus_written

        shutil.copyfile(pool_fa, val_dir / f"{level}.fa")
        n_genera = sum(1 for g in by_genus
                       if any(r.get("_n_kept") for r in by_genus[g]))
        print(f"  pool: {n_written_total:,} reads over {n_genera} genera")
        print(f"    {pool_fa}")

        for r in level_rows:
            meta_rows.append({
                "level": level,
                "genus": r["genus"],
                "genus_class": r["genus_class"],
                "accession": r["accession"],
                "organism_name": r["organism_name"],
                "max_ani_same_genus": r["max_ani_same_genus"],
                "closest_training_ref_same_genus": r["closest_training_ref_same_genus"],
                "n_training_species_in_genus": r["n_training_species_in_genus"],
                "n_simulated": r.get("_n_simulated", 0),
                "n_in_pool": r.get("_n_kept", 0),
            })

    meta_cols = ["level", "genus", "genus_class", "accession", "organism_name",
                 "max_ani_same_genus", "closest_training_ref_same_genus",
                 "n_training_species_in_genus", "n_simulated", "n_in_pool"]
    # Merge, never overwrite: this script is run once per level group, and
    # clobbering the table would silently strip the ANI metadata that tb_07
    # needs for every rung it is not currently simulating.
    meta_path = test_root / "genome_metadata.tsv"
    kept_existing = []
    if meta_path.exists():
        regenerating = set(wanted_levels)
        for r in read_tsv(str(meta_path)):
            if r.get("level") not in regenerating:
                kept_existing.append(r)
    with open(meta_path, "w") as fh:
        fh.write("\t".join(meta_cols) + "\n")
        for r in kept_existing + meta_rows:
            fh.write("\t".join(str(r.get(c, "")) for c in meta_cols) + "\n")
    print(f"\n{meta_path}  ({len(kept_existing)} rows kept from other levels, "
          f"{len(meta_rows)} written)")


if __name__ == "__main__":
    main()
