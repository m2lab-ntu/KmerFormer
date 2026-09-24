#!/usr/bin/env python3
"""
Track B – Step 6: Kraken2 (matched 1,535-genome DB) on one ladder level.

Runs kraken2 + bracken on the pool built by tb_04 and writes predictions in the
same .npz layout the neural models use, so tb_07 can score every method with
one code path.

Kraken2 taxids in this DB follow the project convention  taxid = species_class + 2
(see scripts/baselines/build_kraken2_db_1535.py). Species IDs are mapped to genus
through the training label table; genus nodes use taxid = 10000 + genus_class.

Reads are matched to Kraken2 output by read id, never by line order, so a
reordered kraken output cannot silently misalign the labels.

Output
------
  out/<level>/kraken2/kraken.out, kraken.report
  out/<level>/kraken2/preds.npz          genus preds (-1 = unclassified)
  out/<level>/kraken2/bracken.species.tsv
  out/<level>/kraken2/bracken_genus_abundance.tsv
"""

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAXID_OFFSET = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--level", required=True)
    p.add_argument("--pool_fa", required=True)
    p.add_argument("--pool_labels", required=True,
                   help="<level>_labels.tsv from tb_04")
    p.add_argument("--db", default=os.environ.get(
        "KRAKEN2_DB", os.path.join(os.environ.get("KF_OUT", "."), "kraken2_db_1535")))
    p.add_argument("--train_labels",
                   default=os.path.join(os.environ.get("KF_DATA", "."),
                                        "val_100K", "labels_100K_val.tsv"),
                   help="Provides species_class -> genus_class for the 1,535 refs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--read_len", type=int, default=150)
    p.add_argument("--skip_bracken", action="store_true")
    p.add_argument("--reuse", action="store_true",
                   help="Reuse kraken.out if it already exists")
    return p.parse_args()


def build_taxid_lut(train_labels: str) -> tuple[np.ndarray, int]:
    df = pd.read_csv(train_labels, sep="\t",
                     usecols=["species_class", "genus_class"])
    sp2gn = dict(zip(df["species_class"].astype(int),
                     df["genus_class"].astype(int)))
    n_genera = int(df["genus_class"].max()) + 1
    genus_ids = sorted(set(sp2gn.values()))
    max_taxid = max(max(sp2gn) + TAXID_OFFSET, 10000 + max(genus_ids))
    lut = np.full(max_taxid + 1, -1, dtype=np.int64)
    for sp, gn in sp2gn.items():
        lut[sp + TAXID_OFFSET] = gn
    # Genus-aware indexes can classify directly at an internal genus node.
    # Those are valid calls, not missing-species assignments.
    for genus in genus_ids:
        lut[10000 + genus] = genus
    print(f"Crosswalk: {len(sp2gn)} species -> {n_genera} genera "
          f"(taxid range 2..{max_taxid})")
    return lut, n_genera


def run_kraken2(db: str, pool_fa: str, out_dir: Path, threads: int,
                reuse: bool) -> tuple[Path, Path]:
    kout = out_dir / "kraken.out"
    kreport = out_dir / "kraken.report"
    if reuse and kout.exists() and kout.stat().st_size > 0:
        print(f"  reusing {kout}")
        return kout, kreport
    cmd = ["kraken2", "--db", db, "--threads", str(threads),
           "--output", str(kout), "--report", str(kreport), pool_fa]
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr[-2000:])
    if proc.returncode != 0:
        sys.exit(f"kraken2 failed (rc={proc.returncode})")
    return kout, kreport


def run_bracken(db: str, kreport: Path, out_dir: Path, read_len: int) -> Path | None:
    out_tsv = out_dir / "bracken.species.tsv"
    cmd = ["bracken", "-d", db, "-i", str(kreport), "-o", str(out_tsv),
           "-w", str(out_dir / "bracken.report"), "-r", str(read_len), "-l", "S"]
    print("  " + " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        # Bracken only feeds the abundance table; the genus read-level
        # predictions are already written.  Never let a missing binary throw
        # away a completed Kraken2 run.
        print("  ! bracken not on PATH — skipping abundance table")
        return None
    if proc.returncode != 0:
        print(f"  ! bracken failed: {proc.stderr.strip()[-500:]}")
        return None
    return out_tsv


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lut, n_genera = build_taxid_lut(args.train_labels)

    labels_df = pd.read_csv(args.pool_labels, sep="\t")
    order = {sid: i for i, sid in enumerate(labels_df["seq_id"])}
    y_true = labels_df["genus_class"].to_numpy(np.int64)
    n = len(y_true)
    print(f"Pool: {n:,} reads, {labels_df['genus_name'].nunique()} genera")

    kout, kreport = run_kraken2(args.db, args.pool_fa, out_dir,
                                args.threads, args.reuse)

    preds = np.full(n, -1, dtype=np.int64)
    n_lines = n_unmatched = n_classified = n_no_genus = 0
    with open(kout) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            n_lines += 1
            status, read_id, taxid_field = parts[0], parts[1], parts[2]
            idx = order.get(read_id)
            if idx is None:
                n_unmatched += 1
                continue
            if status != "C":
                continue
            # Kraken sometimes writes "name (taxid 123)"; keep the numeric part.
            taxid = taxid_field
            if "taxid" in taxid_field:
                taxid = taxid_field.rsplit("taxid", 1)[1].strip(" )")
            try:
                taxid = int(taxid)
            except ValueError:
                continue
            n_classified += 1
            if 0 <= taxid < len(lut) and lut[taxid] >= 0:
                preds[idx] = lut[taxid]
            else:
                # Classified, but to an internal/LCA node above species -> the
                # read is committed yet gives us no genus.  Stays -1 and is
                # reported separately so it is not mistaken for "unclassified".
                n_no_genus += 1

    if n_lines != n:
        print(f"  ! kraken lines ({n_lines:,}) != pool reads ({n:,})")
    if n_unmatched:
        print(f"  ! {n_unmatched:,} kraken read ids not found in the pool")

    committed = preds >= 0
    acc_all = float((preds == y_true).mean())
    acc_committed = (float((preds[committed] == y_true[committed]).mean())
                     if committed.any() else float("nan"))

    np.savez_compressed(out_dir / "preds.npz", preds=preds, labels=y_true)

    stats = {
        "level": args.level,
        "n_reads": int(n),
        "kraken_classified_rate": n_classified / n if n else 0.0,
        "genus_committed_rate": float(committed.mean()),
        "classified_but_no_genus": int(n_no_genus),
        "genus_acc_all_reads": acc_all,
        "genus_acc_committed_only": acc_committed,
        "n_genera_in_pool": int(labels_df["genus_name"].nunique()),
    }

    if not args.skip_bracken:
        bt = run_bracken(args.db, kreport, out_dir, args.read_len)
        if bt is not None and bt.exists():
            br = pd.read_csv(bt, sep="\t")
            br["species_class"] = (br["taxonomy_id"].astype(int) - TAXID_OFFSET)
            br["genus_class"] = br["species_class"].map(
                lambda s: lut[s + TAXID_OFFSET] if 0 <= s + TAXID_OFFSET < len(lut) else -1)
            br = br[br["genus_class"] >= 0]
            g = (br.groupby("genus_class", as_index=False)["fraction_total_reads"]
                   .sum().rename(columns={"fraction_total_reads": "pred_fraction"}))
            g.to_csv(out_dir / "bracken_genus_abundance.tsv", sep="\t", index=False)
            stats["bracken_genera"] = int(len(g))
            stats["bracken_fraction_sum"] = float(g["pred_fraction"].sum())

    (out_dir / "kraken_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
