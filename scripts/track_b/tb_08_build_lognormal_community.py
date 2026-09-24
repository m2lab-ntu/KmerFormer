#!/usr/bin/env python3
"""
Track B – Step 8: build a non-uniform (log-normal) community from a ladder level.

E43 asks whether genus-level *abundance* survives when every read comes from a
species the model has never seen.  The balanced ladder pools cannot answer that:
Track A already showed that a uniform pool makes sample-level correlation a
sampling artefact (r ~ 0 for reasons that have nothing to do with the model).

So we resample one ladder level into a realistic log-normal community and emit
it in exactly the layout tb_04 produces, which means tb_05 (neural) and tb_06
(Kraken2 + Bracken) then run on it unchanged.

    python tb_08_build_lognormal_community.py --source_level L2_species \
        --name L2_lognormal --sigma 1.5 --n_reads 200000

Output
------
  test_data/<name>.fa, <name>_labels.tsv, <name>/val_dir/<name>.fa
  test_data/<name>_expected_abundance.tsv    ground-truth genus fractions
"""

import os
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--track_b_dir", default=os.environ.get(
        "KF_TRACK_B", os.path.join(os.environ.get("KF_OUT", "."), "track_b")))
    p.add_argument("--source_level", default="L2_species")
    p.add_argument("--name", default="L2_lognormal")
    p.add_argument("--sigma", type=float, default=1.5,
                   help="Log-normal sigma for the genus abundance profile")
    p.add_argument("--n_reads", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def read_fasta_index(path: Path) -> list[tuple[str, str]]:
    records = []
    with open(path) as fh:
        hdr = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                hdr = line[1:]
            elif hdr is not None:
                records.append((hdr, line))
                hdr = None
    return records


def main():
    args = parse_args()
    tb = Path(args.track_b_dir)
    test_dir = tb / "test_data"
    src_fa = test_dir / f"{args.source_level}.fa"
    src_tsv = test_dir / f"{args.source_level}_labels.tsv"
    for f in (src_fa, src_tsv):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    rng = np.random.default_rng(args.seed)
    pool = pd.read_csv(src_tsv, sep="\t")
    records = read_fasta_index(src_fa)
    if len(records) != len(pool):
        raise SystemExit(f"{len(records)} fasta records vs {len(pool)} label rows")

    genera = sorted(pool["genus_name"].unique())
    weights = rng.lognormal(mean=0.0, sigma=args.sigma, size=len(genera))
    weights /= weights.sum()

    idx_by_genus = {g: np.flatnonzero((pool["genus_name"] == g).to_numpy())
                    for g in genera}

    # Draw without replacement so no read is duplicated; a genus that is asked
    # for more reads than it has is capped, and the realised fractions (not the
    # requested ones) are what gets written as ground truth.
    chosen, requested, realised = [], {}, {}
    for g, w in zip(genera, weights):
        want = int(round(args.n_reads * w))
        avail = idx_by_genus[g]
        take = min(want, len(avail))
        requested[g] = want
        if take > 0:
            chosen.append(rng.choice(avail, size=take, replace=False))
        realised[g] = take
    sel = np.concatenate([c for c in chosen if len(c)]) if chosen else np.array([], int)
    rng.shuffle(sel)

    n_total = len(sel)
    capped = {g: (requested[g], realised[g]) for g in genera
              if realised[g] < requested[g]}
    print(f"{args.name}: {n_total:,} reads over "
          f"{sum(1 for g in genera if realised[g] > 0)} genera "
          f"(requested {args.n_reads:,})")
    if capped:
        print(f"  {len(capped)} genera capped by read availability; "
              f"expected abundances use the realised counts")

    out_fa = test_dir / f"{args.name}.fa"
    out_tsv = test_dir / f"{args.name}_labels.tsv"
    val_dir = test_dir / args.name / "val_dir"
    val_dir.mkdir(parents=True, exist_ok=True)

    sub = pool.iloc[sel].reset_index(drop=True)
    with open(out_fa, "w") as fa:
        for i in sel:
            hdr, seq = records[i]
            fa.write(f">{hdr}\n{seq}\n")
    sub.to_csv(out_tsv, sep="\t", index=False)
    shutil.copyfile(out_fa, val_dir / f"{args.name}.fa")

    exp = (sub.groupby(["genus_class", "genus_name"], as_index=False)
              .size().rename(columns={"size": "n_reads"}))
    exp["expected_fraction"] = exp["n_reads"] / n_total
    exp = exp.sort_values("expected_fraction", ascending=False)
    exp.to_csv(test_dir / f"{args.name}_expected_abundance.tsv",
               sep="\t", index=False)

    top = exp.head(5)[["genus_name", "expected_fraction"]]
    print("  top genera:\n" + top.to_string(index=False))
    print(f"  dynamic range: {exp['expected_fraction'].max():.4f} .. "
          f"{exp['expected_fraction'].min():.6f}")
    print(f"\n  {out_fa}\n  {out_tsv}")


if __name__ == "__main__":
    main()
