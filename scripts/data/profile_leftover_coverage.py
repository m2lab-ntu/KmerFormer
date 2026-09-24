#!/usr/bin/env python
"""
Profile leftover (held-out-able) read coverage per class.

We want a CLEAN test set for v14/v15 = reads from the 1535sp source that were
NOT used by balanced_250M (v14/v15's training data). Because balanced_250M caps
each species at ~162K reads, small species may be fully consumed (zero leftover)
while large species have a surplus. This script quantifies that:

    leftover[class] = source_total[class] - used_by_250M[class]

Inputs:
  --src_counts : TSV produced by the companion grep|awk pass over the source
                 reads.fa headers, with columns:  genus_class  species_class  count
                 (one row per (genus,species) pair seen in the source)
  --idx        : balanced_250M/reads_250M.idx.npy (structured: offset,genus,species)

Outputs (written next to --out_prefix):
  <out>_per_species.tsv : species_class, source, used, leftover
  <out>_per_genus.tsv   : genus_class,   source, used, leftover
  + a summary to stdout: how many species/genera have leftover >= thresholds,
    how many are fully depleted (leftover<=0), and the max clean test size
    achievable at various per-class quotas.
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_counts", required=True,
                    help="TSV: genus_class<TAB>species_class<TAB>count (source headers)")
    ap.add_argument("--idx", required=True, help="balanced_250M reads_250M.idx.npy")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    # ── source totals per (genus, species) ──
    src = pd.read_csv(args.src_counts, sep="\t",
                      names=["genus_class", "species_class", "src_count"],
                      dtype={"genus_class": np.int32, "species_class": np.int32,
                             "src_count": np.int64})
    print(f"Source: {src['src_count'].sum():,} reads across "
          f"{src['species_class'].nunique():,} species / "
          f"{src['genus_class'].nunique():,} genera")

    # ── used counts from the 250M index (fast bincount) ──
    idx = np.load(args.idx)
    used_sp = pd.Series(np.bincount(idx["species"].astype(np.int64))).rename("used")
    used_gn = pd.Series(np.bincount(idx["genus"].astype(np.int64))).rename("used")
    print(f"balanced_250M: {len(idx):,} reads used")

    # ── per-species leftover ──
    sp = (src.groupby("species_class")["src_count"].sum().rename("source")
          .to_frame().join(used_sp, how="outer").fillna(0).astype(np.int64))
    sp["leftover"] = sp["source"] - sp["used"]
    sp = sp.sort_index()
    sp.to_csv(f"{args.out_prefix}_per_species.tsv", sep="\t",
              index_label="species_class")

    # ── per-genus leftover ──
    gn_src = src.groupby("genus_class")["src_count"].sum().rename("source")
    gn = (gn_src.to_frame().join(used_gn, how="outer").fillna(0).astype(np.int64))
    gn["leftover"] = gn["source"] - gn["used"]
    gn = gn.sort_index()
    gn.to_csv(f"{args.out_prefix}_per_genus.tsv", sep="\t", index_label="genus_class")

    # ── summary ──
    def coverage(df, level):
        n = len(df)
        depleted = int((df["leftover"] <= 0).sum())
        print(f"\n=== {level} coverage ({n} classes) ===")
        print(f"  fully depleted (leftover<=0): {depleted}  ({depleted/n*100:.1f}%)")
        for q in (100, 500, 1000, 5000):
            ok = int((df["leftover"] >= q).sum())
            print(f"  >= {q:>5} leftover reads: {ok:>5} classes ({ok/n*100:.1f}%)")
        # max clean test at a per-class quota
        for q in (100, 500, 1000):
            achievable = int(np.minimum(df["leftover"].clip(lower=0), q).sum())
            print(f"  clean test size @ {q}/class cap: {achievable:,} reads")
        print(f"  total leftover pool: {int(df['leftover'].clip(lower=0).sum()):,} reads")

    coverage(sp, "SPECIES")
    coverage(gn, "GENUS")
    print(f"\nWrote: {args.out_prefix}_per_species.tsv / _per_genus.tsv")


if __name__ == "__main__":
    main()
