#!/usr/bin/env python
"""Score the depth-sweep Track A runs on the contamination-filtered pool.

run_genus_rctta.py reports accuracy over all 595,000 reads. metrics_clean.json
excludes three accessions that also appear in training, leaving 580,000 reads
over 116 genera; the published Track A numbers are all on that clean pool, so
these must be too. rctta.npz stores predictions in the order the runner kept
reads, which is fasta order filtered to genera present in the training label
space -- reproduced here to recover the read identity each prediction belongs to.

Balanced accuracy is reported alongside Top-1 because the genus distribution
here is not uniform and the two diverge.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# SWEEP holds the depth-sweep run outputs (one directory per arm under out/);
# TRACKA holds the out-of-genome pool. See docs/DATA.md.
SWEEP = Path(os.environ.get("KF_OUT", "."))
TRACKA = Path(os.environ.get("KF_TRACK_A",
                            os.path.join(os.environ.get("KF_DATA", "."), "track_a")))
OUT = SWEEP / "track_a_depth"

EXCLUDE = ("GCF_000010185", "GCF_000158275", "GCF_000312005")


def fasta_ids(path):
    ids = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].strip())
    return ids


def main():
    train = pd.read_csv(os.path.join(os.environ.get("KF_DATA", "."),
                                     "balanced_50M", "labels_50M.tsv"), sep="\t")
    genus2id = {g: i for i, g in enumerate(sorted(train["genus_name"].unique()))}

    test = pd.read_csv(TRACKA / "test_data/newgenome_test_labels.tsv", sep="\t")
    sid2gn = dict(zip(test["seq_id"], test["genus_name"]))

    # same filter the runner applied, in the same order
    kept_ids = [s for s in fasta_ids(TRACKA / "test_data/newgenome_test.fa")
                if sid2gn.get(s) in genus2id]
    keep = np.array([not any(a in s for a in EXCLUDE) for s in kept_ids])
    print(f"matched reads {len(kept_ids):,} -> clean {int(keep.sum()):,} "
          f"(dropped {int((~keep).sum()):,})")

    rows = []
    OUT.mkdir(exist_ok=True)
    for d in sorted(p for p in OUT.iterdir() if (p / "rctta.npz").exists()):
        z = np.load(d / "rctta.npz")
        preds, labels = z["preds"], z["labels"]
        if len(preds) != len(kept_ids):
            print(f"!! {d.name}: {len(preds):,} preds vs {len(kept_ids):,} reads "
                  f"-- order assumption broken, skipping")
            continue
        p, y = preds[keep], labels[keep]
        top1 = float((p == y).mean() * 100)
        recalls = [float((p[y == g] == g).mean()) for g in np.unique(y)]
        rows.append({"model": d.name, "top1_clean_pct": round(top1, 2),
                     "balanced_acc_pct": round(float(np.mean(recalls)) * 100, 2),
                     "top1_all_pct": round(float((preds == labels).mean() * 100), 2),
                     "genera": len(recalls), "reads": int(keep.sum())})

    if not rows:
        print("no scored runs found")
        return
    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    (OUT / "metrics_depth_clean.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT / 'metrics_depth_clean.json'}")
    print("\nreference (same clean pool): MT 13-mer 250M 32.67 | MT 13-mer 50M 27.05 | "
          "NT-v2 500M 25.81 | 13-mer exact 16L 23.14 | 13-mer hashed 16L 21.23")


if __name__ == "__main__":
    main()
