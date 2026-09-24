#!/usr/bin/env python3
"""
Prepare Kraken2 predictions for evaluate_sample.py by handling unclassified
reads (preds == -1).  Two output strategies:

  unknown_class : map -1 to a new pseudo-class n_classes (= max_label + 1).
                  All 100K reads are kept; unclassified reads bin into the
                  "unknown" slot.  evaluate_sample.py will compute metrics
                  with one extra class — its true count is always 0, its
                  predicted abundance is the unclassified rate per sample.
                  This is the "Kraken2's actual sample-level utility"
                  measure (unclassified reads = lost signal).

  filtered      : drop reads where preds == -1.  Output has ~70K reads.
                  Pearson r / BC / ROC are computed only on reads Kraken2
                  was willing to commit to.  Inflates Kraken2 but answers
                  "how good is Kraken2 *on its confident calls*".

Usage:
  python prep_kraken2_for_sample_eval.py \
      --input  predictions_kraken2_twcc.npz \
      --output_unknown predictions_kraken2_unknown.npz \
      --output_filtered predictions_kraken2_filtered.npz
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",            required=True)
    ap.add_argument("--output_unknown",   required=True)
    ap.add_argument("--output_filtered",  required=True)
    args = ap.parse_args()

    d = np.load(args.input)
    preds  = d["preds"].astype(np.int64)
    labels = d["labels"].astype(np.int64)

    n = len(preds)
    classified = preds >= 0
    n_clf = int(classified.sum())
    n_unclf = n - n_clf
    max_label = int(labels.max())
    n_classes_real = max_label + 1
    unknown_id = n_classes_real

    print(f"N reads:           {n}")
    print(f"Classified:        {n_clf} ({n_clf/n*100:.2f}%)")
    print(f"Unclassified:      {n_unclf} ({n_unclf/n*100:.2f}%)")
    print(f"Real classes:      {n_classes_real}")
    print(f"Unknown class id:  {unknown_id} (added)")

    # ── Strategy 1: unknown_class — keep all reads, remap -1 -> unknown_id ─
    preds_unknown = np.where(preds < 0, unknown_id, preds).astype(np.int64)
    # To make evaluate_sample.py compute n_classes = unknown_id + 1, we must
    # ensure labels also span up to unknown_id.  Add one synthetic read at
    # the end so labels.max() = unknown_id.  This adds one row to the npz
    # with label=unknown_id and pred=unknown_id (perfectly classified ghost
    # read) → contributes a constant 1/(N+1) to both true and pred for the
    # unknown class, but doesn't affect real species accuracy noticeably.
    # Cleaner alternative: don't add the ghost row, instead trust that
    # preds.max() reaches unknown_id which is independent of labels.max().
    # evaluate_sample.py uses labels.max() — so we MUST patch this.
    labels_unknown = labels.copy()
    # Append one ghost read at end: label = unknown_id, pred = unknown_id
    preds_unknown_aug = np.concatenate([preds_unknown, np.array([unknown_id], dtype=np.int64)])
    labels_unknown_aug = np.concatenate([labels_unknown, np.array([unknown_id], dtype=np.int64)])

    np.savez_compressed(args.output_unknown,
                        preds=preds_unknown_aug, labels=labels_unknown_aug)
    acc_all = (preds_unknown_aug == labels_unknown_aug).mean()
    print(f"\n[unknown_class]   saved {args.output_unknown}")
    print(f"                  N={len(preds_unknown_aug)} (including 1 ghost row), acc={acc_all*100:.2f}%")
    print(f"                  → Pearson r will compare relative abundance across {unknown_id+1} bins")
    print(f"                  → Unclassified reads bin into class {unknown_id}; their 'true' count is always 0")

    # ── Strategy 2: filtered — drop unclassified reads ─────────────────────
    preds_filt  = preds[classified]
    labels_filt = labels[classified]
    np.savez_compressed(args.output_filtered,
                        preds=preds_filt, labels=labels_filt)
    acc_filt = (preds_filt == labels_filt).mean()
    print(f"\n[filtered]        saved {args.output_filtered}")
    print(f"                  N={len(preds_filt)} (dropped {n_unclf} unclassified), acc={acc_filt*100:.2f}%")
    print(f"                  → Pearson r computed only on reads Kraken2 committed to")


if __name__ == "__main__":
    main()
