#!/usr/bin/env python3
"""
Remap Kraken2 species predictions to genus, using labels_100K.tsv as the
species_class -> genus_class lookup.  Unclassified reads (preds == -1)
stay as -1 in the genus output.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species_npz",  required=True)
    ap.add_argument("--labels_tsv",   required=True)
    ap.add_argument("--output_npz",   required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_tsv, sep="\t")
    sp2gn = dict(zip(df["species_class"].astype(int), df["genus_class"].astype(int)))
    # Build vectorised LUT
    max_sp = max(sp2gn.keys())
    lut = np.full(max_sp + 1, -1, dtype=np.int64)
    for sp_id, gn_id in sp2gn.items():
        lut[sp_id] = gn_id
    n_genera = int(df["genus_class"].max()) + 1
    print(f"Species classes: 0–{max_sp}, Genera: 0–{n_genera-1}")

    d = np.load(args.species_npz)
    sp_preds  = d["preds"].astype(np.int64)
    sp_labels = d["labels"].astype(np.int64)
    n = len(sp_preds)

    # Remap labels (always valid)
    gn_labels = lut[sp_labels]
    if (gn_labels < 0).any():
        raise RuntimeError("Some species labels have no genus mapping")

    # Remap preds: keep -1 as -1, otherwise lookup
    gn_preds = np.where(sp_preds < 0, -1, lut[np.clip(sp_preds, 0, max_sp)])

    # Sanity: no positive species pred should map to -1 unless out-of-range
    bad = (sp_preds >= 0) & (gn_preds < 0)
    if bad.any():
        print(f"WARNING: {bad.sum()} species preds had no genus mapping; setting to -1")

    np.savez_compressed(args.output_npz, preds=gn_preds, labels=gn_labels)

    n_clf = int((gn_preds >= 0).sum())
    correct = int((gn_preds == gn_labels).sum())
    correct_clf = int((gn_preds[gn_preds >= 0] == gn_labels[gn_preds >= 0]).sum())
    print(f"\nN reads:    {n}")
    print(f"Classified: {n_clf} ({n_clf/n*100:.2f}%)")
    print(f"Genus Top-1 (all):        {correct/n*100:.2f}%")
    print(f"Genus Top-1 (classified): {correct_clf/n_clf*100:.2f}%")
    print(f"Saved: {args.output_npz}")


if __name__ == "__main__":
    main()
