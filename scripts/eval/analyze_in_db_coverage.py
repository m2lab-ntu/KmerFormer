#!/usr/bin/env python3
"""
Fair-comparison analysis vs Kraken2:
  - Kraken2 DB covers 1,316 / 1,535 species in the 100K test set
  - 219 GCF species absent from DB → 14,181 reads where Kraken2 = 0%
  - Restrict NT-v2 / MT predictions to the 85,819 in-DB reads
  - Report all-100K vs in-DB-only top-1
  - Per-species accuracy on the 219 missing species
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Root of the run outputs this analysis reads (predictions and metrics).
REPO = Path(os.environ.get("KF_OUT", "."))
LOCAL = REPO / "local_predictions"
RESULTS = REPO / "results"

# ─── Load mask and missing species ───────────────────────────────────────────
mask = np.load(LOCAL / "in_db_mask.npy")
missing = pd.read_csv(LOCAL / "missing_species.tsv", sep="\t")
print(f"In-DB mask:  N={len(mask)},  True={int(mask.sum())},  False={int((~mask).sum())}")
print(f"Missing species: {len(missing)} entries, sum n_reads = {missing['n_reads'].sum()}")
print()

missing_species_classes = set(missing["species_class"].astype(int).tolist())

# Verify mask alignment by re-deriving from labels of a known prediction file
ref = np.load(RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz")
labels = ref["labels"]
derived_mask = ~np.isin(labels, list(missing_species_classes))
if (derived_mask == mask).all():
    print("✓ Mask alignment verified (matches labels in eval_test100k/predictions.npz)")
else:
    print(f"⚠ Mask mismatch: derived has {int(derived_mask.sum())} True vs given {int(mask.sum())}")
print()

# ─── Models to evaluate ──────────────────────────────────────────────────────
MODELS = [
    ("NT-Species flat",              RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz"),
    ("NT-Species hier (NT-Genus router)", RESULTS / "nt_token_species_v4_50M/eval_hier_stream_test100k/predictions.npz"),
    ("NT-v2 per-genus (predicted)",  LOCAL / "predictions_predicted.npz"),
    ("NT-v2 per-genus (ORACLE)",     LOCAL / "predictions_oracle.npz"),
    ("MT 6-mer flat",                RESULTS / "mt_6mer_species_flat/mt_6mer_species_flat_preds_100K.npz"),
    ("MT 6-mer hier.",               RESULTS / "mt_6mer_hierarchical/mt_6mer_hierarchical_preds_100K.npz"),
    ("MT 13-mer flat",               RESULTS / "mt_species_flat/mt_species_flat_preds_100K.npz"),
    ("MT 13-mer hier.",              RESULTS / "mt_hierarchical/mt_hierarchical_preds_100K.npz"),
]

# ─── Compute top-1 on subsets ────────────────────────────────────────────────
print("="*92)
print(f"{'Model':<38} {'All 100K':>12} {'In-DB 85,819':>14} {'OOD 14,181':>12} {'Δ (In-DB – All)':>16}")
print("="*92)

rows = []
for name, path in MODELS:
    if not path.exists():
        print(f"{name:<38} {'MISSING FILE':>12}  ({path})")
        continue
    d = np.load(path)
    preds = d["preds"]
    lbls = d["labels"]
    # Sanity: labels must match ref labels (same test set, same ordering)
    if len(lbls) != len(labels):
        print(f"{name:<38} SHAPE MISMATCH: {len(lbls)} reads (expected {len(labels)})")
        continue
    if not (lbls == labels).all():
        # Not all models share identical label ordering; for the ones from local_predictions
        # and other dirs they should. Print warning if any deviation.
        n_diff = int((lbls != labels).sum())
        print(f"  [warn] {name}: labels differ from ref at {n_diff} positions — proceeding anyway")

    correct = (preds == lbls)
    acc_all    = correct.mean()
    acc_in_db  = correct[mask].mean()
    acc_ood    = correct[~mask].mean()
    delta      = acc_in_db - acc_all

    print(f"{name:<38} {acc_all*100:>11.2f}% {acc_in_db*100:>13.2f}% {acc_ood*100:>11.2f}% {delta*100:>+15.2f} pp")
    rows.append(dict(model=name, all=acc_all, in_db=acc_in_db, ood=acc_ood, delta=delta))

# Add Kraken2 reference row
print("-"*92)
print(f"{'Kraken2 (in-DB, reference)':<38} {'66.23%':>12} {'77.18%':>14} {'0.00%':>12} {'+10.95 pp':>16}")
print(f"{'  Kraken2 (classified-only)':<38} {'94.61%':>12} {'99.33%':>14} {'—':>12}")
print("="*92)
print()

# ─── Per-species accuracy on the 219 missing species ─────────────────────────
print("="*92)
print(f"Per-species accuracy on the 219 OOD species (14,181 reads · Kraken2 = 0%)")
print("="*92)

ood_rows = []
for name, path in MODELS:
    if not path.exists():
        continue
    d = np.load(path)
    preds = d["preds"]
    lbls = d["labels"]
    mask_ood = ~mask
    correct_ood = (preds == lbls)[mask_ood]
    lbls_ood = lbls[mask_ood]
    overall_ood_acc = correct_ood.mean()

    # Per-species
    per_sp = {}
    for sp_id in missing_species_classes:
        sel = (lbls_ood == sp_id)
        n = int(sel.sum())
        if n > 0:
            per_sp[sp_id] = (correct_ood[sel].mean(), n)
    # Summary stats on per-species accuracy
    per_sp_accs = np.array([a for a, _ in per_sp.values()])
    n_nonzero = int((per_sp_accs > 0).sum())
    median_acc = float(np.median(per_sp_accs))
    mean_acc   = float(per_sp_accs.mean())
    max_acc    = float(per_sp_accs.max())

    print(f"{name:<38}  overall OOD acc = {overall_ood_acc*100:6.2f}%   "
          f"species_w/_signal = {n_nonzero:>3}/{len(per_sp)}   "
          f"per-sp mean={mean_acc*100:5.2f}%  median={median_acc*100:5.2f}%  max={max_acc*100:5.2f}%")
    ood_rows.append((name, overall_ood_acc, n_nonzero, len(per_sp), mean_acc, median_acc, max_acc))

print()
print("="*92)
print("Top-10 OOD species with highest NT-Species flat accuracy (illustrative):")
print("="*92)

d = np.load(RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz")
preds = d["preds"]; lbls = d["labels"]
correct = (preds == lbls)
rows_top = []
for sp_id in missing_species_classes:
    sel = (lbls == sp_id) & ~mask
    n = int(sel.sum())
    if n > 0:
        rows_top.append((sp_id, correct[sel].mean(), n))
rows_top.sort(key=lambda x: -x[1])

# Map species_class -> name from missing_species.tsv
name_lookup = dict(zip(missing["species_class"], missing["species_name"]))
genus_lookup = dict(zip(missing["species_class"], missing["genus_name"]))
print(f"{'species_class':>14}  {'species_name':<22}  {'genus':<22}  {'n_reads':>8}  {'NT-Species acc':>14}")
for sp_id, acc, n in rows_top[:10]:
    print(f"{sp_id:>14}  {name_lookup[sp_id]:<22}  {genus_lookup[sp_id]:<22}  {n:>8}  {acc*100:>13.2f}%")

# Save summary to CSV
out_csv = LOCAL / "in_db_coverage_summary.csv"
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
