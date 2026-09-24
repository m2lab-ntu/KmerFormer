#!/usr/bin/env python3
"""
Genus-level fair comparison on in-DB 85,819 reads.

Approach:
  1. Build species_class → genus_class lookup from labels_100K.tsv (NT-v2 space).
  2. For NT-v2 species predictions: remap directly using NT-v2 lookup.
  3. For MT species predictions: first remap MT class → NT class using label
     correspondence (both arrays index the same 100K reads in order), then
     apply NT-v2 lookup.
  4. Apply in_db_mask, compute genus-level metrics.
  5. Kraken2 genus already on 100K test (predictions_kraken2_twcc_genus.npz).
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Root of the run outputs this analysis reads (predictions and metrics).
REPO    = Path(os.environ.get("KF_OUT", "."))
LOCAL   = REPO / "local_predictions"
RESULTS = REPO / "results"

mask = np.load(LOCAL / "in_db_mask.npy")

# ─── NT-v2 species → genus lookup ─────────────────────────────────────────────
df = pd.read_csv(os.path.join(os.environ.get("KF_DATA", "."), "val_100K", "labels_100K_val.tsv"),
                  sep="\t")
nt_sp_to_gn = dict(zip(df["species_class"].astype(int), df["genus_class"].astype(int)))
max_nt_sp = max(nt_sp_to_gn.keys())
nt_lut = np.full(max_nt_sp + 1, -1, dtype=np.int64)
for sp, gn in nt_sp_to_gn.items():
    nt_lut[sp] = gn
n_genus = int(df["genus_class"].max()) + 1
print(f"Genera: {n_genus}, NT species range: 0..{max_nt_sp}")


def load_pl(p):
    with zipfile.ZipFile(p) as zf:
        with zf.open("preds.npy") as f:
            preds = np.load(io.BytesIO(f.read()))
        with zf.open("labels.npy") as f:
            labels = np.load(io.BytesIO(f.read()))
    return preds.astype(np.int64), labels.astype(np.int64)


# Reference NT labels (the "true" species at each read position in NT space)
nt_ref = np.load(RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz")
nt_labels = nt_ref["labels"]
true_genus_labels = nt_lut[nt_labels]  # universal genus labels per read


def remap_nt_species_to_genus(species_preds):
    """NT-v2-style species → genus via NT lookup."""
    safe = np.clip(species_preds, 0, max_nt_sp)
    return np.where((species_preds >= 0) & (species_preds <= max_nt_sp),
                    nt_lut[safe], -1)


def remap_mt_via_label_correspondence(mt_species_preds, mt_species_labels):
    """
    MT class space → NT class space by label-position correspondence,
    then NT lookup → genus.
    """
    # Build MT class → NT class mapping from labels (both index the same reads)
    mt_to_nt = {}
    for i in range(len(mt_species_labels)):
        mt_c = int(mt_species_labels[i]); nt_c = int(nt_labels[i])
        # Multiple reads of same species should give same mapping;
        # first occurrence wins (verified consistent below)
        mt_to_nt.setdefault(mt_c, nt_c)
    # Build LUT
    max_mt = max(mt_to_nt.keys())
    mt_lut = np.full(max_mt + 1, -1, dtype=np.int64)
    for m, n in mt_to_nt.items():
        mt_lut[m] = n
    # Verify consistency
    inconsistencies = 0
    for i in range(min(10000, len(mt_species_labels))):
        if mt_lut[int(mt_species_labels[i])] != int(nt_labels[i]):
            inconsistencies += 1
    if inconsistencies > 0:
        print(f"  [warn] MT label inconsistency in {inconsistencies}/10000 sampled positions")
    # Remap species preds via MT→NT, then NT species→genus
    safe = np.clip(mt_species_preds, 0, max_mt)
    nt_space_preds = np.where((mt_species_preds >= 0) & (mt_species_preds <= max_mt),
                              mt_lut[safe], -1)
    return remap_nt_species_to_genus(nt_space_preds)


def pred_counts_safe(sp, n):
    v = sp >= 0
    return np.bincount(sp[v], minlength=n)


def sample_eval(preds, labels, n_classes, rng,
                rps=1000, n_part=85, n_sparse=200, genera_present=50):
    n = len(preds); idx = rng.permutation(n)
    r_p, r_s, bcs = [], [], []
    n_part_act = min(n_part, n // rps)
    for i in range(n_part_act):
        s = idx[i*rps:(i+1)*rps]
        sp, sl = preds[s], labels[s]
        tc = np.bincount(sl, minlength=n_classes) / rps
        pc = pred_counts_safe(sp, n_classes) / rps
        if tc.sum() == 0 or pc.sum() == 0: continue
        rp, _ = pearsonr(tc, pc); rs, _ = spearmanr(tc, pc)
        bc = np.sum(np.abs(tc - pc)) / np.sum(tc + pc) if np.sum(tc + pc) > 0 else float("nan")
        r_p.append(rp); r_s.append(rs); bcs.append(bc)
    r_p, r_s, bcs = np.array(r_p), np.array(r_s), np.array(bcs)
    # ROC AUC via sparse community
    present_classes = np.unique(labels)
    by_class = {c: np.where(labels == c)[0] for c in present_classes}
    gp = min(genera_present, len(present_classes))
    all_scores, all_truth = [], []
    for _ in range(n_sparse):
        present = rng.choice(present_classes, size=gp, replace=False)
        present_set = set(present.tolist())
        avail = np.concatenate([by_class[c] for c in present])
        chosen = rng.choice(avail, size=rps, replace=False)
        sp = preds[chosen]
        pc = pred_counts_safe(sp, n_classes) / rps
        for c in present_classes:
            all_scores.append(pc[c]); all_truth.append(1 if c in present_set else 0)
    all_scores, all_truth = np.array(all_scores), np.array(all_truth)
    order = np.argsort(-all_scores, kind="stable")
    t = all_truth[order]; n_pos = t.sum(); n_neg = len(t) - n_pos
    cum_tp = np.cumsum(t); tpr = cum_tp / n_pos
    fpr = (np.arange(1, len(t)+1) - cum_tp) / n_neg
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    elig = fpr <= 0.05
    sens95 = float(tpr[np.where(elig)[0].max()]) if elig.any() else float("nan")
    return dict(pearson_r=float(r_p.mean()), bc=float(bcs.mean()),
                roc_auc=float(auc), sens95=sens95)


# ─── Model genus predictions on 100K test (NT-v2 + Kraken2 only) ─────────────
# MT predictions are in different read order than NT-v2 (verified empirically:
# each MT class maps to ~64 different NT classes on average → scrambled order).
# MT genus comparison requires re-running MT inference in TWCC's read order
# or providing seq_id metadata. Skipped here.
genus_preds = {}

for tag, path in [
    ("NT-v2 species→genus (flat)",       RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz"),
    ("NT-v2 species→genus (hier)",       RESULTS / "nt_token_species_v4_50M/eval_hier_stream_test100k/predictions.npz"),
    ("NT-v2 per-genus (predicted)→genus",LOCAL / "predictions_predicted.npz"),
    ("NT-v2 per-genus (ORACLE)→genus",   LOCAL / "predictions_oracle.npz"),
]:
    sp_preds, sp_labels = load_pl(path)
    if not (sp_labels == nt_labels).all():
        n_diff = int((sp_labels != nt_labels).sum())
        print(f"[warn] {tag}: labels differ at {n_diff} positions — skipping")
        continue
    genus_preds[tag] = remap_nt_species_to_genus(sp_preds)

# MT 13-mer / 6-mer genus (aligned, dedicated genus classifiers)
# These are MT genus models trained from scratch — class IDs in MT's own genus space.
# To compare against NT-v2 genus space, build MT-genus-class → NT-genus-class LUT from labels.
def remap_mt_genus_via_correspondence(mt_genus_preds, mt_genus_labels, true_nt_genus_labels):
    mt_to_nt = {}
    for i in range(len(mt_genus_labels)):
        mt_to_nt.setdefault(int(mt_genus_labels[i]), int(true_nt_genus_labels[i]))
    # Verify
    inconsistent = sum(1 for i in range(len(mt_genus_labels))
                       if mt_to_nt[int(mt_genus_labels[i])] != int(true_nt_genus_labels[i]))
    if inconsistent > 0:
        print(f"  [warn] MT genus label inconsistent at {inconsistent} positions")
    max_mt = max(mt_to_nt.keys())
    lut = np.full(max_mt+1, -1, dtype=np.int64)
    for m, n in mt_to_nt.items():
        lut[m] = n
    safe = np.clip(mt_genus_preds, 0, max_mt)
    return np.where((mt_genus_preds >= 0) & (mt_genus_preds <= max_mt), lut[safe], -1)

for tag, fn in [
    ("MT 13-mer genus (dedicated)", "mt_genus_13mer_preds_100K_twcc.npz"),
    ("MT 6-mer genus (dedicated)",  "mt_genus_6mer_preds_100K_twcc.npz"),
]:
    mt_p, mt_l = load_pl(LOCAL / fn)
    genus_preds[tag] = remap_mt_genus_via_correspondence(mt_p, mt_l, true_genus_labels)

# Kraken2 genus (verified same read order)
kp, kl = load_pl(LOCAL / "predictions_kraken2_twcc_genus.npz")
if not (kl == true_genus_labels).all():
    n_diff = int((kl != true_genus_labels).sum())
    print(f"[warn] Kraken2 genus labels differ from NT-v2 mapping at {n_diff} positions")
genus_preds["Kraken2 (genus)"] = kp

# Optional: include dedicated NT-Genus v9 100K test predictions if available
ntg_path = RESULTS / "nt_token_genus_lora_v9_50M/eval_test100k/predictions.npz"
if ntg_path.exists():
    p, l = load_pl(ntg_path)
    if (l == true_genus_labels).all():
        genus_preds["NT-Genus v9 (dedicated)"] = p
        print(f"  Added: NT-Genus v9 (dedicated, 100K test)")
    else:
        print(f"  [warn] NT-Genus v9 labels mismatch; skipping")
else:
    print(f"  Note: NT-Genus v9 100K test predictions not yet available (job 218570 pending)")

# ─── In-DB fair comparison ──────────────────────────────────────────────────
print()
print("="*92)
print(f"GENUS-level fair comparison on in-DB 85,819 reads (read-level + sample-level)")
print("="*92)
print(f"{'Model':<35} {'Read Acc':>10} {'Pearson r':>11} {'BC':>7} {'ROC AUC':>9} {'Sens@95':>9}")
print("-"*92)
rows = []
for name, gp in genus_preds.items():
    gp_db = gp[mask]
    gl_db = true_genus_labels[mask]
    # Read-level acc (treat -1 as wrong)
    acc = (gp_db == gl_db).mean()
    # Sample-level
    rng = np.random.default_rng(42)
    sm = sample_eval(gp_db, gl_db, n_genus, rng)
    print(f"{name:<35} {acc*100:>9.2f}% {sm['pearson_r']:>11.4f} {sm['bc']:>7.3f} "
          f"{sm['roc_auc']:>9.3f} {sm['sens95']*100:>8.1f}%")
    rows.append(dict(model=name, read_acc=acc, **sm))

# Kraken2 classified-only on in-DB pool
kp_db = kp[mask]; kl_db = true_genus_labels[mask]
n_clf = int((kp_db >= 0).sum())
acc_clf = (kp_db[kp_db>=0] == kl_db[kp_db>=0]).mean()
print(f"\nKraken2 genus, classified-only on in-DB: {n_clf}/{len(kp_db)} = {n_clf/len(kp_db)*100:.1f}% classified, acc={acc_clf*100:.2f}%")

df_out = pd.DataFrame(rows)
out = LOCAL / "in_db_only_genus_sample_metrics.csv"
df_out.to_csv(out, index=False)
print(f"\nSaved: {out}")
