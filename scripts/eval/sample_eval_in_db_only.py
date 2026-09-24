#!/usr/bin/env python3
"""
Fair-comparison sample-level evaluation on Kraken2 DB-covered reads only.

For each model:
  - Filter the 100K predictions to the 85,819 reads whose true species is in
    the Kraken2 DB (using in_db_mask.npy).
  - Compute read-level Top-1, Pearson r (abundance), Bray-Curtis, ROC AUC,
    Sens@95%spec — all on the same filtered pool.
  - For Kraken2, treat preds==-1 (unclassified) as "lost signal".

Output: in_db_only_sample_metrics.csv
"""

import os
import numpy as np
import pandas as pd
import io, zipfile
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Root of the run outputs this analysis reads (predictions and metrics).
REPO    = Path(os.environ.get("KF_OUT", "."))
LOCAL   = REPO / "local_predictions"
RESULTS = REPO / "results"

mask = np.load(LOCAL / "in_db_mask.npy")
N_DB = int(mask.sum())

# Reference labels (from NT-v2 eval_test100k) and the in-DB labels
ref = np.load(RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz")
ref_labels = ref["labels"]
n_classes = int(ref_labels.max()) + 1  # 1535


def load_preds_labels(npz_path):
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("preds.npy") as f:
            preds = np.load(io.BytesIO(f.read()))
        with zf.open("labels.npy") as f:
            labels = np.load(io.BytesIO(f.read()))
    return preds.astype(np.int64), labels.astype(np.int64)


def pred_counts_safe(sp, n_classes):
    """Bincount that ignores negative preds (Kraken2 unclassified)."""
    v = sp >= 0
    return np.bincount(sp[v], minlength=n_classes)


def sample_metrics(preds, labels, n_classes, rng,
                   reads_per_sample=1000, n_partition=100,
                   n_sparse=200, genera_present=50):
    """
    Partition samples → Pearson r / Spearman / BC.
    Sparse community samples → ROC AUC + sens at fixed spec.
    """
    n = len(preds)
    idx = rng.permutation(n)
    # Partition (overlapping if pool too small)
    r_p, r_s, bcs = [], [], []
    n_partition_actual = min(n_partition, n // reads_per_sample)
    for i in range(n_partition_actual):
        s = idx[i*reads_per_sample:(i+1)*reads_per_sample]
        sp, sl = preds[s], labels[s]
        tc = np.bincount(sl, minlength=n_classes) / reads_per_sample
        pc = pred_counts_safe(sp, n_classes) / reads_per_sample
        if tc.sum() == 0 or pc.sum() == 0:
            continue
        rp, _ = pearsonr(tc, pc)
        rs, _ = spearmanr(tc, pc)
        bc = np.sum(np.abs(tc - pc)) / np.sum(tc + pc) if np.sum(tc + pc) > 0 else float("nan")
        r_p.append(rp); r_s.append(rs); bcs.append(bc)
    r_p, r_s, bcs = np.array(r_p), np.array(r_s), np.array(bcs)

    # Sparse community: pick genera_present out of present-classes, sample reads
    present_classes = np.unique(labels)
    by_class = {c: np.where(labels == c)[0] for c in present_classes}
    n_classes_present = len(present_classes)
    gp = min(genera_present, n_classes_present)

    all_scores, all_truth = [], []
    for _ in range(n_sparse):
        present = rng.choice(present_classes, size=gp, replace=False)
        present_set = set(present.tolist())
        avail = np.concatenate([by_class[c] for c in present])
        chosen = rng.choice(avail, size=reads_per_sample, replace=False)
        sp = preds[chosen]; sl = labels[chosen]
        pc = pred_counts_safe(sp, n_classes) / reads_per_sample
        for c in present_classes:
            all_scores.append(pc[c])
            all_truth.append(1 if c in present_set else 0)
    all_scores = np.array(all_scores); all_truth = np.array(all_truth)
    order = np.argsort(-all_scores, kind="stable")
    t = all_truth[order]
    n_pos = t.sum(); n_neg = len(t) - n_pos
    cum_tp = np.cumsum(t)
    tpr = cum_tp / n_pos
    fpr = (np.arange(1, len(t)+1) - cum_tp) / n_neg
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)

    # Sens at 95% spec
    elig = fpr <= 0.05
    sens95 = float(tpr[np.where(elig)[0].max()]) if elig.any() else float("nan")

    return dict(
        pearson_r_mean=float(r_p.mean()), pearson_r_std=float(r_p.std()),
        spearman_r_mean=float(r_s.mean()),
        bray_curtis_mean=float(bcs.mean()), bray_curtis_std=float(bcs.std()),
        roc_auc=float(auc), sens95=sens95,
    )


MODELS = [
    ("NT-Species flat",              RESULTS / "nt_token_species_v4_50M/eval_test100k/predictions.npz"),
    ("NT-Species hier.",             RESULTS / "nt_token_species_v4_50M/eval_hier_stream_test100k/predictions.npz"),
    ("NT-v2 per-genus (predicted)",  LOCAL / "predictions_predicted.npz"),
    ("NT-v2 per-genus (ORACLE)",     LOCAL / "predictions_oracle.npz"),
    ("MT 6-mer flat (aligned)",      LOCAL / "mt_6mer_species_flat_preds_100K_twcc.npz"),
    ("MT 6-mer hier. (aligned)",     LOCAL / "mt_6mer_hierarchical_preds_100K_twcc.npz"),
    ("MT 13-mer flat (aligned)",     LOCAL / "mt_species_flat_preds_100K_twcc.npz"),
    # MT 13-mer hier: Taiwana-2 checkpoint broken (stopped at batch 10000, val_loss=9.29); excluded.
    ("Kraken2 (in-DB)",              LOCAL / "predictions_kraken2_twcc.npz"),
]

rows = []
print(f"{'Model':<32} {'Read Acc':>10} {'Pearson r':>11} {'BC':>7} {'ROC AUC':>9} {'Sens@95':>9}")
print("─"*84)
for name, path in MODELS:
    if not path.exists():
        print(f"{name:<32} (missing file)")
        continue
    preds, labels = load_preds_labels(path)
    # Filter to in-DB reads only
    p_db = preds[mask]; l_db = labels[mask]
    # Read-level acc (treat -1 as wrong if Kraken2)
    acc = (p_db == l_db).mean()
    # Sample-level
    rng = np.random.default_rng(42)
    sm = sample_metrics(p_db, l_db, n_classes, rng,
                        reads_per_sample=1000, n_partition=85,
                        n_sparse=200, genera_present=50)
    print(f"{name:<32} {acc*100:>9.2f}% {sm['pearson_r_mean']:>11.4f} {sm['bray_curtis_mean']:>7.3f} "
          f"{sm['roc_auc']:>9.3f} {sm['sens95']*100:>8.1f}%")
    rows.append(dict(model=name, read_acc=acc, **sm))

# For Kraken2 also report classified-only acc
print()
preds, labels = load_preds_labels(LOCAL / "predictions_kraken2_twcc.npz")
p_db = preds[mask]; l_db = labels[mask]
n_clf = int((p_db >= 0).sum())
acc_clf = (p_db[p_db>=0] == l_db[p_db>=0]).mean()
print(f"Kraken2 classified-only on in-DB pool:  {n_clf}/{len(p_db)} classified, "
      f"accuracy = {acc_clf*100:.2f}%")

df = pd.DataFrame(rows)
out = LOCAL / "in_db_only_sample_metrics.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
