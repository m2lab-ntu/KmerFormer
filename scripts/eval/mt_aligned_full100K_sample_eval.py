#!/usr/bin/env python3
"""
Compute sample-level metrics on FULL 100K test set (no in-DB restriction) for the
NEWLY ALIGNED MT predictions. These replace the older thesis table values which
were computed on MT's own val subset (a different 100K reads).
"""

import os
import io, zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

LOCAL   = Path(os.environ.get("KF_OUT", ".")) / "local_predictions"

def load_pl(p):
    with zipfile.ZipFile(p) as zf:
        with zf.open("preds.npy") as f:
            preds = np.load(io.BytesIO(f.read()))
        with zf.open("labels.npy") as f:
            labels = np.load(io.BytesIO(f.read()))
    return preds.astype(np.int64), labels.astype(np.int64)


def pred_counts_safe(sp, n):
    v = sp >= 0
    return np.bincount(sp[v], minlength=n)


def sample_eval(preds, labels, n_classes, rng,
                rps=1000, n_part=100, n_sparse=200, gp=50):
    n = len(preds); idx = rng.permutation(n)
    r_p, r_s, bcs = [], [], []
    for i in range(min(n_part, n // rps)):
        s = idx[i*rps:(i+1)*rps]
        sp, sl = preds[s], labels[s]
        tc = np.bincount(sl, minlength=n_classes) / rps
        pc = pred_counts_safe(sp, n_classes) / rps
        if tc.sum() == 0 or pc.sum() == 0:
            continue
        rp, _ = pearsonr(tc, pc); rs, _ = spearmanr(tc, pc)
        bc = np.sum(np.abs(tc - pc)) / np.sum(tc + pc) if np.sum(tc+pc) > 0 else float("nan")
        r_p.append(rp); r_s.append(rs); bcs.append(bc)
    r_p, r_s, bcs = np.array(r_p), np.array(r_s), np.array(bcs)

    present = np.unique(labels)
    by_c = {c: np.where(labels == c)[0] for c in present}
    gp_eff = min(gp, len(present))
    scores, truth = [], []
    for _ in range(n_sparse):
        pr = rng.choice(present, size=gp_eff, replace=False)
        pset = set(pr.tolist())
        avail = np.concatenate([by_c[c] for c in pr])
        chosen = rng.choice(avail, size=rps, replace=False)
        sp = preds[chosen]
        pc = pred_counts_safe(sp, n_classes) / rps
        for c in present:
            scores.append(pc[c]); truth.append(1 if c in pset else 0)
    scores, truth = np.array(scores), np.array(truth)
    order = np.argsort(-scores, kind="stable")
    t = truth[order]; n_pos = t.sum(); n_neg = len(t) - n_pos
    cum = np.cumsum(t); tpr = cum / n_pos
    fpr = (np.arange(1, len(t)+1) - cum) / n_neg
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    elig = fpr <= 0.05
    sens95 = float(tpr[np.where(elig)[0].max()]) if elig.any() else float("nan")
    return dict(pearson_r=float(r_p.mean()),
                spearman_r=float(r_s.mean()),
                bc=float(bcs.mean()),
                roc_auc=float(auc), sens95=sens95)


# ─── Species level (full 100K) ──────────────────────────────────────────
print("="*80)
print("MT species predictions on FULL 100K (aligned, replacing older thesis values)")
print("="*80)
print(f"{'Model':<35} {'Read':>7} {'r':>7} {'ρ':>7} {'BC':>7} {'AUC':>7} {'S@95':>7}")
print("-"*80)
for tag, fn in [
    ("MT 13-mer flat (aligned)",    "mt_species_flat_preds_100K_twcc.npz"),
    ("MT 6-mer flat (aligned)",     "mt_6mer_species_flat_preds_100K_twcc.npz"),
    ("MT 6-mer hier. (aligned)",    "mt_6mer_hierarchical_preds_100K_twcc.npz"),
]:
    preds, labels = load_pl(LOCAL / fn)
    n_classes = int(labels.max()) + 1
    rng = np.random.default_rng(42)
    sm = sample_eval(preds, labels, n_classes, rng,
                     rps=1000, n_part=100, n_sparse=200, gp=50)
    acc = (preds == labels).mean()
    print(f"{tag:<35} {acc*100:>6.2f}% {sm['pearson_r']:>7.3f} {sm['spearman_r']:>7.3f} "
          f"{sm['bc']:>7.3f} {sm['roc_auc']:>7.3f} {sm['sens95']*100:>6.1f}%")

# ─── Genus level (full 100K) ────────────────────────────────────────────
print()
print("="*80)
print("MT genus predictions on FULL 100K (aligned)")
print("="*80)
print(f"{'Model':<35} {'Read':>7} {'r':>7} {'ρ':>7} {'BC':>7} {'AUC':>7} {'S@95':>7}")
print("-"*80)
for tag, fn in [
    ("MT 13-mer genus (aligned)",   "mt_genus_13mer_preds_100K_twcc.npz"),
    ("MT 6-mer genus (aligned)",    "mt_genus_6mer_preds_100K_twcc.npz"),
]:
    preds, labels = load_pl(LOCAL / fn)
    n_classes = int(labels.max()) + 1
    rng = np.random.default_rng(42)
    sm = sample_eval(preds, labels, n_classes, rng,
                     rps=1000, n_part=100, n_sparse=200, gp=50)
    acc = (preds == labels).mean()
    print(f"{tag:<35} {acc*100:>6.2f}% {sm['pearson_r']:>7.4f} {sm['spearman_r']:>7.3f} "
          f"{sm['bc']:>7.3f} {sm['roc_auc']:>7.3f} {sm['sens95']*100:>6.1f}%")
