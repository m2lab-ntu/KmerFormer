#!/usr/bin/env python3
"""
Sample-level evaluation specifically for Kraken2 predictions, where
preds may contain -1 (unclassified).

Treatment: unclassified reads are "lost signal" — they contribute to
neither true nor predicted counts for any of the 1535 real species,
but they DO count toward the sample size denominator (so a sample with
30% unclassified has lower total predicted abundance, naturally
reflecting Kraken2's commit rate).

This is the most honest framing: Kraken2's predicted abundance for a
real species is (count of reads classified as that species) / 1000,
where 1000 includes the unclassified ones.
"""

import argparse
import json
import io
import zipfile
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


def load_preds_labels(npz_path: str):
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("preds.npy") as f:
            preds = np.load(io.BytesIO(f.read()))
        with zf.open("labels.npy") as f:
            labels = np.load(io.BytesIO(f.read()))
    return preds.astype(np.int64), labels.astype(np.int64)


def pred_counts_safe(sample_preds, n_classes):
    """Bincount that ignores negative predictions (treats them as lost signal)."""
    valid = sample_preds >= 0
    return np.bincount(sample_preds[valid], minlength=n_classes)


def true_counts(sample_labels, n_classes):
    return np.bincount(sample_labels, minlength=n_classes)


def abundance_metrics_one_sample(sp, sl, n_classes, reads_per_sample):
    tc = true_counts(sl, n_classes) / reads_per_sample
    pc = pred_counts_safe(sp, n_classes) / reads_per_sample
    r_p, _ = pearsonr(tc, pc)
    r_s, _ = spearmanr(tc, pc)
    bc = np.sum(np.abs(tc - pc)) / np.sum(tc + pc) if np.sum(tc + pc) > 0 else float("nan")
    return r_p, r_s, bc


def random_partition_samples(preds, labels, n_samples, reads_per_sample, rng):
    n = len(preds)
    idx = rng.permutation(n)
    samples = []
    for i in range(n_samples):
        start = i * reads_per_sample
        end = start + reads_per_sample
        if end > n:
            break
        s_idx = idx[start:end]
        samples.append((preds[s_idx], labels[s_idx]))
    return samples


def sparse_community_samples(preds, labels, n_classes, n_samples,
                             reads_per_sample, genera_present, rng):
    by_genus = {g: np.where(labels == g)[0] for g in range(n_classes)}
    samples = []
    for _ in range(n_samples):
        present = rng.choice(n_classes, size=genera_present, replace=False)
        present_set = set(present.tolist())
        absent_set = set(range(n_classes)) - present_set
        available = np.concatenate([by_genus[g] for g in present])
        chosen = rng.choice(available, size=reads_per_sample, replace=False)
        samples.append((preds[chosen], labels[chosen], present_set, absent_set))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reads_per_sample", type=int, default=1000)
    ap.add_argument("--n_partition_samples", type=int, default=100)
    ap.add_argument("--n_sparse_samples", type=int, default=200)
    ap.add_argument("--genera_present", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    preds, labels = load_preds_labels(args.predictions)
    n_classes = int(labels.max()) + 1
    n = len(preds)
    n_clf = int((preds >= 0).sum())

    print(f"N reads:     {n}")
    print(f"Classified:  {n_clf} ({n_clf/n*100:.2f}%)")
    print(f"N classes:   {n_classes}")
    print(f"Read acc:    {((preds == labels).mean())*100:.2f}%")
    print(f"Read acc on classified: {((preds[preds>=0] == labels[preds>=0]).mean())*100:.2f}%")
    print()

    # Part 1: random partition samples → abundance metrics
    print("Part 1: random partition samples → Pearson r / Bray-Curtis …")
    p_samples = random_partition_samples(preds, labels, args.n_partition_samples,
                                          args.reads_per_sample, rng)
    r_ps, r_ss, bcs = [], [], []
    for sp, sl in p_samples:
        r_p, r_s, bc = abundance_metrics_one_sample(sp, sl, n_classes, args.reads_per_sample)
        r_ps.append(r_p); r_ss.append(r_s); bcs.append(bc)
    r_ps = np.array(r_ps); r_ss = np.array(r_ss); bcs = np.array(bcs)

    print(f"  Pearson r:  mean={r_ps.mean():.4f}  std={r_ps.std():.4f}")
    print(f"  Spearman ρ: mean={r_ss.mean():.4f}  std={r_ss.std():.4f}")
    print(f"  BC:         mean={bcs.mean():.4f}  std={bcs.std():.4f}")
    print()

    # Part 2: sparse community samples → ROC AUC
    print("Part 2: sparse community samples → ROC AUC (species detection) …")
    s_samples = sparse_community_samples(preds, labels, n_classes,
                                          args.n_sparse_samples, args.reads_per_sample,
                                          args.genera_present, rng)
    # Build pooled scores for ROC: pred frac as score, present/absent as labels
    all_scores, all_truth = [], []
    for sp, sl, present_set, absent_set in s_samples:
        pc = pred_counts_safe(sp, n_classes) / args.reads_per_sample
        for c in range(n_classes):
            if c in present_set:
                all_scores.append(pc[c]); all_truth.append(1)
            elif c in absent_set:
                all_scores.append(pc[c]); all_truth.append(0)
    all_scores = np.array(all_scores); all_truth = np.array(all_truth)

    # ROC AUC computation
    order = np.argsort(-all_scores, kind="stable")
    truth_sorted = all_truth[order]
    n_pos = truth_sorted.sum(); n_neg = len(truth_sorted) - n_pos
    cum_tp = np.cumsum(truth_sorted)
    tp_at_threshold = cum_tp / n_pos
    fp_at_threshold = (np.arange(1, len(truth_sorted)+1) - cum_tp) / n_neg
    trap = getattr(np, "trapezoid", None) or np.trapz
    auc = trap(tp_at_threshold, fp_at_threshold)
    print(f"  ROC AUC: {auc:.4f}")

    # Operating points: sensitivity at fixed specificity
    op_points = {}
    for target_spec in [0.80, 0.90, 0.95, 0.99]:
        # specificity = 1 - fp_rate; find largest threshold where 1-fpr >= target
        eligible = fp_at_threshold <= (1 - target_spec)
        if eligible.any():
            i_last = np.where(eligible)[0].max()
            sens = tp_at_threshold[i_last]
            spec = 1 - fp_at_threshold[i_last]
            thresh = all_scores[order[i_last]]
        else:
            sens = float("nan"); spec = float("nan"); thresh = float("nan")
        op_points[f"spec_{int(target_spec*100)}pct"] = dict(
            target_specificity=target_spec, achieved_specificity=float(spec),
            sensitivity=float(sens), threshold_pred_frac=float(thresh))
        print(f"  Spec ≥ {int(target_spec*100)}%: sens = {sens:.4f}  (threshold = {thresh:.5f})")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "read_level_accuracy": float((preds == labels).mean()),
        "read_level_accuracy_classified_only": float((preds[preds>=0] == labels[preds>=0]).mean()),
        "classification_rate": float(n_clf / n),
        "n_reads_total": int(n),
        "n_classified": int(n_clf),
        "n_genera": n_classes,
        "abundance_estimation": dict(
            pearson_r_mean=float(r_ps.mean()), pearson_r_std=float(r_ps.std()),
            pearson_r_median=float(np.median(r_ps)),
            spearman_r_mean=float(r_ss.mean()), spearman_r_std=float(r_ss.std()),
            bray_curtis_mean=float(bcs.mean()), bray_curtis_std=float(bcs.std()),
            n_samples=args.n_partition_samples, reads_per_sample=args.reads_per_sample,
        ),
        "roc_detection": dict(auc=float(auc), operating_points=op_points),
        "sample_config": dict(
            reads_per_sample=args.reads_per_sample,
            n_partition_samples=args.n_partition_samples,
            n_sparse_samples=args.n_sparse_samples,
            genera_present_per_sparse_sample=args.genera_present,
        ),
    }
    with open(out_dir / "sample_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_dir}/sample_metrics.json")


if __name__ == "__main__":
    main()
