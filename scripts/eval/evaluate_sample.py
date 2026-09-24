#!/usr/bin/env python3
"""
Sample-level evaluation for the genus classifier.

Aggregates per-read predictions into synthetic metagenomic samples to compute
application-relevant metrics:
  - Binary detection (presence/absence) at multiple abundance thresholds
  - Relative abundance estimation accuracy
  - Sensitivity / specificity / precision curves

Input: predictions_rc_tta.npz (or predictions.npz) from evaluate.py
       → contains 'preds' [N] int64 and 'labels' [N] int64

Usage:
    python scripts/evaluate_sample.py \\
        --predictions results/nt_token_genus_lora_v9_50M/eval_rc_tta/predictions_rc_tta.npz \\
        --report     results/nt_token_genus_lora_v9_50M/eval_rc_tta/classification_report.csv \\
        --out_dir    results/nt_token_genus_lora_v9_50M/eval_sample_level
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ──────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────

def load_preds_labels(npz_path: str):
    """Load only preds and labels from an npz file without loading large logit arrays."""
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("preds.npy") as f:
            preds = np.load(io.BytesIO(f.read()))
        with zf.open("labels.npy") as f:
            labels = np.load(io.BytesIO(f.read()))
    return preds, labels


def load_genus_names(report_csv: str, n_classes: int):
    """Return genus names in class-index order, or numeric strings if unavailable."""
    if report_csv and os.path.isfile(report_csv):
        df = pd.read_csv(report_csv, index_col=0)
        # Drop sklearn summary rows
        summary_rows = {"accuracy", "macro avg", "weighted avg"}
        names = [r for r in df.index if r not in summary_rows]
        if len(names) == n_classes:
            return names
    return [str(i) for i in range(n_classes)]


# ──────────────────────────────────────────────
# Sample construction
# ──────────────────────────────────────────────

def random_partition_samples(preds, labels, n_samples: int, reads_per_sample: int, rng):
    """
    Randomly partition the reads into n_samples non-overlapping samples.
    Each sample contains reads from all genera that have data in that slice.
    Returns list of (sample_preds, sample_labels).
    """
    n = len(preds)
    idx = rng.permutation(n)
    samples = []
    for i in range(n_samples):
        start = i * reads_per_sample
        end   = start + reads_per_sample
        if end > n:
            break
        s_idx = idx[start:end]
        samples.append((preds[s_idx], labels[s_idx]))
    return samples


def sparse_community_samples(
    preds, labels, n_classes: int,
    n_samples: int, reads_per_sample: int,
    genera_present: int, rng
):
    """
    Build samples where only `genera_present` genera (out of n_classes) are present.
    For each sample, select a random subset of genera, then sample reads from those
    genera only. The absent genera serve as true negatives for detection analysis.

    Returns list of (sample_preds, sample_labels, present_set, absent_set).
    """
    # Build per-genus index lists for fast sampling
    by_genus = {g: np.where(labels == g)[0] for g in range(n_classes)}

    samples = []
    for _ in range(n_samples):
        present = rng.choice(n_classes, size=genera_present, replace=False)
        present_set = set(present.tolist())
        absent_set  = set(range(n_classes)) - present_set

        # Sample reads_per_sample reads uniformly from the present genera
        available = np.concatenate([by_genus[g] for g in present])
        chosen    = rng.choice(available, size=reads_per_sample, replace=False)
        samples.append((preds[chosen], labels[chosen], present_set, absent_set))

    return samples


# ──────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────

def pred_counts(sample_preds, n_classes: int):
    return np.bincount(sample_preds, minlength=n_classes)


def true_counts(sample_labels, n_classes: int):
    return np.bincount(sample_labels, minlength=n_classes)


def abundance_metrics_one_sample(sp, sl, n_classes: int):
    """
    Pearson/Spearman correlation and Bray-Curtis dissimilarity for one sample.
    """
    tc = true_counts(sl, n_classes) / len(sl)
    pc = pred_counts(sp, n_classes) / len(sp)
    r_p, _  = pearsonr(tc, pc)
    r_s, _  = spearmanr(tc, pc)
    bc = np.sum(np.abs(tc - pc)) / np.sum(tc + pc)
    return r_p, r_s, bc


def detection_at_threshold(tc, pc, n_total, threshold_frac):
    """
    For a single sample, compute TP/FP/TN/FN for each class given an abundance threshold.
    threshold_frac: minimum TRUE fraction for a genus to be called "truly present".
    Detection criterion: any predicted reads (pred_count >= 1) is the detection signal.
    Returns per-class (tp, fp, tn, fn) arrays.
    """
    truly_present = (tc / n_total) >= threshold_frac  # [n_classes] bool
    detected      = pc >= 1                            # [n_classes] bool

    tp = ( truly_present &  detected).sum()
    fp = (~truly_present &  detected).sum()
    tn = (~truly_present & ~detected).sum()
    fn = ( truly_present & ~detected).sum()
    return tp, fp, tn, fn


def detection_metrics_vs_count_threshold(tc, pc, n_total):
    """
    Sweep detection threshold (min predicted reads) and true presence threshold
    fixed at ≥1 read.  Returns a dict with arrays for each threshold value.
    """
    truly_present = tc >= 1
    results = {}
    for det_thresh in [1, 2, 5, 10, 20, 50]:
        detected = pc >= det_thresh
        tp = int(( truly_present &  detected).sum())
        fp = int((~truly_present &  detected).sum())
        tn = int((~truly_present & ~detected).sum())
        fn = int(( truly_present & ~detected).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        results[det_thresh] = dict(tp=tp, fp=fp, tn=tn, fn=fn,
                                   sensitivity=sens, specificity=spec, precision=prec)
    return results


def detection_metrics_vs_abundance_threshold(tc, pc, n_total):
    """
    Both detection AND true-presence thresholds are expressed as relative abundance.
    A genus is 'truly present' if its true fraction >= true_thresh.
    A genus is 'detected' if its predicted fraction >= true_thresh (same threshold).
    Sweeps over abundance thresholds relevant to metagenomics practice.
    """
    results = {}
    for pct_thresh in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        frac = pct_thresh / 100.0
        truly_present = (tc / n_total) >= frac if frac > 0 else tc >= 1
        detected      = (pc / n_total) >= frac if frac > 0 else pc >= 1
        tp = int(( truly_present &  detected).sum())
        fp = int((~truly_present &  detected).sum())
        tn = int((~truly_present & ~detected).sum())
        fn = int(( truly_present & ~detected).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        label = f">={pct_thresh:.2f}%"
        results[label] = dict(
            threshold_pct=pct_thresh,
            sensitivity=round(sens, 5), specificity=round(spec, 5), precision=round(prec, 5),
            tp=tp, fp=fp, tn=tn, fn=fn,
        )
    return results


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_abundance_scatter(true_abund_list, pred_abund_list, out_path, task_label="genus", exp_name=""):
    """One dot per (sample, class) pair showing true vs predicted abundance."""
    true_all = np.concatenate(true_abund_list)
    pred_all = np.concatenate(pred_abund_list)
    # Subsample for visual clarity
    if len(true_all) > 20000:
        idx = np.random.choice(len(true_all), 20000, replace=False)
        true_all, pred_all = true_all[idx], pred_all[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_all, pred_all, s=2, alpha=0.3, color="steelblue")
    lim = max(true_all.max(), pred_all.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="perfect")
    ax.set_xlabel("True relative abundance")
    ax.set_ylabel("Predicted relative abundance")
    title = f"{task_label.capitalize()}-level relative abundance estimation"
    if exp_name:
        title = f"{exp_name}\n{title}"
    ax.set_title(title)
    r_p, _ = pearsonr(true_all, pred_all)
    ax.text(0.05, 0.92, f"Pearson r = {r_p:.3f}", transform=ax.transAxes,
            fontsize=10, color="navy")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_detection_curves(agg_by_thresh, out_path, task_label="genus", exp_name=""):
    """
    Bar chart: sensitivity, specificity, precision at each detection threshold
    (min predicted reads required to call a genus present).
    """
    thresholds = sorted(agg_by_thresh.keys())
    sens = [agg_by_thresh[t]["sensitivity"] for t in thresholds]
    spec = [agg_by_thresh[t]["specificity"] for t in thresholds]
    prec = [agg_by_thresh[t]["precision"]   for t in thresholds]

    x   = np.arange(len(thresholds))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, sens, w, label="Sensitivity (recall)", color="#4C72B0")
    ax.bar(x,     spec, w, label="Specificity",           color="#55A868")
    ax.bar(x + w, prec, w, label="Precision",             color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{t}" for t in thresholds])
    ax.set_xlabel("Detection threshold (min predicted reads per sample)")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    title = f"Sample-level {task_label} detection (sparse-community samples)"
    if exp_name:
        title = f"{exp_name}  —  {title}"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sensitivity_by_abundance(sens_by_abund, out_path, exp_name=""):
    """
    Sensitivity as a function of true relative abundance bin.
    Shows that low read-level accuracy → high detection sensitivity for
    genera at ≥0.1% abundance.
    """
    bins    = sorted(sens_by_abund.keys())
    means   = [sens_by_abund[b]["mean"]  for b in bins]
    counts  = [sens_by_abund[b]["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(bins)), means, "o-", color="steelblue", lw=2)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b for b in bins], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Detection sensitivity")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("True relative abundance bin")
    title = "Detection sensitivity by true abundance\n(detection threshold: ≥1 predicted read)"
    if exp_name:
        title = f"{exp_name}  —  {title}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Annotate with counts
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.annotate(f"n={c}", (i, m + 0.02), ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_roc_curve(sparse_samples, n_classes: int):
    """
    ROC curve for genus presence/absence detection using sparse-community samples.

    Ground truth: binary present_set / absent_set from sample construction.
    Score: predicted fraction (pred_count / reads_per_sample).
    Sweeps detection threshold from high to low to build the ROC curve.

    Returns:
        fpr_arr:    array of false-positive rates (1 - specificity)
        tpr_arr:    array of true-positive rates (sensitivity)
        thresholds: array of predicted-fraction thresholds
        auc:        area under the ROC curve
        ops:        dict of operating points {spec: (threshold, sensitivity)}
    """
    # Collect (is_present, pred_frac) for every (sample, genus) pair
    labels_list, scores_list = [], []
    for sp, sl, present_set, absent_set in sparse_samples:
        n = len(sp)
        pc = np.bincount(sp, minlength=n_classes)
        for g in range(n_classes):
            labels_list.append(1 if g in present_set else 0)
            scores_list.append(pc[g] / n)

    y_true = np.array(labels_list, dtype=np.int32)
    y_score = np.array(scores_list, dtype=np.float64)

    # Sort by score descending
    order = np.argsort(-y_score)
    y_true_s = y_true[order]
    y_score_s = y_score[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    # Build ROC by sweeping threshold
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend (0, 0) to close the curve
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[np.inf], y_score_s])

    # AUC via trapezoidal rule
    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))

    # Operating points at fixed specificities
    target_specs = [0.80, 0.90, 0.95, 0.99]
    ops = {}
    for sp_target in target_specs:
        fpr_target = 1.0 - sp_target
        # Find last index where fpr <= fpr_target
        idx = np.searchsorted(fpr, fpr_target, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        ops[sp_target] = {
            "threshold": float(thresholds[idx]),
            "sensitivity": float(tpr[idx]),
            "specificity": float(1.0 - fpr[idx]),
        }

    return fpr, tpr, thresholds, auc, ops


def plot_roc_curve(fpr, tpr, auc, ops, out_path, task_label="genus", exp_name=""):
    """ROC curve for genus binary detection with fixed-ground-truth."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random")

    # Mark key operating points
    colors = {0.95: "red", 0.90: "orange", 0.80: "green"}
    for sp_target, info in ops.items():
        if sp_target not in colors:
            continue
        fp = 1.0 - info["specificity"]
        tp = info["sensitivity"]
        ax.scatter([fp], [tp], s=60, zorder=5, color=colors[sp_target])
        ax.annotate(
            f"Spec={sp_target:.0%}\nSens={tp:.1%}",
            xy=(fp, tp), xytext=(fp + 0.04, tp - 0.06),
            fontsize=7, color=colors[sp_target],
            arrowprops=dict(arrowstyle="-", color=colors[sp_target], lw=0.8),
        )

    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity)")
    title = (f"ROC — {task_label.capitalize()} Presence/Absence Detection\n"
             "(sparse-community samples, binary ground truth)")
    if exp_name:
        title = f"{exp_name}  —  {title}"
    ax.set_title(title)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="Path to predictions_rc_tta.npz or predictions.npz")
    ap.add_argument("--report", default=None,
                    help="classification_report.csv for genus names (optional)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_partition_samples", type=int, default=100,
                    help="Number of random-partition samples for abundance analysis")
    ap.add_argument("--reads_per_sample", type=int, default=50000,
                    help="Reads per synthetic sample")
    ap.add_argument("--n_sparse_samples", type=int, default=200,
                    help="Number of sparse-community samples for detection analysis")
    ap.add_argument("--genera_present", type=int, default=60,
                    help="Number of genera present in each sparse-community sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name", default=None,
                    help="Experiment label for plot titles. Auto-derived from out_dir if not set.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load data ──
    print(f"Loading predictions from {args.predictions} …")
    preds, labels = load_preds_labels(args.predictions)
    n_classes = int(labels.max()) + 1
    task_label = "genus" if n_classes <= 200 else "species"
    # Derive experiment name from out_dir if not provided
    # e.g. results/mt_genus_13mer/eval_sample_level → "mt_genus_13mer"
    if args.exp_name:
        exp_name = args.exp_name
    else:
        parts = Path(args.out_dir).parts
        res_idx = next((i for i, p in enumerate(parts) if p == "results"), None)
        exp_name = parts[res_idx + 1] if res_idx is not None and res_idx + 1 < len(parts) else Path(args.out_dir).parent.name

    print(f"  {len(preds):,} reads, {n_classes} {task_label}s  [{exp_name}]")
    print(f"  Read-level accuracy: {(preds == labels).mean():.4f}")

    genus_names = load_genus_names(args.report, n_classes)

    # ══════════════════════════════════════════
    # Part 1: Random-partition → abundance metrics
    # ══════════════════════════════════════════
    print(f"\nPart 1: random-partition samples "
          f"(n={args.n_partition_samples}, size={args.reads_per_sample:,}) …")

    partition_samples = random_partition_samples(
        preds, labels,
        n_samples=args.n_partition_samples,
        reads_per_sample=args.reads_per_sample,
        rng=rng,
    )

    pearson_vals, spearman_vals, bc_vals = [], [], []
    true_abund_list, pred_abund_list = [], []

    for sp, sl in partition_samples:
        n = len(sp)
        tc = true_counts(sl, n_classes)
        pc = pred_counts(sp, n_classes)
        true_abund_list.append(tc / n)
        pred_abund_list.append(pc / n)
        r_p, r_s, bc = abundance_metrics_one_sample(sp, sl, n_classes)
        pearson_vals.append(r_p)
        spearman_vals.append(r_s)
        bc_vals.append(bc)

    abund_results = {
        "pearson_r_mean":   float(np.nanmean(pearson_vals)),
        "pearson_r_std":    float(np.nanstd(pearson_vals)),
        "pearson_r_median": float(np.nanmedian(pearson_vals)),
        "spearman_r_mean":  float(np.nanmean(spearman_vals)),
        "spearman_r_std":   float(np.nanstd(spearman_vals)),
        "bray_curtis_mean": float(np.nanmean(bc_vals)),
        "bray_curtis_std":  float(np.nanstd(bc_vals)),
        "n_samples":        len(partition_samples),
        "reads_per_sample": args.reads_per_sample,
    }
    print(f"  Abundance Pearson r  = {abund_results['pearson_r_mean']:.4f} "
          f"± {abund_results['pearson_r_std']:.4f}")
    print(f"  Abundance Spearman r = {abund_results['spearman_r_mean']:.4f} "
          f"± {abund_results['spearman_r_std']:.4f}")
    print(f"  Bray-Curtis dissim.  = {abund_results['bray_curtis_mean']:.4f} "
          f"± {abund_results['bray_curtis_std']:.4f}")

    plot_abundance_scatter(
        true_abund_list, pred_abund_list,
        os.path.join(args.out_dir, "abundance_scatter.png"),
        task_label=task_label, exp_name=exp_name,
    )

    # ══════════════════════════════════════════
    # Part 2: Sparse-community → detection metrics
    # ══════════════════════════════════════════
    print(f"\nPart 2: sparse-community samples "
          f"(n={args.n_sparse_samples}, size={args.reads_per_sample:,}, "
          f"genera_present={args.genera_present}/{n_classes}) …")

    # Each sparse sample draws reads_per_sample reads from genera_present genera.
    # Feasibility: need at least reads_per_sample/genera_present reads per genus.
    min_per_genus = min((labels == g).sum() for g in range(n_classes))
    max_rps = int(min_per_genus) * args.genera_present
    rps = min(args.reads_per_sample, max_rps)
    if rps < args.reads_per_sample:
        print(f"  [warn] reads_per_sample capped to {rps:,} (min genus size={min_per_genus})")

    sparse_samples = sparse_community_samples(
        preds, labels, n_classes,
        n_samples=args.n_sparse_samples,
        reads_per_sample=rps,
        genera_present=args.genera_present,
        rng=rng,
    )

    # Aggregate detection metrics across all samples
    agg_thresh: dict[int, dict] = {}
    agg_abund_thresh: dict[str, dict] = {}
    # Per-genus sensitivity aggregated by true abundance bucket
    abund_buckets = {
        "<0.01%":  dict(tp=0, fn=0),
        "0.01–0.1%": dict(tp=0, fn=0),
        "0.1–1%":  dict(tp=0, fn=0),
        "1–5%":    dict(tp=0, fn=0),
        ">5%":     dict(tp=0, fn=0),
    }

    for sp, sl, present_set, absent_set in sparse_samples:
        n = len(sp)
        tc = true_counts(sl, n_classes)
        pc = pred_counts(sp, n_classes)

        # Detection metrics at each prediction-count threshold
        m = detection_metrics_vs_count_threshold(tc, pc, n)
        for thresh, vals in m.items():
            if thresh not in agg_thresh:
                agg_thresh[thresh] = dict(tp=0, fp=0, tn=0, fn=0)
            for k in ("tp", "fp", "tn", "fn"):
                agg_thresh[thresh][k] += vals[k]

        # Detection metrics at each relative-abundance threshold
        m2 = detection_metrics_vs_abundance_threshold(tc, pc, n)
        for label, vals in m2.items():
            if label not in agg_abund_thresh:
                agg_abund_thresh[label] = dict(tp=0, fp=0, tn=0, fn=0)
            for k in ("tp", "fp", "tn", "fn"):
                agg_abund_thresh[label][k] += vals[k]

        # Sensitivity by abundance bucket (using detection threshold ≥1 read)
        for g in range(n_classes):
            frac = tc[g] / n if n > 0 else 0.0
            detected = pc[g] >= 1
            if frac == 0:
                continue  # truly absent, skip sensitivity
            if frac < 0.0001:
                bucket = "<0.01%"
            elif frac < 0.001:
                bucket = "0.01–0.1%"
            elif frac < 0.01:
                bucket = "0.1–1%"
            elif frac < 0.05:
                bucket = "1–5%"
            else:
                bucket = ">5%"
            abund_buckets[bucket]["tp"] += int(detected)
            abund_buckets[bucket]["fn"] += int(not detected)

    # Compute aggregate sensitivity/specificity/precision per threshold
    detection_results = {}
    print("\n  Detection metrics (sparse-community, aggregated over all samples):")
    print(f"  {'Threshold':>12}  {'Sensitivity':>11}  {'Specificity':>11}  {'Precision':>9}  "
          f"{'TP':>6}  {'FP':>6}  {'FN':>6}")
    for thresh in sorted(agg_thresh.keys()):
        d = agg_thresh[thresh]
        sens = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else float("nan")
        spec = d["tn"] / (d["tn"] + d["fp"]) if (d["tn"] + d["fp"]) > 0 else float("nan")
        prec = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else float("nan")
        detection_results[f">={thresh}_reads"] = dict(
            threshold_reads=thresh,
            sensitivity=round(sens, 5), specificity=round(spec, 5), precision=round(prec, 5),
            tp=d["tp"], fp=d["fp"], tn=d["tn"], fn=d["fn"],
        )
        print(f"  ≥{thresh:>11d}  {sens:>11.4f}  {spec:>11.4f}  {prec:>9.4f}  "
              f"{d['tp']:>6d}  {d['fp']:>6d}  {d['fn']:>6d}")

    # Sensitivity by abundance bucket
    sens_by_abund = {}
    print("\n  Sensitivity by true relative abundance (detection threshold: ≥1 read):")
    bucket_order = ["<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", ">5%"]
    for b in bucket_order:
        d = abund_buckets[b]
        total = d["tp"] + d["fn"]
        s = d["tp"] / total if total > 0 else float("nan")
        sens_by_abund[b] = {"mean": round(s, 5), "count": total}
        print(f"    {b:>10s}: sensitivity={s:.4f}  (n={total} genus-instances)")

    # Abundance-threshold detection results
    abund_det_results = {}
    print("\n  Detection metrics (abundance threshold — same threshold for presence AND detection):")
    print(f"  {'Threshold':>10}  {'Sensitivity':>11}  {'Specificity':>11}  {'Precision':>9}  "
          f"{'TP':>6}  {'FP':>6}  {'FN':>6}")
    for label in sorted(agg_abund_thresh.keys(),
                        key=lambda x: float(x.replace(">=", "").replace("%", ""))):
        d = agg_abund_thresh[label]
        sens = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else float("nan")
        spec = d["tn"] / (d["tn"] + d["fp"]) if (d["tn"] + d["fp"]) > 0 else float("nan")
        prec = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else float("nan")
        abund_det_results[label] = dict(
            sensitivity=round(sens, 5), specificity=round(spec, 5), precision=round(prec, 5),
            tp=d["tp"], fp=d["fp"], tn=d["tn"], fn=d["fn"],
        )
        print(f"  {label:>10s}  {sens:>11.4f}  {spec:>11.4f}  {prec:>9.4f}  "
              f"{d['tp']:>6d}  {d['fp']:>6d}  {d['fn']:>6d}")

    plot_detection_curves(
        {t: dict(
            sensitivity=detection_results[k]["sensitivity"],
            specificity=detection_results[k]["specificity"],
            precision=detection_results[k]["precision"],
         )
         for k, t in [(f">={t}_reads", t) for t in sorted(agg_thresh.keys())]},
        os.path.join(args.out_dir, "detection_curves.png"),
        task_label=task_label, exp_name=exp_name,
    )
    plot_sensitivity_by_abundance(
        sens_by_abund,
        os.path.join(args.out_dir, "sensitivity_by_abundance.png"),
        exp_name=exp_name,
    )

    # ──────────────────────────────────────────
    # ROC analysis (fixed binary ground truth, sweep detection threshold)
    # ──────────────────────────────────────────
    print("\n  ROC analysis (binary ground truth from sparse-community construction) …")
    roc_fpr, roc_tpr, roc_thresh, roc_auc, roc_ops = compute_roc_curve(sparse_samples, n_classes)
    print(f"  AUC = {roc_auc:.4f}")
    for sp_tgt, info in sorted(roc_ops.items()):
        print(f"    Spec ≥ {sp_tgt:.0%}:  sensitivity = {info['sensitivity']:.4f}  "
              f"(threshold = {info['threshold']:.5f})")
    plot_roc_curve(
        roc_fpr, roc_tpr, roc_auc, roc_ops,
        os.path.join(args.out_dir, "roc_detection.png"),
        task_label=task_label, exp_name=exp_name,
    )

    roc_results = {
        "auc": round(roc_auc, 5),
        "operating_points": {
            f"spec_{int(sp*100)}pct": {
                "target_specificity": sp,
                "achieved_specificity": round(info["specificity"], 5),
                "sensitivity": round(info["sensitivity"], 5),
                "threshold_pred_frac": round(info["threshold"], 6),
            }
            for sp, info in roc_ops.items()
        },
    }

    # ══════════════════════════════════════════
    # Part 3: Per-genus recall summary
    # ══════════════════════════════════════════
    print("\nPart 3: per-genus recall summary …")
    per_genus = []
    for g in range(n_classes):
        mask = labels == g
        n_g  = mask.sum()
        sens = (preds[mask] == g).mean() if n_g > 0 else float("nan")
        per_genus.append({
            "genus": genus_names[g],
            "n_reads": int(n_g),
            "recall": round(float(sens), 5),
        })
    per_genus_df = pd.DataFrame(per_genus).sort_values("recall")
    per_genus_df.to_csv(os.path.join(args.out_dir, "per_genus_recall.csv"), index=False)
    print(f"  Bottom-5 recall genera:")
    print(per_genus_df.head(5).to_string(index=False))
    print(f"  Top-5 recall genera:")
    print(per_genus_df.tail(5).to_string(index=False))

    # ══════════════════════════════════════════
    # Save combined results
    # ══════════════════════════════════════════
    results = {
        "read_level_accuracy": float((preds == labels).mean()),
        "n_reads_total": int(len(preds)),
        "n_genera": n_classes,
        "abundance_estimation": abund_results,
        "binary_detection_by_count_threshold": detection_results,
        "binary_detection_by_abundance_threshold": abund_det_results,
        "sensitivity_by_abundance": sens_by_abund,
        "roc_detection": roc_results,
        "sample_config": {
            "reads_per_sample": rps,
            "n_partition_samples": len(partition_samples),
            "n_sparse_samples": args.n_sparse_samples,
            "genera_present_per_sparse_sample": args.genera_present,
        },
    }
    out_json = os.path.join(args.out_dir, "sample_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
