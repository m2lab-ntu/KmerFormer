#!/usr/bin/env python3
"""
Fair Kraken2(1535) vs NT/MT comparison on the TWCC 100K pool.

Computes genus read accuracy, abundance (Pearson r / Bray-Curtis), on:
  - full 100K pool (matched reference after rebuilding Kraken with GCF)
  - legacy in_db_mask (85,819) for continuity with the paper table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def bray_curtis(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    denom = (x + y).sum()
    return float(np.abs(x - y).sum() / denom) if denom > 0 else float("nan")


def load_preds(path):
    d = np.load(path)
    return d["preds"].astype(np.int64), d["labels"].astype(np.int64)


def abundance(preds, labels, n_classes, n_total=None):
    """Abundance = counts / n_total (includes abstention as missing mass)."""
    n_total = n_total or len(labels)
    true = np.bincount(labels, minlength=n_classes).astype(float) / n_total
    valid = preds >= 0
    pred = np.bincount(preds[valid], minlength=n_classes).astype(float) / n_total
    return pred, true


def metrics_on_mask(preds, labels, mask, n_classes, name):
    p = preds[mask]
    y = labels[mask]
    n = len(y)
    clf = p >= 0
    top1_all = float((p == y).mean())
    top1_clf = float((p[clf] == y[clf]).mean()) if clf.any() else float("nan")
    pred_ab, true_ab = abundance(p, y, n_classes, n_total=n)
    r = float(pearsonr(pred_ab, true_ab)[0])
    bc = bray_curtis(pred_ab, true_ab)
    return {
        "name": name,
        "n": int(n),
        "classified_rate": float(clf.mean()),
        "read_acc_all": top1_all,
        "read_acc_classified": top1_clf,
        "pearson_r": r,
        "bray_curtis": bc,
    }


def bracken_genus_abundance(bracken_tsv, crosswalk_tsv, n_genera, n_total):
    cw = pd.read_csv(crosswalk_tsv, sep="\t")
    # species_N column may be named species_N; taxid in DB = species_class_val + 2
    # For new DB all 1535 species: rebuild crosswalk from labels if needed
    if "species_N" in cw.columns:
        N2g = dict(zip(cw["species_N"].astype(int), cw["genus_class"].astype(int)))
    else:
        raise SystemExit(f"unexpected crosswalk cols: {cw.columns.tolist()}")

    br = pd.read_csv(bracken_tsv, sep="\t")
    name_col = "name" if "name" in br.columns else br.columns[0]
    reads_col = "new_est_reads" if "new_est_reads" in br.columns else br.columns[-2]
    gvec = np.zeros(n_genera)
    missing = 0.0
    for name, reads in zip(br[name_col].astype(str), br[reads_col].astype(float)):
        try:
            N = int(str(name).split("_")[-1])
        except ValueError:
            continue
        g = N2g.get(N)
        if g is None:
            missing += reads
            continue
        if 0 <= g < n_genera:
            gvec[g] += reads
    return gvec / n_total, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--legacy_mask", required=True)
    ap.add_argument("--kraken_genus_npz", required=True)
    ap.add_argument("--mt13_npz", required=True)
    ap.add_argument("--mt6_npz", required=True)
    ap.add_argument("--nt_npz", default=None)
    ap.add_argument("--bracken_tsv", default=None)
    ap.add_argument("--crosswalk", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels, sep="\t")
    n_genera = int(labels_df["genus_class"].max()) + 1
    legacy_mask = np.load(args.legacy_mask).astype(bool)
    full_mask = np.ones(len(legacy_mask), dtype=bool)

    models = {
        "Kraken2_raw": load_preds(args.kraken_genus_npz),
        "MT_13mer": load_preds(args.mt13_npz),
        "MT_6mer": load_preds(args.mt6_npz),
    }
    if args.nt_npz and Path(args.nt_npz).exists():
        models["NT_v2"] = load_preds(args.nt_npz)

    # Align labels to Kraken genus labels as reference truth
    _, y_ref = models["Kraken2_raw"]
    results = {"full_100K": [], "legacy_in_db_85819": []}
    for name, (preds, labels) in models.items():
        # Prefer each file's own labels; warn if mismatch with kraken
        if not np.array_equal(labels, y_ref):
            # try to use y_ref if same length (order-aligned TWCC pool)
            if len(labels) == len(y_ref):
                # genus labels should match; if not, keep model's labels
                agree = float((labels == y_ref).mean())
                print(f"[warn] {name} labels agree with Kraken genus labels: {agree:.4f}")
        for split, mask in [("full_100K", full_mask), ("legacy_in_db_85819", legacy_mask)]:
            results[split].append(metrics_on_mask(preds, labels, mask, n_genera, name))

    # Bracken on full pool (report is always full-sample; cannot subset per-read)
    if args.bracken_tsv and args.crosswalk:
        brk_ab, missing = bracken_genus_abundance(
            args.bracken_tsv, args.crosswalk, n_genera, n_total=len(y_ref)
        )
        true_ab = np.bincount(y_ref, minlength=n_genera).astype(float) / len(y_ref)
        results["bracken_full_100K"] = {
            "pearson_r": float(pearsonr(brk_ab, true_ab)[0]),
            "bray_curtis": bray_curtis(brk_ab, true_ab),
            "reads_unmapped_to_genus": float(missing),
            "note": "Bracken redistributes the full report; not restricted to legacy mask",
        }

    (out / "comparison_kraken1535.json").write_text(json.dumps(results, indent=2))

    # Markdown table
    lines = [
        "# Kraken2 matched-reference (1,535) vs NT/MT — TWCC 100K",
        "",
        "## Full pool (100,000 reads) — fair after GCF included",
        "",
        "| Model | Classified | Read acc | Pearson r | Bray–Curtis |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["full_100K"]:
        lines.append(
            f"| {row['name']} | {row['classified_rate']*100:.1f}% | "
            f"{row['read_acc_all']*100:.1f}% | {row['pearson_r']:.3f} | {row['bray_curtis']:.3f} |"
        )
    if "bracken_full_100K" in results:
        b = results["bracken_full_100K"]
        lines += [
            "",
            f"**Kraken2+Bracken (full report):** r={b['pearson_r']:.3f}, "
            f"BC={b['bray_curtis']:.3f}",
        ]
    lines += [
        "",
        "## Legacy coverage-matched subset (85,819; old DB in-set only)",
        "",
        "| Model | Classified | Read acc | Pearson r | Bray–Curtis |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["legacy_in_db_85819"]:
        lines.append(
            f"| {row['name']} | {row['classified_rate']*100:.1f}% | "
            f"{row['read_acc_all']*100:.1f}% | {row['pearson_r']:.3f} | {row['bray_curtis']:.3f} |"
        )
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out/'comparison_kraken1535.json'} and COMPARISON.md")


if __name__ == "__main__":
    main()
