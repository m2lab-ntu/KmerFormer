#!/usr/bin/env python3
"""
Track B – Step 14 (E44): can a model tell that a read's genus is not in its label space?

Scores confidence-based rejection with the in-distribution set being reads whose
genus IS one of the 120 (default: L1_strain) and the out-of-distribution set
being the novel-genus pool (L3_genus).  The intermediate rungs are reported at
the same operating point, because the interesting question is not only "can it
reject a novel genus" but "does the rejection signal distinguish a novel genus
from a novel species of a known genus" -- if it cannot, confidence-based
rejection buys less than it appears to.

Scores (higher = more likely OOD):
  msp      1 - max softmax probability
  entropy  Shannon entropy of the softmax, normalised by log(n_classes)
  margin   1 - (top1 - top2 probability)

Energy is not computed: the inference scripts store softmax, and the logsumexp
normaliser cannot be recovered from it.

Kraken2 has no confidence vector; its detector is whether it commits to a genus
at all, so it appears as a single operating point rather than a curve.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _coverage import add_coverage_flag, report_unscored

SCORES = ("msp", "entropy", "margin")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--track_b_dir", default=os.environ.get(
        "KF_TRACK_B", os.path.join(os.environ.get("KF_OUT", "."), "track_b")))
    p.add_argument("--in_level", default="L1_strain")
    p.add_argument("--ood_level", default="L3_genus")
    p.add_argument("--mid_levels", default="L2_species,L2_far")
    add_coverage_flag(p)
    p.add_argument("--models", default="mt_250M,mt_50M,mt_6mer,nt_v9")
    p.add_argument("--subsample", type=int, default=100_000,
                   help="Reads per level per model (probs arrays are N x 120)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / rank-based AUROC; ties get average rank."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within tie groups
    sortedv = allv[order]
    i = 0
    while i < len(sortedv):
        j = i
        while j + 1 < len(sortedv) and sortedv[j + 1] == sortedv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def load_scores(npz: Path, subsample: int, seed: int) -> dict[str, np.ndarray] | None:
    d = np.load(npz)
    if "probs" not in d:
        return None
    probs = d["probs"]
    n = probs.shape[0]
    idx = np.arange(n)
    if subsample and n > subsample:
        idx = np.sort(np.random.default_rng(seed).choice(n, subsample, replace=False))
    p = probs[idx].astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    msp = top2[:, -1]
    second = top2[:, -2]
    ent = -(p * np.log(p)).sum(1) / np.log(p.shape[1])
    return {"msp": 1.0 - msp,
            "entropy": ent,
            "margin": 1.0 - (msp - second)}


def committed_rate(npz: Path) -> float | None:
    if not npz.exists():
        return None
    d = np.load(npz)
    preds = d["preds"]
    return float((preds >= 0).mean())


def main():
    args = parse_args()
    tb = Path(args.track_b_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = [args.in_level] + [l for l in args.mid_levels.split(",") if l] + [args.ood_level]
    models = [m for m in args.models.split(",") if m]

    rows, mean_rows = [], []
    result = {"in_level": args.in_level, "ood_level": args.ood_level,
              "models": {}, "kraken2_operating_point": {}}

    # Kraken2 rejects by abstaining, which is a single operating point rather
    # than a score.  Computed first so the neural detectors can be evaluated at
    # a matched OOD rejection rate.  Note that on the novel-genus pool
    # abstention is the *correct* answer, so tb_06's "genus_acc_all_reads" for
    # L3 is really the correct-rejection rate and must not be read as accuracy.
    for lv in levels:
        cr = committed_rate(tb / "out" / lv / "kraken2" / "preds.npz")
        if cr is not None:
            result["kraken2_operating_point"][lv] = cr
    k_in = result["kraken2_operating_point"].get(args.in_level)
    k_ood = result["kraken2_operating_point"].get(args.ood_level)
    k_reject_ood = (1.0 - k_ood) if k_ood is not None else None
    if k_in is not None and k_ood is not None:
        result["kraken2_detector"] = {
            "reject_rate_ood": 1.0 - k_ood,
            "reject_rate_in_dist": 1.0 - k_in,
            "note": "abstention as the detector; one operating point, no curve",
        }

    for _lv in levels:
        report_unscored(tb / "out" / _lv, models, level=_lv,
                        allow_unscored=args.allow_unscored)

    for model in models:
        per_level = {}
        for lv in levels:
            npz = tb / "out" / lv / model / "preds.npz"
            if not npz.exists():
                print(f"  ! {model}/{lv}: missing {npz}")
                continue
            s = load_scores(npz, args.subsample, args.seed)
            if s is None:
                print(f"  ! {model}/{lv}: no probs stored — rerun with --save_probs")
                continue
            per_level[lv] = s
        if args.in_level not in per_level or args.ood_level not in per_level:
            print(f"  ! {model}: need both {args.in_level} and {args.ood_level}, skipping")
            continue

        result["models"][model] = {}
        for score in SCORES:
            ind = per_level[args.in_level][score]
            ood = per_level[args.ood_level][score]
            a = auroc(ood, ind)
            # Operating point: accept the 95% of in-distribution reads with the
            # lowest OOD score; what share of novel-genus reads sneaks through?
            thr = np.quantile(ind, 0.95)
            leak = float((ood <= thr).mean())
            entry = {"auroc": a, "fpr_at_95tpr_in": leak,
                     "threshold_at_95pct_in": float(thr),
                     "n_in": int(len(ind)), "n_ood": int(len(ood))}
            for lv in levels:
                if lv in per_level:
                    entry[f"accept_rate_{lv}"] = float(
                        (per_level[lv][score] <= thr).mean())
            # Matched-TPR view: Kraken2 rejects a fixed share of novel-genus
            # reads by abstaining, so the only fair question is what the same
            # OOD rejection rate costs a confidence threshold in false
            # rejections of in-distribution reads.
            if k_reject_ood is not None and 0 < k_reject_ood < 1:
                thr_m = np.quantile(ood, 1.0 - k_reject_ood)
                entry["matched_ood_reject_rate"] = float(k_reject_ood)
                entry["in_dist_reject_rate_at_matched"] = float((ind > thr_m).mean())
            result["models"][model][score] = entry
            rows.append({"model": model, "score": score, **entry})

        for lv in levels:
            if lv in per_level:
                mean_rows.append({
                    "model": model, "level": lv,
                    "mean_max_softmax": float(1.0 - per_level[lv]["msp"].mean()),
                    "mean_norm_entropy": float(per_level[lv]["entropy"].mean()),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "ood_detection.tsv", sep="\t", index=False)
        print(f"\n== OOD detection: in={args.in_level}  out={args.ood_level}")
        for score in SCORES:
            sub = df[df["score"] == score].sort_values("auroc", ascending=False)
            print(f"\n  score = {score}")
            for _, r in sub.iterrows():
                extra = ""
                if "in_dist_reject_rate_at_matched" in r and pd.notna(
                        r["in_dist_reject_rate_at_matched"]):
                    extra = (f"   | to match Kraken2's "
                             f"{r['matched_ood_reject_rate']*100:.1f}% OOD rejection it "
                             f"must reject {r['in_dist_reject_rate_at_matched']*100:.1f}% "
                             f"of in-dist reads")
                print(f"    {r['model']:<9} AUROC={r['auroc']:.3f}   "
                      f"novel-genus accepted at 95% in-dist acceptance = "
                      f"{r['fpr_at_95tpr_in']*100:5.1f}%{extra}")
    if mean_rows:
        pd.DataFrame(mean_rows).to_csv(out_dir / "ood_confidence_by_level.tsv",
                                       sep="\t", index=False)
        print("\n== mean max-softmax by rung (is confidence even calibrated to distance?)")
        m = pd.DataFrame(mean_rows).pivot(index="model", columns="level",
                                          values="mean_max_softmax")
        print(m.reindex(columns=[l for l in levels if l in m.columns]).round(3).to_string())

    if result["kraken2_operating_point"]:
        print("\n== Kraken2: abstention as the detector (one operating point)")
        for lv, cr in result["kraken2_operating_point"].items():
            print(f"    {lv:<12} commits {cr*100:5.1f}%  /  rejects {(1-cr)*100:5.1f}%")
        kd = result.get("kraken2_detector")
        if kd:
            print(f"    -> rejects {kd['reject_rate_ood']*100:.1f}% of novel-genus "
                  f"reads at the cost of rejecting "
                  f"{kd['reject_rate_in_dist']*100:.1f}% of in-distribution reads")

    (out_dir / "ood_detection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n  {out_dir / 'ood_detection.tsv'}")


if __name__ == "__main__":
    main()
