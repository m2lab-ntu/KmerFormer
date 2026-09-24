#!/usr/bin/env python3
"""
Track B – Step 9: sample-level abundance metrics on the log-normal community (E43).

Reports, per method, genus-level Pearson / Spearman r, Bray-Curtis dissimilarity,
detection sensitivity at the >=1% level and false-positive genus count, with
bootstrap CIs over genera.  Kraken2 is scored twice: raw read assignment and
Bracken-reestimated abundance, because Bracken is what closed the abundance gap
in the in-database setting and is therefore the baseline to beat here.

Abundance convention matches scripts/eval_kraken1535_vs_neural.py: fractions are
over *all* reads, so a method that abstains carries that as missing mass rather
than being silently renormalised.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from _coverage import add_coverage_flag, report_unscored

DETECT_THRESHOLD = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--track_b_dir", default=os.environ.get(
        "KF_TRACK_B", os.path.join(os.environ.get("KF_OUT", "."), "track_b")))
    p.add_argument("--level", default="L2_lognormal")
    p.add_argument("--models", default="mt_250M,mt_50M,mt_6mer,nt_v9,kraken2")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--genus_map",
                   default=os.path.join(os.environ.get("KF_TRACK_A",
                       os.path.join(os.environ.get("KF_OUT", "."), "track_a")),
                                        "assets", "genus_map.tsv"),
                   help="Defines the full label space; the pool itself only "
                        "covers a subset of genera but models can predict any.")
    add_coverage_flag(p)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def bray_curtis(x: np.ndarray, y: np.ndarray) -> float:
    denom = (x + y).sum()
    return float(np.abs(x - y).sum() / denom) if denom > 0 else float("nan")


def score(expected: np.ndarray, predicted: np.ndarray,
          keep: np.ndarray) -> dict:
    """Correlation on the genera actually in the community; detection and
    false positives over the full 120-genus label space.

    Restricting everything to in-pool genera would hide the failure mode the
    mock-community work already found -- reads piling into a genus that is not
    in the sample at all.  Those land outside `keep` by construction.
    """
    e_in, p_in = expected[keep], predicted[keep]
    present = expected >= DETECT_THRESHOLD          # true of in-pool genera only
    detected = predicted >= DETECT_THRESHOLD        # anywhere in the label space
    fp_mask = detected & ~present
    return {
        "pearson_r": float(pearsonr(e_in, p_in)[0]),
        "spearman_r": float(spearmanr(e_in, p_in)[0]),
        "bray_curtis": bray_curtis(e_in, p_in),
        "detection_sens_ge1pct": (float((detected & present).sum() / present.sum())
                                  if present.any() else float("nan")),
        "n_expected_ge1pct": int(present.sum()),
        "fp_genera_ge1pct": int(fp_mask.sum()),
        "fp_genera_out_of_pool": int((fp_mask & ~keep).sum()),
        "fp_mass_out_of_pool": float(predicted[~keep].sum()),
        "predicted_mass_in_pool": float(p_in.sum()),
    }


def bootstrap_ci(expected: np.ndarray, predicted: np.ndarray,
                 n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(expected)
    rs, bcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e, p = expected[idx], predicted[idx]
        if e.std() == 0 or p.std() == 0:
            continue
        rs.append(pearsonr(e, p)[0])
        bcs.append(bray_curtis(e, p))
    out = {}
    if rs:
        out["pearson_r_ci95"] = [float(np.percentile(rs, 2.5)),
                                 float(np.percentile(rs, 97.5))]
    if bcs:
        out["bray_curtis_ci95"] = [float(np.percentile(bcs, 2.5)),
                                   float(np.percentile(bcs, 97.5))]
    return out


def main():
    args = parse_args()
    tb = Path(args.track_b_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_path = tb / "test_data" / f"{args.level}_expected_abundance.tsv"
    labels_path = tb / "test_data" / f"{args.level}_labels.tsv"
    for f in (exp_path, labels_path):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    pool = pd.read_csv(labels_path, sep="\t")
    y = pool["genus_class"].to_numpy(np.int64)
    n_total = len(y)
    # Size arrays by the full training label space, not by what happens to be
    # in this pool: a model may predict any of the 120 genera, and those
    # off-pool predictions are exactly the false positives we want to count.
    n_genera = int(pd.read_csv(args.genus_map, sep="\t")["genus_class"].max()) + 1

    exp_df = pd.read_csv(exp_path, sep="\t")
    expected = np.zeros(n_genera)
    expected[exp_df["genus_class"].to_numpy()] = exp_df["expected_fraction"].to_numpy()

    # Only genera that exist in this community are informative points; keeping
    # the 100+ structural zeros would inflate r for every method equally.
    keep = expected > 0
    print(f"{args.level}: {n_total:,} reads, {keep.sum()} genera present")

    rows, result = [], {"level": args.level, "n_reads": n_total,
                        "n_genera_present": int(keep.sum()), "methods": {}}

    for model in [m for m in args.models.split(",") if m]:
        npz = tb / "out" / args.level / model / "preds.npz"
        if not npz.exists():
            print(f"  - {model}: missing {npz}")
            continue
        d = np.load(npz)
        preds = d["preds"].astype(np.int64)
        if "labels" in d and not np.array_equal(d["labels"].astype(np.int64), y):
            raise SystemExit(f"{model}: npz labels disagree with the pool table")

        valid = preds >= 0
        pred_ab = (np.bincount(preds[valid], minlength=n_genera).astype(float)
                   / n_total)
        m = score(expected, pred_ab, keep)
        m.update(bootstrap_ci(expected[keep], pred_ab[keep], args.n_boot, args.seed))
        m["committed_rate"] = float(valid.mean())
        result["methods"][model] = m
        rows.append({"method": model, **m})
        print(f"  - {model:<9} r={m['pearson_r']:.3f}  BC={m['bray_curtis']:.3f}  "
              f"det={m['detection_sens_ge1pct']*100:.1f}%  FP={m['fp_genera_ge1pct']}")

        if model == "kraken2":
            br_path = npz.parent / "bracken_genus_abundance.tsv"
            if br_path.exists():
                br = pd.read_csv(br_path, sep="\t")
                br_ab = np.zeros(n_genera)
                sel = br["genus_class"].to_numpy() < n_genera
                br_ab[br.loc[sel, "genus_class"].to_numpy()] = \
                    br.loc[sel, "pred_fraction"].to_numpy()
                mb = score(expected, br_ab, keep)
                mb.update(bootstrap_ci(expected[keep], br_ab[keep],
                                       args.n_boot, args.seed))
                result["methods"]["kraken2_bracken"] = mb
                rows.append({"method": "kraken2_bracken", **mb})
                print(f"  - {'kraken2+brk':<9} r={mb['pearson_r']:.3f}  "
                      f"BC={mb['bray_curtis']:.3f}  "
                      f"det={mb['detection_sens_ge1pct']*100:.1f}%  "
                      f"FP={mb['fp_genera_ge1pct']}")
            else:
                print(f"  ! no bracken table at {br_path}")

    report_unscored(tb / "out" / args.level,
                    [m for m in args.models.split(",") if m],
                    level=args.level, allow_unscored=args.allow_unscored)

    pd.DataFrame(rows).to_csv(out_dir / f"{args.level}_abundance.tsv",
                              sep="\t", index=False)
    (out_dir / f"{args.level}_abundance.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(f"\n  {out_dir / f'{args.level}_abundance.tsv'}")


if __name__ == "__main__":
    main()
