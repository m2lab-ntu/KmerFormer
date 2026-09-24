#!/usr/bin/env python3
"""
Track B – Step 7: score every method on the ladder and build the ANI curve.

Covers E39 (ladder accuracy), E40 (accuracy vs ANI), E42 (tokenization
crossover) and E45 (per-genus, vs number of training species in that genus).

Safety: for every prediction file the labels stored in the .npz are checked
against the pool label table position by position.  Any mismatch aborts -- a
silently misaligned prediction array would produce plausible-looking but
meaningless accuracies.

Output (in --out_dir)
---------------------
  ladder_accuracy.tsv    level x model
  ani_curve.tsv          model x ANI bin
  per_genus.tsv          level x model x genus
  metrics.json           everything above, plus the closed-set anchors
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from _coverage import add_coverage_flag, report_unscored

# Closed-set (L0) anchors on the matched full-100K TWCC pool, from
# RESULTS_SUMMARY.md section D2.  Used only as the reference row of the ladder;
# nothing here recomputes them.
CLOSED_SET_ANCHOR = {
    "mt_250M": 0.943,
    "mt_50M": 0.875,
    "mt_6mer": 0.488,
    "nt_v9": 0.674,
    "kraken2": 0.800,
}

ANI_BINS = [(None, 80.0), (80.0, 85.0), (85.0, 90.0), (90.0, 93.0),
            (93.0, 95.0), (95.0, 97.0), (97.0, 99.0), (99.0, 100.01)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--track_b_dir", required=True)
    p.add_argument("--levels", default="L1_strain,L2_species,L2_far")
    add_coverage_flag(p)
    p.add_argument("--models",
                   default="mt_250M,mt_50M,mt_6mer,nt_v9,kraken2,nb_13mer,nb_6mer")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def bin_label(lo, hi) -> str:
    if lo is None:
        return f"<{hi:.0f}"
    if hi > 100:
        return f">={lo:.0f}"
    return f"{lo:.0f}-{hi:.0f}"


def assign_bin(ani: float | None) -> str:
    for lo, hi in ANI_BINS:
        if lo is None:
            if ani is None or ani < hi:
                return bin_label(lo, hi)
        elif ani is not None and lo <= ani < hi:
            return bin_label(lo, hi)
    return bin_label(*ANI_BINS[-1])


def load_preds(path: Path, y_true: np.ndarray, seq_ids: np.ndarray,
               name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, sel) where sel indexes the pool rows these preds cover.

    Whole-pool models give sel = every row.  The k-mer NB baselines score a
    subsample (a full 13-mer count table over 750K reads does not fit in RAM),
    so they carry seq_ids and are aligned by id rather than by position.
    """
    d = np.load(path, allow_pickle=False)
    preds = d["preds"].astype(np.int64)

    if "seq_ids" in d:
        pos = {sid: i for i, sid in enumerate(seq_ids)}
        sel = np.array([pos[s] for s in d["seq_ids"].astype(str)], dtype=np.int64)
        if len(sel) != len(preds):
            raise SystemExit(f"{name}: seq_ids/preds length mismatch")
    else:
        if len(preds) != len(y_true):
            raise SystemExit(
                f"{name}: {len(preds)} preds vs {len(y_true)} pool reads")
        sel = np.arange(len(y_true), dtype=np.int64)

    if "labels" in d:
        stored = d["labels"].astype(np.int64)
        if not np.array_equal(stored, y_true[sel]):
            n_bad = int((stored != y_true[sel]).sum())
            raise SystemExit(
                f"{name}: npz labels disagree with the pool label table at "
                f"{n_bad}/{len(sel)} positions -- predictions are misaligned, "
                f"refusing to report metrics.")
    return preds, sel


def summarise(preds: np.ndarray, y: np.ndarray, genus: np.ndarray) -> dict:
    committed = preds >= 0
    per_genus_acc = []
    for g in np.unique(genus):
        m = genus == g
        per_genus_acc.append(float((preds[m] == y[m]).mean()))
    return {
        "n_reads": int(len(y)),
        "n_genera": int(len(np.unique(genus))),
        "committed_rate": float(committed.mean()),
        "read_acc_all": float((preds == y).mean()),
        "read_acc_committed": (float((preds[committed] == y[committed]).mean())
                              if committed.any() else float("nan")),
        "macro_genus_acc": float(np.mean(per_genus_acc)),
    }


def main():
    args = parse_args()
    tb = Path(args.track_b_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = [lv for lv in args.levels.split(",") if lv]
    models = [m for m in args.models.split(",") if m]

    meta = pd.read_csv(tb / "test_data" / "genome_metadata.tsv", sep="\t")
    ani_of = dict(zip(meta["accession"], meta["max_ani_same_genus"]))
    ntrain_of = dict(zip(meta["accession"], meta["n_training_species_in_genus"]))

    ladder_rows, ani_rows, genus_rows = [], [], []
    for _lv in levels:
        report_unscored(tb / "out" / _lv, models, level=_lv,
                        allow_unscored=args.allow_unscored)

    result = {"levels": {}, "closed_set_anchor": CLOSED_SET_ANCHOR}
    # Kept so the ladder can be recomputed on genera common to every level --
    # otherwise an L1-vs-L2 drop partly reflects which genera happened to have
    # a same-species genome available, not the difficulty of novel species.
    cache: dict[str, dict] = {}

    # ANI curve pools L1 and L2 together: the point of E40 is one continuous
    # axis, so the ladder split must not fragment it.
    ani_acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # Per (model, genus, bin) counts.  The raw curve is confounded -- which
    # genera populate a bin depends on what relatives exist in RefSeq, and the
    # high-ANI bins are dominated by well-sampled genera that the 6-mer models
    # happen to be bad at.  These counts drive the genus-controlled variants.
    ani_genus_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))

    for level in levels:
        labels_path = tb / "test_data" / f"{level}_labels.tsv"
        if not labels_path.exists():
            print(f"  skip {level}: no label table")
            continue
        pool = pd.read_csv(labels_path, sep="\t")
        y = pool["genus_class"].to_numpy(np.int64)
        genus = pool["genus_name"].to_numpy()
        acc_col = pool["accession"].to_numpy()
        read_ani = np.array([
            float(ani_of[a]) if a in ani_of and str(ani_of[a]) not in ("", "nan")
            else np.nan for a in acc_col])
        read_bin = np.array([assign_bin(None if np.isnan(v) else float(v))
                             for v in read_ani])
        read_ntrain = np.array([int(ntrain_of.get(a, 0)) for a in acc_col])

        result["levels"][level] = {"n_reads": int(len(y)),
                                   "n_genera": int(len(set(genus))),
                                   "models": {}}
        print(f"\n== {level}: {len(y):,} reads, {len(set(genus))} genera")

        for model in models:
            npz = tb / "out" / level / model / "preds.npz"
            if not npz.exists():
                print(f"  - {model}: missing {npz}")
                continue
            preds, sel = load_preds(npz, y, pool["seq_id"].to_numpy(),
                                    f"{level}/{model}")
            ys, gs = y[sel], genus[sel]
            stats = summarise(preds, ys, gs)
            stats["subsampled"] = int(len(sel) != len(y))
            cache[(level, model)] = (preds, ys, gs)
            result["levels"][level]["models"][model] = stats
            ladder_rows.append({"level": level, "model": model, **stats})
            note = "  [subsample]" if stats["subsampled"] else ""
            print(f"  - {model:<9} acc_all={stats['read_acc_all']*100:6.2f}%  "
                  f"macro={stats['macro_genus_acc']*100:6.2f}%  "
                  f"committed={stats['committed_rate']*100:6.2f}%{note}")

            correct = preds == ys
            bins_sel, ani_sel, ntrain_sel = read_bin[sel], read_ani[sel], read_ntrain[sel]
            for b in np.unique(bins_sel):
                m = bins_sel == b
                cell = ani_acc[model][b]
                cell[0] += int(correct[m].sum())
                cell[1] += int(m.sum())
                for g in np.unique(gs[m]):
                    gm = m & (gs == g)
                    gcell = ani_genus_acc[model][g][b]
                    gcell[0] += int(correct[gm].sum())
                    gcell[1] += int(gm.sum())

            for g in sorted(set(gs)):
                m = gs == g
                genus_rows.append({
                    "level": level, "model": model, "genus": g,
                    "n_reads": int(m.sum()),
                    "read_acc": float(correct[m].mean()),
                    "n_training_species_in_genus": int(ntrain_sel[m][0]),
                    "mean_max_ani": (float(np.nanmean(ani_sel[m]))
                                     if not np.all(np.isnan(ani_sel[m])) else float("nan")),
                })

    # --- matched-genus ladder ------------------------------------------------
    present_levels = sorted({lv for lv, _ in cache})
    matched_rows = []
    if len(present_levels) >= 2:
        # One genus set per level is enough; all models share that level's pool.
        per_level_sets = {}
        for lv in present_levels:
            for m in models:
                if (lv, m) in cache:
                    per_level_sets[lv] = set(cache[(lv, m)][2].tolist())
                    break
        common = set.intersection(*per_level_sets.values()) if per_level_sets else set()
        result["matched_genera"] = {"levels": present_levels,
                                    "n_common_genera": len(common),
                                    "genera": sorted(common)}
        print(f"\n== matched-genus ladder: {len(common)} genera present in all of "
              f"{', '.join(present_levels)}")
        if not common:
            print("  (no genus is present in every level -- matched table skipped)")
        for lv in present_levels:
            for m in models:
                if (lv, m) not in cache:
                    continue
                preds, y, genus = cache[(lv, m)]
                mask = np.isin(genus, list(common))
                if not mask.any():
                    continue
                st = summarise(preds[mask], y[mask], genus[mask])
                matched_rows.append({"level": lv, "model": m, **st})
                print(f"  {lv:<11} {m:<9} acc_all={st['read_acc_all']*100:6.2f}%  "
                      f"macro={st['macro_genus_acc']*100:6.2f}%  n={st['n_reads']:,}")

    order = {bin_label(lo, hi): i for i, (lo, hi) in enumerate(ANI_BINS)}
    for model, bins in ani_acc.items():
        for b, (ncorr, ntot) in sorted(bins.items(), key=lambda kv: order.get(kv[0], 99)):
            ani_rows.append({"model": model, "ani_bin": b, "bin_order": order.get(b, 99),
                             "n_reads": ntot,
                             "read_acc": ncorr / ntot if ntot else float("nan")})

    # --- genus-controlled ANI views ------------------------------------------
    NEAR = {"95-97", "97-99", ">=99"}
    matched_rows_ani, paired_rows = [], []
    for model, by_genus in ani_genus_acc.items():
        # A genus only controls the comparison if it has reads on both sides of
        # the species boundary; otherwise "near vs far" is just "genus A vs B".
        spanning = {g: bins for g, bins in by_genus.items()
                    if any(b in NEAR for b in bins) and any(b not in NEAR for b in bins)}
        for b in {b for bins in spanning.values() for b in bins}:
            accs = [c[0] / c[1] for g, bins in spanning.items()
                    if (c := bins.get(b)) and c[1] > 0]
            if accs:
                matched_rows_ani.append({
                    "model": model, "ani_bin": b, "bin_order": order.get(b, 99),
                    "n_genera": len(accs), "macro_read_acc": float(np.mean(accs))})
        near_far = []
        for g, bins in spanning.items():
            n_c = [sum(c[0] for b, c in bins.items() if b in NEAR),
                   sum(c[1] for b, c in bins.items() if b in NEAR)]
            f_c = [sum(c[0] for b, c in bins.items() if b not in NEAR),
                   sum(c[1] for b, c in bins.items() if b not in NEAR)]
            if n_c[1] and f_c[1]:
                near_far.append((g, n_c[0] / n_c[1], f_c[0] / f_c[1]))
        if near_far:
            near = np.array([x[1] for x in near_far])
            far = np.array([x[2] for x in near_far])
            d = far - near
            paired_rows.append({
                "model": model, "n_genera_paired": len(near_far),
                "mean_acc_near_ge95": float(near.mean()),
                "mean_acc_far_lt95": float(far.mean()),
                "mean_delta": float(d.mean()),
                "delta_ci95_lo": float(np.percentile(d, 2.5)),
                "delta_ci95_hi": float(np.percentile(d, 97.5)),
                "n_genera_worse_when_far": int((d < 0).sum()),
            })
    if paired_rows:
        print("\n== within-genus paired: near (ANI>=95) vs far (<95)")
        for r in sorted(paired_rows, key=lambda r: r["mean_delta"]):
            print(f"  {r['model']:<9} near={r['mean_acc_near_ge95']*100:6.2f}%  "
                  f"far={r['mean_acc_far_lt95']*100:6.2f}%  "
                  f"delta={r['mean_delta']*100:+6.2f} pp  "
                  f"({r['n_genera_worse_when_far']}/{r['n_genera_paired']} genera worse)")
        pd.DataFrame(paired_rows).to_csv(out_dir / "ani_paired_near_far.tsv",
                                         sep="\t", index=False)
        result["ani_paired_near_far"] = paired_rows
    if matched_rows_ani:
        pd.DataFrame(matched_rows_ani).sort_values(["model", "bin_order"]).to_csv(
            out_dir / "ani_curve_matched_genera.tsv", sep="\t", index=False)

    pd.DataFrame(ladder_rows).to_csv(out_dir / "ladder_accuracy.tsv",
                                     sep="\t", index=False)
    if matched_rows:
        pd.DataFrame(matched_rows).to_csv(
            out_dir / "ladder_accuracy_matched_genera.tsv", sep="\t", index=False)
    pd.DataFrame(ani_rows).sort_values(["model", "bin_order"]).to_csv(
        out_dir / "ani_curve.tsv", sep="\t", index=False)
    pd.DataFrame(genus_rows).to_csv(out_dir / "per_genus.tsv",
                                    sep="\t", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nWrote:\n  {out_dir/'ladder_accuracy.tsv'}"
          f"\n  {out_dir/'ani_curve.tsv'}"
          f"\n  {out_dir/'per_genus.tsv'}\n  {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
