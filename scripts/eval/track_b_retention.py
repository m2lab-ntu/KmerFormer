#!/usr/bin/env python3
"""Recompute the ANI-ladder retention table in docs/RESULTS.md.

Retention is far-rung accuracy as a fraction of same-species accuracy. It is the
paper's cleanest separation of the two tokenizer families: every long-k arm keeps
13-22% of its first-rung accuracy at the far rung and every 6-mer arm keeps
49-54%, with no overlap, across 1 to 29 encoder layers, exact and hashed
vocabularies, 0.9M to 500M parameters, and two arms with no neural network at all.

THE RESTRICTION MATTERS. Accuracies are computed only over the genera present at
*every* rung -- fourteen of them. That is what makes the three points on a line
differ in distance from the reference rather than in class count, and it is why
the ladder is built the way it is. Recomputing over each rung's full pool gives
different numbers, and they are not comparable across rungs.

Usage::

    python scripts/eval/track_b_retention.py --track_b_out $KF_OUT/track_b
    python scripts/eval/track_b_retention.py --track_b_out ... --markdown

Expects one directory per rung, each holding one directory per arm with a
``preds.npz`` carrying ``preds`` and ``labels``::

    <track_b_out>/L1_strain/L29/preds.npz
    <track_b_out>/L2_species/L29/preds.npz
    <track_b_out>/L2_far/L29/preds.npz

Arms are discovered from the directory names, so it reports whatever is present.
Rows are labelled through ARM_LABELS below; an unknown directory is reported
under its own name and left unclassified. Abstaining methods (Kraken 2) are
omitted unless --include_abstaining is passed, so the default output matches
docs/RESULTS.md -- see the note on that row in ARM_LABELS.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUNGS = ["L1_strain", "L2_species", "L2_far"]

# directory name -> (display label, tokenizer family)
# "family" is what the band claim is about: the two k families separate, and
# nothing else in the table does.
ARM_LABELS = {
    # L29 is the depth series' endpoint AND the 6-mer representative the
    # manuscript's band claim uses; L1/L8/L16 are the same architecture at other
    # depths, which is why they are reported as a separate grouping below.
    "L29":                ("KmerFormer 6-mer, 29 layers",       "6-mer"),
    "L16":                ("KmerFormer 6-mer, 16 layers",       "6-mer-depth"),
    "L8":                 ("KmerFormer 6-mer, 8 layers",        "6-mer-depth"),
    "L1":                 ("KmerFormer 6-mer, 1 layer",         "6-mer-depth"),
    "nt_v9":              ("NT-v2 500M + LoRA, 6-mer",          "6-mer"),
    "mt_6mer":            ("MetaTransformer 6-mer",             "6-mer"),
    "nb_6mer":            ("6-mer naive Bayes",                 "6-mer"),
    "mt_250M":            ("MetaTransformer 13-mer, 250M",      "long-k"),
    "mt_50M":             ("MetaTransformer 13-mer, 50M",       "long-k"),
    "kf_exact13_1L_250M": ("KmerFormer exact 13-mer, 1L, 250M", "long-k"),
    "kf_exact13_1L_50M":  ("KmerFormer exact 13-mer, 1L, 50M",  "long-k"),
    "kf_exact13_16L":     ("KmerFormer exact 13-mer, 16L",      "long-k"),
    "kf_hash13_d128_1L":  ("KmerFormer hashed d128, 1L",        "long-k"),
    "kf_hash13_d128_16L": ("KmerFormer hashed d128, 16L",       "long-k"),
    "kf_hash13_d64_16L":  ("KmerFormer hashed d64, 16L",        "long-k"),
    "nb_13mer":           ("13-mer naive Bayes",                "long-k"),
    # Kraken 2 is scored on request but OMITTED BY DEFAULT, matching the
    # manuscript and docs/RESULTS.md, which report no Kraken 2 row on this
    # ladder. Read-level accuracy on these rungs is not a like-for-like quantity
    # for a method that abstains: a low far-rung number means "declined to call"
    # where a neural arm's means "called it wrong". Pass --include_abstaining to
    # see it. Note what adding it would mean: Kraken 2 appears in the paper's
    # framing, methods, compute section, discussion and supplementary but in none
    # of its four results sections, so putting it on the ladder is not adding a
    # row -- it is its first appearance in a results section. That is a question
    # about the paper's structure and is open. Where the paper does compare
    # Kraken 2 is docs/RESULTS.md section 5.
    "kraken2":            ("Kraken 2 (1,535-genome index)",     "abstaining"),
}

BAND_FAMILIES = ("6-mer", "long-k")

# The manuscript states the band over the four architecturally distinct 6-mer arms.
# The depth series is the same architecture at 1, 8 and 16 layers, and including it
# widens the 6-mer band without touching the separation. Both are reported, because
# a reader who regenerates this table and compares it with the paper needs to see
# which grouping each number belongs to.
SIX_MER_ALL = ("6-mer", "6-mer-depth")


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["preds"], d["labels"]


def common_genera(root):
    """Genera present at every rung. Labels are a property of the pool, so any
    one arm per rung answers this."""
    per_rung = []
    for rung in RUNGS:
        found = None
        for arm_dir in sorted((root / rung).iterdir()):
            f = arm_dir / "preds.npz"
            if f.exists():
                found = set(np.unique(load(f)[1]).tolist())
                break
        if found is None:
            raise SystemExit(f"no preds.npz under {root / rung}")
        per_rung.append(found)
    return np.array(sorted(set.intersection(*per_rung)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track_b_out", type=Path, required=True,
                    help="directory holding one subdirectory per rung")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table instead of aligned text")
    ap.add_argument("--include_abstaining", action="store_true",
                    help="also score abstaining methods (Kraken 2). Omitted by "
                         "default so the output matches docs/RESULTS.md, which "
                         "follows the manuscript in reporting no Kraken 2 row on "
                         "this ladder.")
    ap.add_argument("--json_out", type=Path, default=None)
    args = ap.parse_args()

    root = args.track_b_out
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    genera = common_genera(root)
    print(f"# genera present at every rung: {len(genera)}", file=sys.stderr)
    print(f"# genus label ids: {genera.tolist()}", file=sys.stderr)

    arms = sorted({d.name for rung in RUNGS for d in (root / rung).iterdir()
                   if (d / "preds.npz").exists()})
    if not args.include_abstaining:
        dropped = [a for a in arms
                   if ARM_LABELS.get(a, (a, ""))[1] == "abstaining"]
        arms = [a for a in arms if a not in dropped]
        for a in dropped:
            print(f"# omitted (abstaining method, not on the manuscript's ladder): "
                  f"{a} -- pass --include_abstaining to score it", file=sys.stderr)

    rows = []
    for arm in arms:
        label, family = ARM_LABELS.get(arm, (arm, "unclassified"))
        accs = []
        for rung in RUNGS:
            f = root / rung / arm / "preds.npz"
            if not f.exists():
                accs.append(None)
                continue
            preds, labels = load(f)
            keep = np.isin(labels, genera)
            accs.append(100.0 * float((preds[keep] == labels[keep]).mean()))
        retention = (100.0 * accs[2] / accs[0]) if (accs[0] and accs[2]) else None
        rows.append({"arm": arm, "label": label, "family": family,
                     "L1_strain": accs[0], "L2_species": accs[1],
                     "L2_far": accs[2], "retention": retention})

    # families first, then best first rung inside a family
    # The depth series sorts after the four architecturally distinct 6-mer arms
    # rather than interleaving with them. Interleaved, its 45.0% reads as a member
    # of a band it is not in, and the two groupings printed below stop being
    # legible in the table they summarise.
    order = {"6-mer": 0, "6-mer-depth": 1, "long-k": 2, "abstaining": 3,
             "unclassified": 4}
    rows.sort(key=lambda r: (order.get(r["family"], 9), -(r["L1_strain"] or 0)))

    def cell(v, suffix=""):
        return f"{v:.2f}{suffix}" if v is not None else "--"

    if args.markdown:
        print("| Arm | L1 strain | L2 species | L2_far | Retention |")
        print("|---|---:|---:|---:|---:|")
        for r in rows:
            print(f"| {r['label']} | {cell(r['L1_strain'], '%')} | "
                  f"{cell(r['L2_species'], '%')} | {cell(r['L2_far'], '%')} | "
                  f"{cell(r['retention'], '%')} |")
    else:
        w = max(len(r["label"]) for r in rows)
        print(f"{'Arm':<{w}} {'L1':>9} {'L2_sp':>9} {'L2_far':>9} {'retention':>10}  family")
        for r in rows:
            print(f"{r['label']:<{w}} {cell(r['L1_strain']):>9} "
                  f"{cell(r['L2_species']):>9} {cell(r['L2_far']):>9} "
                  f"{cell(r['retention']):>10}  {r['family']}")

    print()
    for family in BAND_FAMILIES:
        vals = [r["retention"] for r in rows
                if r["family"] == family and r["retention"] is not None]
        if vals:
            print(f"{family:>7} retention band: {min(vals):.1f}% - {max(vals):.1f}% "
                  f"(n={len(vals)})")
    def band(families):
        return [r["retention"] for r in rows
                if r["family"] in families and r["retention"] is not None]

    six_all = band(SIX_MER_ALL)
    longk = band(("long-k",))
    if six_all and longk:
        print(f"\n  6-mer incl. depth series: {min(six_all):.1f}% - {max(six_all):.1f}% "
              f"(n={len(six_all)})")
        gap = min(six_all) - max(longk)
        print(f"the two bands {'do NOT overlap' if gap > 0 else 'OVERLAP'} "
              f"(gap {gap:+.1f} points, narrowest grouping)")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"n_common_genera": len(genera),
             "common_genus_ids": genera.tolist(),
             "rungs": RUNGS, "rows": rows}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
