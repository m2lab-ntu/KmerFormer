#!/usr/bin/env python3
"""Check that a run's selected epoch agrees between its two independent records.

There are two places a run says which epoch it kept:

  training_history.csv   the epoch whose val_acc is the maximum
  eval_*/eval_metrics_*.json   the `checkpoint_epoch` evaluate.py recorded

They come from different scripts at different times and nothing ties them
together. That was harmless while the run root also carried metrics.json, but an
earlier train.py crashed before writing it (docs/MISSING_ASSETS.md), so for nine
runs these two are the *only* record — and the paper now leans on it: a repeat of
the headline 13-mer arm selected epoch 14 where the original selected 12, and part
of the 0.113-point difference between the two draws is that choice. If
checkpoint_epoch ever drifted from the val_acc peak, the paper would be wrong in a
way nothing downstream would reveal.

Tie-breaking is not arbitrary and this encodes the real rule: EarlyStopping.step
takes a new best only on `metric > best`, strictly, so best.pt stays with the
*earliest* epoch that reaches the maximum.

    python scripts/check_epoch_selection.py $KF_OUT/L29_50M
    python scripts/check_epoch_selection.py $KF_OUT/*/          # every run

Exit 0 only when at least one pair was compared and all pairs agree. Finding
nothing to compare is a failure, not a pass.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def peak_epoch(history: Path):
    """(epoch, val_acc, tied_epochs) for the maximum val_acc, earliest on a tie."""
    rows = []
    with history.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("val_acc") or "").strip():
                continue
            try:
                rows.append((int(r["epoch"]), float(r["val_acc"])))
            except (ValueError, KeyError):
                continue
    if not rows:
        return None
    best = max(v for _, v in rows)
    tied = sorted(e for e, v in rows if v == best)
    return tied[0], best, tied


def eval_epochs(run_dir: Path):
    """[(path, checkpoint_epoch)] over every eval JSON that records one."""
    out = []
    for f in sorted(run_dir.glob("eval_*/eval_metrics_*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "checkpoint_epoch" in d:
            out.append((f, int(d["checkpoint_epoch"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    compared, mismatches, skipped = 0, [], []

    for run in args.run_dir:
        if not run.is_dir():
            skipped.append(f"{run}: not a directory")
            continue
        history = run / "training_history.csv"
        if not history.exists():
            skipped.append(f"{run.name}: no training_history.csv")
            continue
        peak = peak_epoch(history)
        if peak is None:
            skipped.append(f"{run.name}: training_history.csv has no val_acc rows")
            continue
        want, acc, tied = peak

        evals = eval_epochs(run)
        if not evals:
            skipped.append(f"{run.name}: no eval JSON records checkpoint_epoch")
            continue

        note = f"  (val_acc {acc:.5f}" + (f", tied at epochs {tied}" if len(tied) > 1 else "") + ")"
        print(f"{run.name}: history peak = epoch {want}{note}")
        for f, got in evals:
            compared += 1
            ok = got == want
            if not ok:
                mismatches.append((run.name, f.parent.name, want, got, acc))
            if args.verbose or not ok:
                flag = "" if ok else "   <-- MISMATCH"
                print(f"    {f.parent.name}: checkpoint_epoch = {got}{flag}")

    print()
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped:
            print(f"  {s}")
        print()

    if compared == 0:
        print("NOTHING COMPARED. That is a failure of this check, not a pass -- "
              "point it at run directories that hold both training_history.csv and "
              "an eval_*/ subdirectory.")
        return 2

    if mismatches:
        print(f"MISMATCH in {len(mismatches)} of {compared} comparisons:")
        for run, ev, want, got, acc in mismatches:
            print(f"  {run}/{ev}: history says epoch {want} (val_acc {acc:.5f}), "
                  f"eval JSON says {got}")
        print("\nOne of the two is wrong about which checkpoint was scored. Before "
              "changing anything, check whether best.pt was overwritten by a later "
              "run, or the eval was run against last.pt rather than best.pt.")
        return 1

    print(f"all {compared} comparison(s) agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
