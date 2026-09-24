#!/usr/bin/env python3
"""Tests for scripts/check_epoch_selection.py, on synthetic run directories.

The checker itself has to run against real run outputs, which are not in this
repository, so it cannot be a test. Its *logic* can be, and needs to be: it
encodes a tie-breaking rule that is easy to get backwards, and its whole value is
that it fails when two records disagree — a checker that quietly passes is worse
than none, which this repository has already learned once from a check that
reported "0 compared, all passed".

The tie rule is not a guess. `EarlyStopping.step` takes a new best only on
`metric > best`, strictly, so on equal val_acc the *earliest* epoch keeps best.pt.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CHECKER = REPO / "scripts" / "check_epoch_selection.py"

sys.path.insert(0, str(REPO / "scripts"))


def make_run(tmp_path, name, history, eval_epochs):
    """history: [(epoch, val_acc)]; eval_epochs: {eval_dir: checkpoint_epoch}"""
    run = tmp_path / name
    run.mkdir(parents=True)
    with (run / "training_history.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_acc", "val_f1"])
        for e, v in history:
            w.writerow([e, 0.5, v, v])
    for d, ep in eval_epochs.items():
        sub = run / d
        sub.mkdir()
        (sub / "eval_metrics_rc_tta.json").write_text(
            json.dumps({"checkpoint_epoch": ep, "micro_accuracy": 0.9}))
    return run


def run_checker(*dirs):
    r = subprocess.run([sys.executable, str(CHECKER), *map(str, dirs)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_agreement_passes(tmp_path):
    run = make_run(tmp_path, "ok", [(1, .80), (2, .91), (3, .85)],
                   {"eval_clean_common_rctta": 2})
    code, out = run_checker(run)
    assert code == 0, out
    assert "agree" in out


def test_disagreement_fails_and_names_both_numbers(tmp_path):
    run = make_run(tmp_path, "drift", [(1, .80), (2, .91), (3, .85)],
                   {"eval_clean_common_rctta": 3})
    code, out = run_checker(run)
    assert code == 1, out
    assert "MISMATCH" in out
    assert "epoch 2" in out and "says 3" in out


def test_tie_resolves_to_the_earliest_epoch(tmp_path):
    """`metric > best` means a later equal epoch does NOT replace best.pt."""
    run = make_run(tmp_path, "tie", [(1, .80), (2, .91), (3, .91), (4, .85)],
                   {"eval_clean_common_rctta": 2})
    code, out = run_checker(run)
    assert code == 0, out
    assert "tied at epochs [2, 3]" in out

    later = make_run(tmp_path, "tie_wrong", [(1, .80), (2, .91), (3, .91)],
                     {"eval_clean_common_rctta": 3})
    code, out = run_checker(later)
    assert code == 1, "a later tied epoch must not be accepted as the selection"


def test_every_eval_dir_is_compared(tmp_path):
    """A run scored on several pools must have all of them checked, not the first."""
    run = make_run(tmp_path, "multi", [(1, .80), (2, .91)],
                   {"eval_clean_common_rctta": 2, "eval_track_a": 1})
    code, out = run_checker(run)
    assert code == 1, out
    assert "eval_track_a" in out


def test_nothing_to_compare_is_a_failure_not_a_pass(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out = run_checker(empty)
    assert code == 2, out
    assert "NOTHING COMPARED" in out

    # history but no eval JSON: also nothing compared
    only_history = make_run(tmp_path, "hist_only", [(1, .9)], {})
    code, out = run_checker(only_history)
    assert code == 2, out


def test_rows_without_val_acc_are_ignored_not_treated_as_zero(tmp_path):
    """A crashed or in-progress epoch leaves val_acc blank; it must not win or
    displace the real peak."""
    run = tmp_path / "blank"
    run.mkdir()
    with (run / "training_history.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "val_acc"])
        w.writerow([1, "0.80"])
        w.writerow([2, "0.91"])
        w.writerow([3, ""])
    sub = run / "eval_clean_common_rctta"
    sub.mkdir()
    (sub / "eval_metrics_rc_tta.json").write_text(json.dumps({"checkpoint_epoch": 2}))
    code, out = run_checker(run)
    assert code == 0, out


def test_tie_rule_matches_EarlyStopping(tmp_path):
    """Assert against the real class rather than a restatement of it."""
    from kmerformer.utils import EarlyStopping
    es = EarlyStopping(patience=5, mode="max")
    selected = None
    for epoch, acc in [(1, .80), (2, .91), (3, .91), (4, .85)]:
        if es.step(acc):
            selected = epoch
    assert selected == 2, "EarlyStopping should keep the earliest tied epoch"

    from check_epoch_selection import peak_epoch
    run = make_run(tmp_path, "cmp", [(1, .80), (2, .91), (3, .91), (4, .85)], {})
    assert peak_epoch(run / "training_history.csv")[0] == selected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
