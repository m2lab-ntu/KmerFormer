#!/usr/bin/env python3
"""Tests for the prediction arrays behind docs/RESULTS.md's abundance table.

WHY. Two 100,000-read pools exist over the same 120 genera, and their
genus_class vectors agree on 8.00% of positions -- neither identity nor chance.
Every candidate file for these arms is 100,000 long over 120 classes, so shape,
length, class count and accuracy all pass on the wrong pool: the same 6-mer
checkpoint scores 69.53% on one and 69.88% on the other. The table has already
been published on the wrong pool once.

The only thing that distinguishes them is the stored label vector, so that is
what these check -- and they check it without needing the pool file, which is
not in the repository. Every committed array must carry the same labels, and
that shared hash must be the one the JSON says the table was computed on.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
SUMMARY = ASSETS / "s2_abundance_vs_detection.json"

# clean_common's genus_class column. Independently confirmed on the A6000,
# which stores predictions for two of these arms from separate jobs.
CLEAN_COMMON_LABELS_SHA1 = "9b9ec0c313f7f5fc"


def sha1(a):
    return hashlib.sha1(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def arrays():
    found = sorted(ASSETS.glob("*_preds_clean_common.npz"))
    assert found, "no prediction arrays in docs/assets/"
    return found


@pytest.mark.parametrize("path", arrays(), ids=lambda p: p.stem)
def test_labels_are_clean_commons(path):
    """The check that separates the two pools. Nothing cheaper does."""
    d = np.load(path)
    assert set(d.files) == {"preds", "labels"}, f"{path.name}: unexpected keys"
    assert sha1(d["labels"]) == CLEAN_COMMON_LABELS_SHA1, (
        f"{path.name} is not on clean_common. Its labels hash to "
        f"{sha1(d['labels'])}. An array from the other 100K pool agrees with "
        f"this one on 8% of positions and is otherwise indistinguishable.")


@pytest.mark.parametrize("path", arrays(), ids=lambda p: p.stem)
def test_predictions_are_in_range(path):
    d = np.load(path)
    preds = d["preds"]
    assert preds.shape == d["labels"].shape
    assert preds.min() >= -1, f"{path.name}: preds below -1"
    assert preds.max() <= 119, f"{path.name}: preds above 119"


def test_every_table_row_has_a_committed_array():
    """A row whose array was dropped cannot be rechecked, and *.npz is
    gitignored here by default -- which has already silently dropped one."""
    rows = json.loads(SUMMARY.read_text())["rows"]
    have = {p.stem.replace("_preds_clean_common", "").lower()
            for p in arrays()}
    missing = [r["arm"] for r in rows if r["arm"].lower() not in have]
    assert not missing, f"table rows with no committed predictions: {missing}"


def test_arrays_reproduce_the_tables_read_accuracy():
    """Ties the committed arrays to the printed numbers, so a swapped array
    fails here rather than quietly changing what the table means."""
    rows = {r["arm"].lower(): r for r in json.loads(SUMMARY.read_text())["rows"]}
    for path in arrays():
        arm = path.stem.replace("_preds_clean_common", "").lower()
        row = rows.get(arm)
        if row is None:
            continue
        d = np.load(path)
        acc = float((d["preds"] == d["labels"]).mean())
        assert acc == pytest.approx(row["read_acc_full"], abs=5e-6), (
            f"{path.name}: array gives {acc:.6f}, table says "
            f"{row['read_acc_full']:.6f}")
        assert sha1(d["preds"]) == row["preds_sha1"], (
            f"{path.name}: predictions changed since the table was written")


def test_summary_names_the_pool_it_used():
    s = json.loads(SUMMARY.read_text())
    assert s["pool_labels_sha1"] == CLEAN_COMMON_LABELS_SHA1
    assert "clean_common" in s["pool_identity"], (
        f"table was computed on {s['pool_identity']!r}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
