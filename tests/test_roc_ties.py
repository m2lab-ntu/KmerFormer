#!/usr/bin/env python3
"""Sample-level ROC must group tied scores.

The detection score is a genus's predicted read fraction, and every genus a
classifier never assigns scores exactly 0, so the pooled scores are dominated by
one huge tie. Accumulating TP/FP one pair at a time put the 95%-specificity
operating point inside that tie -- a point no threshold can reach -- and
overstated sensitivity by up to 10.5 points. These tests pin the grouped
behaviour to scikit-learn, which groups ties, on data shaped like the real
problem.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from _roc import operating_point, tie_grouped_roc  # noqa: E402


def zero_heavy(seed, n=24_000, zero_share=0.5):
    """Scores like the detection pool: half exact zeros, the rest coarse."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = np.where(rng.random(n) < zero_share, 0.0,
                     np.round(rng.random(n) * (0.2 + 0.6 * y), 3))
    return y, score


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_points_and_auc_match_sklearn(seed):
    y, s = zero_heavy(seed)
    fpr, tpr, thr, auc = tie_grouped_roc(y, s)
    ref_fpr, ref_tpr, ref_thr = roc_curve(y, s, drop_intermediate=False)
    assert np.allclose(fpr, ref_fpr) and np.allclose(tpr, ref_tpr)
    assert np.allclose(thr[1:], ref_thr[1:])
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("spec", [0.80, 0.90, 0.95, 0.99])
def test_operating_point_matches_sklearn(seed, spec):
    y, s = zero_heavy(seed)
    op = operating_point(*tie_grouped_roc(y, s)[:3], spec)
    ref_fpr, ref_tpr, _ = roc_curve(y, s)
    assert op["sensitivity"] == pytest.approx(ref_tpr[ref_fpr <= 1 - spec].max())
    assert op["specificity"] >= spec


def test_every_point_is_reachable_by_a_threshold():
    """The defect in one line: one point per distinct score, none inside a tie."""
    y, s = zero_heavy(3)
    _, _, thr, _ = tie_grouped_roc(y, s)
    assert len(thr) == len(np.unique(s)) + 1          # plus the (0, 0) anchor
    assert np.all(np.diff(thr[1:]) < 0)


def test_the_zero_block_is_a_single_point():
    """All-zero scores contribute one step, however many pairs share them."""
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    s = np.array([.9, .8, 0, 0, 0, 0, 0, 0])
    fpr, tpr, thr, auc = tie_grouped_roc(y, s)
    assert list(thr[1:]) == [.9, .8, 0.0]
    assert auc == pytest.approx(roc_auc_score(y, s))


@pytest.mark.parametrize("script", ["evaluate_sample.py", "evaluate_sample_kraken2.py"])
def test_both_scorers_use_the_grouped_roc(script):
    """A scorer that goes back to its own loop would reintroduce the defect."""
    src = (REPO / "scripts" / "eval" / script).read_text()
    assert "from _roc import" in src and "tie_grouped_roc(" in src
    assert "argsort(-" not in src, f"{script} sorts scores itself again"
