#!/usr/bin/env python3
"""The two 50M exact 13-mer runs behind the supplement's repeat analysis.

Supplementary Section S15 reports that the two archived runs share one label
vector, differ on 2,155 + 2,268 reads, and give a nominal paired standard error
of 0.067 points. Those numbers were computed from archive directories that are
not public, so a reader following the supplement's snippet found no file. Both
runs are now in the repository; these tests pin the published numbers to the
published files so the claim and the evidence cannot drift apart.
"""

import hashlib
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
RUN1 = ASSETS / "KmerFormer_exact13mer_1L_50M_preds_clean_common.npz"
RUN2 = ASSETS / "repeats" / "KmerFormer_exact13mer_1L_50M_run2_preds_clean_common.npz"


def load(path):
    d = np.load(path)
    return d["preds"], d["labels"]


def test_both_runs_share_the_closed_set_label_vector():
    for path in (RUN1, RUN2):
        _, labels = load(path)
        assert hashlib.sha1(labels.tobytes()).hexdigest()[:16] == "9b9ec0c313f7f5fc", path.name


def test_the_run_accuracies_are_the_reported_pair():
    for path, expected in ((RUN1, 91.152), (RUN2, 91.265)):
        preds, labels = load(path)
        assert round(float((preds == labels).mean()) * 100, 3) == expected, path.name


def test_the_discordant_counts_and_paired_se_match_the_supplement():
    a, labels = load(RUN1)
    b, labels2 = load(RUN2)
    assert np.array_equal(labels, labels2)
    right_then_wrong = int(((a == labels) & (b != labels)).sum())
    wrong_then_right = int(((a != labels) & (b == labels)).sum())
    assert (right_then_wrong, wrong_then_right) == (2155, 2268)
    se = np.sqrt(right_then_wrong + wrong_then_right) / labels.size * 100
    assert round(se, 3) == 0.067


def test_the_second_run_is_outside_the_regeneration_glob():
    """REPRODUCE.md loops over docs/assets/*_preds_clean_common.npz as the
    table's arms; the second run must not become a fifteenth row."""
    assert RUN2 not in set(ASSETS.glob("*_preds_clean_common.npz"))
