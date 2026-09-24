#!/usr/bin/env python3
"""The two train/validation split paths do not agree, and that is load-bearing.

`train.py` picks between them on `data.lazy`. A comment in that file used to
claim both paths give the same split for a given (seed, val_ratio). They do not,
and the claim is the kind someone relies on instead of reading both functions:
it would license writing "identical training indices" into a manuscript for
comparisons that do not have them.

These tests reproduce both algorithms exactly as the two modules implement them
and pin what actually holds -- deterministic within a path, independent across
them -- so the wrong claim cannot come back unnoticed.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
SEED, VAL_RATIO = 42, 0.1


def labels(n=100_000, n_classes=120, skew_seed=7):
    """A long-tailed 120-genus label vector, like the real pool."""
    rng = np.random.default_rng(skew_seed)
    w = rng.pareto(1.2, n_classes) + 1
    return rng.choice(n_classes, size=n, p=w / w.sum()).astype(np.int64)


def split_lazy(n, seed=SEED, val_ratio=VAL_RATIO):
    """dataset_lazy.py: default_rng permutation, unstratified."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_ratio)
    return perm[n_val:], perm[:n_val]


def split_eager(y, seed=SEED, val_ratio=VAL_RATIO):
    """data_loader.py: sklearn train_test_split, stratified on the task label."""
    return train_test_split(np.arange(len(y)), test_size=val_ratio,
                            random_state=seed, stratify=y)


def test_each_path_is_deterministic_on_its_own():
    """The half of the old comment that was true."""
    y = labels()
    a1, v1 = split_lazy(len(y)); a2, v2 = split_lazy(len(y))
    assert np.array_equal(v1, v2) and np.array_equal(a1, a2)
    b1, w1 = split_eager(y); b2, w2 = split_eager(y)
    assert np.array_equal(w1, w2) and np.array_equal(b1, b2)


def test_the_two_paths_disagree_at_the_same_seed():
    """The half that was not. Asserted positively so it cannot pass vacuously."""
    y = labels()
    _, v_lazy = split_lazy(len(y))
    _, v_eager = split_eager(y)
    assert len(v_lazy) == len(v_eager), "same size, different membership"
    assert not np.array_equal(np.sort(v_lazy), np.sort(v_eager)), (
        "the two paths returned the same validation set -- if a split "
        "implementation changed, train.py's comment and the manuscript's "
        "wording about matched training data both need revisiting")


def test_the_overlap_is_what_independent_draws_would_give():
    """Not merely different: different in the way two unrelated draws are.

    A seed shared across the paths buys nothing, so 'same seed' must never be
    offered as evidence that two arms saw the same partition.
    """
    y = labels(); n = len(y)
    _, v_lazy = split_lazy(n)
    _, v_eager = split_eager(y)
    overlap = len(set(v_lazy.tolist()) & set(v_eager.tolist()))
    expected = len(v_lazy) * len(v_eager) / n          # 1,000 at 100K and 10%
    assert abs(overlap - expected) < 0.15 * expected, (
        f"overlap {overlap} against {expected:.0f} expected by chance")


def test_only_one_path_holds_the_class_balance():
    """The mechanism behind the difference, and the one with a research cost.

    Unstratified splitting leaves rare genera with wildly uneven validation
    shares, so validation accuracy on the lazy path weights genera differently
    from the eager path even before any model is trained.
    """
    y = labels(); n = len(y)
    counts = np.bincount(y, minlength=120)
    present = counts > 0

    def share(val_idx):
        v = np.bincount(y[val_idx], minlength=120)
        return v[present] / counts[present]

    _, v_lazy = split_lazy(n)
    _, v_eager = split_eager(y)
    s_lazy, s_eager = share(v_lazy), share(v_eager)
    assert s_eager.std() < s_lazy.std() / 3, (
        f"stratification should tighten the per-genus validation share: "
        f"eager sd {s_eager.std():.5f}, lazy sd {s_lazy.std():.5f}")
    assert s_eager.max() - s_eager.min() < 0.05


@pytest.mark.parametrize("path,expected", [
    ("13mer/exact_1L_50M_meanpool.yaml", True),
    ("13mer/exact_1L_250M_meanpool.yaml", True),
    ("13mer/exact_16L_50M.yaml", True),
    ("13mer/hashed_d128_1L_50M.yaml", False),
    ("13mer/hashed_d64_16L_50M.yaml", False),
    ("6mer/L29_50M.yaml", False),
    ("6mer/L1_50M.yaml", False),
    ("baselines/ntv2_lora_50M.yaml", False),
])
def test_which_arms_take_which_path(path, expected):
    """Pins the grouping the comment names, so a config edit that silently
    moves an arm across the boundary fails here rather than in a claim."""
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / path).read_text())
    assert bool(cfg["data"].get("lazy", False)) is expected


def test_historical_configs_name_their_split_strategy():
    import yaml
    for path in (REPO / "configs").rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        expected = "random" if cfg["data"].get("lazy", False) else "stratified"
        assert cfg["data"]["split_strategy"] == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
