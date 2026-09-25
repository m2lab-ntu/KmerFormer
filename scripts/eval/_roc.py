"""ROC for pooled genus-detection scores, with tied scores grouped.

WHY THIS EXISTS. Both sample-level scorers used to sort the pooled scores and
accumulate TP/FP one pair at a time. Each position in that ordering became an
ROC point, including positions *inside* a run of equal scores. No threshold can
separate pairs that share a score, so those interior points are not achievable
operating points; they depend only on the sort's tie order.

The score is a genus's predicted read fraction in a sparse community, and a
genus the classifier never assigns scores exactly 0. Across the 24,000 pooled
genus-community pairs of the closed-set table, 10,289-13,074 score 0. The 95%
specificity operating point fell inside that zero block, so every reported
threshold printed as 0.00000 and the sensitivity at 95% specificity was
overstated by up to 10.5 points (exact 13-mer, 250M: 88.02% ungrouped, 77.49%
grouped).

Here an ROC point is formed only at the end of each group of equal scores, with
the curve anchored at (0, 0). The AUC is then the trapezoid over those points,
which counts tied positive-negative pairs as one half, as the Mann-Whitney
statistic does. tests/test_roc_ties.py checks both against scikit-learn.
"""

import numpy as np


def tie_grouped_roc(y_true, y_score):
    """Return (fpr, tpr, thresholds, auc) with one point per distinct score.

    thresholds[i] is the lowest score counted as a call at point i; the first
    point is (0, 0) at an infinite threshold.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score differ in shape")
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs at least one positive and one negative")

    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    truth = y_true[order]
    # Last index of every group of equal scores.
    ends = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    tps = np.cumsum(truth)[ends]
    fps = (ends + 1) - tps

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, scores[ends]]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return fpr, tpr, thresholds, float(trapezoid(tpr, fpr))


def operating_point(fpr, tpr, thresholds, target_specificity):
    """Highest-sensitivity achievable point with specificity >= target."""
    eligible = np.flatnonzero(fpr <= 1.0 - target_specificity)
    i = int(eligible.max())          # index 0, (0, 0), is always eligible
    return {"sensitivity": float(tpr[i]),
            "specificity": float(1.0 - fpr[i]),
            "threshold": float(thresholds[i])}
