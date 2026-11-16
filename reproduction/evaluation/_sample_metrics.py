"""Sample-level scoring extracted from the archived evaluation; protocol unchanged."""
import numpy as np
from scipy.stats import pearsonr
NC, READS, NPART, NSPARSE, PRESENT, SEED = 120, 1000, 100, 200, 60, 42
SPEC_GRID = (0.95, 0.98, 0.99, 0.995, 0.999)
NBOOT, BOOT_SEED = 1000, 7

def tie_grouped_roc(y_true, y_score):
    """One ROC point per distinct score, anchored at (0, 0); AUC = trapezoid.
    Same algorithm as scripts/eval/_roc.py."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    order = np.argsort(-y_score, kind='mergesort')
    scores, truth = y_score[order], y_true[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    tps = np.cumsum(truth)[ends]
    fps = (ends + 1) - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, scores[ends]]
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    return fpr, tpr, thresholds, float(trapezoid(tpr, fpr))

def operating_point(fpr, tpr, thresholds, target_specificity):
    """Highest-sensitivity realizable point with specificity >= target."""
    i = int(np.flatnonzero(fpr <= 1.0 - target_specificity).max())
    return dict(sensitivity=float(tpr[i]), specificity=float(1.0 - fpr[i]), threshold=float(thresholds[i]))

def interp_sens(fpr, tpr, spec):
    """Sensitivity at exactly `spec`, linear between adjacent realizable points."""
    x = 1.0 - spec
    left = int(np.flatnonzero(fpr <= x + 1e-15).max())
    if abs(fpr[left] - x) < 1e-15 or left == len(fpr) - 1:
        return float(tpr[left])
    w = (x - fpr[left]) / (fpr[left + 1] - fpr[left])
    return float(tpr[left] + w * (tpr[left + 1] - tpr[left]))

def draw_samples(labels, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    part = [perm[i * READS:(i + 1) * READS] for i in range(NPART) if (i + 1) * READS <= n]
    by_genus = {g: np.where(labels == g)[0] for g in range(NC)}
    sparse = []
    for _ in range(NSPARSE):
        present = rng.choice(NC, size=PRESENT, replace=False)
        avail = np.concatenate([by_genus[g] for g in present])
        sparse.append((present, rng.choice(avail, size=READS, replace=False)))
    return part, sparse

def read_level(preds, labels):
    clf = preds >= 0
    return dict(n=int(len(labels)), accuracy=float((preds == labels).mean()), coverage=float(clf.mean()),
                n_declined=int((~clf).sum()), n_wrong_committed=int((clf & (preds != labels)).sum()),
                classified_accuracy=float((preds[clf] == labels[clf]).mean()))

def profile(sp, sl, mode):
    n = len(sl)
    tc = np.bincount(sl, minlength=NC) / n
    pc = np.bincount(sp[sp >= 0], minlength=NC) / n
    if mode == 'uncbin':
        tc, pc = np.r_[tc, 0.0], np.r_[pc, (sp < 0).sum() / n]
    elif mode == 'renorm':
        s = pc.sum()
        pc = pc / s if s > 0 else pc
    return tc, pc

def abundance(preds, labels, part, mode='lost', keep=None):
    rs, bcs = [], []
    for idx in part:
        if keep is not None:
            idx = idx[keep[idx]]
        tc, pc = profile(preds[idx], labels[idx], mode)
        rs.append(pearsonr(tc, pc)[0])
        bcs.append(np.abs(tc - pc).sum() / (tc + pc).sum())
    return dict(pearson_r=float(np.mean(rs)), bray_curtis=float(np.mean(bcs)), n_samples=len(part))

def detection_arrays(preds, labels, sparse, keep=None):
    truth, score, count, ntrue = [], [], [], []
    for present, idx in sparse:
        if keep is not None:
            idx = idx[keep[idx]]
        sp = preds[idx]
        pc = np.bincount(sp[sp >= 0], minlength=NC)
        t = np.zeros(NC, dtype=np.int64)
        t[present] = 1
        truth.append(t); count.append(pc); score.append(pc / len(idx))
        ntrue.append(np.bincount(labels[idx], minlength=NC))
    return np.array(truth), np.array(score), np.array(count), np.array(ntrue)

def detection(truth, score, count, ntrue):
    y, s, c = truth.ravel(), score.ravel(), count.ravel()
    fpr, tpr, thr, auc = tie_grouped_roc(y, s)
    op = operating_point(fpr, tpr, thr, 0.95)
    out = dict(auc=auc, sens_rule=op['sensitivity'], spec_rule=op['specificity'],
               threshold_reads_rule=int(round(op['threshold'] * READS)) if np.isfinite(op['threshold']) else None,
               interp={f'{sp:.3f}': interp_sens(fpr, tpr, sp) for sp in SPEC_GRID},
               ceiling=float((ntrue[truth == 1] >= 1).mean()))
    for k in (1, 2):
        call = c >= k
        out[f'count_ge{k}'] = dict(sens=float(call[y == 1].mean()), spec=float(1 - call[y == 0].mean()))
    return out

def score_cfg(preds, labels, part, sparse, mode='lost', keep=None):
    rl = read_level(preds if keep is None else preds[keep], labels if keep is None else labels[keep])
    ab = abundance(preds, labels, part, mode, keep)
    if mode == 'lost':
        ab['bray_curtis_conditional'] = abundance(preds, labels, part, 'renorm', keep)['bray_curtis']
    arrays = detection_arrays(preds, labels, sparse, keep)
    return dict(read=rl, abundance=ab, detection=detection(*arrays)), arrays

def paired_boot(arr_a, arr_b, specs=(0.95, 0.99)):
    """Paired bootstrap over the 200 sparse samples; differences a - b."""
    (ta, sa, _, _), (tb, sb, _, _) = arr_a, arr_b
    rng = np.random.default_rng(BOOT_SEED)
    n = ta.shape[0]
    res = {'auc': []}
    res.update({f'interp_{sp:.3f}': [] for sp in specs})
    for _ in range(NBOOT):
        i = rng.integers(0, n, n)
        fa, pa, _, aa = tie_grouped_roc(ta[i].ravel(), sa[i].ravel())
        fb, pb, _, ab = tie_grouped_roc(tb[i].ravel(), sb[i].ravel())
        res['auc'].append(aa - ab)
        for sp in specs:
            res[f'interp_{sp:.3f}'].append(interp_sens(fa, pa, sp) - interp_sens(fb, pb, sp))
    return {k: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5))) for k, v in res.items()}
