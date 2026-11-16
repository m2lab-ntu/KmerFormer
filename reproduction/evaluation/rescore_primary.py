#!/usr/bin/env python3
"""Recompute the primary common-read comparison from bundled scores, CPU only.

No temperature fitting or new inference occurs here. Native MetaTransformer uses
raw MSP >= 0.5. The separate ts-control mode reproduces the added diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import _sample_metrics as s1
NBOOT, BOOT_SEED = 1000, 7
ROOT = Path(__file__).resolve().parent

def abstain(preds, conf, tau):
    p = preds.astype(np.int64).copy()
    p[conf < tau] = -1
    return p

def mcnemar(a_ok, b_ok):
    """Exact two-sided McNemar on paired correctness; returns discordant counts and p."""
    from scipy.stats import binomtest
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(a_right_b_wrong=b, a_wrong_b_right=c, p_exact=float(p))

def boot_diff(a_ok, b_ok, groups=None):
    """95% interval of acc(a) - acc(b) on the same reads; resample reads, or whole groups (genomes)."""
    rng = np.random.default_rng(BOOT_SEED)
    d = a_ok.astype(float) - b_ok.astype(float)
    if groups is None:
        n = len(d); vals = [d[rng.integers(0, n, n)].mean() for _ in range(NBOOT)]
    else:
        u, inv = np.unique(groups, return_inverse=True)
        s = np.bincount(inv, weights=d); c = np.bincount(inv)
        vals = []
        for _ in range(NBOOT):
            i = rng.integers(0, len(u), len(u))
            vals.append(s[i].sum() / c[i].sum())
    return dict(diff=float(d.mean()), lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)))

def verdict(ci, a="KF", b="other"):
    return f"{a} higher" if ci["lo"] > 0 else (f"{b} higher" if ci["hi"] < 0 else "not distinguishable")

def fair_reads(P, labels, groups=None):
    keep = np.all([p >= 0 for p in P.values()], axis=0)
    out = dict(n_pool=int(len(labels)), n_fair=int(keep.sum()), frac_fair=float(keep.mean()),
               declined={k: int((p < 0).sum()) for k, p in P.items()},
               declined_only_by={k: int(((p < 0) & np.all([q >= 0 for kk, q in P.items() if kk != k], axis=0)).sum())
                                 for k, p in P.items()},
               declined_by_all=int(np.all([p < 0 for p in P.values()], axis=0).sum()))
    ok = {k: (p[keep] == labels[keep]) for k, p in P.items()}
    out["accuracy"] = {k: float(v.mean()) for k, v in ok.items()}
    out["n_wrong"] = {k: int((~v).sum()) for k, v in ok.items()}
    g = None if groups is None else groups[keep]
    for a, b in [("KF", "K2"), ("KF", "MT"), ("MT", "K2")]:
        if a in ok and b in ok:
            ci = boot_diff(ok[a], ok[b], g)
            out[f"{a}_minus_{b}"] = dict(ci, verdict=verdict(ci, a, b), mcnemar=mcnemar(ok[a], ok[b]))
    return keep, out

def detection_block(P, labels, keep):
    """(i) repool: seed-42 sampler on the fair pool; (ii) mask: original samples, removed reads masked out."""
    res = {}
    lab_f = labels[keep]
    part, sparse = s1.draw_samples(lab_f)
    arrs = {}
    res["repool"] = {}
    for k, p in P.items():
        r, arrs[k] = s1.score_cfg(p[keep], lab_f, part, sparse)
        res["repool"][k] = dict(accuracy=r["read"]["accuracy"], **r["abundance"],
                                auc=r["detection"]["auc"], sens95=r["detection"]["interp"]["0.950"],
                                sens99=r["detection"]["interp"]["0.990"], ceiling=r["detection"]["ceiling"])
    res["repool"]["n_partition_samples"] = len(part)
    for a, b in [("KF", "K2"), ("KF", "MT")]:
        if a in P and b in P:
            res["repool"][f"{a}_minus_{b}_boot95"] = s1.paired_boot(arrs[a], arrs[b])
    part0, sparse0 = s1.draw_samples(labels)
    res["mask"] = {}
    for k, p in P.items():
        r, _ = s1.score_cfg(p, labels, part0, sparse0, keep=keep)
        res["mask"][k] = dict(auc=r["detection"]["auc"], sens95=r["detection"]["interp"]["0.950"],
                              sens99=r["detection"]["interp"]["0.990"])
    return res


def load_inputs(root=ROOT):
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {record['path']}")
    with np.load(root / "primary_scores.npz", allow_pickle=False) as source:
        data = dict(source)
    protocol = json.loads((root / "protocol.json").read_text())
    labels = data["labels"]
    if labels.shape != (protocol["primary_n"],) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected the original ordered primary label vector")
    if np.any((labels < 0) | (labels >= s1.NC)):
        raise ValueError("Labels outside the 120-genus space")
    for name, values in data.items():
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid shape or nonfinite values: {name}")
        if name.endswith("msp") and np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid confidence: {name}")
        if name.endswith("preds") and (not np.issubdtype(values.dtype, np.integer)
                or np.any((values < (-1 if name == "kraken_preds" else 0)) | (values >= s1.NC))):
            raise ValueError(f"Invalid predictions: {name}")
    return data, protocol


def policy_predictions(data, protocol, policy):
    if policy not in ("native", "ts-control"):
        raise ValueError(f"Unknown policy: {policy}")
    score = data["mt_raw_msp"] if policy == "native" else data["mt_ts_msp"]
    cutoff = protocol["mt_native_tau"] if policy == "native" else protocol["mt_ts_control_tau"]
    return {"KF": abstain(data["kf_preds"], data["kf_ts_msp"], protocol["kf_tau"]),
            "MT": abstain(data["mt_preds"], score, cutoff),
            "K2": data["kraken_preds"]}


def compare(actual, expected, name="result"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, (float, int)):
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise AssertionError(f"{name}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{name}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["native", "ts-control"], default="native")
    parser.add_argument("--check", action="store_true", help="compare every stored primary metric")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data, protocol = load_inputs()
    predictions = policy_predictions(data, protocol, args.policy)
    keep, result = fair_reads(predictions, data["labels"])
    result["detection"] = detection_block(predictions, data["labels"], keep)
    if args.check:
        name = "primary_reference.json" if args.policy == "native" else "primary_ts_control.json"
        compare(result, json.loads((ROOT / name).read_text()))
        print("PASS: all stored read, abundance, detection and paired interval values reproduced")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"policy": args.policy, "n_fair": result["n_fair"],
                      "accuracy": result["accuracy"], "n_wrong": result["n_wrong"],
                      "declined": result["declined"]}, indent=2))


if __name__ == "__main__":
    main()
