#!/usr/bin/env python3
"""Verify bundled predictions; optionally recompute full-pool sample metrics.

No FASTA, TSV, checkpoint, GPU or network is needed. The species-coverage-masked
accuracy cannot be reconstructed from genus-only arrays and is not verified here.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from s2_abundance_vs_detection import PROTOCOL, score  # noqa: E402


def fingerprint(values):
    return hashlib.sha1(np.asarray(values, dtype="<i8").tobytes()).hexdigest()[:16]


def verify(assets, full_metrics=False):
    summary = json.loads((assets / "s2_abundance_vs_detection.json").read_text())
    rows = summary["rows"]
    if len(rows) != 14 or len({r["arm"] for r in rows}) != len(rows):
        raise ValueError("expected 14 distinct archived arms")
    if summary["protocol"] != PROTOCOL:
        raise ValueError("archived scoring protocol differs from the scorer")
    if summary["pool_labels_sha1"] != "9b9ec0c313f7f5fc":
        raise ValueError("summary does not identify the manuscript's closed-set labels")
    files = {p.stem.removesuffix("_preds_clean_common"): p
             for p in assets.glob("*_preds_clean_common.npz")}
    if set(files) != {r["arm"] for r in rows}:
        raise ValueError("prediction files and summary arms do not match")
    checked = []
    with tempfile.TemporaryDirectory(prefix="kmerformer-metrics-") as scratch:
        for row in rows:
            path = files[row["arm"]]
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"labels", "preds"}:
                    raise ValueError(f"{path.name}: expected labels and preds only")
                labels, preds = data["labels"], data["preds"]
            if labels.shape != (100000,) or preds.shape != labels.shape:
                raise ValueError(f"{path.name}: wrong read count or shape")
            if not all(np.issubdtype(a.dtype, np.integer) for a in (labels, preds)):
                raise ValueError(f"{path.name}: class IDs must be integers")
            if labels.min() < 0 or labels.max() > 119 or preds.min() < -1 or preds.max() > 119:
                raise ValueError(f"{path.name}: invalid class IDs")
            if fingerprint(labels) != summary["pool_labels_sha1"]:
                raise ValueError(f"{path.name}: wrong pool or read order")
            if fingerprint(preds) != row["preds_sha1"]:
                raise ValueError(f"{path.name}: predictions differ from the archived summary")
            measured = {"read_acc_full": float((preds == labels).mean()),
                        "unclassified_frac": float((preds < 0).mean())}
            if full_metrics:
                metrics = score(path, Path(scratch) / row["arm"])
                measured.update(
                    pearson_r=metrics["abundance_estimation"]["pearson_r_mean"],
                    bray_curtis=metrics["abundance_estimation"]["bray_curtis_mean"],
                    roc_auc=metrics["roc_detection"]["auc"],
                    sens_at_95spec=metrics["roc_detection"]["operating_points"]
                                         ["spec_95pct"]["sensitivity"])
            for key, value in measured.items():
                if not np.isfinite(value) or not np.isclose(value, row[key], rtol=0, atol=1e-8):
                    raise ValueError(f"{row['arm']} {key}: {value} != archived {row[key]}")
            checked.append({"arm": row["arm"], **measured})
            print(f"PASS {row['arm']}: {measured['read_acc_full'] * 100:.4f}%", flush=True)
    return {"arms": checked, "full_metrics": full_metrics,
            "pool_labels_sha1": summary["pool_labels_sha1"],
            "not_checked": ["species-coverage-masked accuracy", "fresh inference",
                            "training reproducibility", "ANI ladder and real mocks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=REPO / "docs" / "assets")
    parser.add_argument("--full-metrics", action="store_true")
    parser.add_argument("--json", type=Path, help="write the verification report")
    args = parser.parse_args()
    try:
        report = verify(args.assets, args.full_metrics)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("Verified 14 bundled arms; coverage-masked results were not checked.")


if __name__ == "__main__":
    main()
