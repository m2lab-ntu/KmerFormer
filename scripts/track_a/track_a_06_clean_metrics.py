#!/usr/bin/env python3
"""Recompute Track A Top-1 after excluding training-reference accessions."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-fasta", required=True)
    parser.add_argument("--exclude-accessions", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=NPZ",
        help="Repeat for each model; NPZ must contain preds and labels.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def accession_base(value):
    match = re.match(r"(GC[AF]_\d+)", value)
    return match.group(1) if match else value.split(".", 1)[0]


def read_exclusions(path):
    with open(path) as handle:
        return {
            accession_base(line.strip())
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def read_fasta_metadata(path):
    accessions, genus_classes, genus_names = [], [], []
    with open(path) as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            fields = line[1:].rstrip().split("|")
            if len(fields) < 5:
                raise ValueError(f"Unexpected Track A header: {line.rstrip()}")
            accessions.append(fields[2])
            genus_classes.append(int(fields[3]))
            genus_names.append(fields[4].rsplit("-", 1)[0])
    return np.asarray(accessions), np.asarray(genus_classes), np.asarray(genus_names)


def main():
    args = parse_args()
    exclusions = read_exclusions(args.exclude_accessions)
    accessions, header_labels, genus_names = read_fasta_metadata(args.test_fasta)
    accession_bases = np.asarray([accession_base(value) for value in accessions])
    excluded = np.isin(accession_bases, list(exclusions))
    keep = ~excluded

    excluded_counts = Counter(accessions[excluded])
    excluded_found = sorted(set(accession_bases[excluded]))
    all_genera = sorted(set(genus_names))
    clean_genera = sorted(set(genus_names[keep]))

    result = {
        "test_reads_total": int(len(accessions)),
        "excluded_reads": int(excluded.sum()),
        "excluded_pct": round(float(excluded.mean() * 100), 4),
        "clean_reads": int(keep.sum()),
        "genera_total": len(all_genera),
        "genera_clean": len(clean_genera),
        "genera_removed": sorted(set(all_genera) - set(clean_genera)),
        "excluded_accessions_found": excluded_found,
        "excluded_accession_read_counts": dict(sorted(excluded_counts.items())),
        "models": {},
    }

    for specification in args.prediction:
        if "=" not in specification:
            raise ValueError(f"Expected NAME=NPZ, got: {specification}")
        name, npz_path = specification.split("=", 1)
        data = np.load(npz_path)
        predictions = data["preds"]
        labels = data["labels"]
        if len(labels) != len(accessions):
            raise ValueError(
                f"{name}: {len(labels):,} predictions but "
                f"{len(accessions):,} FASTA records"
            )
        if not np.array_equal(labels, header_labels):
            raise ValueError(f"{name}: NPZ labels do not align with FASTA headers")
        result["models"][name] = {
            "original_correct": int((predictions == labels).sum()),
            "original_top1_pct": round(float((predictions == labels).mean() * 100), 4),
            "clean_correct": int((predictions[keep] == labels[keep]).sum()),
            "clean_top1_pct": round(
                float((predictions[keep] == labels[keep]).mean() * 100), 4
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
