#!/usr/bin/env python3
"""Check the released 6-mer bundle against the fixed 16-read reference output."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from kmerformer import ModelBundle, iter_reads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=Path, default=Path(__file__).resolve().parents[1] / "examples")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = ModelBundle(args.model, device=args.device)
    actual = list(model.predict(iter_reads(args.examples / "reads.fa")))
    with (args.examples / "expected_predictions.tsv").open() as source:
        expected = list(csv.DictReader(source, delimiter="\t"))
    assert len(actual) == len(expected) == 16
    for found, reference in zip(actual, expected):
        assert found["read_id"] == reference["read_id"]
        assert found["class_id"] == int(reference["class_id"])
        assert found["taxon_name"] == reference["taxon_name"]
        assert found["model_id"] == reference["model_id"]
        np.testing.assert_allclose(found["score"], float(reference["score"]), rtol=1e-4, atol=1e-6)
    print(f"Verified {len(actual)} actual-model predictions on {args.device}")


if __name__ == "__main__":
    main()
