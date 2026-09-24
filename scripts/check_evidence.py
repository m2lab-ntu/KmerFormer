#!/usr/bin/env python3
"""Verify packaged evidence checksums and explicitly scoped result records."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "reproduction/manifest.json").read_text())
    seen = set()
    included, unavailable = 0, []
    for record in manifest["files"]:
        relative = record["path"]
        if relative in seen:
            raise ValueError(f"Duplicate evidence destination: {relative}")
        seen.add(relative)
        if record["status"] != "included":
            unavailable.append(relative)
            continue
        path = ROOT / "reproduction" / relative
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == record["sha256"], relative
        assert path.stat().st_size == record["size_bytes"], relative
        if "n_reads" in record:
            with np.load(path, allow_pickle=False) as data:
                assert data["labels"].size == record["n_reads"], relative
                assert abs(float(np.mean(data["preds"] == data["labels"])) - record["array_accuracy"]) < 1e-12, relative
        included += 1
    for budget in ("5M", "50M"):
        cfg = yaml.safe_load((ROOT / f"reproduction/soil/configs/metatransformer_6mer_e128_{budget}.yaml").read_text())
        assert cfg["mdl_common"]["kmer_size"] == 6
        assert cfg["model"]["embed_dim"] == 128
    results = json.loads((ROOT / "docs/results_manifest.json").read_text())
    for row in results["readme_rows"]:
        assert (ROOT / row["prediction_asset"]).is_file()
        if row["config"]:
            assert (ROOT / row["config"]).is_file()
    print(f"Verified {included} evidence files; {len(unavailable)} explicitly unavailable source records")
    for relative in unavailable:
        print("Unavailable:", relative)


if __name__ == "__main__":
    main()
