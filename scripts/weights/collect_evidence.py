#!/usr/bin/env python3
"""Collect existing paper evidence with hashes; perform no inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import yaml



# Machine-specific prefixes rewritten in recorded evidence. The repository is
# public, and a path like /nas2/... names one workstation's disk layout rather
# than anything a reader can resolve; the relative remainder is what identifies
# the file. source_sha256 in the manifest still ties each record to its original.
PORTABLE_PREFIXES = (
    ("/nas2/gfm-classifier/", ""),
    ("/nas2/hierachical_test/data/", "${KF_DATA}/"),
    ("/nas2/data/", "${KF_DATA}/"),
    ("/work/ymj1123ntu/vocab_file/", "${KF_VOCAB}/"),
)


def portable_paths(value):
    """Recursively rewrite machine-specific path prefixes in JSON/YAML data."""
    if isinstance(value, dict):
        return {k: portable_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_paths(v) for v in value]
    if isinstance(value, str):
        for prefix, replacement in PORTABLE_PREFIXES:
            if value.startswith(prefix):
                return replacement + value[len(prefix):]
    return value

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reproduction"))
    args = parser.parse_args()
    output, records = args.output, []
    output.mkdir(parents=True, exist_ok=True)

    def collect(source, relative, family, *, arrays=False, portable_config=False):
        if not source.exists():
            records.append({"path": relative, "source_family": family, "status": "source_not_found"})
            return
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"path": relative, "source_family": family, "source_filename": source.name,
                  "source_sha256": digest(source), "status": "included"}
        if arrays:
            with np.load(source, allow_pickle=False) as archive:
                values = {key: archive[key] for key in ("preds", "labels", "acc_fwd", "acc_rc", "acc_tta",
                    "acc_top1", "macro_f1") if key in archive}
                if not values:
                    # Some archived scalar records use other numeric key names.
                    values = {key: archive[key] for key in archive.files if archive[key].size < 100}
                np.savez_compressed(target, **values)
            record["transformation"] = "Retained prediction/label arrays and small archived metric scalars; omitted probability/logit matrices"
            if "preds" in values and "labels" in values:
                if values["preds"].shape != values["labels"].shape:
                    raise ValueError(f"Prediction/label shapes disagree: {source}")
                record.update(n_reads=int(values["labels"].size),
                              array_accuracy=float(np.mean(values["preds"] == values["labels"])))
        elif portable_config:
            cfg = yaml.safe_load(source.read_text())
            if "data" in cfg and "model" in cfg:
                for field in ("fasta_path", "labels_path"):
                    if field in cfg["data"]:
                        cfg["data"][field] = "${KF_DATA}/soil/ntv2_data/" + Path(cfg["data"][field]).name
                cfg.setdefault("output", {})["dir"] = "${KF_OUT}/soil/" + target.stem
                cfg["model"].setdefault("type", "gfm")
                cfg["data"]["split_strategy"] = "random" if cfg["data"].get("lazy") else "stratified"
                cfg["training"]["seed"] = None
            else:
                # MetaTransformer source configs retain their third-party schema.
                cfg.get("paths", {})["data_path_root"] = "${KF_DATA}/soil"
                cfg = portable_paths(cfg)
            target.write_text(yaml.safe_dump(cfg, sort_keys=False))
            record["transformation"] = "Normalized data/output roots; made KmerFormer model type and historical split/seed explicit"
        elif source.suffix == ".json":
            data = json.loads(source.read_text())
            portable = portable_paths(data)
            if portable != data:
                target.write_text(json.dumps(portable, indent=2) + "\n")
                record["transformation"] = "Rewrote machine-specific path prefixes; values otherwise unchanged"
            else:
                shutil.copyfile(source, target)
                record["transformation"] = "Byte-for-byte copy"
        else:
            shutil.copyfile(source, target)
            record["transformation"] = "Byte-for-byte copy"
        record.update(sha256=digest(target), size_bytes=target.stat().st_size)
        records.append(record)
        print(relative, record.get("n_reads", ""), flush=True)

    paper = args.paper / "supplementary_data"
    for name in ("gut_catalogue.csv", "gut_genera.csv", "soil_genera.csv", "track_a_genomes.csv",
                 "track_a_source.tsv", "track_b_ani_ladder.csv", "sequence_uniqueness_50M.json",
                 "mori_mock_expected.csv", "mori_mock_provenance.json"):
        collect(paper / name, "catalogues/" + name, "manuscript_supplementary_data")
    for folder in ("mori_mock_rescore", "d6331_mock_rescore", "ladder_cluster_bootstrap"):
        for source in sorted((paper / folder).glob("*")):
            if source.suffix in (".csv", ".json"):
                collect(source, "summaries/" + folder + "/" + source.name, "manuscript_supplementary_data")
    nt = paper / "nt_99742_provenance"
    for source in sorted((nt / "predictions").glob("*.npz")):
        collect(source, "nt_99742/predictions/" + source.name, "nt_99742_provenance", arrays=True)
    collect(nt / "membership.tsv.gz", "nt_99742/membership.tsv.gz", "nt_99742_provenance")
    for source in sorted((nt / "filter").glob("*.txt")):
        collect(source, "nt_99742/filter/" + source.name, "nt_99742_provenance")
    for source in sorted((nt / "training").glob("*.csv")):
        collect(source, "nt_99742/training/" + source.name, "nt_99742_provenance")

    for rung in ("L1_strain", "L2_species", "L2_far"):
        for source in sorted((args.experiments / "track_b/out" / rung).glob("*/*.npz")):
            if source.name in ("preds.npz", "rctta.npz"):
                name = source.parent.name + ("_forward" if source.name == "preds.npz" and (source.parent / "rctta.npz").exists() else "")
                collect(source, f"ladder/{rung}/{name}.npz", "track_b_archive", arrays=True)
    for name in ("per_genus.tsv", "ladder_accuracy_matched_genera.tsv", "ani_curve_matched_genera.tsv"):
        collect(args.experiments / "track_b/results" / name, "ladder/" + name, "track_b_archive")
    soil = args.backup / "work/gfm-classifier/soil_tokenization_study/results/predictions"
    for name in ("mt5m_6mer_e128_test_final.npz", "mt50m_6mer_e128_test_final.npz",
                 "ntv2_5M_rctta.npz", "ntv2_50M_rctta.npz"):
        collect(soil / name, "soil/predictions/" + name, "taiwania2_backup_20260828", arrays=True)
    for budget in ("5M", "50M"):
        source = args.backup / f"work/mt5m_soil/results/shallow29_soil_{budget}_15ep/config.yaml"
        collect(source, f"soil/configs/kmerformer_L29_{budget}.yaml", "taiwania2_backup_20260828", portable_config=True)
        source = args.paper / f"reviews/final_consistency_records_20260908/nt_soil_{budget}.yaml"
        collect(source, f"soil/configs/ntv2_lora_{budget}.yaml", "recovered_soil_configs", portable_config=True)
    for relative, name in (("work/mt5m_soil/experiments/genus_5M_soil_6mer_e128_971959/config.yaml", "metatransformer_6mer_e128_5M.yaml"),
                           ("work/mt50m_soil/experiments/genus_50M_soil_6mer_e128_970180/config.yaml", "metatransformer_6mer_e128_50M.yaml")):
        collect(args.backup / relative, "soil/configs/" + name, "taiwania2_backup_20260828", portable_config=True)
    (output / "manifest.json").write_text(json.dumps({"schema_version": 1,
        "operation": "Recovery and packaging of existing evidence; no new model inference or experiments",
        "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
