"""Exercise the real training, resume and evaluation commands on tiny local data."""
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


REPO = Path(__file__).resolve().parents[1]


def fixture_config(directory, exact=False):
    directory.mkdir(parents=True, exist_ok=True)
    fasta = directory / "reads.fa"
    rows, records = [], []
    for i in range(31):
        label = i % 3
        name = f"lbl|{label}|genome{i}|{label}|read{i}"
        sequence = ("ACGTACGTACGT", "TTACGGTTACGG", "GCCAGCGCCAGC")[label]
        # Wrapped reads exercise the lazy reader through the real CLI.
        records.append(f">{name}\n{sequence[:6]}\n{sequence[6:]}\n")
        rows.append(dict(idx=i, seq_id=name, species_class=label, genus_class=label,
                         genus_name=f"Genus{label}", species_name=f"Species{label}"))
    fasta.write_text("".join(records))
    labels = directory / "labels.tsv"
    pd.DataFrame(rows).to_csv(labels, sep="\t", index=False)
    tokenizer = dict(type="hashed_kmer", k=3, n_buckets=32, canonical=False)
    if exact:
        vocab = directory / "vocab.txt"
        vocab.write_text("\n".join("".join(x) for x in itertools.product("ACGT", repeat=3)) + "\n")
        tokenizer = dict(type="exact_kmer", k=3, vocab_path=str(vocab), canonical=False)
    cfg = {
        "model": {"type": "shallow_transformer", "backbone": "unused",
                  "vocab_size": 67 if exact else 35, "pad_id": 0, "max_seq_len": 16,
                  "head_type": "mean_pool", "head_config": {"hidden_dim": 8, "dropout": 0.0},
                  "shallow_config": {"d_model": 8, "nhead": 2, "d_ff": 16, "num_layers": 1,
                                     "dropout": 0.0, "sparse_embedding": exact}},
        "data": {"task": "genus", "lazy": True, "fasta_path": str(fasta),
                 "labels_path": str(labels), "val_ratio": 0.17, "seed": 17,
                 "rc_augment": False, "max_token_length": 16, "tokenizer": tokenizer},
        "training": {"batch_size": 4, "eval_batch_size": 4, "num_workers": 0,
                     "gradient_accumulation_steps": 4, "num_epochs": 1,
                     "learning_rate": 0.002, "backbone_lr": 0.001, "class_weights": False,
                     "amp": False, "lr_schedule": "none", "early_stopping_patience": 5},
        "output": {"dir": str(directory / "output")},
    }
    return cfg, records, rows


def run_cli(directory, name, arguments):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, *arguments], cwd=directory, env=env,
                            capture_output=True, text=True, timeout=100)
    log = directory / f"{name}.log"
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0, f"{name} failed; {log}:\n{log.read_text()[-6000:]}"
    return result.stdout


@pytest.mark.parametrize("exact", [False, True], ids=["hashed", "exact_sparse"])
def test_cpu_train_resume_and_subset_rc_evaluation(tmp_path, exact):
    cfg, records, rows = fixture_config(tmp_path, exact)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run_cli(tmp_path, "train", ["-m", "kmerformer.train", "--config", str(config)])
    output = Path(cfg["output"]["dir"])
    last = output / "last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["scheduler_state_dict"]["last_epoch"] == 2  # six batches, accumulation four
    if exact:
        assert checkpoint["sparse_scheduler_state_dict"]["last_epoch"] == 2
        assert checkpoint["sparse_optimizer_state_dict"]["state"]
        checkpoint["best_val_acc"] = 1.0
        checkpoint["patience_counter"] = 4
    else:
        checkpoint["best_val_acc"] = 0.0
        checkpoint["val_acc"] = 1.0  # best=0 must not fall through to this value
    torch.save(checkpoint, last)
    cfg["training"]["num_epochs"] = 2
    config.write_text(yaml.safe_dump(cfg))
    resume_log = run_cli(tmp_path, "resume", ["-m", "kmerformer.train", "--config", str(config), "--resume", str(last)])
    resumed = torch.load(last, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert resumed["scheduler_state_dict"]["last_epoch"] == 4
    assert "scaler_state_dict" in resumed
    if exact:
        assert resumed["patience_counter"] == 5
        assert "Early stopping at epoch 2" in resume_log
    else:
        assert "Restored early stopping: best=0.0000" in resume_log

    # A subset contains IDs 1 and 2, while the trained head has three classes.
    subset_fasta = tmp_path / "subset.fa"
    subset_labels = tmp_path / "subset.tsv"
    subset_fasta.write_text(records[1] + records[2])
    pd.DataFrame([rows[1], rows[2]]).to_csv(subset_labels, sep="\t", index=False)
    cfg["data"]["task"] = "species" if exact else "genus"
    config.write_text(yaml.safe_dump(cfg))
    run_cli(tmp_path, "evaluate", ["-m", "kmerformer.evaluate", "--config", str(config),
                                  "--checkpoint", str(last), "--test_fasta", str(subset_fasta),
                                  "--test_labels", str(subset_labels), "--num_classes", "3", "--rc_tta"])
    with np.load(output / "predictions_rc_tta.npz") as predictions:
        np.testing.assert_array_equal(predictions["labels"], [1, 2])
        assert predictions["logits"].shape == (2, 3)
        np.testing.assert_allclose(predictions["logits"],
                                   (predictions["fwd_logits"] + predictions["rc_logits"]) / 2)
    assert (output / "eval_metrics_rc_tta.json").is_file()


def test_torchrun_resume_and_incomplete_epoch_checkpoint(tmp_path):
    cfg, _, _ = fixture_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    launch = ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
              "--module", "kmerformer.train_ddp", "--config", str(config)]
    run_cli(tmp_path, "ddp_train", launch)
    last = Path(cfg["output"]["dir"]) / "last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_state_dict"]["last_epoch"] == 1
    assert checkpoint["optimizer_state_dict"]["param_groups"][1]["params"]
    checkpoint["best_val_acc"] = 1.0
    checkpoint["patience_counter"] = 2
    torch.save(checkpoint, last)
    cfg["training"]["num_epochs"] = 2
    config.write_text(yaml.safe_dump(cfg))
    run_cli(tmp_path, "ddp_resume", launch + ["--resume", str(last)])
    resumed = torch.load(last, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert resumed["scheduler_state_dict"]["last_epoch"] == 2
    assert resumed["best_val_acc"] == 1.0
    assert resumed["patience_counter"] == 3
    cfg["output"]["dir"] = str(tmp_path / "timeout_output")
    config.write_text(yaml.safe_dump(cfg))
    run_cli(tmp_path, "ddp_timeout", launch + ["--time_limit_sec", "0"])
    stopped = torch.load(Path(cfg["output"]["dir"]) / "last.pt", map_location="cpu", weights_only=False)
    assert stopped["epoch"] == 0
    assert stopped["scheduler_state_dict"]["last_epoch"] == 0
    assert not (Path(cfg["output"]["dir"]) / "training_history.csv").exists()
