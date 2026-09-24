"""Public inference contracts, independent of training data and downloads."""
import csv
import gzip
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from kmerformer import ModelBundle, Read, create_model, iter_reads
from kmerformer.bundle import export_bundle, verify_bundle
from kmerformer.data_loader import load_data
from kmerformer.dataset_lazy import LazyFASTADataset
from kmerformer.kmer_tokenizers import HashedKmerTokenizer
from kmerformer.reproducibility import capture_rng, restore_rng, seed_training
from kmerformer.splits import split_indices
from kmerformer.stream_evaluation import OnlineMetrics


@pytest.fixture
def bundle_fixture(tmp_path):
    torch.manual_seed(4)
    cfg = {"model": {"type": "kmerformer", "vocab_size": 67, "pad_id": 0,
                     "head_type": "mean_pool", "head_config": {"hidden_dim": 8, "dropout": 0.0},
                     "max_seq_len": 32, "shallow_config": {"d_model": 8, "nhead": 2,
                     "d_ff": 16, "num_layers": 1, "dropout": 0.0}},
           "data": {"task": "genus", "max_token_length": 32,
                    "tokenizer": {"type": "hashed_kmer", "k": 3, "n_buckets": 64}},
           "training": {"batch_size": 2, "amp": False}, "output": {"dir": str(tmp_path / "eval")}}
    model = create_model(cfg["model"], 3).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint)
    labels = tmp_path / "labels.tsv"
    labels.write_text("seq_id\tgenus_class\tgenus_name\na\t0\tAlpha\nb\t1\tBeta\nc\t2\tGamma\n")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "bundle"
    export_bundle(checkpoint, cfg, labels, output, model_id="fixture")
    return output, cfg, checkpoint, labels, config


def test_unlabelled_reads_predict_and_roundtrip(bundle_fixture):
    path, _, _, _, _ = bundle_fixture
    bundle = ModelBundle(path)
    reads = [Read("first", "ACGTACGT"), Read("second", "TGCATGCA"), Read("third", "AAAAACCC")]
    one = list(bundle.predict(iter(reads), batch_size=1))
    several = list(bundle.predict(iter(reads), batch_size=2))
    assert [p["read_id"] for p in one] == [r.read_id for r in reads]
    assert [p["class_id"] for p in one] == [p["class_id"] for p in several]
    np.testing.assert_allclose([p["score"] for p in one], [p["score"] for p in several], rtol=1e-5)
    assert all(p["taxon_name"] in ("Alpha", "Beta", "Gamma") for p in one)


def test_bundle_corruption_is_rejected_before_model_load(bundle_fixture):
    path, *_ = bundle_fixture
    (path / "labels.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        ModelBundle(path)


def test_export_requires_explicit_weight_terms(bundle_fixture):
    path, *_ = bundle_fixture
    manifest = verify_bundle(path)
    assert manifest["weights_licence"] == "LicenseRef-Pending"
    assert manifest["tokenizer_licence"] == "MIT"
    assert "WEIGHTS_LICENSE.txt" not in manifest["files"]
    assert "TOKENIZER_LICENSE.txt" in manifest["files"]


def test_apache_export_preserves_weights_and_tokenizer_terms(bundle_fixture, tmp_path):
    path, cfg, checkpoint, labels, _ = bundle_fixture
    target = tmp_path / "apache-bundle"
    manifest = export_bundle(checkpoint, cfg, labels, target, model_id="fixture", licence="Apache-2.0")
    original = verify_bundle(path)
    assert manifest["weights_licence"] == "Apache-2.0"
    assert manifest["weights_licence_scope"] == "model.safetensors"
    assert manifest["tokenizer_licence"] == original["tokenizer_licence"] == "MIT"
    for name in ("model.safetensors", "config.json", "labels.json", "TOKENIZER_LICENSE.txt"):
        assert manifest["files"][name] == original["files"][name]
    assert b"Grant of Patent License" in (target / "WEIGHTS_LICENSE.txt").read_bytes()
    (target / "WEIGHTS_LICENSE.txt").write_text("Changed licence text")
    with pytest.raises(ValueError, match="WEIGHTS_LICENSE.txt"):
        verify_bundle(target)


def test_bundle_manifest_rejects_path_traversal(bundle_fixture):
    path, *_ = bundle_fixture
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["../outside"] = {"sha256": "x", "size_bytes": 1}
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        verify_bundle(path)


def test_export_rejects_wrong_label_catalogue(bundle_fixture, tmp_path):
    _, cfg, checkpoint, _, _ = bundle_fixture
    labels = tmp_path / "wrong.tsv"
    labels.write_text("genus_class\tgenus_name\n0\tOnlyOne\n")
    with pytest.raises(ValueError, match="output width"):
        export_bundle(checkpoint, cfg, labels, tmp_path / "bad", model_id="bad")


def test_multiline_gzipped_fastq_and_fasta(tmp_path):
    path = tmp_path / "reads.fastq.gz"
    with gzip.open(path, "wt") as f:
        f.write("@one description\nacgt\nNN\n+\n!!!!\n!!\n@two\nTGCA\n+\n@@@@\n")
    assert list(iter_reads(path)) == [Read("one description", "ACGTNN"), Read("two", "TGCA")]
    path = tmp_path / "reads.fa"
    path.write_text(">first\nACGT\nTGCA\n>second\nN\n")
    assert list(iter_reads(path)) == [Read("first", "ACGTTGCA"), Read("second", "N")]


@pytest.mark.parametrize("text", ["@read\nACGT\n+\n!!\n", "@read\nACGT\n+\n!!!!!\n", ">empty\n", ">bad\nACX\n", "", "@one\nAC\n+\n!!\n\n@two\nTG\n+\n!!\n"])
def test_malformed_reads_fail_explicitly(tmp_path, text):
    path = tmp_path / "bad.reads"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_reads(path))


def test_prediction_normalizes_api_reads_and_restores_runtime_settings(bundle_fixture):
    from kmerformer.predict import inference_kernels
    bundle = ModelBundle(bundle_fixture[0])
    assert list(bundle.predict([Read("a", "acgt")])) == list(bundle.predict([Read("a", "ACGT")]))
    with pytest.raises(ValueError, match="IUPAC"):
        list(bundle.predict([Read("bad", "ACX")]))
    fastpath = torch.backends.mha.get_fastpath_enabled()
    precision = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError):
        with inference_kernels("fp32"):
            assert not torch.backends.mha.get_fastpath_enabled()
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("model failure")
    assert torch.backends.mha.get_fastpath_enabled() == fastpath
    assert torch.get_float32_matmul_precision() == precision


def test_online_metrics_match_full_array_metrics():
    from kmerformer.evaluate import compute_metrics
    rng = np.random.default_rng(10)
    probabilities = rng.random((103, 12))
    probabilities /= probabilities.sum(1, keepdims=True)
    labels = rng.choice([0, 3, 11], size=103)
    online = OnlineMetrics(12)
    for offset in range(0, len(labels), 7):
        online.update(labels[offset:offset + 7], probabilities[offset:offset + 7])
    reference = compute_metrics(labels, probabilities.argmax(1), probabilities, 12)
    assert online.result() == pytest.approx(reference)


@pytest.mark.parametrize("strategy", ["stratified", "random"])
def test_eager_and_indexed_paths_share_manifest(tmp_path, strategy):
    fasta, labels = tmp_path / "reads.fa", tmp_path / "labels.tsv"
    lines, rows = [], ["seq_id\tgenus_class\tgenus_name\tspecies_class\tspecies_name"]
    for i in range(30):
        header = f"lbl|{7 if i % 2 else 119}|genome|{i % 2}|read{i}"
        lines.append(f">{header}\nACGTACGT\n")
        rows.append(f"{header}\t{i % 2}\tg{i % 2}\t{7 if i % 2 else 119}\ts{i % 2}")
    fasta.write_text("".join(lines)); labels.write_text("\n".join(rows) + "\n")
    manifest = tmp_path / "split.npz"
    eager = load_data(fasta, labels, .2, 42, split_strategy=strategy, split_manifest=manifest)
    lazy = LazyFASTADataset(fasta, labels, HashedKmerTokenizer(k=3, n_buckets=64), val_ratio=.2,
                           split="val", split_strategy=strategy, split_manifest=manifest)
    with np.load(manifest) as split:
        assert np.array_equal(lazy._idx, split["val"])
    assert np.array_equal(eager["val_genus_labels"], lazy.get_labels())
    species = load_data(fasta, labels, .2, 42, task="species")
    assert species["num_species"] == 120
    assert set(species["train_species_labels"]) == {7, 119}
    # Same labels and lengths, different sequence contents: fail the manifest.
    fasta.write_text(fasta.read_text().replace("ACGTACGT", "TGCATGCA"))
    with pytest.raises(ValueError, match="does not match"):
        load_data(fasta, labels, .2, 42, split_strategy=strategy, split_manifest=manifest)


def test_rng_checkpoint_restores_all_cpu_generators():
    seed_training(7)
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    seed_training(99)
    assert restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


@pytest.mark.parametrize("skip", [False, True])
def test_streamed_evaluation_outputs_match_bundle_predictions(bundle_fixture, tmp_path, skip):
    from argparse import Namespace
    from kmerformer.stream_evaluation import evaluate_stream
    path, cfg, checkpoint, labels, _ = bundle_fixture
    fasta = tmp_path / "reads.fa"
    fasta.write_text(">a\nACGTACGT\n>b\nTGCAACGT\n>c\nAAAACCCC\n")
    args = Namespace(output_dir=str(tmp_path / "stream"), device="cpu", checkpoint=str(checkpoint),
                     num_classes=None, precision="fp32", batch_size=2, rc_tta=True,
                     test_fasta=str(fasta), test_labels=str(labels), skip_save_logits=skip)
    result = evaluate_stream(args, cfg)
    predicted = list(ModelBundle(path).predict(iter_reads(fasta), batch_size=2))
    with np.load(Path(args.output_dir) / "predictions_rc_tta.npz") as output:
        assert output["preds"].tolist() == [r["class_id"] for r in predicted]
        assert output["labels"].tolist() == [0, 1, 2]
        if skip:
            assert set(output.files) == {"preds", "labels"}
        else:
            np.testing.assert_allclose(output["logits"], (output["fwd_logits"] + output["rc_logits"]) / 2)
    assert result["num_reads"] == 3 and result["streaming"]
    assert not list(Path(args.output_dir).glob(".evaluation-*"))
