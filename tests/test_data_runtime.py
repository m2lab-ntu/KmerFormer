"""Read and vocabulary integrity across ambiguous bases, wrapping and caches."""

import itertools
import pickle
from pathlib import Path

import numpy as np
import pytest

from kmerformer.dataset_lazy import LazyFASTADataset
from kmerformer.kmer_tokenizers import (
    CLS_ID, UNK_ID, ExactKmerTokenizer, HashedKmerTokenizer,
)


@pytest.fixture
def vocabulary(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("".join("".join(k) + "\n"
                         for k in itertools.product("ACGT", repeat=3)))
    return p


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("read", ["NAA", "ANN", "NNN", "?TT", "nAC"])
def test_exact_ambiguous_kmers_are_unknown(vocabulary, read, canonical):
    tok = ExactKmerTokenizer(vocabulary, k=3, canonical=canonical)
    assert tok(read)["input_ids"] == [CLS_ID, UNK_ID]


def test_exact_cache_follows_vocabulary_order(vocabulary):
    first = ExactKmerTokenizer(vocabulary, k=3, canonical=False)
    before = first("AAAC")["input_ids"]
    lines = vocabulary.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    vocabulary.write_text("\n".join(lines) + "\n")
    second = ExactKmerTokenizer(vocabulary, k=3, canonical=False)
    assert second("AAAC")["input_ids"] == [CLS_ID, before[2], before[1]]


def test_incomplete_cache_is_rebuilt(vocabulary):
    tok = ExactKmerTokenizer(vocabulary, k=3, canonical=False)
    expected = tok("ACGT")["input_ids"]
    cache = vocabulary.with_name(vocabulary.name + ".table_k3.npy")
    cache.write_bytes(b"incomplete")
    assert ExactKmerTokenizer(vocabulary, k=3, canonical=False)("ACGT")["input_ids"] == expected


@pytest.mark.parametrize("kw", [{"k": 0}, {"k": 33}, {"stride": 0},
                                {"n_buckets": 0}, {"hash_seed": -1}])
def test_invalid_hash_configuration_fails_early(kw):
    with pytest.raises(ValueError):
        HashedKmerTokenizer(**kw)


def _dataset(tmp_path, *, name="reads.fa", sequence="ACGT\nTGCA", **kwargs):
    fasta = tmp_path / name
    if not fasta.exists():
        fasta.write_text(f">lbl|0|g|0|sp/0\n{sequence}\n"
                         ">lbl|1|h|1|sp/1\nTTTT\nCCCC\n")
    labels = tmp_path / "labels.tsv"
    labels.write_text("species_class\tgenus_class\n0\t0\n1\t1\n")
    tok = HashedKmerTokenizer(k=3, n_buckets=32)
    return LazyFASTADataset(fasta, labels, tok, split="train", val_ratio=0, **kwargs)


def test_lazy_reads_all_sequence_lines_and_stops_at_next_header(tmp_path):
    data = _dataset(tmp_path)
    assert data._read_seq(0) == "ACGTTGCA"
    assert data._read_seq(1) == "TTTTCCCC"


def test_lazy_can_be_pickled_after_parent_reads(tmp_path):
    data = _dataset(tmp_path)
    data._read_seq(0)
    restored = pickle.loads(pickle.dumps(data))
    assert restored._read_seq(1) == "TTTTCCCC"
    assert data._read_seq(0) == "ACGTTGCA"


def test_lazy_cache_does_not_alias_fasta_extensions(tmp_path):
    _dataset(tmp_path, name="reads.fa")
    other = _dataset(tmp_path, name="reads.fasta", sequence="A" * 200)
    assert other._read_seq(1) == "TTTTCCCC"


def test_lazy_rebuilds_index_when_source_changes(tmp_path):
    data = _dataset(tmp_path)
    path = tmp_path / "reads.fa"
    path.write_text(path.read_text().replace("ACGT\nTGCA", "A" * 200))
    refreshed = _dataset(tmp_path)
    assert refreshed._read_seq(1) == "TTTTCCCC"
    np.testing.assert_array_equal(data._idx, refreshed._idx)


def test_lazy_reports_invalid_labels_in_header(tmp_path):
    (tmp_path / "reads.fa").write_text(">unlabelled\nACGT\n")
    with pytest.raises(ValueError, match="labelled FASTA header"):
        _dataset(tmp_path)
def test_competing_cache_writer_cannot_relabel_another_array(tmp_path, monkeypatch):
    from kmerformer import array_cache
    cache = tmp_path / "cache.npy"
    first, second = np.arange(4), np.arange(4) + 100
    replace = array_cache.os.replace
    raced = False

    def interleaved_replace(source, destination):
        nonlocal raced
        replace(source, destination)
        if Path(destination) == cache and not raced:
            raced = True
            array_cache.save_array_cache(cache, second, {"source": "second"})

    monkeypatch.setattr(array_cache.os, "replace", interleaved_replace)
    array_cache.save_array_cache(cache, first, {"source": "first"})
    loaded = array_cache.load_array_cache(cache, {"source": "first"})
    assert loaded is None or np.array_equal(loaded, first)


def test_valid_json_with_invalid_cache_metadata_is_rebuilt(tmp_path):
    from kmerformer.array_cache import load_array_cache
    cache = tmp_path / "cache.npy"
    Path(str(cache) + ".meta.json").write_text("[]")
    assert load_array_cache(cache, {}) is None
