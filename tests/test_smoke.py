#!/usr/bin/env python3
"""CPU-only smoke tests: does the install work, and do the tokenizers still agree?

Runs in seconds, needs no GPU, no data and no network::

    pytest tests/ -v
    python tests/test_smoke.py       # same checks without pytest

These are not accuracy tests -- see docs/RESULTS.md for those. What they cover is
the layer where a broken install or a subtle refactor does silent damage: token
ids. A tokenizer change that shifts ids by one, or that stops being deterministic
across processes, leaves every checkpoint in weights/ quietly wrong rather than
loudly broken, because the ids *are* the model's input vocabulary.
"""

import numpy as np
import pytest
import torch

from kmerformer import (
    HashedKmerTokenizer,
    build_tokenizer,
    create_model,
)
from kmerformer.kmer_tokenizers import CLS_ID, N_SPECIAL, PAD_ID, UNK_ID

# A fixed 150 bp read -- the length every arm in the paper is trained and
# evaluated on, so the token counts below are the real ones.
READ = ("ACCCAGGCATAGTACGAGATAGACCGGCGAATTTCCGGCACGTGGGGAGT"
        "TGAAACTGTGCAGCGCTTGTGGCGAGGGTTCGTCGTAGCTAGTCTTATCC"
        "ACCTCACTAGTGCTCCCTATACTCGGCTGCTATCAGCTATCATGAACAAG")


def test_read_is_150bp():
    """The whole paper is about 150 bp reads; keep the fixture honest."""
    assert len(READ) == 150


# --------------------------------------------------------------------------
# Hashed tokenizer
# --------------------------------------------------------------------------

def test_hashed_token_count_matches_stride():
    tok = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    out = tok(READ)
    # one token per stride-1 13-mer, plus [CLS]
    assert len(out["input_ids"]) == (150 - 13 + 1) + 1
    assert out["input_ids"][0] == CLS_ID


def test_hashed_ids_are_in_range():
    tok = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    ids = np.asarray(tok(READ)["input_ids"])
    assert tok.vocab_size == (1 << 22) + N_SPECIAL
    assert ids.min() >= 0 and ids.max() < tok.vocab_size


def test_hashed_is_deterministic_across_instances():
    """Determinism is load-bearing, not a nicety.

    Python's built-in hash() is salted per process, so using it would give each
    DataLoader worker and each DDP rank a different vocabulary -- corrupting
    training silently. The tokenizer uses a seeded MurmurHash3 finalizer over
    fixed-width uint64 instead, so two instances must agree exactly.
    """
    a = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    b = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    assert a(READ)["input_ids"] == b(READ)["input_ids"]


def test_hashed_ids_match_the_values_the_checkpoints_were_trained_on():
    """Golden ids. Do not update these to make a change pass.

    The hash is a silent property of the released weights. Swap the mixer, the
    seed, the bit-packing order or the special-token offset and every id is still
    a valid index into the embedding table, so the model runs, produces
    confident nonsense, and nothing raises. A second implementation of this
    class exists in the project's history using splitmix64 and a default seed of
    1234 instead of MurmurHash3's fmix64 and 42; over 2,000 k-mer codes the two
    agree on exactly zero bucket assignments, even at a matched seed. It is not
    in this repository and must not be.

    `test_hashed_is_deterministic_across_instances` cannot catch this: two
    instances of the *same wrong code* agree perfectly. Only pinned values can.

    The numbers below were taken from the implementation that trained every
    released hashed arm, whose checkpoints record `hash_seed: 42` and
    `n_buckets: 4194304` in their own config. If this test fails, the tokenizer
    changed and those weights no longer map onto it.
    """
    tok = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    ids = tok(READ)["input_ids"]
    assert len(ids) == 139
    assert ids[:12] == [1, 1017791, 2504391, 3259711, 1841450, 2710497,
                        3228718, 3101940, 1022617, 285247, 3895636, 3324040]
    assert ids[-4:] == [509013, 3574203, 1435382, 897110]
    assert sum(ids) == 317111491

    six = HashedKmerTokenizer(k=6, stride=6, n_buckets=1 << 12, hash_seed=42)
    ids6 = six(READ)["input_ids"]
    assert ids6[:8] == [1, 2493, 2633, 1617, 340, 1328, 1543, 1041]
    assert sum(ids6) == 52247

    # The DEFAULT seed is load-bearing too, and pinning ids while passing the seed
    # explicitly does not pin it: changing the default alone leaves both blocks
    # above green while silently re-bucketing every config that omits hash_seed.
    # Every config in configs/ sets it, but nothing stops a new one from not.
    assert HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22).hash_seed == 42
    assert HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22)(READ)["input_ids"] == ids


def test_build_tokenizer_round_trips_a_checkpoints_own_config():
    """Loading a checkpoint and rebuilding its tokenizer must not lose the seed.

    Every released hashed checkpoint carries its tokenizer block -- type, k,
    stride, n_buckets, hash_seed -- so `build_tokenizer(ckpt["config"])` is the
    reconstruction that cannot silently pick a different vocabulary. This checks
    the path that makes that true, using a config shaped exactly like a stored
    one, seed included and not defaulted.
    """
    stored = {"model": {"backbone": "unused"},
              "data": {"tokenizer": {"type": "hashed_kmer", "k": 13, "stride": 1,
                                     "n_buckets": 4194304, "hash_seed": 42}}}
    tok = build_tokenizer(stored)
    assert (tok.hash_seed, tok.n_buckets, tok.k, tok.stride) == (42, 4194304, 13, 1)
    assert tok(READ)["input_ids"][:4] == [1, 1017791, 2504391, 3259711]

    # a different recorded seed must produce a different vocabulary, or the seed
    # is not being honoured and the round-trip above proves nothing
    other = {"model": {"backbone": "unused"},
             "data": {"tokenizer": {**stored["data"]["tokenizer"], "hash_seed": 1234}}}
    assert build_tokenizer(other)(READ)["input_ids"] != tok(READ)["input_ids"]


def test_hashed_seed_changes_the_vocabulary():
    a = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    b = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=1234)
    assert a(READ)["input_ids"] != b(READ)["input_ids"]


def test_ambiguous_bases_become_unk_not_a_wrong_kmer():
    """An N must not silently alias onto a real k-mer.

    Exactly the windows spanning the N go to UNK: one when it sits at the very
    start of the read, k when it sits anywhere with k-1 bases on both sides.
    """
    tok = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    clean = np.asarray(tok(READ)["input_ids"])

    at_start = np.asarray(tok("N" + READ[1:])["input_ids"])
    assert (at_start == UNK_ID).sum() == 1
    assert at_start[1] == UNK_ID                       # index 0 is [CLS]
    np.testing.assert_array_equal(at_start[2:], clean[2:])

    mid = np.asarray(tok(READ[:70] + "N" + READ[71:])["input_ids"])
    assert (mid == UNK_ID).sum() == 13
    assert (mid[1 + 70 - 12 : 1 + 70 + 1] == UNK_ID).all()


def test_padding_and_mask_line_up():
    """max_token_length 140 holds a 150 bp read's 138 stride-1 13-mers plus [CLS].

    That leaves exactly one PAD, which is why the MetaTransformer-direction
    ablation has a 139-token variant: the two projects disagree by that one slot.
    """
    tok = HashedKmerTokenizer(k=13, stride=1, n_buckets=1 << 22, hash_seed=42)
    out = tok([READ, READ[:60]], max_length=140, padding="max_length",
              truncation=True, return_tensors="pt")
    ids, mask = out["input_ids"], out["attention_mask"]
    assert ids.shape == mask.shape == (2, 140)
    # every masked-out position is PAD, and every PAD position is masked out
    assert (ids[mask == 0] == PAD_ID).all()
    assert (ids[mask == 1] != PAD_ID).all()
    assert mask[0].sum() == (150 - 13 + 1) + 1 == 139
    assert mask[1].sum() == (60 - 13 + 1) + 1 == 49


def test_stride_k_gives_non_overlapping_tokens():
    tok = HashedKmerTokenizer(k=6, stride=6, n_buckets=1 << 12, hash_seed=42)
    out = tok(READ)
    assert len(out["input_ids"]) == 150 // 6 + 1


# --------------------------------------------------------------------------
# Exact tokenizer
#
# The released arms use MetaTransformer's 33,545,099-entry vocab_13mer.txt, which
# is 448 MiB and not in this repo. These build the same thing at k=4 and k=5 in a
# tmpdir, so the code path under test -- fixed-width parse, int32 lookup table,
# canonical auto-detection, cache round-trip -- is the real one. Anything about
# canonicalisation needs an odd k, as 13 is; see the note below.
# --------------------------------------------------------------------------

def _write_vocab(path, kmers):
    """Fixed-width vocab file: k characters plus a newline, one k-mer per line."""
    path.write_text("".join(f"{km}\n" for km in kmers))
    return path


def _all_kmers(k):
    from itertools import product
    return ["".join(p) for p in product("ACGT", repeat=k)]


def _rc(kmer):
    return kmer[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _canonical_vocab(k):
    """One member of each reverse-complement pair, as MetaTransformer's vocab is."""
    return sorted({min(km, _rc(km)) for km in _all_kmers(k)})


def test_odd_k_has_no_self_reverse_complement_kmers():
    """Why the canonical vocabulary is exactly half of 4^k at k=13.

    A k-mer equal to its own reverse complement would be counted once rather than
    twice. Those exist only at even k, so at k=13 the canonical vocabulary covers
    49.99% of 4^13 -- the coverage the tokenizer's auto-detection reports.
    """
    assert not [km for km in _all_kmers(5) if _rc(km) == km]
    assert [km for km in _all_kmers(4) if _rc(km) == km]
    assert len(_canonical_vocab(5)) == 4 ** 5 // 2


def test_exact_tokenizer_on_a_full_vocabulary(tmp_path):
    from kmerformer import ExactKmerTokenizer

    k = 4
    kmers = _all_kmers(k)
    vocab = _write_vocab(tmp_path / "vocab.txt", kmers)
    tok = ExactKmerTokenizer(vocab_path=str(vocab), k=k, stride=1, canonical=False)

    assert tok.n_kmers == 4 ** k
    assert tok.vocab_size == 4 ** k + N_SPECIAL

    ids = np.asarray(tok(READ)["input_ids"])
    assert len(ids) == (150 - k + 1) + 1
    assert not (ids == UNK_ID).any()          # a full vocabulary has no misses
    # the id of a k-mer is its line number, offset past the special tokens
    assert ids[1] == kmers.index(READ[:k]) + N_SPECIAL


def test_exact_tokenizer_autodetects_a_canonical_vocabulary(tmp_path):
    """Without canonicalisation ~half of all real k-mers would go to UNK.

    A canonical vocabulary holds only min(kmer, rc(kmer)), so a raw lookup misses
    every k-mer that lost its coin flip. The tokenizer samples the table to detect
    this and then canonicalises each query -- get it wrong and the model silently
    trains on half-UNK input.

    k must be ODD here, as 13 is: at even k a k-mer can be its own reverse
    complement (ACGT), those palindromes make the "RC also present" rate nonzero,
    and the detector then correctly reports the vocabulary as non-canonical. The
    real vocab_13mer.txt measures 0.00%.
    """
    from kmerformer import ExactKmerTokenizer

    k = 5
    vocab = _write_vocab(tmp_path / "canon.txt", _canonical_vocab(k))
    tok = ExactKmerTokenizer(vocab_path=str(vocab), k=k, stride=1)  # canonical=None

    assert tok.canonical is True
    assert not (np.asarray(tok(READ)["input_ids"]) == UNK_ID).any()

    # forcing it off is what the failure looks like
    naive = ExactKmerTokenizer(vocab_path=str(vocab), k=k, stride=1, canonical=False)
    assert (np.asarray(naive(READ)["input_ids"]) == UNK_ID).any()


def test_exact_tokenizer_canonical_vocab_makes_rc_an_order_permutation(tmp_path):
    """A documented consequence: with a canonical vocabulary a read and its
    reverse complement give the SAME ids in reversed order, so RC augmentation
    permutes positions rather than changing content."""
    from kmerformer import ExactKmerTokenizer

    k = 5   # odd, as 13 is -- see the note in the autodetect test
    vocab = _write_vocab(tmp_path / "canon.txt", _canonical_vocab(k))
    tok = ExactKmerTokenizer(vocab_path=str(vocab), k=k, stride=1)
    assert tok.canonical is True

    fwd = np.asarray(tok(READ)["input_ids"])[1:]
    rev = np.asarray(tok(_rc(READ))["input_ids"])[1:]
    np.testing.assert_array_equal(fwd, rev[::-1])


def test_exact_tokenizer_caches_its_lookup_table(tmp_path):
    from kmerformer import ExactKmerTokenizer

    k = 4
    vocab = _write_vocab(tmp_path / "vocab.txt", _all_kmers(k))
    first = ExactKmerTokenizer(vocab_path=str(vocab), k=k, stride=1, canonical=False)
    cache = tmp_path / f"vocab.txt.table_k{k}.npy"
    assert cache.exists()

    second = ExactKmerTokenizer(vocab_path=str(vocab), k=k, stride=1, canonical=False)
    assert second.vocab_size == first.vocab_size
    assert second(READ)["input_ids"] == first(READ)["input_ids"]


def test_exact_tokenizer_rejects_a_ragged_vocab_file(tmp_path):
    """The parser is vectorised over fixed-width lines, so a ragged file must
    fail loudly instead of producing a shifted vocabulary."""
    from kmerformer import ExactKmerTokenizer

    bad = tmp_path / "ragged.txt"
    bad.write_text("ACGT\nACG\nACGTA\n")
    with pytest.raises(ValueError):
        ExactKmerTokenizer(vocab_path=str(bad), k=4, stride=1, canonical=False)


# --------------------------------------------------------------------------
# build_tokenizer dispatch
# --------------------------------------------------------------------------

def test_build_tokenizer_reads_the_config():
    cfg = {"model": {"backbone": "unused"},
           "data": {"tokenizer": {"type": "hashed_kmer", "k": 13, "stride": 1,
                                  "n_buckets": 1 << 22, "hash_seed": 42}}}
    tok = build_tokenizer(cfg)
    assert isinstance(tok, HashedKmerTokenizer)
    assert tok.vocab_size == (1 << 22) + N_SPECIAL


def test_build_tokenizer_rejects_an_unknown_type():
    cfg = {"model": {"backbone": "unused"},
           "data": {"tokenizer": {"type": "no_such_tokenizer"}}}
    with pytest.raises(ValueError, match="unknown tokenizer type"):
        build_tokenizer(cfg)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def _tiny(**shallow):
    """A KmerFormer small enough to run on a CPU, with the real code path.

    vocab_size is set so no NT-v2 tokenizer is downloaded; everything else is
    the architecture the paper's arms use, only narrower.
    """
    cfg = {
        "type": "shallow_transformer",
        "backbone": "unused",
        "head_type": shallow.pop("head_type", "attention_pool"),
        "head_config": {"hidden_dim": 64, "num_attention_heads": 2, "dropout": 0.1},
        "max_seq_len": 32,
        "vocab_size": 4107,
        "pad_id": 0,
        "shallow_config": {"d_model": 32, "nhead": 2, "d_ff": 64,
                           "num_layers": 2, "dropout": 0.1, **shallow},
    }
    return create_model(cfg, num_classes=120)


@pytest.mark.parametrize("kwargs", [
    {},                                                  # the default arm
    {"head_type": "mean_pool"},                          # the k=13 headline pooling
    {"conv_kernels": [2, 3, 5]},                         # the negative-result front-end
    {"pos_encoding": "sinusoidal", "norm_first": False,   # walked toward MetaTransformer
     "final_norm": True, "embed_scale": "sqrt_d"},
])
def test_forward_shape(kwargs):
    model = _tiny(**kwargs)
    ids = torch.randint(N_SPECIAL, 4107, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    assert model(ids, mask).shape == (4, 120)


def test_padding_does_not_change_a_prediction():
    """Masked positions must not leak. In eval mode a read's logits must be the
    same whether or not the batch happens to pad it."""
    model = _tiny().eval()
    ids = torch.randint(N_SPECIAL, 4107, (1, 20))
    mask = torch.ones(1, 20, dtype=torch.long)
    padded_ids = torch.cat([ids, torch.zeros(1, 12, dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 12, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = model(ids, mask)
        b = model(padded_ids, padded_mask)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_one_training_step_moves_the_loss():
    model = _tiny()
    ids = torch.randint(N_SPECIAL, 4107, (8, 32))
    mask = torch.ones(8, 32, dtype=torch.long)
    target = torch.randint(0, 120, (8,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = torch.nn.functional.cross_entropy(model(ids, mask), target)
    for _ in range(5):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(ids, mask), target).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        after = torch.nn.functional.cross_entropy(model(ids, mask), target)
    assert after < before


def test_sparse_embedding_yields_a_sparse_gradient():
    """This is why the exact-13-mer arms cannot use DDP.

    torch's data-parallel reducer rejects sparse gradients, so the flag that
    makes the 8 GiB table trainable is also what forces single-GPU training.
    If this ever stops being true, the guard in train_ddp.py can go.
    """
    model = _tiny(sparse_embedding=True)
    ids = torch.randint(N_SPECIAL, 4107, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    torch.nn.functional.cross_entropy(
        model(ids, mask), torch.randint(0, 120, (4,))).backward()
    assert model.token_embedding.weight.grad.is_sparse


def test_parameter_count_is_dominated_by_the_vocabulary_at_large_k():
    """The paper's caveat about "one layer": at k=13 the encoder is a rounding
    error next to the embedding table. Checked here at a small scale so the
    claim has a test rather than only a sentence."""
    model = _tiny()
    embed = model.token_embedding.weight.numel()
    rest = sum(p.numel() for n, p in model.named_parameters()
               if not n.startswith("token_embedding"))
    assert embed > 0 and rest > 0
    # At 4,107 tokens x 32 dims the table is small; the point is the ratio scales
    # with vocab_size, which is 8,000x larger for the exact 13-mer arms.
    assert embed == 4107 * 32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
