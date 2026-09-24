#!/usr/bin/env python3
"""
Custom k-mer tokenizers for the tokenization-axis study.

Motivation: NT-v2's tokenizer only knows 6-mers (vocab 4107). To put 13-mers on
the same architecture we need our own vocabulary. Two variants:

  HashedKmerTokenizer — hashes each stride-1 k-mer into n_buckets slots.
      13-mer, 2^22 buckets, d_model 128 -> 2 GB embedding instead of the
      48 GB a full 4^13 table needs with Adam moments. This is the recipe the
      paper recommends.

  ExactKmerTokenizer  — one id per k-mer actually observed in a vocab file
      (MetaTransformer's `vocab_13mer.txt`, 33,545,099 entries). Used by Job A.

Both expose the minimal HuggingFace-tokenizer surface the datasets rely on:
    tok(text, max_length=, padding="max_length"|False, truncation=, return_tensors="pt"|None)
      -> {"input_ids": ..., "attention_mask": ...}
    .vocab_size, .pad_token_id, .unk_token_id, .cls_token_id

Determinism (critical): hashing uses the MurmurHash3 64-bit finalizer over the
2-bit-packed k-mer with an explicit seed from the config. Python's built-in
hash() is NOT used — it is salted per process (PYTHONHASHSEED), which would give
different ids in every DataLoader worker and every DDP rank, silently corrupting
training. All arithmetic is fixed-width numpy uint64, so ids are reproducible
across processes, nodes, and runs.

Token layout mirrors the NT-v2 path so the attention-pool head sees the same
structure: [CLS] + one token per k-mer, right-padded with PAD.
"""

import numpy as np
from pathlib import Path

from .array_cache import file_identity, load_array_cache, save_array_cache

PAD_ID, CLS_ID, UNK_ID = 0, 1, 2
N_SPECIAL = 3

# base -> 2-bit code; anything else (N, IUPAC codes) -> 4 == invalid
_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _b, _v in zip(b"ACGT", (0, 1, 2, 3)):
    _BASE_LUT[_b] = _v
for _b, _v in zip(b"acgt", (0, 1, 2, 3)):
    _BASE_LUT[_b] = _v

_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_S33 = np.uint64(33)


def _fmix64(h: np.ndarray, seed: int) -> np.ndarray:
    """MurmurHash3 64-bit finalizer — deterministic avalanche mix, seeded."""
    h = (h ^ np.uint64(seed)).astype(np.uint64)
    h ^= h >> _S33
    h *= _M1
    h ^= h >> _S33
    h *= _M2
    h ^= h >> _S33
    return h


class _BaseKmerTokenizer:
    """Shared k-merisation, padding and tensor packing."""

    def __init__(self, k: int, stride: int = 1):
        self.k = int(k)
        self.stride = int(stride)
        if not 1 <= self.k <= 32:
            raise ValueError("k must be between 1 and 32 for uint64 k-mer codes")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.pad_token_id = PAD_ID
        self.cls_token_id = CLS_ID
        self.unk_token_id = UNK_ID

    # ---- subclasses map k-mer codes -> token ids -----------------------------
    def _codes_to_ids(self, codes: np.ndarray, valid: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _encode_one(self, seq: str) -> np.ndarray:
        b = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        codes_b = _BASE_LUT[b]
        n = len(codes_b) - self.k + 1
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        w = np.lib.stride_tricks.sliding_window_view(codes_b, self.k)
        if self.stride != 1:
            w = w[:: self.stride]
        valid = ~(w > 3).any(axis=1)
        # 2-bit pack: most-significant base first (stable, order-preserving)
        powers = (np.uint64(4) ** np.arange(self.k - 1, -1, -1, dtype=np.uint64))
        codes = w.astype(np.uint64) @ powers
        return self._codes_to_ids(codes, valid)

    def __call__(self, text, max_length=None, padding=False, truncation=False,
                 return_tensors=None, **kw):
        single = isinstance(text, str)
        seqs = [text] if single else list(text)
        rows = [self._encode_one(s) for s in seqs]

        out_ids, out_mask = [], []
        for ids in rows:
            seq_ids = np.concatenate(([CLS_ID], ids)).astype(np.int64)
            if truncation and max_length is not None and len(seq_ids) > max_length:
                seq_ids = seq_ids[:max_length]
            mask = np.ones(len(seq_ids), dtype=np.int64)
            if padding == "max_length" and max_length is not None:
                pad = max_length - len(seq_ids)
                if pad > 0:
                    seq_ids = np.concatenate((seq_ids, np.full(pad, PAD_ID, dtype=np.int64)))
                    mask = np.concatenate((mask, np.zeros(pad, dtype=np.int64)))
            out_ids.append(seq_ids)
            out_mask.append(mask)

        if return_tensors == "pt":
            import torch
            # only stackable when every row is the same length (padding="max_length")
            if len({len(r) for r in out_ids}) == 1:
                return {"input_ids": torch.from_numpy(np.stack(out_ids)),
                        "attention_mask": torch.from_numpy(np.stack(out_mask))}
            return {"input_ids": [torch.from_numpy(r) for r in out_ids],
                    "attention_mask": [torch.from_numpy(m) for m in out_mask]}
        if single:
            return {"input_ids": out_ids[0].tolist(),
                    "attention_mask": out_mask[0].tolist()}
        return {"input_ids": [r.tolist() for r in out_ids],
                "attention_mask": [m.tolist() for m in out_mask]}


class HashedKmerTokenizer(_BaseKmerTokenizer):
    """Stride-1 k-mers hashed into n_buckets slots (Job C).

    NOT CANONICALISED, unlike ExactKmerTokenizer. A k-mer and its reverse
    complement land in unrelated buckets here, where the exact vocabulary maps
    both to one row. That asymmetry is a real confound in the exact-vs-hashed
    contrast the paper reports, and it is disclosed there rather than corrected:
    the trained arms are what they are, and adding canonicalisation now would
    describe a model nobody trained. Do not "fix" it.

    The mixer and the default seed are load-bearing. A separate implementation of
    this class exists in the project's history using splitmix64 with a default
    seed of 1234; over 2,000 k-mer codes it agrees with this one on exactly zero
    bucket assignments, even at a matched seed. Swapping either leaves every id a
    valid table index, so a checkpoint trained here runs under it and produces
    confident nonsense with no error. tests/test_smoke.py pins the ids against
    that, and every released hashed checkpoint records its own hash_seed and
    n_buckets so build_tokenizer(ckpt["config"]) reconstructs the right one.
    """

    def __init__(self, k=13, stride=1, n_buckets=1 << 22, hash_seed=42):
        super().__init__(k, stride)
        self.n_buckets = int(n_buckets)
        self.hash_seed = int(hash_seed)
        if not 1 <= self.n_buckets <= np.iinfo(np.int64).max - N_SPECIAL:
            raise ValueError("n_buckets must be positive and fit in int64 token IDs")
        if not 0 <= self.hash_seed <= np.iinfo(np.uint64).max:
            raise ValueError("hash_seed must fit in an unsigned 64-bit integer")
        self._pow2 = (self.n_buckets & (self.n_buckets - 1)) == 0
        self._mask = np.uint64(self.n_buckets - 1)
        self.vocab_size = self.n_buckets + N_SPECIAL

    def _codes_to_ids(self, codes, valid):
        h = _fmix64(codes, self.hash_seed)
        bucket = (h & self._mask) if self._pow2 else (h % np.uint64(self.n_buckets))
        ids = (bucket + np.uint64(N_SPECIAL)).astype(np.int64)
        ids[~valid] = UNK_ID
        return ids


def _rc_codes(cd: np.ndarray, k: int) -> np.ndarray:
    """Reverse-complement of 2-bit-packed k-mer codes (vectorised)."""
    cd = cd.copy()
    out = np.zeros_like(cd)
    four, three = np.uint64(4), np.uint64(3)
    for _ in range(k):
        out = out * four + (three - (cd % four))
        cd = cd // four
    return out


class ExactKmerTokenizer(_BaseKmerTokenizer):
    """One id per k-mer listed in a vocab file (Job A, MetaTransformer vocab).

    The vocab file is a plain text list of fixed-width k-mers, one per line; the
    line number is the id. Lookup is an int32 table indexed by the 2-bit code:
    4^k * 4 bytes (13-mer -> 256 MiB), O(1), no Python dict. The table is cached
    next to the vocab file so later runs load it in seconds.

    CANONICAL VOCABULARIES: MetaTransformer's vocab_13mer.txt holds only one
    member of each reverse-complement pair — verified: it covers 49.99% of 4^13
    and 0.00% of its k-mers have their RC also present, every entry being the
    min() of its pair. Looking up raw codes against such a vocab would send ~50%
    of real k-mers to UNK, so codes must be canonicalised to min(code, rc(code))
    first. This is auto-detected (canonical=None) and can be forced either way.

    Side effect worth knowing: with a canonical vocabulary a read and its reverse
    complement yield the SAME token ids in reversed order, so RC augmentation
    becomes an order permutation rather than a content change.
    """

    def __init__(self, vocab_path, k=13, stride=1, canonical=None, cache_path=None):
        super().__init__(k, stride)
        powers = (np.uint64(4) ** np.arange(self.k - 1, -1, -1, dtype=np.uint64))
        cache = Path(cache_path) if cache_path else \
            Path(str(vocab_path) + f".table_k{self.k}.npy")

        codes = None
        # NOTE: loaded fully into RAM on purpose. With mmap_mode="r" the table
        # lives on NFS and every random k-mer lookup becomes an NFS page fault —
        # measured 719 reads/s end-to-end vs a 3,158 reads/s model ceiling.
        # DataLoader workers fork after this, so copy-on-write shares one copy.
        identity = {"vocab": file_identity(vocab_path, content_hash=True),
                    "k": self.k, "version": 1}
        self._table = load_array_cache(cache, identity)
        if self._table is not None and (self._table.shape != (4 ** self.k,)
                                       or self._table.dtype != np.int32):
            self._table = None

        if self._table is None:
            # Vectorised build: the file is fixed width (k chars + newline).
            raw = np.fromfile(str(vocab_path), dtype=np.uint8)
            W = self.k + 1
            if raw.size % W != 0:
                raise ValueError(
                    f"{vocab_path}: size {raw.size} is not a multiple of {W}; "
                    "expected fixed-width lines of k chars + newline")
            raw = raw.reshape(-1, W)
            if not bool((raw[:, self.k] == 10).all()):
                raise ValueError(f"{vocab_path}: line width is not {self.k}+newline")
            c = _BASE_LUT[raw[:, :self.k]]
            if bool((c > 3).any()):
                raise ValueError(f"{vocab_path}: contains non-ACGT characters")
            codes = c.astype(np.uint64) @ powers
            if not len(codes):
                raise ValueError(f"{vocab_path}: vocabulary is empty")
            table = np.full(4 ** self.k, UNK_ID, dtype=np.int32)
            table[codes] = np.arange(len(codes), dtype=np.int32) + N_SPECIAL
            if np.count_nonzero(table != UNK_ID) != len(codes):
                raise ValueError(f"{vocab_path}: duplicate k-mers in vocabulary")
            self._table = table
            try:
                save_array_cache(cache, table, identity)
                print(f"  [ExactKmerTokenizer] cached lookup table -> {cache}")
            except OSError as e:
                print(f"  [ExactKmerTokenizer] could not cache table ({e})")

        n_present = int(len(codes)) if codes is not None else \
            int(np.count_nonzero(np.asarray(self._table) != UNK_ID))
        self.n_kmers = n_present
        self.vocab_size = n_present + N_SPECIAL

        if canonical is None:
            # Auto-detect: in a canonical vocab a sampled entry's RC is absent.
            tbl = np.asarray(self._table)
            rng = np.random.default_rng(0)
            cand = rng.integers(0, 4 ** self.k, 200_000, dtype=np.uint64)
            samp = cand[tbl[cand] != UNK_ID]
            rc_present = float((tbl[_rc_codes(samp, self.k)] != UNK_ID).mean())
            canonical = rc_present < 0.01
            print(f"  [ExactKmerTokenizer] canonical auto-detected: {canonical} "
                  f"(RC-also-present rate {rc_present*100:.2f}%, "
                  f"coverage {n_present / 4**self.k * 100:.2f}% of 4^{self.k})")
        self.canonical = bool(canonical)

    def _codes_to_ids(self, codes, valid):
        # Invalid bases can pack to values outside [0, 4**k). Mask before
        # indexing, while preserving every valid k-mer's trained token ID.
        codes = codes[valid]
        if self.canonical:
            codes = np.minimum(codes, _rc_codes(codes, self.k))
        ids = np.full(valid.shape, UNK_ID, dtype=np.int64)
        ids[valid] = np.asarray(self._table)[codes]
        return ids


def build_tokenizer(cfg: dict):
    """Return the tokenizer described by cfg['data']['tokenizer'], else NT-v2's.

    cfg['data']['tokenizer'] examples:
        {type: hashed_kmer, k: 13, stride: 1, n_buckets: 4194304, hash_seed: 42}
        {type: exact_kmer,  k: 13, stride: 1, vocab_path: /path/vocab_13mer.txt}
    """
    tcfg = (cfg.get("data") or {}).get("tokenizer")
    if not tcfg:
        backbone = cfg["model"].get("backbone", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
        if backbone == "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species":
            from .nt_tokenizer import NucleotideTokenizer
            return NucleotideTokenizer(Path(__file__).parent / "assets" / "ntv2_6mer.json")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            backbone, revision=cfg["model"].get("revision"),
            trust_remote_code=cfg["model"].get("trust_remote_code", True))
    if tcfg["type"] == "ntv2_6mer":
        from .nt_tokenizer import NucleotideTokenizer
        return NucleotideTokenizer(tcfg.get("vocab_path", Path(__file__).parent / "assets" / "ntv2_6mer.json"))

    ttype = tcfg["type"]
    if ttype == "hashed_kmer":
        return HashedKmerTokenizer(k=tcfg.get("k", 13), stride=tcfg.get("stride", 1),
                                   n_buckets=tcfg.get("n_buckets", 1 << 22),
                                   hash_seed=tcfg.get("hash_seed", 42))
    if ttype == "exact_kmer":
        return ExactKmerTokenizer(vocab_path=tcfg["vocab_path"], k=tcfg.get("k", 13),
                                  stride=tcfg.get("stride", 1),
                                  canonical=tcfg.get("canonical"),
                                  cache_path=tcfg.get("cache_path"))
    raise ValueError(f"unknown tokenizer type: {ttype}")
