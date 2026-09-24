#!/usr/bin/env python3
"""
PyTorch Dataset for Token-level GFM Classifier.

Key difference from previous pipeline:
  - Previous: precompute embeddings → mean pool → save .npz → train MLP on embeddings
  - Now: tokenize on-the-fly → feed to model → get token-level output → classify

NO mean pooling. The full token sequence is preserved.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class TokenLevelReadDataset(Dataset):
    """
    Dataset that returns tokenized reads WITHOUT mean pooling.

    Standard mode:
        Each sample: (input_ids, attention_mask, label)

    RC-consistency mode (rc_consistency=True):
        Each sample: (fwd_ids, fwd_mask, rc_ids, rc_mask, label)
        Both forward and reverse-complement tokenized for consistency training.
    """

    def __init__(
        self,
        sequences: list,
        labels: np.ndarray,
        tokenizer,
        max_length: int = 32,
        rc_augment: bool = False,
        rc_consistency: bool = False,
        kmer_preprocess: dict = None,
    ):
        """
        Args:
            sequences: List of DNA strings
            labels: Integer label array
            tokenizer: Pre-loaded HuggingFace tokenizer
            max_length: Max token length after tokenization
            rc_augment: If True, randomly use reverse complement (training only)
            rc_consistency: If True, return both fwd and RC (supersedes rc_augment)
            kmer_preprocess: If set to {"k": int, "stride": int}, split the raw
                DNA string into k-mers of length k with the given stride and join
                them with spaces before passing to the tokenizer.
                Required for DNABERT (k=6, stride=1). Leave as None for NT-v2 /
                DNABERT-2 whose tokenizers handle raw sequences directly.
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rc_augment = rc_augment
        self.rc_consistency = rc_consistency
        self.kmer_preprocess = kmer_preprocess
        if kmer_preprocess is not None:
            assert "k" in kmer_preprocess and "stride" in kmer_preprocess, (
                "kmer_preprocess must contain 'k' and 'stride'"
            )

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def reverse_complement(seq: str) -> str:
        comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N",
                "R": "Y", "Y": "R", "M": "K", "K": "M", "S": "S", "W": "W"}
        return "".join(comp.get(b, "N") for b in reversed(seq.upper()))

    def _preprocess(self, seq: str) -> str:
        """Optionally split seq into k-mers joined by spaces (for DNABERT)."""
        if self.kmer_preprocess is None:
            return seq
        k = self.kmer_preprocess["k"]
        stride = self.kmer_preprocess["stride"]
        if len(seq) < k:
            return seq
        return " ".join(seq[i:i + k] for i in range(0, len(seq) - k + 1, stride))

    def _tokenize(self, seq):
        """Tokenize a single sequence and return (input_ids, attention_mask)."""
        encoding = self.tokenizer(
            self._preprocess(seq),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # ---- RC consistency mode: return both fwd + RC ----
        if self.rc_consistency:
            fwd_ids, fwd_mask = self._tokenize(seq)
            rc_seq = self.reverse_complement(seq)
            rc_ids, rc_mask = self._tokenize(rc_seq)
            return fwd_ids, fwd_mask, rc_ids, rc_mask, label

        # ---- Standard mode: optional random RC augmentation ----
        if self.rc_augment and torch.rand(1).item() < 0.5:
            seq = self.reverse_complement(seq)

        input_ids, attention_mask = self._tokenize(seq)
        return input_ids, attention_mask, label

    def validate_token_lengths(self, n_samples: int = 200) -> dict:
        """
        Sanity check: tokenize a sample of sequences WITHOUT padding
        and report token length statistics.

        Call this once after construction to verify max_length is appropriate.

        Returns dict with min, max, mean, median, pct_truncated, padding_ratio.
        """
        rng = np.random.RandomState(0)
        indices = rng.choice(len(self.sequences), size=min(n_samples, len(self.sequences)), replace=False)
        lengths = []
        for idx in indices:
            enc = self.tokenizer(
                self._preprocess(self.sequences[idx]),
                padding=False,
                truncation=False,
            )
            lengths.append(len(enc["input_ids"]))
        lengths = np.array(lengths)
        n_truncated = int((lengths > self.max_length).sum())
        avg_len = float(lengths.mean())
        stats = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(avg_len, 1),
            "median": int(np.median(lengths)),
            "max_length_cfg": self.max_length,
            "pct_truncated": round(100.0 * n_truncated / len(lengths), 2),
            "avg_padding_ratio": round(100.0 * (self.max_length - min(avg_len, self.max_length)) / self.max_length, 1),
        }
        print(f"  Token length stats (n={len(lengths)}): "
              f"min={stats['min']} max={stats['max']} mean={stats['mean']} "
              f"median={stats['median']} | "
              f"max_length={self.max_length} → "
              f"truncated={stats['pct_truncated']}%, "
              f"avg_padding={stats['avg_padding_ratio']}%")
        assert stats["pct_truncated"] < 5.0, (
            f"WARNING: {stats['pct_truncated']}% of sequences are truncated at "
            f"max_length={self.max_length}. Consider increasing max_length."
        )
        return stats

