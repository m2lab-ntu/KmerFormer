#!/usr/bin/env python3
"""
Classification heads for token-level GFM classifiers.

All heads accept:
    token_embeddings: [batch, seq_len, input_dim]
    attention_mask:   [batch, seq_len]  (1 = real token, 0 = padding)
and return:
    logits: [batch, num_classes]

Architecture reconstructed from checkpoint state dict
(nt_token_species_v4_50M_best.pt):
  head.query:              [1, num_heads, head_dim]
  head.k_proj.{weight,bias}
  head.v_proj.{weight,bias}
  head.classifier.0  → LayerNorm(total_dim)
  head.classifier.1  → Linear(total_dim, hidden_dim)
  head.classifier.2  → GELU
  head.classifier.3  → Dropout
  head.classifier.4  → Linear(hidden_dim, num_classes)
"""

import torch
import torch.nn as nn


class AttentionPoolHead(nn.Module):
    """
    Multi-head cross-attention pooling head.

    A single learnable query vector attends over the token sequence
    (one query per head), producing a fixed-size pooled representation
    that is then passed through a two-layer MLP classifier.

    Key design choices (matching trained checkpoints):
    - head_dim = input_dim // num_attention_heads  (no down-projection)
    - k_proj / v_proj: Linear(input_dim, input_dim)
    - classifier: LayerNorm → Linear → GELU → Dropout → Linear
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_attention_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()

        assert input_dim % num_attention_heads == 0, (
            f"input_dim ({input_dim}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
        self.num_heads = num_attention_heads
        self.head_dim = input_dim // num_attention_heads
        self.total_dim = input_dim  # = num_heads * head_dim

        # Learnable query: one query vector per head
        self.query = nn.Parameter(
            torch.randn(1, self.num_heads, self.head_dim) * 0.02
        )

        # Key / value projections (same dim as backbone hidden)
        self.k_proj = nn.Linear(input_dim, self.total_dim)
        self.v_proj = nn.Linear(input_dim, self.total_dim)

        # MLP classifier: indices must match checkpoint (0=LN, 1=Lin, 2=GELU, 3=Drop, 4=Lin)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.total_dim),
            nn.Linear(self.total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: [batch, seq_len, input_dim]
            attention_mask:   [batch, seq_len]  (1=real, 0=pad)
        Returns:
            logits: [batch, num_classes]
        """
        batch, seq_len, _ = token_embeddings.shape

        # Project to keys and values: [batch, seq_len, total_dim]
        K = self.k_proj(token_embeddings)
        V = self.v_proj(token_embeddings)

        # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand query over batch: [batch, num_heads, 1, head_dim]
        Q = self.query.expand(batch, -1, -1).unsqueeze(2)

        # Attention scores: [batch, num_heads, 1, seq_len]
        scale = self.head_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Mask padding positions
        if attention_mask is not None:
            # [batch, 1, 1, seq_len] — broadcasts over heads and query pos
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [batch, num_heads, 1, seq_len]

        # Weighted sum over sequence: [batch, num_heads, 1, head_dim]
        pooled = torch.matmul(weights, V)

        # Flatten: [batch, total_dim]
        pooled = pooled.squeeze(2).reshape(batch, self.total_dim)

        return self.classifier(pooled)


class MeanPoolHead(nn.Module):
    """Masked mean pooling with the same classifier as AttentionPoolHead.

    Only the pooling differs from AttentionPoolHead: the learnable query and the
    key/value projections are gone, and token embeddings are averaged over
    non-padded positions instead. The classifier is identical, so a difference in
    accuracy is attributable to pooling alone.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_attention_heads: int = 4,   # accepted and unused; keeps configs interchangeable
        hidden_dim: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.total_dim = input_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.total_dim),
            nn.Linear(self.total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_embeddings, attention_mask=None):
        if attention_mask is None:
            pooled = token_embeddings.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        return self.classifier(pooled)


class UnmaskedMeanLinearHead(nn.Module):
    """MetaTransformer's head, reproduced exactly.

    Two differences from MeanPoolHead, and both matter for the ablation:
    the mean runs over every position including padding (MetaTransformer calls
    torch.mean(x, dim=1) with no mask), and the classifier is a single Linear
    rather than LayerNorm/Linear/GELU/Dropout/Linear. Together they are two of
    the six architectural differences Supplementary Table S7 lists as
    unablated.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_attention_heads: int = 4,   # accepted and unused; keeps configs swappable
        hidden_dim: int = 512,          # ditto
        dropout: float = 0.15,          # ditto
    ):
        super().__init__()
        self.total_dim = input_dim
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, token_embeddings, attention_mask=None):
        # attention_mask deliberately ignored -- that is the point of this head
        return self.classifier(token_embeddings.mean(dim=1))


def create_head(
    head_type: str,
    input_dim: int,
    num_classes: int,
    config: dict = None,
) -> nn.Module:
    """Factory for classification heads."""
    cfg = config or {}

    if head_type == "attention_pool":
        return AttentionPoolHead(
            input_dim=input_dim,
            num_classes=num_classes,
            num_attention_heads=cfg.get("num_attention_heads", 4),
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.15),
        )

    if head_type == "unmasked_mean_linear":
        return UnmaskedMeanLinearHead(
            input_dim=input_dim,
            num_classes=num_classes,
            num_attention_heads=cfg.get("num_attention_heads", 4),
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.15),
        )

    if head_type == "mean_pool":
        return MeanPoolHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.15),
        )

    raise ValueError(f"Unknown head_type: {head_type!r}. Supported: ['attention_pool', 'mean_pool']")
