"""KmerFormer encoders and backward-compatible model construction."""
import math
import torch
import torch.nn as nn

from .heads import create_head


class ShallowTransformerClassifier(nn.Module):
    """
    KmerFormer: learned k-mer embeddings, configurable encoder and classifier.

    Set vocabulary size and padding ID for custom tokenizers. Otherwise the
    fixed NT-v2 6-mer vocabulary is used without a network dependency.
    ``backbone_name`` remains a compatibility argument for older callers.
    """

    def __init__(
        self,
        backbone_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        num_classes: int = None,
        head_type: str = "attention_pool",
        head_config: dict = None,
        d_model: int = 128,
        nhead: int = 2,
        d_ff: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        vocab_size: int = None,
        pad_id: int = None,
        sparse_embedding: bool = False,
        pos_encoding: str = "learned",     # "learned" | "sinusoidal"
        norm_first: bool = True,           # False gives post-norm
        activation: str = "gelu",          # "gelu" | "relu"
        embed_scale: str = "layernorm",    # "layernorm" | "sqrt_d"
        final_norm: bool = False,          # LayerNorm after the encoder stack
        conv_kernels=None,                 # e.g. [2, 3, 5]; None disables
        **kwargs,
    ):
        super().__init__()
        if num_classes is None or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        # The four knobs above exist so the model can be walked toward
        # MetaTransformer one group at a time. Every default is what KmerFormer
        # already did, so nothing changes for the arms already reported.
        if pos_encoding not in ("learned", "sinusoidal"):
            raise ValueError(f"pos_encoding: {pos_encoding}")
        if embed_scale not in ("layernorm", "sqrt_d"):
            raise ValueError(f"embed_scale: {embed_scale}")
        self.pos_encoding = pos_encoding
        self.embed_scale = embed_scale

        # vocab_size given => custom k-mer tokenizer (hashed / exact 13-mer);
        # skip loading NT-v2's tokenizer, whose vocab only covers 6-mers.
        if vocab_size is None:
            print(f"Loading tokenizer for vocab: {backbone_name}")
            from .kmer_tokenizers import build_tokenizer
            tokenizer = build_tokenizer({"model": {"backbone": backbone_name}})
            vocab_size = tokenizer.vocab_size
            if pad_id is None:
                pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        self.sparse_embedding = sparse_embedding

        self.hidden_dim = d_model

        print(f"  KmerFormer: d_model={d_model}, layers={num_layers}, "
              f"heads={nhead}, d_ff={d_ff}")
        print(f"  Vocab size: {vocab_size}, pad_id: {pad_id}, "
              f"sparse_embedding: {sparse_embedding}")
        print(f"  Embedding table: {vocab_size * d_model * 4 / 1024**3:.2f} GiB (fp32)")

        # Learned embeddings (from scratch).
        # sparse=True keeps the gradient a sparse COO tensor — required for the
        # exact-13-mer table, where a dense gradient would be table-sized.
        # It also forces a SparseAdam param group (see train.py).
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id,
                                            sparse=sparse_embedding)
        if pos_encoding == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            nn.init.normal_(self.position_embedding.weight, std=0.02)
        else:
            # MetaTransformer's PositionalEncoding2, term for term: a fixed
            # sin/cos table added to the embeddings, registered as a buffer so it
            # carries in the state dict but takes no gradient.
            self.position_embedding = None
            pe = torch.zeros(max_seq_len, d_model)
            pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pos_sinusoid", pe)

        self.embed_norm = (nn.LayerNorm(d_model) if embed_scale == "layernorm"
                           else nn.Identity())
        self.embed_dropout = nn.Dropout(dropout)

        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Optional multiscale convolutional front-end over the token axis.
        # A width-w kernel over w adjacent k-mer tokens builds an explicit
        # composite feature spanning w*k bases (non-overlapping tokens) or
        # k+w-1 bases (stride-1 tokens) -- i.e. it hands the model the long-k
        # composition it would otherwise have to learn through attention.
        # Parallel kernels are summed residually, so d_model is unchanged and
        # the rest of the network is untouched.
        # Enabled only by configs/ablations/L29_50M_conv_frontend.yaml.
        # Primary released checkpoints leave conv_kernels unset.
        self.conv_frontend = None
        if conv_kernels:
            self.conv_kernels = list(conv_kernels)
            self.conv_frontend = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2)
                for k in self.conv_kernels
            ])
            self.conv_proj = nn.Linear(d_model, d_model)
            self.conv_norm = nn.LayerNorm(d_model)
            self.conv_act = nn.GELU()
            print(f"  Conv front-end: kernels={self.conv_kernels} "
                  f"(spans up to {max(self.conv_kernels)} tokens)")

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        # final_norm: MetaTransformer ends its stack with a LayerNorm
        # (transformer_enc.encoder.norm, 2*d_model parameters). We never had one.
        # Found by counting parameters above the embedding table during the
        # MetaTransformer-direction ablation: our reconstruction came out 128
        # parameters short at d_model=64, and this was the whole of the gap.
        # Default False preserves every arm reported so far, including the ones
        # already trained.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if final_norm else None)

        # Classification head (same interface as GFM variant)
        self.head = create_head(
            head_type=head_type,
            input_dim=d_model,
            num_classes=num_classes,
            config=head_config or {},
        )
        print(f"  Head type: {head_type}")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)
        if self.embed_scale == "sqrt_d":
            # MetaTransformer scales the embedding by sqrt(d_model) where we
            # apply a LayerNorm; the two are alternatives, not both.
            x = x * math.sqrt(self.hidden_dim)
        if self.pos_encoding == "learned":
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        else:
            x = x + self.pos_sinusoid[:seq_len, :]
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        if self.conv_frontend is not None:
            # zero out padding so it cannot leak into the convolutions
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).to(x.dtype)
            h = x.transpose(1, 2)                       # [B, d, L]
            acc = None
            for conv in self.conv_frontend:
                c = conv(h)[:, :, :seq_len]             # even kernels pad +1
                acc = c if acc is None else acc + c
            acc = acc.transpose(1, 2)                   # [B, L, d]
            x = self.conv_norm(x + self.conv_proj(self.conv_act(acc)))

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        logits = self.head(x, attention_mask)
        return logits

    def get_backbone_params(self):
        """Return embedding + encoder parameters."""
        backbone_parts = [
            "token_embedding", "position_embedding",
            "embed_norm", "embed_dropout", "encoder",
            "conv_frontend", "conv_proj", "conv_norm",
        ]
        return [
            p for n, p in self.named_parameters()
            if any(bp in n for bp in backbone_parts) and p.requires_grad
        ]

    def get_head_params(self):
        return [
            p for n, p in self.named_parameters()
            if "head" in n and p.requires_grad
        ]

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if "head" not in n:
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True


# ============================================================
# Factory
# ============================================================

def create_model(cfg: dict, num_classes: int):
    """Create model from config dict. Supports 'gfm' and 'shallow_transformer'."""
    model_type = cfg.get("type", "kmerformer")
    backbone_name = cfg.get("backbone", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")

    if model_type in ("kmerformer", "shallow_transformer"):
        shallow_cfg = cfg.get("shallow_config", {})
        return ShallowTransformerClassifier(
            backbone_name=backbone_name,
            num_classes=num_classes,
            head_type=cfg.get("head_type", "attention_pool"),
            head_config=cfg.get("head_config", {}),
            d_model=shallow_cfg.get("d_model", 128),
            nhead=shallow_cfg.get("nhead", 2),
            d_ff=shallow_cfg.get("d_ff", 512),
            num_layers=shallow_cfg.get("num_layers", 1),
            dropout=shallow_cfg.get("dropout", 0.1),
            max_seq_len=cfg.get("max_seq_len", 128),
            vocab_size=cfg.get("vocab_size"),
            pad_id=cfg.get("pad_id"),
            sparse_embedding=shallow_cfg.get("sparse_embedding", False),
            pos_encoding=shallow_cfg.get("pos_encoding", "learned"),
            norm_first=shallow_cfg.get("norm_first", True),
            activation=shallow_cfg.get("activation", "gelu"),
            embed_scale=shallow_cfg.get("embed_scale", "layernorm"),
            conv_kernels=shallow_cfg.get("conv_kernels"),
            final_norm=shallow_cfg.get("final_norm", False),
        )
    elif model_type == "gfm":
        from .baselines import TokenLevelGFMClassifier
        return TokenLevelGFMClassifier(
            backbone_name=backbone_name,
            num_classes=num_classes,
            head_type=cfg.get("head_type", "attention_pool"),
            head_config=cfg.get("head_config", {}),
            freeze_backbone=cfg.get("freeze_backbone", False),
            use_lora=cfg.get("use_lora", True),
            lora_r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            lora_target_modules=cfg.get("lora_target_modules"),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            backbone_loader=cfg.get("backbone_loader", "mlm"),
            trust_remote_code=cfg.get("trust_remote_code", True),
            revision=cfg.get("revision"),
        )

    raise ValueError(f"Unknown model type: {model_type}")


# Historical import path retained for existing experiment scripts.
def __getattr__(name):
    if name == "TokenLevelGFMClassifier":
        from .baselines import TokenLevelGFMClassifier
        return TokenLevelGFMClassifier
    raise AttributeError(name)


KmerFormer = ShallowTransformerClassifier
