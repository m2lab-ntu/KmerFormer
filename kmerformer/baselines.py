"""Optional foundation-model adapters for the paper comparisons."""
import gc
import torch
import torch.nn as nn
from .heads import create_head


class TokenLevelGFMClassifier(nn.Module):
    """
    Token-level GFM classifier.
    NO mean pooling — full token sequence → classification head.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        head_type: str = "attention_pool",
        head_config: dict = None,
        freeze_backbone: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        gradient_checkpointing: bool = True,
        backbone_loader: str = "mlm",
        trust_remote_code: bool = True,
        revision: str = None,
    ):
        super().__init__()
        from transformers import AutoModel, AutoModelForMaskedLM

        # ===== 1. Load GFM backbone =====
        # backbone_loader:
        #   "mlm"  — AutoModelForMaskedLM, then strip the LM head (NT-v2, DNABERT)
        #   "base" — AutoModel directly (DNABERT-2, which has no AutoModelForMaskedLM mapping)
        # trust_remote_code:
        #   True   — NT-v2 / DNABERT-2 need custom code
        #   False  — DNABERT's custom BertConfig conflicts with stock BertConfig
        #            (ValueError on AutoModel*); it's plain BERT-base anyway
        print(f"Loading backbone: {backbone_name}  (loader={backbone_loader}, "
              f"trust_remote_code={trust_remote_code})")
        if backbone_loader == "base":
            # DNABERT-2's bundled Triton flash-attn kernel uses tl.dot(..., trans_b=True)
            # which was removed in Triton >= 3.0. Disable the Triton path so the model
            # falls back to standard attention (slightly slower, but correct).
            def _disable_triton_flash_attn():
                import sys
                for mod_name, mod in list(sys.modules.items()):
                    if mod is None:
                        continue
                    if "DNABERT-2" in mod_name and mod_name.endswith("bert_layers"):
                        if hasattr(mod, "flash_attn_qkvpacked_func"):
                            mod.flash_attn_qkvpacked_func = None
            try:
                self.backbone = AutoModel.from_pretrained(
                    backbone_name, revision=revision, trust_remote_code=trust_remote_code
                )
                _disable_triton_flash_attn()
            except ValueError as e:
                # DNABERT-2 (and similar custom models) trigger AutoModel.register
                # conflicts between stock BertConfig and the remote BertConfig.
                # Fall back to resolving the model class via the dynamic module
                # and loading it directly.
                if "config_class" not in str(e) or not trust_remote_code:
                    raise
                print(f"  [warn] AutoModel register clash — bypassing via dynamic module")
                from transformers import AutoConfig
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                cfg_obj = AutoConfig.from_pretrained(
                    backbone_name, revision=revision, trust_remote_code=True
                )
                if not (hasattr(cfg_obj, "auto_map") and "AutoModel" in cfg_obj.auto_map):
                    raise
                model_ref = cfg_obj.auto_map["AutoModel"]
                model_class = get_class_from_dynamic_module(model_ref, backbone_name, revision=revision)
                self.backbone = model_class.from_pretrained(
                    backbone_name, revision=revision, config=cfg_obj, trust_remote_code=True
                )
                _disable_triton_flash_attn()
        else:
            if backbone_name == "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species":
                from .baseline_compat import load_ntv2_model, NT_V2_REVISION
                full_model = load_ntv2_model(backbone_name, revision or NT_V2_REVISION)
            else:
                full_model = AutoModelForMaskedLM.from_pretrained(
                    backbone_name, revision=revision, trust_remote_code=trust_remote_code
                )
            # NT-v2 wraps the encoder in .esm, BERT-family (DNABERT) wraps it in .bert
            if hasattr(full_model, "esm"):
                self.backbone = full_model.esm
            elif hasattr(full_model, "bert"):
                self.backbone = full_model.bert
            else:
                self.backbone = full_model
            del full_model
        self.hidden_dim = self.backbone.config.hidden_size
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Num layers: {self.backbone.config.num_hidden_layers}")

        gc.collect()

        # ===== 2. Gradient checkpointing =====
        if gradient_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable()
                print("  Gradient checkpointing: enabled")
            except Exception as e:
                print(f"  Gradient checkpointing: not available ({e})")

        # ===== 3. Backbone fine-tune strategy =====
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  Strategy: frozen backbone")
        elif use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)
        else:
            print("  Strategy: full fine-tuning (all params trainable)")

        # ===== 4. Classification Head (operates on token sequence) =====
        self.head = create_head(
            head_type=head_type,
            input_dim=self.hidden_dim,
            num_classes=num_classes,
            config=head_config or {},
        )
        print(f"  Head type: {head_type}")

        # Print parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")

    def _apply_lora(self, r, alpha, dropout, target_modules):
        """Apply LoRA adapters to backbone. Compatible with peft 0.5.0+."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(f"peft not installed: {e}")

        if target_modules is None:
            target_modules = ["query", "key", "value"]

        # Handle TaskType compatibility (may not exist in older peft)
        lora_kwargs = dict(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        try:
            from peft import TaskType
            lora_kwargs["task_type"] = TaskType.FEATURE_EXTRACTION
        except (ImportError, AttributeError):
            pass  # older peft — skip task_type

        lora_config = LoraConfig(**lora_kwargs)
        self.backbone = get_peft_model(self.backbone, lora_config)

        print("  Strategy: LoRA")
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        # ===== Backbone: get token-level embeddings =====
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Most HF models return ModelOutput with .last_hidden_state.
        # DNABERT-2's custom forward returns a tuple (last_hidden_state, pooler_output).
        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            token_embeddings = outputs[0]
        else:
            token_embeddings = outputs

        # ===== Classification Head: operates on FULL token sequence =====
        # NO mean pooling! [batch, seq_len, hidden_dim] → head → [batch, num_classes]
        logits = self.head(token_embeddings, attention_mask)

        return logits

    def get_backbone_params(self):
        """Return backbone parameters (for separate LR group)."""
        return [p for n, p in self.named_parameters()
                if "backbone" in n and p.requires_grad]

    def get_head_params(self):
        """Return head parameters (for separate LR group)."""
        return [p for n, p in self.named_parameters()
                if "head" in n and p.requires_grad]

    def freeze_backbone(self):
        """Temporarily freeze backbone (for Phase 1 training)."""
        for n, p in self.named_parameters():
            if "backbone" in n:
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone (for Phase 2 training)."""
        # Only unfreeze LoRA params or all params depending on config
        for n, p in self.named_parameters():
            if "backbone" in n:
                # For LoRA: only lora_ params should be unfrozen
                if "lora_" in n or "modules_to_save" in n:
                    p.requires_grad = True
                # If not using LoRA and it was frozen for phase1, unfreeze all
                elif not any("lora_" in nn for nn, _ in self.named_parameters()):
                    p.requires_grad = True


# ============================================================
# Shallow Transformer Classifier (MetaTransformer-style ablation)
# ============================================================
