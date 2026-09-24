"""Compatibility boundary for the pinned NT-v2 remote model on Transformers 5.

Only NT-v2's dynamically loaded base class receives the legacy mask helpers.
No installed Transformers files or upstream checkpoint tensors are modified.
"""
from contextlib import contextmanager
import threading
import torch

NT_V2_REVISION = "06615c1660c892fc199840c18123f8385b3542a8"
_IMPORT_LOCK = threading.RLock()


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    heads = set(heads) - already_pruned_heads
    keep = torch.ones(n_heads, head_size, dtype=torch.bool)
    for head in heads:
        shifted = head - sum(previous < head for previous in already_pruned_heads)
        keep[shifted] = False
    return heads, torch.arange(n_heads * head_size)[keep.reshape(-1)]


def _extended_mask(self, attention_mask, input_shape, device=None, dtype=None):
    dtype = dtype or self.dtype
    if attention_mask.ndim == 3:
        mask = attention_mask[:, None, :, :]
    elif attention_mask.ndim == 2:
        mask = attention_mask[:, None, None, :]
        if getattr(self.config, "is_decoder", False):
            length = input_shape[1]
            positions = torch.arange(length, device=attention_mask.device)
            causal = positions[None, :] <= positions[:, None]
            prefix = attention_mask.shape[-1] - length
            if prefix:
                causal = torch.cat((torch.ones(length, prefix, dtype=torch.bool, device=causal.device), causal), dim=-1)
            mask = mask * causal[None, None, :, :]
    else:
        raise ValueError("Attention mask must have two or three dimensions")
    return (1.0 - mask.to(dtype=dtype)) * torch.finfo(dtype).min


def _invert_mask(self, encoder_attention_mask):
    mask = encoder_attention_mask
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    else:
        raise ValueError("Encoder attention mask must have two or three dimensions")
    return (1.0 - mask.to(self.dtype)) * torch.finfo(self.dtype).min


def _head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.ndim == 1:
        head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.ndim == 2:
        head_mask = head_mask[:, None, :, None, None]
    if head_mask.ndim != 5:
        raise ValueError("Invalid attention head mask dimensions")
    head_mask = head_mask.to(self.dtype)
    return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask


def ntv2_classes(backbone, revision=NT_V2_REVISION):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import transformers.pytorch_utils as helpers
    cfg = AutoConfig.from_pretrained(backbone, revision=revision, trust_remote_code=True)
    # Defaults formerly inherited from PretrainedConfig by the pinned class.
    for name, value in {"is_decoder": False, "add_cross_attention": False,
                        "chunk_size_feed_forward": 0, "output_attentions": False,
                        "output_hidden_states": False, "return_dict": True}.items():
        if not hasattr(cfg, name):
            setattr(cfg, name, value)
    # The upstream module imports this removed helper once. Restore the host
    # namespace immediately after import; references remain local to NT-v2.
    with _IMPORT_LOCK:
        previous = getattr(helpers, "find_pruneable_heads_and_indices", None)
        if previous is None:
            helpers.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        try:
            cls = get_class_from_dynamic_module(cfg.auto_map["AutoModelForMaskedLM"], backbone, revision=revision)
        finally:
            if previous is None:
                del helpers.find_pruneable_heads_and_indices
    module = __import__(cls.__module__, fromlist=["EsmPreTrainedModel"])
    base = module.EsmPreTrainedModel
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        cls._tied_weights_keys = {"lm_head.decoder.weight": "esm.embeddings.word_embeddings.weight"}
        if not getattr(base, "_kmerformer_init_compat", False):
            original_init = base.init_weights
            def initialize(self):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.post_init()
                else:
                    original_init(self)
            base.init_weights = initialize
            base._kmerformer_init_compat = True
    for name, function in (("get_extended_attention_mask", _extended_mask),
                           ("invert_attention_mask", _invert_mask), ("get_head_mask", _head_mask)):
        if not hasattr(base, name):
            setattr(base, name, function)
    return cfg, cls


def load_ntv2_model(backbone, revision=NT_V2_REVISION):
    cfg, cls = ntv2_classes(backbone, revision)
    return cls.from_pretrained(backbone, revision=revision, config=cfg)
