"""Bounded runtime checks for optional LoRA and CUDA configurations."""
import os
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kmerformer.model import create_model
from kmerformer.train import train_epoch


@pytest.mark.skipif(os.environ.get("KMERFORMER_TEST_NTV2") != "1",
                    reason="Set KMERFORMER_TEST_NTV2=1 to allow pinned NT-v2 source retrieval")
def test_pinned_ntv2_matches_legacy_tensors_and_reloads(tmp_path):
    pytest.importorskip("peft")
    from kmerformer.baseline_compat import ntv2_classes
    from peft import LoraConfig, get_peft_model
    fixture = Path(__file__).parent / "fixtures" / "ntv2_legacy_tiny.npz"
    meta = json.loads(fixture.with_suffix(".json").read_text())
    cfg, cls = ntv2_classes("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    for key, value in meta["config_changes"].items():
        setattr(cfg, key, value)
    with np.load(fixture) as saved:
        state = {key: torch.from_numpy(saved[key].copy()) for key in saved.files if not key.startswith("reference_")}
        inputs = torch.from_numpy(saved["reference_input_ids"].copy())
        expected = torch.from_numpy(saved["reference_logits"].copy())
    model = cls(cfg).eval()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = model(input_ids=inputs, attention_mask=inputs.ne(1)).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    model.save_pretrained(tmp_path / "tiny")
    restored = cls.from_pretrained(tmp_path / "tiny", config=cfg).eval()
    with torch.no_grad():
        result = restored(input_ids=inputs, attention_mask=inputs.ne(1)).logits
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    adapted = get_peft_model(restored.esm, LoraConfig(r=2, lora_alpha=4,
                                                   target_modules=["query", "value"]))
    adapted(input_ids=inputs, attention_mask=inputs.ne(1)).last_hidden_state.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in adapted.named_parameters() if "lora_" in name)


def test_lora_with_a_local_backbone(tmp_path):
    pytest.importorskip("peft")
    from transformers import BertConfig, BertForMaskedLM

    backbone = tmp_path / "backbone"
    BertForMaskedLM(BertConfig(vocab_size=67, hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=16,
                               max_position_embeddings=32)).save_pretrained(backbone)
    model = create_model({
        "backbone": str(backbone), "type": "gfm",
        "head_config": {"hidden_dim": 8}, "gradient_checkpointing": False,
        "use_lora": True, "lora_r": 2, "lora_alpha": 4,
        "lora_target_modules": ["query", "value"],
    }, 3)
    inputs = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]])
    loss = torch.nn.functional.cross_entropy(model(inputs, torch.ones_like(inputs)),
                                             torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    adapters = [p for name, p in model.named_parameters() if "lora_" in name]
    assert adapters and any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapters)
    assert all(p.grad is not None for p in model.get_head_params())
    frozen = [p for name, p in model.named_parameters() if "base_layer" in name]
    assert frozen and all(not p.requires_grad and p.grad is None for p in frozen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_cuda_accumulation_and_amp(sparse, dtype):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Native bfloat16 device required")
    model = create_model({
        "type": "shallow_transformer", "backbone": "unused", "vocab_size": 67,
        "pad_id": 0, "max_seq_len": 16, "head_type": "mean_pool",
        "head_config": {"hidden_dim": 8, "dropout": 0.0},
        "shallow_config": {"d_model": 8, "nhead": 2, "d_ff": 16, "num_layers": 1,
                           "dropout": 0.0, "sparse_embedding": sparse},
    }, 3).cuda()
    embedding = model.token_embedding.weight
    dense = [p for p in model.parameters() if not sparse or p is not embedding]
    optimizers = [torch.optim.AdamW(dense, lr=0.01)]
    if sparse:
        optimizers.append(torch.optim.SparseAdam([embedding], lr=0.01))
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=0.9) for o in optimizers]
    inputs = torch.randint(3, 67, (6, 10))
    loader = DataLoader(TensorDataset(inputs, torch.ones_like(inputs), torch.arange(6) % 3), batch_size=2)
    before = [p.detach().clone() for p in model.get_head_params()]
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=128.)
    train_epoch(model, loader, torch.nn.CrossEntropyLoss(), optimizers, schedulers, scaler,
                torch.device("cuda"), 4, 1., True, amp_dtype=dtype)
    assert all(s.last_epoch == 1 for s in schedulers)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, old) for p, old in zip(model.get_head_params(), before))
