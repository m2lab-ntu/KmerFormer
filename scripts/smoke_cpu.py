#!/usr/bin/env python3
"""Exercise a tiny synthetic model on CPU, without pretrained models or data."""
import io
import json

import torch

from kmerformer import build_tokenizer, create_model


def run():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cfg = {
        "model": {
            "type": "shallow_transformer", "backbone": "unused", "vocab_size": 1027,
            "pad_id": 0,
            "max_seq_len": 140, "head_type": "mean_pool",
            "shallow_config": {"d_model": 16, "nhead": 2, "d_ff": 32,
                               "num_layers": 1, "dropout": 0.0},
        },
        "data": {"max_token_length": 140,
                 "tokenizer": {"type": "hashed_kmer", "k": 13, "stride": 1,
                               "n_buckets": 1024, "hash_seed": 42}},
    }
    reads = [("ACGT" * 38)[:150], ("GATTACA" * 22)[:150]]
    tokenizer = build_tokenizer(cfg)
    batch = tokenizer(reads, max_length=140, padding="max_length",
                      truncation=True, return_tensors="pt")
    model = create_model(cfg["model"], num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(batch["input_ids"], batch["attention_mask"])
    stream = io.BytesIO()
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    restored = create_model(checkpoint["config"]["model"], num_classes=2).eval()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        actual = restored(batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"device": "cpu", "reads": 2, "token_shape": list(batch["input_ids"].shape),
            "logit_shape": list(actual.shape), "loss": float(loss.detach()),
            "checkpoint_round_trip": "bit-identical",
            "scope": "synthetic software test, not an accuracy reproduction"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
