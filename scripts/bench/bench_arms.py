#!/usr/bin/env python
"""Uniform speed / memory benchmark across every arm in the v4 manuscript.

WHY THIS EXISTS
The compute table the advisor asked for cannot be assembled from the numbers we
already have: the earlier benchmark covers NT-v2 and the MetaTransformer arms
only, our own arms are absent entirely, and the rows were not all taken on one
machine. This script measures every arm through one code path, on one GPU, at
one batch size, so the columns are actually comparable.

WHAT IS MEASURED
Forward-only and forward+backward+step throughput on a fixed synthetic batch of
150 bp reads, with warm-up excluded, so the number reflects the architecture's
compute rather than data loading or NFS. Peak allocated GPU memory is captured
per phase. Parameter counts are split into embedding and non-embedding, because
for long-k arms nearly all of the model is the embedding table and a single
"parameters" figure is misleading.

Two tokenizer families are handled: the HuggingFace tokenizer that the 6-mer
arms and NT-v2 share, and the exact/hashed k-mer tokenizers used by the 13-mer
arms. Both are driven from the same config files the training runs used, so an
arm cannot be benchmarked in a configuration it was never trained in.

Usage:
  python bench_arms.py --config CFG.yaml --label NAME --num_classes 120 \
      --batch 128 --csv out.csv [--infer_only]
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from kmerformer.model import create_model  # noqa: E402


def rand_reads(n, length=150):
    return ["".join(random.choice("ACGT") for _ in range(length)) for _ in range(n)]


def encode(cfg, reads, max_len):
    """Return input_ids, attention_mask for whichever tokenizer the arm uses."""
    tok_cfg = cfg.get("data", {}).get("tokenizer")
    if tok_cfg:
        # the k-mer tokenizers are HuggingFace-compatible through __call__, so
        # both families go through the same call below
        from kmer_tokenizers import build_tokenizer
        tk = build_tokenizer(cfg)          # takes the whole config, not the subtree
        enc = tk(reads, padding="max_length", truncation=True,
                 max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc.get("attention_mask")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(cfg["model"]["backbone"], trust_remote_code=True)
    # kmer_preprocess is either an int k (non-overlapping) or {k, stride}
    kmer = cfg.get("data", {}).get("kmer_preprocess")
    if kmer:
        k = kmer["k"] if isinstance(kmer, dict) else int(kmer)
        stride = kmer.get("stride", k) if isinstance(kmer, dict) else k
        reads = [" ".join(r[i:i + k] for i in range(0, len(r) - k + 1, stride))
                 for r in reads]
    enc = tk(reads, padding="max_length", truncation=True,
             max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc.get("attention_mask")


def param_split(model):
    """Embedding parameters versus everything else.

    For the long-k arms the embedding table is essentially the whole model, so
    a single parameter count hides where the cost actually is.

    The table is found by module type, not by attribute name. Matching the string
    "token_embedding" only ever matched our own ShallowTransformerClassifier: a
    backbone loaded from HuggingFace names its table
    embeddings.word_embeddings, so NT-v2 was reported with a zero-parameter
    vocabulary table and its whole 498.6M in "other" -- printed as an em dash in
    the manuscript, which reads as "has no table" rather than "was not counted".
    Positional embeddings stay in "other", which is what the k-mer arms already
    did and what the column means.
    """
    tables = {n for n, m in model.named_modules()
              if isinstance(m, nn.Embedding) and "position" not in n.lower()}
    emb = sum(p.numel() for n, p in model.named_parameters()
              if any(n.startswith(t + ".") for t in tables))
    other = sum(p.numel() for p in model.parameters()) - emb
    return emb, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--num_classes", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--infer_only", action="store_true",
                    help="skip the training phase (for arms whose optimiser "
                         "state does not fit on this GPU)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    random.seed(0)
    torch.manual_seed(0)

    cfg = yaml.safe_load(open(args.config))
    max_len = cfg.get("data", {}).get("max_token_length", 128)

    input_ids, attn = encode(cfg, rand_reads(args.batch), max_len)
    input_ids = input_ids.to(device)
    attn = attn.to(device) if attn is not None else None
    labels = torch.randint(0, args.num_classes, (args.batch,), device=device)
    n_tokens = int(input_ids.shape[1])

    mc = dict(cfg["model"])
    mc["gradient_checkpointing"] = False
    model = create_model(mc, args.num_classes).to(device)
    emb_p, other_p = param_split(model)

    crit = nn.CrossEntropyLoss()
    dense = [p for n, p in model.named_parameters()
             if p.requires_grad and "token_embedding" not in n]
    sparse = [p for n, p in model.named_parameters()
              if p.requires_grad and "token_embedding" in n]
    opts = [torch.optim.AdamW(dense, lr=1e-4)] if dense else []
    if sparse:
        is_sparse = cfg["model"].get("shallow_config", {}).get("sparse_embedding", False)
        opts.append(torch.optim.SparseAdam(sparse, lr=1e-4) if is_sparse
                    else torch.optim.AdamW(sparse, lr=1e-4))
    scaler = torch.amp.GradScaler("cuda")

    def train_step():
        for o in opts:
            o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = crit(model(input_ids, attn), labels)
        scaler.scale(loss).backward()
        for o in opts:
            scaler.step(o)
        scaler.update()

    def infer_step():
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(input_ids, attn)

    def timed(fn, eval_mode):
        model.eval() if eval_mode else model.train()
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            fn()
        torch.cuda.synchronize()
        el = time.time() - t0
        reads = args.steps * args.batch
        return reads / el, torch.cuda.max_memory_allocated() / (1024 ** 2)

    phases = [("infer", infer_step, True)]
    if not args.infer_only:
        phases.append(("train", train_step, False))

    header = ["model", "phase", "batch", "tokens_per_read", "reads_per_sec",
              "peak_gpu_mib", "embed_params", "other_params"]
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for phase, fn, ev in phases:
            try:
                rps, peak = timed(fn, ev)
            except torch.cuda.OutOfMemoryError:
                print(f"[{args.label}] {phase}: OOM on this GPU", flush=True)
                w.writerow([args.label, phase, args.batch, n_tokens, "OOM", "OOM",
                            emb_p, other_p])
                torch.cuda.empty_cache()
                continue
            print(f"[{args.label}] {phase:5s} {rps:>10,.0f} reads/s  "
                  f"peak {peak:>8,.0f} MiB  tok/read {n_tokens:>4d}  "
                  f"emb {emb_p/1e6:>8.1f}M  other {other_p/1e6:.2f}M", flush=True)
            w.writerow([args.label, phase, args.batch, n_tokens,
                        f"{rps:.0f}", f"{peak:.0f}", emb_p, other_p])


if __name__ == "__main__":
    main()
