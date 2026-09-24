#!/usr/bin/env python3
"""
Controlled TRAINING-throughput probe for the NT family (NT-v2 + LoRA, DNABERT,
DNABERT-2) — i.e. any model built by create_model() in model.py.

Measures the pure forward+backward+step loop (warmup excluded) on a fixed
synthetic batch of 150 bp reads, so the number reflects the architecture's
training compute, not data loading / NFS. Appends one row to a shared CSV that
pairs with the inference benchmark_summary.csv.

Usage:
  python probe_train_speed.py --config CONFIG.yaml --num_classes 120 \
      --label NT-v2_v9_genus --batch 128 --steps 30 --warmup 8 --csv OUT.csv
"""
import argparse, csv, os, random, time
from pathlib import Path
import yaml
import torch
import torch.nn as nn

import sys
from kmerformer.model import create_model
from transformers import AutoTokenizer


def rand_reads(n, length=150):
    return ["".join(random.choice("ACGT") for _ in range(length)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--read_len", type=int, default=150)
    ap.add_argument("--max_token_length", type=int, default=None,
                    help="override cfg.data.max_token_length")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    random.seed(0); torch.manual_seed(0)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    backbone = model_cfg["backbone"]
    max_len = args.max_token_length or cfg.get("data", {}).get("max_token_length", 128)

    print(f"[{args.label}] backbone={backbone} max_token_length={max_len} "
          f"batch={args.batch} num_classes={args.num_classes}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        backbone, trust_remote_code=model_cfg.get("trust_remote_code", True))

    # One fixed synthetic batch, tokenized once, reused every step (compute-only).
    enc = tok(rand_reads(args.batch, args.read_len),
              padding="max_length", truncation=True, max_length=max_len,
              return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else None
    labels = torch.randint(0, args.num_classes, (args.batch,), device=device)
    n_tokens = int(input_ids.shape[1])

    model = create_model(model_cfg, args.num_classes).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attn)
            loss = crit(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    def infer_step():
        with torch.no_grad(), torch.cuda.amp.autocast():
            model(input_ids, attn)

    def timed(fn, set_eval):
        model.eval() if set_eval else model.train()
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            fn()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        reads = args.steps * args.batch
        return reads / elapsed, elapsed / reads * 1000, \
            torch.cuda.max_memory_allocated() / (1024**2)

    header = ["model", "phase", "batch", "tokens_per_read",
              "reads_per_sec", "ms_per_read", "peak_gpu_mib"]
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for phase, fn, ev in [("infer", infer_step, True), ("train", train_step, False)]:
            rps, ms, peak = timed(fn, ev)
            print(f"[{args.label}] {phase:5s} {rps:,.1f} reads/sec  {ms:.3f} ms/read  "
                  f"peak {peak:,.0f} MiB  (tokens/read={n_tokens})", flush=True)
            w.writerow([args.label, phase, args.batch, n_tokens,
                        f"{rps:.1f}", f"{ms:.4f}", f"{peak:.0f}"])


if __name__ == "__main__":
    main()
