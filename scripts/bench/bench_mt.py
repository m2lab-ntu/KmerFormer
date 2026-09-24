#!/usr/bin/env python
"""Benchmark MetaTransformer under the same protocol as bench_arms.py.

WHY: the manuscript's compute table is measured on one GPU through one code
path so its rows are comparable, but MetaTransformer lives in a separate
codebase and was therefore missing from it -- which left the advisor's request
for a comparison of "each method" only partly answered. This runs MT's own model
and its own tokenizer transform on the same RTX 4090, at the same batch size,
with the same warm-up-excluded timing and the same peak-memory accounting.

The one thing that cannot be matched is the training loop: MT's optimiser and
schedule are its own. We therefore time forward+backward+step with a plain AdamW
(SparseAdam where the embedding is sparse), which is what bench_arms.py does for
our arms too, so the training column measures the architecture rather than
either project's recipe.

Usage (MetaTransformer env):
  PYTHONPATH=<MT_src> python bench_mt.py --exp_dir <dir> --vocab <file> \
      --label "MetaTransformer 13-mer 1L" --batch 128 --csv out.csv
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

try:
    from models.model_utils import instantiate_model_by_str_name, read_transforms_for_input_layer
    from utils.utils import load_vocabulary
    from utils.device_handler import init_device_handler
except ImportError:
    sys.exit("MetaTransformer modules not found; set PYTHONPATH to its src/")


def rand_reads(n, length=150):
    return ["".join(random.choice("ACGT") for _ in range(length)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    random.seed(0)
    torch.manual_seed(0)

    exp_dir = Path(args.exp_dir)
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    vocab, _ = load_vocabulary(args.vocab)   # returns (vocab, reverse_map)
    dev = cfg.get("device_settings", {})
    init_device_handler(use_cpu=False, gpu_count=1,
                        gpu_ids=dev.get("gpu_ids", None), split_gpus=False)

    net = instantiate_model_by_str_name(cfg.model.name, cfg, len(vocab)).to(device)
    transforms = read_transforms_for_input_layer(
        cfg.mdl_common.input_module, cfg, vocab, train=False)
    kmer_transform = transforms[0]

    seqs = rand_reads(args.batch)
    toks = [torch.as_tensor(np.asarray(kmer_transform(s)), dtype=torch.long) for s in seqs]
    n_tokens = max(t.numel() for t in toks)
    x = torch.zeros(args.batch, n_tokens, dtype=torch.long)
    for i, t in enumerate(toks):
        x[i, :t.numel()] = t
    x = x.to(device)
    n_classes = int(cfg.mdl_common.num_classes)
    y = torch.randint(0, n_classes, (args.batch,), device=device)

    emb = sum(p.numel() for n_, p in net.named_parameters() if "embed" in n_.lower())
    other = sum(p.numel() for n_, p in net.named_parameters() if "embed" not in n_.lower())

    crit = nn.CrossEntropyLoss()
    # nn.Embedding(sparse=True) makes the GRADIENT sparse, not the parameter, so
    # p.is_sparse is False here and AdamW would reject the gradient at step time.
    # Find the sparse embeddings from the modules instead.
    sparse_ids = set()
    for mod in net.modules():
        if isinstance(mod, nn.Embedding) and getattr(mod, "sparse", False):
            sparse_ids.add(id(mod.weight))
    sparse = [p for p in net.parameters() if p.requires_grad and id(p) in sparse_ids]
    dense = [p for p in net.parameters() if p.requires_grad and id(p) not in sparse_ids]
    opts = []
    if dense:
        opts.append(torch.optim.AdamW(dense, lr=1e-4))
    if sparse:
        opts.append(torch.optim.SparseAdam(sparse, lr=1e-4))
    print(f"  optimisers: {len(dense)} dense tensors (AdamW), "
          f"{len(sparse)} sparse embedding tensors (SparseAdam)", flush=True)
    scaler = torch.amp.GradScaler("cuda")

    def fwd():
        try:
            return net(x)
        except TypeError:
            return net(x, None)

    def train_step():
        for o in opts:
            o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = fwd()
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        for o in opts:
            scaler.step(o)
        scaler.update()

    def infer_step():
        with torch.no_grad(), torch.amp.autocast("cuda"):
            fwd()

    def timed(fn, eval_mode):
        net.eval() if eval_mode else net.train()
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            fn()
        torch.cuda.synchronize()
        el = time.time() - t0
        return args.steps * args.batch / el, torch.cuda.max_memory_allocated() / (1024 ** 2)

    header = ["model", "phase", "batch", "tokens_per_read", "reads_per_sec",
              "peak_gpu_mib", "embed_params", "other_params"]
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for phase, fn, ev in [("infer", infer_step, True), ("train", train_step, False)]:
            try:
                rps, peak = timed(fn, ev)
            except torch.cuda.OutOfMemoryError:
                print(f"[{args.label}] {phase}: OOM", flush=True)
                w.writerow([args.label, phase, args.batch, n_tokens, "OOM", "OOM", emb, other])
                torch.cuda.empty_cache()
                continue
            print(f"[{args.label}] {phase:5s} {rps:>10,.0f} reads/s  peak {peak:>8,.0f} MiB  "
                  f"tok/read {n_tokens:>4d}  emb {emb/1e6:>8.1f}M  other {other/1e6:.2f}M",
                  flush=True)
            w.writerow([args.label, phase, args.batch, n_tokens, f"{rps:.0f}",
                        f"{peak:.0f}", emb, other])


if __name__ == "__main__":
    main()
