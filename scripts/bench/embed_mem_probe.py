#!/usr/bin/env python3
"""Empirically measure GPU memory for the three tokenization embedding tables
on this V100-32GB. One mode per process (CUDA OOM is unrecoverable in-process).

Modes:
  A_full : full 4^13 = 67,108,864 x 64, sparse Embedding + SparseAdam
  A_obs  : observed 33,545,099 x 64, sparse Embedding + SparseAdam (MT's real vocab)
  C      : 2^22 = 4,194,304 x 128, dense Embedding + AdamW
Reports peak allocated MiB after one fwd+bwd+optim.step(), or the OOM point.
"""
import argparse, torch, torch.nn as nn

MODES = {
    "A_full": (67_108_864, 64, True),
    "A_obs":  (33_545_099, 64, True),
    "C":      (4_194_304, 128, False),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=138)
    a = ap.parse_args()
    V, D, sparse = MODES[a.mode]
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    GiB = 1024**3
    try:
        emb = nn.Embedding(V, D, sparse=sparse).to(dev)
        after_param = torch.cuda.memory_allocated() / GiB
        print(f"[{a.mode}] table {V:,}x{D} sparse={sparse}  param={after_param:.2f} GiB", flush=True)
        opt = (torch.optim.SparseAdam(emb.parameters()) if sparse
               else torch.optim.AdamW(emb.parameters()))
        idx = torch.randint(0, V, (a.batch, a.tokens), device=dev)
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            out = emb(idx)
            loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()
        peak = torch.cuda.max_memory_allocated() / GiB
        print(f"[{a.mode}] OK  peak_after_optimizer={peak:.2f} GiB  "
              f"(V100 cap = 32.00 GiB)", flush=True)
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated() / GiB
        msg = str(e).splitlines()[0]
        print(f"[{a.mode}] OOM/ERROR at peak={peak:.2f} GiB :: {msg}", flush=True)

if __name__ == "__main__":
    main()
