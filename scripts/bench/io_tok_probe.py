#!/usr/bin/env python3
"""Isolate the Job-B dataloader bottleneck: NFS random-seek vs CPU tokenization.

Job B measured 528 reads/s/GPU against a 1,790 reads/s/GPU compute ceiling, so
the DataLoader is the limit. This tells us which half — and therefore whether
raising num_workers (fixes I/O latency) would actually help.
"""
import os
import time, numpy as np, random
from transformers import AutoTokenizer

_D = os.environ.get("KF_DATA", ".")
FASTA = os.environ.get("KF_FASTA", os.path.join(_D, "balanced_50M", "reads_50M.fa"))
# byte-offset index, written next to the FASTA on first lazy load
IDX   = os.environ.get("KF_FASTA_IDX", FASTA.replace(".fa", ".idx.npy"))
N     = 2000

idx = np.load(IDX, allow_pickle=False)
rng = random.Random(0)
picks = [int(idx["offset"][rng.randrange(len(idx))]) for _ in range(N)]

# ---- 1. pure NFS random-seek read (what one DataLoader worker does) ----
f = open(FASTA, "rb")
t0 = time.time()
seqs = []
for off in picks:
    f.seek(off)
    f.readline()                 # header
    seqs.append(f.readline().decode().strip())
t_io = time.time() - t0
print(f"[io]  random seek+read : {N/t_io:8,.0f} reads/s  ({t_io/N*1000:.3f} ms/read)")

# ---- 2. pure tokenization, stride-1 6-mer (no I/O) ----
tk = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)

def prep(seq, k=6, stride=1):
    return " ".join(seq[i:i+k] for i in range(0, len(seq)-k+1, stride))

t0 = time.time()
joined = [prep(s) for s in seqs]
t_join = time.time() - t0
t0 = time.time()
for s in joined:
    tk(s, return_tensors="pt", padding="max_length", truncation=True, max_length=150)
t_tok = time.time() - t0
print(f"[cpu] kmer string join : {N/t_join:8,.0f} reads/s  ({t_join/N*1000:.3f} ms/read)")
print(f"[cpu] tokenizer call   : {N/t_tok:8,.0f} reads/s  ({t_tok/N*1000:.3f} ms/read)")
tot = t_io + t_join + t_tok
print(f"[sum] one worker total : {N/tot:8,.0f} reads/s  ({tot/N*1000:.3f} ms/read)")
print(f"      breakdown: io {t_io/tot*100:.0f}%  join {t_join/tot*100:.0f}%  tok {t_tok/tot*100:.0f}%")
print(f"      -> 4 workers/rank would give ~{4*N/tot:,.0f} reads/s/rank "
      f"(observed 528)")
