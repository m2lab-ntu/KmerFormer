#!/usr/bin/env python3
"""Count total vs UNIQUE seq_ids in label TSVs and/or FASTA headers, + pairwise
overlap of the first two inputs. Hash-based (blake2b 8-byte). Quantifies the
read-duplication discovered during clean-pool build."""
import sys, hashlib
import numpy as np

def h64(s):
    return np.int64(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big", signed=True))

def hashes_of(path):
    parts_buf, out = [], []
    is_fa = path.endswith(".fa") or path.endswith(".fasta")
    n = 0
    with open(path) as f:
        if not is_fa:
            header = f.readline().rstrip("\n").split("\t")
            sid_col = header.index("seq_id") if "seq_id" in header else 1
        for line in f:
            if is_fa:
                if not line.startswith(">"): continue
                sid = line[1:].strip()
            else:
                p = line.rstrip("\n").split("\t")
                if len(p) <= 1: continue
                sid = p[sid_col]
            parts_buf.append(sid); n += 1
            if len(parts_buf) >= 5_000_000:
                out.append(np.fromiter((h64(s) for s in parts_buf), dtype=np.int64, count=len(parts_buf))); parts_buf=[]
    if parts_buf:
        out.append(np.fromiter((h64(s) for s in parts_buf), dtype=np.int64, count=len(parts_buf)))
    arr = np.concatenate(out) if out else np.array([], dtype=np.int64)
    return arr, n

results = {}
for path in sys.argv[1:]:
    arr, total = hashes_of(path)
    uniq = np.unique(arr)
    results[path] = uniq
    print(f"{path}\n   total rows/headers : {total:,}\n   UNIQUE seq_ids     : {len(uniq):,}\n   duplication factor : {total/max(len(uniq),1):.2f}x", flush=True)

keys = list(results)
if len(keys) >= 2:
    a, b = results[keys[0]], results[keys[1]]
    inter = np.intersect1d(a, b, assume_unique=True)
    print(f"\nOverlap {keys[0]} ∩ {keys[1]}: {len(inter):,} unique ids "
          f"(= {100*len(inter)/len(a):.1f}% of first, {100*len(inter)/len(b):.1f}% of second)")
