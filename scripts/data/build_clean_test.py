#!/usr/bin/env python
"""
Build a CLEAN, genus-balanced held-out test set for v14/v15 evaluation.

Clean = every read exists in the 1535sp SOURCE but NOT in balanced_250M (the
training data). Disjointness is guaranteed by a seq_id set check against the
training labels (method B), independent of how the subsample RNG worked.

Single streaming pass over the ~50GB source, memory-light:
  1. Load all training seq_ids from balanced_250M/labels_250M.tsv, blake2b-hash
     each to an 8-byte int -> sorted numpy int64 array (~2GB for 250M). A hash
     collision can only DROP a genuine leftover (false "in-train"), never admit
     a training read -> the test stays clean (conservative).
  2. Stream source reads.fa (60-col WRAPPED -> join lines into full reads).
     Process reads in BATCHES: batch-hash, vectorised searchsorted vs the
     training hashes (fast), then feed leftovers (not in train) to a per-genus
     reservoir sampler (size N).
  3. Emit reservoirs as SINGLE-LINE FASTA + labels TSV (dodges the _read_seq
     wrap-truncation reader bug at eval time).

Genera fully consumed by training (leftover=0) get 0 test reads; summary
reports coverage.
"""
import argparse
import hashlib
import os
import random
import numpy as np

BATCH = 100_000


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "little", signed=True)


def parse_header(line: str):
    """'>lbl|<sp>|<spname>|<gen>|<genname>-<n>' -> (seq_id, sp_class, sp_name, gen_class, gen_name)."""
    sid = line[1:].strip() if line.startswith(">") else line.strip()
    p = sid.split("|")
    return sid, int(p[1]), p[2], int(p[3]), p[4].rsplit("-", 1)[0]


def load_train_hashes(labels_path: str) -> np.ndarray:
    print(f"[1/3] hashing training seq_ids from {labels_path} ...", flush=True)
    hashes = []
    with open(labels_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        sid_col = header.index("seq_id") if "seq_id" in header else 1
        for i, line in enumerate(f):
            hashes.append(h64(line.rstrip("\n").split("\t")[sid_col]))
            if (i + 1) % 20_000_000 == 0:
                print(f"    {i+1:,} train ids hashed", flush=True)
    arr = np.array(hashes, dtype=np.int64)
    arr.sort()
    print(f"    {len(arr):,} training hashes (sorted)", flush=True)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_fa", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_genus", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    train_hashes = load_train_hashes(args.train_labels)

    reservoir = {}       # gen_class -> list of recs
    seen_leftover = {}   # gen_class -> count seen (reservoir prob)

    def consider(rec):
        g = rec[3]
        seen_leftover[g] = seen_leftover.get(g, 0) + 1
        buf = reservoir.setdefault(g, [])
        if len(buf) < args.per_genus:
            buf.append(rec)
        else:
            j = rng.randint(0, seen_leftover[g] - 1)
            if j < args.per_genus:
                buf[j] = rec

    batch_recs = []

    def flush_batch():
        if not batch_recs:
            return
        hs = np.fromiter((h64(r[0]) for r in batch_recs), dtype=np.int64,
                         count=len(batch_recs))
        idx = np.searchsorted(train_hashes, hs)
        in_tr = np.zeros(len(hs), dtype=bool)
        valid = idx < len(train_hashes)
        in_tr[valid] = train_hashes[idx[valid]] == hs[valid]
        for k, rec in enumerate(batch_recs):
            if not in_tr[k]:
                consider(rec)
        batch_recs.clear()

    print(f"[2/3] streaming source {args.source_fa} ...", flush=True)
    cur_hdr = None
    cur_seq = []
    n_reads = 0

    def emit_read():
        nonlocal cur_hdr, cur_seq
        if cur_hdr is None:
            return
        sid, spc, spn, gc, gn = parse_header(cur_hdr)
        batch_recs.append((sid, spc, spn, gc, gn, "".join(cur_seq)))
        cur_hdr, cur_seq = None, []
        if len(batch_recs) >= BATCH:
            flush_batch()

    with open(args.source_fa) as f:
        for line in f:
            if line.startswith(">"):
                emit_read()
                cur_hdr = line
                n_reads += 1
                if n_reads % 20_000_000 == 0:
                    print(f"    {n_reads:,} source reads scanned", flush=True)
            else:
                cur_seq.append(line.strip())
        emit_read()
        flush_batch()
    print(f"    scanned {n_reads:,} source reads", flush=True)

    # ── emit single-line FASTA + labels ──
    print(f"[3/3] writing test set to {args.out_dir} ...", flush=True)
    fa_path = os.path.join(args.out_dir, "reads.fa")
    tsv_path = os.path.join(args.out_dir, "labels.tsv")
    n_out = 0
    with open(fa_path, "w") as fa, open(tsv_path, "w") as tsv:
        tsv.write("idx\tseq_id\tspecies_class\tgenus_class\tgenus_name\tspecies_name\n")
        for g in sorted(reservoir):
            for (sid, spc, spn, gc, gn, seq) in reservoir[g]:
                fa.write(f">{sid}\n{seq}\n")
                tsv.write(f"{n_out}\t{sid}\t{spc}\t{gc}\t{gn}\t{spn}\n")
                n_out += 1

    covered = len(reservoir)
    full = sum(1 for g in reservoir if len(reservoir[g]) >= args.per_genus)
    print("=" * 60)
    print(f"DONE. Test set: {n_out:,} reads across {covered} genera")
    print(f"  genera at full quota ({args.per_genus}): {full}")
    print(f"  genera under quota: {covered - full}")
    print(f"Wrote: {fa_path}")
    print(f"       {tsv_path}")


if __name__ == "__main__":
    main()
