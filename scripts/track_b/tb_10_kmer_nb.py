#!/usr/bin/env python3
"""
Track B – Step 10: multinomial k-mer Naive Bayes baseline on a ladder pool (E42).

This is the mechanism control for the tokenization-crossover claim.  If MT
13-mer's closed-set advantage is essentially long-k token matching, then a
parameter-free 13-mer NB should lose roughly as much accuracy on novel species
as MT 13-mer does, while 6-mer NB should lose much less.  Comparing the two NB
curves isolates k from architecture, pre-training and data scale.

Ported from scripts/kmer_lookup_analysis/multinomial_nb.py (TWCC-hardcoded) with
three changes: k is a parameter, k=6 skips the sparse-index trick (4^6 fits
densely), and predictions are written as a preds.npz so tb_07 scores this
baseline through exactly the same code path as the neural models.

    score_g(read) = log_prior_g + sum_kmers [ log(count[kmer,g]+a) - log(N_g + a*V) ]

Memory: for k=13 the dense count table is (distinct test k-mers) x 120 float32.
Keep --subsample around 100K reads, which is what the published E11 number used;
a 500K-read pool would need ~30 GB.
"""

import os
import argparse
import time
from pathlib import Path

import numpy as np

LUT = np.full(256, 255, np.uint8)
for _i, _b in enumerate(b"ACGT"):
    LUT[_b] = _i


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=13)
    p.add_argument("--ref_fa", default=os.path.join(
        os.environ.get("KF_DATA", "."), "balanced_50M", "reads_50M.fa"))
    p.add_argument("--ref_max", type=int, default=50_000_000)
    p.add_argument("--test", action="append", required=True, metavar="LEVEL=FASTA",
                   help="Repeatable. All levels are scored in ONE pass over the "
                        "reference pool -- that pass is the expensive part, so "
                        "never run this script once per level.")
    p.add_argument("--out_dir", required=True,
                   help="writes <out_dir>/<level>/nb_<k>mer/preds.npz")
    p.add_argument("--n_genera", type=int, default=120)
    p.add_argument("--read_len", type=int, default=150)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=200_000)
    p.add_argument("--subsample", type=int, default=100_000,
                   help="0 = use every test read")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encode_batch(seqs: list[bytes], k: int, read_len: int):
    w = read_len - k + 1
    b = len(seqs)
    arr = LUT[np.frombuffer(b"".join(seqs), np.uint8).reshape(b, read_len)]
    code = np.zeros((b, w), np.int64)
    valid = np.ones((b, w), bool)
    for j in range(k):
        code = code * 4 + arr[:, j:j + w].astype(np.int64)
        valid &= (arr[:, j:j + w] != 255)
    return code, valid


def iter_reads(path: str, read_len: int):
    """Yield (genus_class, seq_id, seq) from a Track A/B style labelled FASTA."""
    with open(path, "rb") as fh:
        while True:
            hdr = fh.readline()
            seq = fh.readline()
            if not seq:
                break
            if hdr[:1] != b">":
                continue
            seq = seq.rstrip(b"\n")
            if len(seq) != read_len:
                continue
            sid = hdr[1:].rstrip(b"\n").decode()
            yield int(hdr.split(b"|", 4)[3]), sid, seq


def load_test(path: str, read_len: int, subsample: int, seed: int):
    genera, sids, seqs = [], [], []
    for g, sid, s in iter_reads(path, read_len):
        genera.append(g)
        sids.append(sid)
        seqs.append(s)
    genera = np.asarray(genera, np.int64)
    sids = np.asarray(sids, dtype=object)
    if subsample and len(seqs) > subsample:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(seqs), size=subsample, replace=False))
        seqs = [seqs[i] for i in sel]
        genera = genera[sel]
        sids = sids[sel]
        print(f"    subsampled to {len(seqs):,} reads")
    return genera, sids, seqs


def main():
    args = parse_args()
    t0 = time.time()
    k, rl, ng = args.k, args.read_len, args.n_genera
    ncode = 4 ** k
    print(f"k={k}  V={ncode:,}  ref={args.ref_fa}")

    # Load every requested level up front and concatenate: the reference pass
    # below is the cost, and it is shared across all of them.
    levels, y_parts, sid_parts, seq_parts, spans = [], [], [], [], []
    cursor = 0
    for spec in args.test:
        if "=" not in spec:
            raise SystemExit(f"expected LEVEL=FASTA, got {spec}")
        level, path = spec.split("=", 1)
        print(f"  {level}: {path}")
        yy, ss, qq = load_test(path, rl, args.subsample, args.seed)
        levels.append(level)
        y_parts.append(yy)
        sid_parts.append(ss)
        seq_parts.append(qq)
        spans.append((cursor, cursor + len(qq)))
        cursor += len(qq)

    y_true = np.concatenate(y_parts)
    all_sids = np.concatenate(sid_parts)
    test_seqs = [s for part in seq_parts for s in part]
    n_test = len(test_seqs)
    del y_parts, sid_parts, seq_parts
    print(f"  test reads (all levels): {n_test:,}")

    # ---- index the k-mer space we actually need ----
    if k <= 8:
        n_slots = ncode
        idx_map = np.arange(ncode, dtype=np.int64)
        print(f"  dense k-mer table ({ncode:,} slots)")
    else:
        present = np.zeros(ncode, bool)
        for i in range(0, n_test, args.batch):
            code, valid = encode_batch(test_seqs[i:i + args.batch], k, rl)
            present[code[valid]] = True
        slots = np.flatnonzero(present)
        n_slots = slots.size
        idx_map = np.full(ncode, -1, np.int32)
        idx_map[slots] = np.arange(n_slots, dtype=np.int32)
        del present, slots
        print(f"  distinct test {k}-mers: {n_slots:,} "
              f"({100 * n_slots / ncode:.2f}% of space)  "
              f"table={n_slots * ng * 4 / 1e9:.1f} GB | {time.time()-t0:.0f}s")

    counts = np.zeros((n_slots, ng), np.float32)
    n_g = np.zeros(ng, np.float64)
    reads_g = np.zeros(ng, np.float64)

    # ---- pass over the reference read pool ----
    buf_seqs, buf_g, n_ref = [], [], 0

    def flush_ref():
        nonlocal buf_seqs, buf_g
        if not buf_seqs:
            return
        code, valid = encode_batch(buf_seqs, k, rl)
        g = np.broadcast_to(np.asarray(buf_g, np.int64)[:, None], code.shape)
        c, gv = code[valid], g[valid]
        n_g[:] += np.bincount(gv, minlength=ng)
        idx = idx_map[c]
        keep = idx >= 0
        np.add.at(counts, (idx[keep], gv[keep]), 1.0)
        buf_seqs, buf_g = [], []

    for genus, _sid, seq in iter_reads(args.ref_fa, rl):
        reads_g[genus] += 1
        buf_g.append(genus)
        buf_seqs.append(seq)
        n_ref += 1
        if len(buf_seqs) >= args.batch:
            flush_ref()
        if n_ref % 10_000_000 == 0:
            print(f"  ref {n_ref/1e6:.0f}M | {time.time()-t0:.0f}s", flush=True)
        if n_ref >= args.ref_max:
            break
    flush_ref()
    print(f"  reference counts from {n_ref:,} reads | {time.time()-t0:.0f}s")

    log_denom = np.log(n_g + args.alpha * ncode)
    log_prior = np.log(reads_g / reads_g.sum() + 1e-12)
    counts += args.alpha
    np.log(counts, out=counts)

    # ---- classify ----
    preds = np.empty(n_test, np.int64)
    pos = 0
    for i in range(0, n_test, args.batch):
        chunk = test_seqs[i:i + args.batch]
        code, valid = encode_batch(chunk, k, rl)
        for j in range(len(chunk)):
            idx = idx_map[code[j][valid[j]]]
            idx = idx[idx >= 0]
            score = counts[idx].sum(0) - idx.size * log_denom + log_prior
            preds[pos] = int(np.argmax(score))
            pos += 1
    assert pos == n_test

    # Split back per level.  seq_ids travel with the predictions so tb_07 can
    # align this subsample against the full pool instead of assuming order.
    print(f"\n{k}-mer NB genus accuracy ({n_ref:,} reference reads)")
    for level, (lo, hi) in zip(levels, spans):
        out = Path(args.out_dir) / level / f"nb_{k}mer" / "preds.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, preds=preds[lo:hi], labels=y_true[lo:hi],
                            seq_ids=all_sids[lo:hi].astype(str))
        acc = float((preds[lo:hi] == y_true[lo:hi]).mean())
        print(f"  {level:<12} {acc*100:6.2f}%  ({hi-lo:,} reads)  -> {out}")
    print(f"  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
