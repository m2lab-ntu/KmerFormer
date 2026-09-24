#!/usr/bin/env python3
"""
Build a COMMON clean read pool for cross-setting sample-level abundance tests.

Clean = read exists in the 1535sp SOURCE but NOT in ANY training set. We exclude
the UNION of seq_ids from every model's training data (reads_50M for v9/MT-50M
AND balanced_250M for v14/v15/gbal/sbal/MT-250M; the 17M sets are subsets of
balanced_250M, MT-250M is a split of it). Disjointness via a sorted hash set
(method B; a collision can only DROP a real leftover, never admit a train read).

Unlike build_clean_test.py (per-genus reservoir → flat), this keeps a single
GLOBAL reservoir → the pool follows the natural leftover distribution, which is
what a real metagenomic sample looks like. The reservoir is shuffled before
writing so evaluate_sample.py's sequential random_partition yields mixed samples.

  python build_clean_pool.py --source_fa SRC \
      --train_labels labels_50M.tsv,balanced_250M/labels_250M.tsv \
      --out_dir DIR --n_pool 6000000 --seed 1234
"""
import argparse, os, random, hashlib, io
import numpy as np

BATCH = 1_000_000


def parse_header(h):
    clean = h.lstrip(">").strip()
    p = clean.split("|")
    sid = clean
    sp_class = int(p[1]); sp_name = p[2]; gen_class = int(p[3])
    gen_name = p[4].split("-")[0]
    return sid, sp_class, sp_name, gen_class, gen_name


def h64(s):
    return np.int64(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big", signed=True))


def load_train_hashes(label_paths):
    all_h = []
    for lp in label_paths:
        print(f"[1] hashing training seq_ids from {lp} ...", flush=True)
        cnt = 0
        with open(lp) as f:
            header = f.readline().rstrip("\n").split("\t")
            sid_col = header.index("seq_id") if "seq_id" in header else 1
            buf = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= sid_col:
                    continue
                buf.append(parts[sid_col]); cnt += 1
                if len(buf) >= 5_000_000:
                    all_h.append(np.fromiter((h64(s) for s in buf), dtype=np.int64, count=len(buf)))
                    buf = []
            if buf:
                all_h.append(np.fromiter((h64(s) for s in buf), dtype=np.int64, count=len(buf)))
        print(f"    {cnt:,} ids from {lp}", flush=True)
    arr = np.unique(np.concatenate(all_h))   # sorted + dedup union
    print(f"    union: {len(arr):,} unique training hashes", flush=True)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_fa", required=True)
    ap.add_argument("--train_labels", required=True, help="comma-separated label TSVs (union excluded)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pool", type=int, default=6_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    train_hashes = load_train_hashes(args.train_labels.split(","))

    reservoir = []          # global reservoir of kept (clean) records
    n_leftover = 0          # total clean reads seen (reservoir prob)
    batch_recs = []

    def consider(rec):
        nonlocal n_leftover
        n_leftover += 1
        if len(reservoir) < args.n_pool:
            reservoir.append(rec)
        else:
            j = rng.randint(0, n_leftover - 1)
            if j < args.n_pool:
                reservoir[j] = rec

    def flush_batch():
        if not batch_recs:
            return
        hs = np.fromiter((h64(r[0]) for r in batch_recs), dtype=np.int64, count=len(batch_recs))
        idx = np.searchsorted(train_hashes, hs)
        in_tr = np.zeros(len(hs), dtype=bool)
        valid = idx < len(train_hashes)
        in_tr[valid] = train_hashes[idx[valid]] == hs[valid]
        for k, rec in enumerate(batch_recs):
            if not in_tr[k]:
                consider(rec)
        batch_recs.clear()

    print(f"[2] streaming source {args.source_fa} ...", flush=True)
    cur_hdr = None; cur_seq = []; n_reads = 0

    def emit():
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
                emit(); cur_hdr = line; n_reads += 1
                if n_reads % 20_000_000 == 0:
                    print(f"    {n_reads:,} scanned, {n_leftover:,} clean so far", flush=True)
            else:
                cur_seq.append(line.strip())
        emit(); flush_batch()
    print(f"    scanned {n_reads:,}; total clean leftovers seen: {n_leftover:,}", flush=True)

    print(f"[3] shuffling {len(reservoir):,} pooled reads + writing ...", flush=True)
    rng.shuffle(reservoir)
    fa = os.path.join(args.out_dir, "reads.fa")
    tsv = os.path.join(args.out_dir, "labels.tsv")
    with open(fa, "w") as ffa, open(tsv, "w") as ftsv:
        ftsv.write("idx\tseq_id\tspecies_class\tgenus_class\tgenus_name\tspecies_name\n")
        for i, (sid, spc, spn, gc, gn, seq) in enumerate(reservoir):
            ffa.write(f">{sid}\n{seq}\n")
            ftsv.write(f"{i}\t{sid}\t{spc}\t{gc}\t{gn}\t{spn}\n")
    print("=" * 60)
    print(f"DONE. Clean common pool: {len(reservoir):,} reads "
          f"(of {n_leftover:,} clean leftovers) → {args.out_dir}")


if __name__ == "__main__":
    main()
