#!/usr/bin/env python3
"""Kraken-lite 13-mer genus lookup — tests the 'MT 13-mer = near-lookup' hypothesis.

Build a 13-mer -> genus map from TRAINING reads (unique vs ambiguous), then
classify clean_common TEST reads by majority vote over genus-unique 13-mers.
No neural net. If this dumb lookup reaches ~MT accuracy, the task is near-lookup.
"""
import sys, time, numpy as np

K = 13
W = 150 - K + 1          # 138 windows per 150bp read
NCODE = 4 ** K           # 67,108,864
# Paths come from the environment so this runs anywhere. Defaults follow the
# layout in docs/DATA.md; the genus label is read out of each FASTA header
# (field 3 of "lbl|species|assembly|genus|name"), so no label TSV is needed.
#
#   KF_DATA     root of the read pools                (default: cwd)
#   KF_REF_FA   reference reads to count k-mers over  (default: 50M training pool)
#   KF_TEST_FA  reads to classify                     (default: closed-set 100K)
#
# The original runs pointed KF_TEST_FA at a file named reads_clean_common.fa,
# which is the same pool as val_100K/reads_100K_val.fa under an earlier name.
import os
_D = os.environ.get("KF_DATA", ".")
REF_FA  = os.environ.get("KF_REF_FA",  os.path.join(_D, "balanced_50M", "reads_50M.fa"))
TEST_FA = os.environ.get("KF_TEST_FA", os.path.join(_D, "val_100K", "reads_100K_val.fa"))
if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    print(f"usage: {sys.argv[0]} [MAX_REFERENCE_READS]")
    print()
    print("Set KF_DATA (or KF_REF_FA / KF_TEST_FA) to point at the pools.")
    print(f"  reference: {REF_FA}")
    print(f"  test     : {TEST_FA}")
    raise SystemExit(0)
REF_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
BATCH   = 200_000

LUT = np.full(256, 255, np.uint8)
for i, b in enumerate(b"ACGT"):
    LUT[b] = i

def encode_batch(seqs):
    """seqs: list of equal-length(150) bytes -> (codes (B,W) int64, valid (B,W) bool)."""
    B = len(seqs)
    arr = LUT[np.frombuffer(b"".join(seqs), np.uint8).reshape(B, 150)]
    code = np.zeros((B, W), dtype=np.int64)
    valid = np.ones((B, W), dtype=bool)
    for j in range(K):
        code = code * 4 + arr[:, j:j + W].astype(np.int64)
        valid &= (arr[:, j:j + W] != 255)
    return code, valid

def genus_from_header(h):
    # >lbl|species|seqid|GENUS|name/mate   -> field index 3
    return int(h.split(b"|", 4)[3])

def iread(path):
    """yield (genus:int, seq:bytes) for 150bp reads."""
    with open(path, "rb") as f:
        while True:
            h = f.readline()
            s = f.readline()
            if not s:
                break
            if h[:1] != b">":
                continue
            s = s.rstrip(b"\n")
            if len(s) != 150:
                continue
            yield genus_from_header(h), s

# ---- 1. build reference ------------------------------------------------------
first_genus = np.full(NCODE, -1, dtype=np.int16)   # -1 unseen, >=0 genus
ambig       = np.zeros(NCODE, dtype=bool)          # True = seen in >1 genus
t0 = time.time(); nref = 0
gb, sb = [], []
def flush_ref():
    global gb, sb
    if not sb: return
    code, valid = encode_batch(sb)
    g = np.broadcast_to(np.array(gb, np.int16)[:, None], code.shape)
    cf = code[valid]; gf = g[valid]
    vals = first_genus[cf]
    new = vals == -1
    first_genus[cf[new]] = gf[new]
    vals2 = first_genus[cf]
    conf = (vals2 != -1) & (vals2 != gf)
    ambig[cf[conf]] = True
    gb, sb = [], []

for genus, seq in iread(REF_FA):
    gb.append(genus); sb.append(seq); nref += 1
    if len(sb) >= BATCH:
        flush_ref()
        if nref % 2_000_000 == 0:
            seen = int((first_genus >= 0).sum()); amb = int(ambig.sum())
            print(f"  ref {nref/1e6:.0f}M reads | distinct 13mers seen={seen/1e6:.1f}M "
                  f"unique-genus={100*(seen-amb)/max(seen,1):.1f}% | {time.time()-t0:.0f}s", flush=True)
    if nref >= REF_MAX:
        break
flush_ref()
seen = int((first_genus >= 0).sum()); amb = int(ambig.sum())
print(f"REFERENCE built from {nref:,} reads | distinct 13-mers seen = {seen:,} "
      f"| genus-unique = {seen-amb:,} ({100*(seen-amb)/max(seen,1):.1f}%) "
      f"| ambiguous = {amb:,} | {time.time()-t0:.0f}s", flush=True)

# ---- 2. classify test reads by unique-13mer majority vote --------------------
n=correct=covered=cov_correct=0
tot_km=known_km=uniq_km=hit_true=0
gb, sb = [], []
def flush_test():
    global gb, sb, n, correct, covered, cov_correct, tot_km, known_km, uniq_km, hit_true
    if not sb: return
    code, valid = encode_batch(sb)
    for i in range(len(sb)):
        vc = code[i][valid[i]]
        tot_km += vc.size
        fg = first_genus[vc]; am = ambig[vc]
        known = fg >= 0
        uniq = known & (~am)
        known_km += int(known.sum()); uniq_km += int(uniq.sum())
        truth = gb[i]
        votes = fg[uniq]
        hit_true += int((votes == truth).any())
        n += 1
        if votes.size:
            covered += 1
            pred = np.bincount(votes).argmax()
            ok = int(pred == truth)
            correct += ok; cov_correct += ok
    gb, sb = [], []

for genus, seq in iread(TEST_FA):
    gb.append(genus); sb.append(seq)
    if len(sb) >= BATCH:
        flush_test()
flush_test()

print("\n================  13-mer LOOKUP CLASSIFIER (no neural net)  ================")
print(f"test reads                         : {n:,}")
print(f"coverage (>=1 unique 13-mer)        : {100*covered/n:.2f}%")
print(f"lookup acc  (over ALL reads)        : {100*correct/n:.2f}%")
print(f"lookup acc  (over COVERED reads)    : {100*cov_correct/max(covered,1):.2f}%")
print(f"reads with >=1 unique 13-mer hitting TRUE genus : {100*hit_true/n:.2f}%")
print(f"per-read 13-mers: known-in-ref={100*known_km/max(tot_km,1):.1f}%  "
      f"genus-unique={100*uniq_km/max(tot_km,1):.1f}%  (of all valid 13-mers)")
print(f"genus-unique share of KNOWN 13-mers : {100*uniq_km/max(known_km,1):.1f}%")
print("compare: MT 13-mer @250M = 98.7% ; @50M = 87.5% ; NT-v2 6-mer = 67%")
