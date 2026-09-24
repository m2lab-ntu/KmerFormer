#!/usr/bin/env python3
"""6-mer non-neural baselines (EXACT, since 4^6=4096 vocab is tiny).
Same reference (50M train reads) + test (clean_common) as the 13-mer run.
Computes, for BOTH stride=1 (overlap, matches our 13-mer run) and stride=6
(non-overlap, matches NT-v2's actual tokenization):
  1. unique-key lookup   2. dominant-genus mode vote   3. multinomial Naive Bayes
Goal: isolate k-length as the only variable vs the 13-mer numbers
(0.72% / 35.3% / 74.9%) and NT-v2 6-mer neural (67%).
"""
import sys, time, numpy as np

K = 6; L = 150; NCODE = 4 ** K; NG = 120; ALPHA = 1.0
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
BATCH = 200_000
Wov = L - K + 1                     # overlapping windows = 145
S6_IDX = np.arange(0, Wov, K)       # non-overlap positions: 0,6,...,144 -> 25 tokens

LUT = np.full(256, 255, np.uint8)
for i, b in enumerate(b"ACGT"): LUT[b] = i

def encode(seqs):
    B = len(seqs)
    arr = LUT[np.frombuffer(b"".join(seqs), np.uint8).reshape(B, L)]
    code = np.zeros((B, Wov), np.int64); valid = np.ones((B, Wov), bool)
    for j in range(K):
        code = code * 4 + arr[:, j:j + Wov].astype(np.int64)
        valid &= (arr[:, j:j + Wov] != 255)
    return code, valid

def iread(path):
    with open(path, "rb") as f:
        while True:
            h = f.readline(); s = f.readline()
            if not s: break
            if h[:1] != b">": continue
            s = s.rstrip(b"\n")
            if len(s) != L: continue
            yield int(h.split(b"|", 4)[3]), s

# exact counts[code, genus] for both strides
cnt1 = np.zeros((NCODE, NG), np.float64)   # stride 1 (overlap)
cnt6 = np.zeros((NCODE, NG), np.float64)   # stride 6 (non-overlap)
t0 = time.time(); nref = 0; sb = []; gb = []
def flush_ref():
    global sb, gb
    if not sb: return
    code, valid = encode(sb)
    g = np.array(gb, np.int64)
    # stride 1
    gg = np.broadcast_to(g[:, None], code.shape)
    c1 = code[valid]; g1 = gg[valid]
    np.add.at(cnt1, (c1, g1), 1.0)
    # stride 6
    c6 = code[:, S6_IDX]; v6 = valid[:, S6_IDX]; gg6 = np.broadcast_to(g[:, None], c6.shape)
    cc6 = c6[v6]; g6 = gg6[v6]
    np.add.at(cnt6, (cc6, g6), 1.0)
    sb = []; gb = []
for genus, seq in iread(REF_FA):
    gb.append(genus); sb.append(seq); nref += 1
    if len(sb) >= BATCH: flush_ref()
    if nref % 10_000_000 == 0: print(f"  ref {nref/1e6:.0f}M | {time.time()-t0:.0f}s", flush=True)
    if nref >= REF_MAX: break
flush_ref()
print(f"REF counts built {nref:,} reads | {time.time()-t0:.0f}s", flush=True)

def build_models(cnt):
    seen = cnt.sum(1) > 0
    ng_per_code = (cnt > 0).sum(1)
    uniq_code = seen & (ng_per_code == 1)      # genus-unique 6-mer
    dom = np.where(seen, cnt.argmax(1), -1)     # dominant genus (exact argmax)
    uniq_genus = np.where(uniq_code, cnt.argmax(1), -1)
    N_g = cnt.sum(0)
    LC = np.log(cnt + ALPHA)                    # log(count+a)
    logdenom = np.log(N_g + ALPHA * NCODE)
    prior = np.log(N_g / N_g.sum() + 1e-12)
    return dict(uniq_genus=uniq_genus, dom=dom, LC=LC, logdenom=logdenom, prior=prior,
                pct_seen=100*seen.mean(), pct_uniq=100*uniq_code.sum()/max(seen.sum(),1))

def classify(seqs_gen, m):
    nb=nb_ok=mv=mv_ok=uk=uk_ok=uk_cov=0
    sb=[]; gb=[]
    def flush():
        nonlocal sb,gb,nb,nb_ok,mv,mv_ok,uk,uk_ok,uk_cov
        if not sb: return
        code, valid = encode(sb)
        for i in range(len(sb)):
            vc = code[i][valid[i]]; truth = gb[i]
            # NB
            S = m["LC"][vc].sum(0) - vc.size*m["logdenom"] + m["prior"]
            nb+=1; nb_ok += int(S.argmax()==truth)
            # mode vote
            dv = m["dom"][vc]; dv = dv[dv>=0]
            mv+=1
            if dv.size: mv_ok += int(np.bincount(dv).argmax()==truth)
            # unique-key vote
            uv = m["uniq_genus"][vc]; uv = uv[uv>=0]
            uk+=1
            if uv.size: uk_cov+=1; uk_ok += int(np.bincount(uv).argmax()==truth)
        sb=[]; gb=[]
    for genus, seq in seqs_gen:
        gb.append(genus); sb.append(seq)
        if len(sb)>=BATCH: flush()
    flush()
    return dict(nb=100*nb_ok/nb, mv=100*mv_ok/mv, uk_all=100*uk_ok/uk,
                uk_cov=100*uk_cov/uk)

for tag, cnt, ntok in [("stride=1 overlap (145 tok)", cnt1, 145),
                       ("stride=6 non-overlap (25 tok, = NT-v2)", cnt6, 25)]:
    m = build_models(cnt)
    r = classify(iread(TEST_FA), m)
    print(f"\n===== 6-mer {tag} =====")
    print(f"  distinct 6-mers seen={m['pct_seen']:.1f}% of 4096 | genus-unique={m['pct_uniq']:.2f}%")
    print(f"  unique-key lookup : {r['uk_all']:.2f}%  (coverage {r['uk_cov']:.1f}%)")
    print(f"  mode vote         : {r['mv']:.2f}%")
    print(f"  multinomial NB    : {r['nb']:.2f}%")
print("\ncompare 13-mer(overlap): unique 0.72% | mode 35.3% | NB 74.9% | MT neural 87.5% ; NT-v2 6-mer neural=67%")
