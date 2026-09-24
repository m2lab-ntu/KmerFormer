#!/usr/bin/env python3
"""Proper multinomial k-mer Naive Bayes (no neural net) — fairest non-neural
composition baseline. Uses full P(13-mer | genus) distribution.

score_g(read) = log_prior_g + sum_kmers [ log(count[kmer,g]+a) - log(N_g + a*V) ]
Classify each clean_common test read; compare to MT / NT-v2.
"""
import sys, time, numpy as np

K=13; W=150-K+1; NCODE=4**K
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
REF_MAX = int(sys.argv[1]) if len(sys.argv)>1 else 50_000_000
NG = 120; ALPHA = 1.0; BATCH=200_000

LUT=np.full(256,255,np.uint8)
for i,b in enumerate(b"ACGT"): LUT[b]=i

def encode_batch(seqs):
    B=len(seqs)
    arr=LUT[np.frombuffer(b"".join(seqs),np.uint8).reshape(B,150)]
    code=np.zeros((B,W),np.int64); valid=np.ones((B,W),bool)
    for j in range(K):
        code=code*4+arr[:,j:j+W].astype(np.int64)
        valid&=(arr[:,j:j+W]!=255)
    return code,valid

def iread(path):
    with open(path,"rb") as f:
        while True:
            h=f.readline(); s=f.readline()
            if not s: break
            if h[:1]!=b">": continue
            s=s.rstrip(b"\n")
            if len(s)!=150: continue
            yield int(h.split(b"|",4)[3]), s

t0=time.time()
# ---- Pass A: distinct 13-mers present in TEST reads -> dense index ----
present=np.zeros(NCODE,bool); gb,sb=[],[]
def flushA():
    global sb
    if not sb: return
    code,valid=encode_batch(sb); present[code[valid]]=True; sb.clear()
tg=[]
for genus,seq in iread(TEST_FA):
    tg.append(genus); sb.append(seq)
    if len(sb)>=BATCH: flushA()
flushA()
test_codes=np.flatnonzero(present); T=test_codes.size
idx_map=np.full(NCODE,-1,np.int32); idx_map[test_codes]=np.arange(T,dtype=np.int32)
print(f"distinct test 13-mers T={T:,} ({100*T/NCODE:.1f}% of space) | {time.time()-t0:.0f}s",flush=True)

# ---- Pass B: reference counts for test-relevant 13-mers, per genus ----
counts=np.zeros((T,NG),np.float32)   # T x 120
N_g=np.zeros(NG,np.float64)          # total kmers per genus (all, for denom)
reads_g=np.zeros(NG,np.float64)      # read prior
nref=0; sb=[]; gbatch=[]
def flushB():
    global sb,gbatch
    if not sb: return
    code,valid=encode_batch(sb)
    g=np.broadcast_to(np.array(gbatch,np.int64)[:,None],code.shape)
    c=code[valid]; gv=g[valid]
    N_g_local=np.bincount(gv,minlength=NG); N_g[:]+=N_g_local
    idx=idx_map[c]; keep=idx>=0
    np.add.at(counts,(idx[keep],gv[keep]),1.0)
    sb=[]; gbatch=[]
for genus,seq in iread(REF_FA):
    reads_g[genus]+=1; gbatch.append(genus); sb.append(seq); nref+=1
    if len(sb)>=BATCH: flushB()
    if nref%10_000_000==0: print(f"  ref {nref/1e6:.0f}M | {time.time()-t0:.0f}s",flush=True)
    if nref>=REF_MAX: break
flushB()
print(f"REF counts built from {nref:,} reads | {time.time()-t0:.0f}s",flush=True)

# ---- precompute log terms ----
logdenom = np.log(N_g + ALPHA*NCODE)             # (120,)
log_prior= np.log(reads_g/reads_g.sum() + 1e-12) # (120,)
counts += ALPHA; np.log(counts, out=counts)       # LC = log(count+alpha), in place

# ---- classify test reads ----
n=correct=0; sb=[]; tgb=[]
def flushT():
    global sb,tgb,n,correct
    if not sb: return
    code,valid=encode_batch(sb)
    for i in range(len(sb)):
        vc=code[i][valid[i]]; idx=idx_map[vc]; idx=idx[idx>=0]
        nv=idx.size
        S=counts[idx].sum(0)                       # (120,)
        score=S - nv*logdenom + log_prior
        pred=int(np.argmax(score)); n+=1
        correct+=int(pred==tgb[i])
    sb=[]; tgb=[]
tgb=[]
for genus,seq in iread(TEST_FA):
    tgb.append(genus); sb.append(seq)
    if len(sb)>=BATCH: flushT()
flushT()

print("\n============  MULTINOMIAL 13-mer NAIVE BAYES (no neural net)  ============")
print(f"reference reads : {nref:,}   alpha={ALPHA}")
print(f"test reads      : {n:,}")
print(f"NB genus accuracy : {100*correct/n:.2f}%")
print("compare: MT 13-mer @50M=87.5% @250M=98.7% | NT-v2 6-mer=67% | mode-vote=35.3% | unique-key=0.72%")
