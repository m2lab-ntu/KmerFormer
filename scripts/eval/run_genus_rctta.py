#!/usr/bin/env python
"""Genus eval with PROPER reverse-complement TTA (sequence-level RC, averaged logits).

Reports forward-only, RC-only, and forward+RC (TTA) Top-1 — so the TTA delta is clear.
The existing run_nt_genus_test100k.py --rc_tta just flips token IDs (not a real RC); this
does RC on the DNA string before tokenizing.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from kmerformer.dataset import TokenLevelReadDataset
from kmerformer.model import create_model
from kmerformer.utils import load_config

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(seq): return seq.translate(_COMP)[::-1]

def load_fasta(path):
    ids, seqs, cid, cur = [], [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cid is not None: ids.append(cid); seqs.append("".join(cur))
                cid = line[1:]; cur = []
            elif line: cur.append(line)
    if cid is not None: ids.append(cid); seqs.append("".join(cur))
    return ids, seqs

@torch.no_grad()
def infer(model, seqs, labels, tokenizer, max_len, kmer, bs, device, use_amp, desc):
    ds = TokenLevelReadDataset(seqs, labels, tokenizer=tokenizer, max_length=max_len,
                               rc_augment=False, kmer_preprocess=kmer)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    out = []
    for input_ids, attn, _ in tqdm(loader, desc=desc):
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attn)
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_fasta", required=True)
    ap.add_argument("--test_labels", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_labels, sep="\t")
    genus2id = {g: i for i, g in enumerate(sorted(train_df["genus_name"].unique()))}
    n_classes = len(genus2id)
    test_df = pd.read_csv(args.test_labels, sep="\t")
    sid2gn = dict(zip(test_df["seq_id"], test_df["genus_name"]))
    fids, fseqs = load_fasta(args.test_fasta)
    seqs, labels = [], []
    for sid, seq in zip(fids, fseqs):
        gn = sid2gn.get(sid)
        if gn in genus2id:
            seqs.append(seq); labels.append(genus2id[gn])
    labels = np.array(labels, dtype=np.int64)
    print(f"n_classes={n_classes}  matched={len(seqs):,}", flush=True)

    backbone = cfg["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    max_len = cfg["data"].get("max_token_length", 32)
    kmer = cfg["data"].get("kmer_preprocess", None)
    use_amp = cfg["training"].get("amp", True)

    mc = dict(cfg["model"]); mc["gradient_checkpointing"] = False
    model = create_model(mc, n_classes).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]; rm = {}
    for k, v in sd.items():
        p = k.split(".")
        if len(p) >= 2 and p[-1] in ("weight", "bias") and p[-2] in ("query", "key", "value"):
            rm[".".join(p[:-1]) + ".base_layer." + p[-1]] = v
        else: rm[k] = v
    model.load_state_dict(rm, strict=False); model.eval()

    p_fwd = infer(model, seqs, labels, tokenizer, max_len, kmer, args.batch_size, device, use_amp, "fwd")
    p_rc  = infer(model, [rc(s) for s in seqs], labels, tokenizer, max_len, kmer, args.batch_size, device, use_amp, "rc")

    acc_fwd = (p_fwd.argmax(1) == labels).mean()
    acc_rc  = (p_rc.argmax(1) == labels).mean()
    acc_tta = ((p_fwd + p_rc).argmax(1) == labels).mean()
    print(f"\nforward-only : {acc_fwd*100:.2f}%")
    print(f"rc-only      : {acc_rc*100:.2f}%")
    print(f"RC-TTA (avg) : {acc_tta*100:.2f}%   (delta vs fwd: {(acc_tta-acc_fwd)*100:+.2f} pp)")
    preds = (p_fwd + p_rc).argmax(1).astype(np.int64)   # RC-TTA per-read prediction (for sample-level eval)
    np.savez_compressed(os.path.join(args.out_dir, "rctta.npz"),
                        preds=preds, labels=labels, acc_fwd=acc_fwd, acc_rc=acc_rc, acc_tta=acc_tta)

if __name__ == "__main__":
    main()
