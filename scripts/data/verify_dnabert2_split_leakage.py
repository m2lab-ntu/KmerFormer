#!/usr/bin/env python
"""
Leakage check for DNABERT-2 resume (epoch 17 best.pt).

The train/val split is NOT stored in the checkpoint — it is recomputed at
startup by data_loader.load_data() via sklearn train_test_split(seed=42).
epoch 1-17 were trained on TWCC (different machine, possibly different file
order / sklearn version). If the val set recomputed here on Nano4 differs
from TWCC's, then resuming would leak: TWCC-train reads land in this val set.

Decision rule (val_acc of epoch-17 best.pt on the Nano4-recomputed val set):
  ≈ 59.22%  → split is consistent → resume is SAFE.
  >> 59.22% → leakage: model has seen these "val" reads in training → REtrain.
  far off   → split mismatch → do not resume.

This reuses the EXACT same functions as train.py so the comparison is valid.
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml

# Same modules train.py uses — guarantees identical split / dataset / model.
from kmerformer.data_loader import load_data
from kmerformer.dataset import TokenLevelReadDataset
from kmerformer.model import create_model
from train import validate
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fasta", required=True, help="Nano4 reads_50M.fa")
    ap.add_argument("--labels", required=True, help="Nano4 labels_50M.tsv")
    ap.add_argument("--max_val", type=int, default=0,
                    help="0 = full val set (most rigorous); else cap for a quick check")
    ap.add_argument("--expected_acc", type=float, default=0.5922,
                    help="training_history val_acc at epoch 17")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = cfg["data"]["task"]
    val_ratio = cfg["data"].get("val_ratio", 0.1)
    seed = cfg["data"].get("seed", 42)
    max_len = cfg["data"].get("max_token_length", 64)
    kmer_preprocess = cfg["data"].get("kmer_preprocess", None)

    print("=" * 64)
    print("DNABERT-2 resume leakage check")
    print(f"  config       : {args.config}")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  fasta        : {args.fasta}")
    print(f"  val_ratio={val_ratio}  seed={seed}  task={task}  max_len={max_len}")
    print(f"  EXPECTED val_acc (epoch 17): {args.expected_acc:.4f}")
    print("=" * 64)

    # ── 1. EXACT same split as train.py (sklearn train_test_split, seed=42) ──
    data = load_data(args.fasta, args.labels, val_ratio, seed, task=task)
    val_seqs = data["val_sequences"].tolist()
    if task == "genus":
        val_labels = data["val_genus_labels"]
        num_classes = data["num_genera"]
    else:
        val_labels = data["val_species_labels"]
        num_classes = data["num_species"]

    if args.max_val and args.max_val < len(val_seqs):
        print(f"\n[NOTE] capping val to first {args.max_val:,} of {len(val_seqs):,} "
              f"(quick check — full run is more rigorous)")
        val_seqs = val_seqs[:args.max_val]
        val_labels = val_labels[:args.max_val]

    # ── 2. Tokenizer + val dataset (rc_augment=False, exactly like train.py) ──
    backbone = cfg["model"]["backbone"]
    trust = cfg["model"].get("trust_remote_code", True)
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust)
    val_ds = TokenLevelReadDataset(
        val_seqs, val_labels, tokenizer=tokenizer, max_length=max_len,
        rc_augment=False, rc_consistency=False, kmer_preprocess=kmer_preprocess,
    )
    eval_bs = cfg["training"].get("eval_batch_size", 512)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4), pin_memory=True,
    )

    # ── 3. Model + epoch-17 weights (same peft key remap as train.py resume) ──
    model = create_model(cfg["model"], num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    remapped = {}
    for k, v in sd.items():
        parts = k.split(".")
        if len(parts) >= 2 and parts[-1] in ("weight", "bias") and parts[-2] in ("query", "key", "value"):
            remapped[".".join(parts[:-1]) + ".base_layer." + parts[-1]] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:3]} ...")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:3]} ...")
    print(f"  Loaded checkpoint epoch={ckpt.get('epoch')} "
          f"recorded val_acc={ckpt.get('val_acc')}")

    # ── 4. Validate (identical loop to train.py) ──
    use_amp = cfg["training"].get("amp", True)
    criterion = nn.CrossEntropyLoss()
    _loss, acc, f1, _p, _l = validate(model, val_loader, criterion, device, use_amp)

    # ── 5. Verdict ──
    diff = acc - args.expected_acc
    print("\n" + "=" * 64)
    print(f"Recomputed val_acc : {acc:.4f}  (f1={f1:.4f})")
    print(f"Expected (ep17)    : {args.expected_acc:.4f}")
    print(f"Difference         : {diff:+.4f} ({diff*100:+.2f} pp)")
    print("-" * 64)
    if abs(diff) <= 0.01:
        print("VERDICT: ✅ MATCH — split is consistent → resume is SAFE.")
    elif diff > 0.01:
        print("VERDICT: ⚠️ HIGHER than recorded — LIKELY LEAKAGE "
              "(model saw these 'val' reads in training). DO NOT resume; retrain.")
    else:
        print("VERDICT: ❓ LOWER/different — split mismatch or env drift. "
              "Do not resume; investigate or retrain from scratch.")
    print("=" * 64)


if __name__ == "__main__":
    main()
