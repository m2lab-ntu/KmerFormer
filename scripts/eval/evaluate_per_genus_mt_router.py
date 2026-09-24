#!/usr/bin/env python3
"""
Per-genus NT-v2 pipeline with MT genus predictions as router (Exp F).

Replaces the v9 genus router with pre-computed MT genus predictions (npz),
keeping per-genus NT-v2 species classifiers unchanged.

This isolates the contribution of the genus router quality:
  Exp B  v9 router (66.1%)  → per-genus NT-v2
  Exp F  MT router (87.4%)  → per-genus NT-v2    ← this script

The genus_predictions npz must contain:
  preds  [N]  — genus class integers matching FASTA header field 3
  labels [N]  — true genus class integers (used to compute MT genus accuracy)

Usage:
    python scripts/evaluate_per_genus_mt_router.py \
        --test_fasta        /path/to/reads_100K.fa \
        --test_labels       /path/to/labels_100K.tsv \
        --genus_predictions results/mt_genus_for_routing/mt_genus_preds_100K.npz \
        --per_genus_dir     results/per_genus_50M \
        --split_dir         $KF_DATA/balanced_50M/per_genus \
        --output_dir        results/mt_genus_per_genus_eval
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from kmerformer.dataset import TokenLevelReadDataset
from kmerformer.model import create_model
from kmerformer.utils import save_json

PER_GENUS_MODEL_CFG = {
    "backbone": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "head_type": "attention_pool",
    "head_config": {"hidden_dim": 256, "num_attention_heads": 4, "dropout": 0.0},
    "freeze_backbone": False,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["query", "key", "value"],
    "gradient_checkpointing": False,
}

BACKBONE_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
MAX_TOKEN_LENGTH = 32


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data(fasta_path, labels_path):
    print(f"Loading test data:\n  FASTA:  {fasta_path}\n  Labels: {labels_path}")
    df = pd.read_csv(labels_path, sep="\t")
    print(f"  {len(df):,} reads  |  {df['genus_name'].nunique()} genera  |  {df['species_name'].nunique()} species")

    seq_dict = {}
    with open(fasta_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            else:
                seq_dict[header] = line

    species2id = {s: i for i, s in enumerate(sorted(df["species_name"].unique()))}

    # Build genus_class_int → genus_name mapping (for MT router)
    genus_cls_to_name = {}
    for _, row in df.iterrows():
        genus_cls_to_name[int(row["genus_class"])] = row["genus_name"]

    sequences, species_labels, genus_cls_labels = [], [], []
    for _, row in df.iterrows():
        if row["seq_id"] in seq_dict:
            sequences.append(seq_dict[row["seq_id"]])
            species_labels.append(species2id[row["species_name"]])
            genus_cls_labels.append(int(row["genus_class"]))

    return {
        "sequences":      sequences,
        "species_labels": np.array(species_labels),
        "genus_cls_labels": np.array(genus_cls_labels),
        "species2id":     species2id,
        "id2species":     {v: k for k, v in species2id.items()},
        "genus_cls_to_name": genus_cls_to_name,
        "num_species":    len(species2id),
    }


# ── Per-genus inference ───────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, sequences, tokenizer, device, batch_size=512, desc="Inference"):
    model.eval()
    use_amp = torch.cuda.is_available()
    amp_dtype = (torch.bfloat16
                 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                 else torch.float16)
    dummy = np.zeros(len(sequences), dtype=np.int64)
    dataset = TokenLevelReadDataset(sequences, dummy, tokenizer=tokenizer,
                                    max_length=MAX_TOKEN_LENGTH, rc_augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_logits = []
    for input_ids, attn_mask, _ in tqdm(loader, desc=desc, leave=False):
        input_ids = input_ids.to(device)
        attn_mask  = attn_mask.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(input_ids, attn_mask)
        all_logits.append(logits.float().cpu().numpy())
    return np.vstack(all_logits)


def per_genus_pipeline(sequences, genus_assignments, species2id,
                       per_genus_dir, split_dir, tokenizer, device):
    N = len(sequences)
    num_species = len(species2id)
    final_preds = np.zeros(N, dtype=np.int64)
    final_probs = np.zeros((N, num_species), dtype=np.float32)
    per_genus_dir = Path(per_genus_dir)
    split_dir = Path(split_dir)

    for genus_name, indices in tqdm(genus_assignments.items(), desc="Per-genus inference"):
        if not indices:
            continue
        species_list_path = split_dir / genus_name / "species_list.json"
        if not species_list_path.exists():
            continue
        species_list = json.loads(species_list_path.read_text())
        local2global = {i: species2id[sp] for i, sp in enumerate(species_list) if sp in species2id}
        indices = np.array(indices)

        if len(species_list) == 1:
            gid = species2id.get(species_list[0])
            if gid is not None:
                final_preds[indices] = gid
                final_probs[indices, gid] = 1.0
            continue

        best_pt = per_genus_dir / genus_name / "best.pt"
        if not best_pt.exists():
            continue

        model = create_model(PER_GENUS_MODEL_CFG, len(species_list)).to(device)
        ckpt = torch.load(best_pt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        genus_seqs = [sequences[i] for i in indices]
        logits = run_inference(model, genus_seqs, tokenizer, device,
                               desc=f"  {genus_name} ({len(indices)} reads)")
        probs = scipy_softmax(logits, axis=-1)
        local_preds = logits.argmax(axis=-1)

        global_preds = np.array([local2global.get(lp, 0) for lp in local_preds])
        final_preds[indices] = global_preds
        for local_id, global_id in local2global.items():
            final_probs[indices, global_id] = probs[:, local_id]

        del model
        torch.cuda.empty_cache()

    return final_preds, final_probs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(true_labels, preds, probs, num_classes, name):
    acc  = accuracy_score(true_labels, preds)
    f1   = f1_score(true_labels, preds, average="macro", zero_division=0)
    top5 = top_k_accuracy_score(true_labels, probs, k=5,  labels=range(num_classes))
    top10= top_k_accuracy_score(true_labels, probs, k=10, labels=range(num_classes))
    present = np.unique(true_labels)
    per_f1  = f1_score(true_labels, preds, average=None, zero_division=0, labels=range(num_classes))
    n_zero  = int(sum(1 for c in present if per_f1[c] == 0.0))
    print(f"\n  {name}")
    print(f"    Top-1: {acc:.4f}   Top-5: {top5:.4f}   Top-10: {top10:.4f}   F1: {f1:.4f}   F1=0: {n_zero}/{len(present)}")
    return {"name": name, "top1": float(acc), "top5": float(top5), "top10": float(top10),
            "f1_macro": float(f1), "f1_zero_classes": int(n_zero), "n_present_classes": int(len(present))}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fasta",        required=True)
    parser.add_argument("--test_labels",       required=True)
    parser.add_argument("--genus_predictions", required=True,
                        help="npz with preds/labels arrays using genus_class int IDs (from FASTA header field 3)")
    parser.add_argument("--per_genus_dir",     required=True)
    parser.add_argument("--split_dir",         required=True)
    parser.add_argument("--output_dir",        default="results/mt_genus_per_genus_eval")
    parser.add_argument("--batch_size",        type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  Output: {out_dir}\n")

    # ── Load test data ────────────────────────────────────────────────────────
    data = load_test_data(args.test_fasta, args.test_labels)
    sequences         = data["sequences"]
    true_species      = data["species_labels"]
    genus_cls_labels  = data["genus_cls_labels"]
    species2id        = data["species2id"]
    genus_cls_to_name = data["genus_cls_to_name"]
    num_species       = data["num_species"]
    N = len(sequences)

    # ── Load MT genus predictions ─────────────────────────────────────────────
    print(f"\nLoading MT genus predictions: {args.genus_predictions}")
    npz = np.load(args.genus_predictions)
    mt_genus_preds = npz["preds"]   # genus_class integers
    assert len(mt_genus_preds) == N, (
        f"Size mismatch: genus_predictions has {len(mt_genus_preds)} entries, test set has {N}"
    )

    mt_genus_acc = (mt_genus_preds == genus_cls_labels).mean()
    print(f"  MT genus accuracy on this test set: {mt_genus_acc:.4f}")

    # ── Verify genus ID mapping ───────────────────────────────────────────────
    mt_classes = set(mt_genus_preds.tolist())
    unknown = mt_classes - set(genus_cls_to_name.keys())
    if unknown:
        print(f"  WARNING: {len(unknown)} MT genus class IDs not found in test labels: {sorted(unknown)[:10]}")
        print("  These reads will be skipped in per-genus routing.")

    # ── Build per-genus assignment ────────────────────────────────────────────
    mt_assignments = defaultdict(list)
    for i, cls_int in enumerate(mt_genus_preds):
        gname = genus_cls_to_name.get(int(cls_int))
        if gname is not None:
            mt_assignments[gname].append(i)

    routed = sum(len(v) for v in mt_assignments.values())
    print(f"  Reads routed: {routed:,}/{N:,}  ({routed/N:.1%})")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BACKBONE_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, trust_remote_code=True)

    # ── Run per-genus pipeline ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Per-genus pipeline — MT genus router")
    print("=" * 60)
    preds, probs = per_genus_pipeline(
        sequences, mt_assignments, species2id,
        args.per_genus_dir, args.split_dir, tokenizer, device,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    results = [compute_metrics(true_species, preds, probs, num_species, "Per-genus (MT genus router)")]

    summary = {
        "n_reads": N,
        "num_species": num_species,
        "mt_genus_accuracy": float(mt_genus_acc),
        "modes": results,
    }
    out_json = out_dir / "eval_per_genus_mt_router.json"
    save_json(summary, str(out_json))
    print(f"\nSaved: {out_json}")

    np.savez_compressed(out_dir / "predictions_mt_router.npz",
                        preds=preds, probs=probs, labels=true_species)
    print(f"Saved: {out_dir}/predictions_mt_router.npz")


if __name__ == "__main__":
    main()
