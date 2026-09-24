#!/usr/bin/env python3
"""
MT hierarchical pipeline: MT genus router → masked MT species prediction (Exp E).

For each read:
  1. predicted_genus  = argmax(genus_logits)
  2. valid_species    = genus2species[predicted_genus]   (from FASTA headers)
  3. predicted_species = valid_species[argmax(species_logits[valid_species])]

Saves preds, probs, labels (species) as npz.

Usage:
    PYTHONPATH=/path/to/MetaTransformer/src \
    python extract_mt_hierarchical_predictions.py \
        --genus_exp_dir   $GENUS_EXP \
        --species_exp_dir $SPECIES_EXP \
        --val_dir         $KF_DATA/balanced_50M/metatransformer_format/val \
        --vocab           vocab_file/vocab_13mer.txt \
        --genus_class_indices   3 \
        --species_class_indices 1 \
        --out             mt_hierarchical_preds.npz \
        --batch_size 512
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    from models.model_utils import instantiate_model_by_str_name, read_transforms_for_input_layer
    from utils.utils import load_vocabulary
except ImportError:
    print("ERROR: MetaTransformer modules not found.")
    print("Set PYTHONPATH=/path/to/MetaTransformer/src before running this script.")
    sys.exit(1)

PAD_IDX = 0


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta_dir(val_dir, genus_ci, species_ci):
    """Return seqs, genus_labels, species_labels from all .fa files in val_dir."""
    seqs, g_lbls, s_lbls = [], [], []
    for fa_path in sorted(Path(val_dir).glob("*.fa")):
        header, parts = None, []
        with open(fa_path) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is not None:
                        fields = header.split("|")
                        g_lbls.append(int(fields[genus_ci]))
                        s_lbls.append(int(fields[species_ci]))
                        seqs.append("".join(parts))
                    header, parts = line[1:], []
                else:
                    parts.append(line)
        if header is not None:
            fields = header.split("|")
            g_lbls.append(int(fields[genus_ci]))
            s_lbls.append(int(fields[species_ci]))
            seqs.append("".join(parts))
    return seqs, np.array(g_lbls, dtype=np.int64), np.array(s_lbls, dtype=np.int64)


def build_genus2species(g_labels, s_labels):
    """genus_class_int → sorted array of species_class_int."""
    g2s = defaultdict(set)
    for g, s in zip(g_labels, s_labels):
        g2s[int(g)].add(int(s))
    return {g: np.array(sorted(slist), dtype=np.int64) for g, slist in g2s.items()}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_mt_model(exp_dir, vocab, device):
    exp_dir = Path(exp_dir)
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    net = instantiate_model_by_str_name(cfg.model.name, cfg, len(vocab))
    ckpt_path = exp_dir / "checkpoints" / "classification_transformer_ckpt_best.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_state_dict"], strict=False)
    net.eval().to(device)
    transforms = read_transforms_for_input_layer(cfg.mdl_common.input_module, cfg, vocab, train=False)
    print(f"  Loaded: {ckpt_path.name}  num_classes={cfg.mdl_common.num_classes}")
    return net, transforms[0]


# ── Batched inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, seqs, kmer_transform, device, batch_size, desc="Inference"):
    use_amp = torch.cuda.is_available()
    all_logits = []
    for i in tqdm(range(0, len(seqs), batch_size), desc=f"  {desc}", unit="batch"):
        batch = seqs[i : i + batch_size]
        encoded = [torch.tensor(kmer_transform(s), dtype=torch.long) for s in batch]
        padded = pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(padded)
        all_logits.append(out.float().cpu().numpy())
    return np.vstack(all_logits)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genus_exp_dir",         required=True)
    parser.add_argument("--species_exp_dir",        required=True)
    parser.add_argument("--val_dir",               required=True)
    parser.add_argument("--vocab",                 required=True)
    parser.add_argument("--genus_class_indices",   type=int, default=3)
    parser.add_argument("--species_class_indices", type=int, default=1)
    parser.add_argument("--out",                   required=True)
    parser.add_argument("--batch_size",            type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading vocabulary: {args.vocab}")
    vocab, _ = load_vocabulary(args.vocab)

    print(f"\nParsing reads from: {args.val_dir}")
    seqs, g_labels, s_labels = parse_fasta_dir(
        args.val_dir, args.genus_class_indices, args.species_class_indices
    )
    N = len(seqs)
    num_species = int(s_labels.max()) + 1
    print(f"  {N:,} reads  |  {len(np.unique(g_labels))} genera  |  {len(np.unique(s_labels))} species")

    print("\nBuilding genus→species mapping...")
    g2s = build_genus2species(g_labels, s_labels)
    print(f"  {len(g2s)} genera represented in val set")

    # ── Genus inference ───────────────────────────────────────────────────────
    print(f"\nLoading genus model: {args.genus_exp_dir}")
    genus_model, genus_transform = load_mt_model(args.genus_exp_dir, vocab, device)

    print("\nRunning genus inference...")
    genus_logits = run_inference(genus_model, seqs, genus_transform, device, args.batch_size, "Genus")
    genus_preds = genus_logits.argmax(axis=-1)
    genus_acc = (genus_preds == g_labels).mean()
    print(f"  Genus accuracy: {genus_acc:.4f}")

    del genus_model
    torch.cuda.empty_cache()

    # ── Species inference ─────────────────────────────────────────────────────
    print(f"\nLoading species model: {args.species_exp_dir}")
    species_model, species_transform = load_mt_model(args.species_exp_dir, vocab, device)

    print("\nRunning species inference...")
    species_logits = run_inference(species_model, seqs, species_transform, device, args.batch_size, "Species")

    del species_model
    torch.cuda.empty_cache()

    # ── Hierarchical masking ──────────────────────────────────────────────────
    print("\nApplying hierarchical genus mask...")
    final_preds = np.zeros(N, dtype=np.int64)
    final_probs = np.zeros((N, num_species), dtype=np.float32)

    fallback_count = 0
    for i in range(N):
        pg = int(genus_preds[i])
        valid = g2s.get(pg)
        if valid is None or len(valid) == 0:
            # Genus not seen in val set — fall back to flat argmax
            final_preds[i] = int(species_logits[i].argmax())
            fallback_count += 1
            continue
        masked = species_logits[i][valid]
        best_local = int(masked.argmax())
        final_preds[i] = int(valid[best_local])
        # Store masked softmax probs for the valid species
        masked_probs = np.exp(masked - masked.max())
        masked_probs /= masked_probs.sum()
        final_probs[i, valid] = masked_probs

    if fallback_count > 0:
        print(f"  Fallback (genus not in val mapping): {fallback_count:,} reads")

    sp_acc = (final_preds == s_labels).mean()
    print(f"  Species top-1 (hierarchical): {sp_acc:.4f}")
    print(f"  Species top-1 (flat baseline): {(species_logits.argmax(-1) == s_labels).mean():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, preds=final_preds, probs=final_probs, labels=s_labels)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
