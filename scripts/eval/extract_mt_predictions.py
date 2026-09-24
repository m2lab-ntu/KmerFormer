#!/usr/bin/env python3
# This script runs from inside a MetaTransformer checkout, so it must import
# MetaTransformer's own `utils` package. Deprioritize this script's directory so
# that package wins over anything shadowing the name here.
import sys as _sys, os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
if _script_dir in _sys.path:
    _sys.path.remove(_script_dir)
    _sys.path.append(_script_dir)
del _sys, _os

"""
Extract MT model predictions on a FASTA directory, save preds+probs+labels as npz.

Usage (Exp D — MT species flat):
    PYTHONPATH=/path/to/MetaTransformer/src \
    python extract_mt_predictions.py \
        --exp_dir    $SPECIES_EXP \
        --val_dir    $KF_DATA/balanced_50M/metatransformer_format/val \
        --vocab      vocab_file/vocab_13mer.txt \
        --out        mt_species_flat_preds.npz \
        --class_indices 1 \
        --batch_size 1024

    class_indices: field index in FASTA header after splitting on "|"
        1 → species class,  3 → genus class
"""

import argparse
import sys
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

PAD_IDX = 0   # SPECIAL_TOKENS_2_INDEX['<pad>']


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta_dir(val_dir, class_index):
    """Yield all (seq, label) pairs from *.fa files in val_dir."""
    seqs, labels = [], []
    for fa_path in sorted(Path(val_dir).glob("*.fa")):
        header, parts = None, []
        with open(fa_path) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is not None:
                        fields = header.split("|")
                        labels.append(int(fields[class_index]))
                        seqs.append("".join(parts))
                    header, parts = line[1:], []
                else:
                    parts.append(line)
        if header is not None:
            fields = header.split("|")
            labels.append(int(fields[class_index]))
            seqs.append("".join(parts))
    return seqs, np.array(labels, dtype=np.int64)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_mt_model(exp_dir, vocab, device):
    exp_dir = Path(exp_dir)
    cfg = OmegaConf.load(exp_dir / "config.yaml")

    # init_device_handler must be called before ClassificationTransformer.__init__
    # which calls DeviceHandler.get_device() internally.
    from utils.device_handler import init_device_handler
    dev = cfg.get("device_settings", {})
    init_device_handler(
        use_cpu=dev.get("use_cpu", False),
        gpu_count=dev.get("gpu_count", 1),
        gpu_ids=dev.get("gpu_ids", None),
        split_gpus=dev.get("split_gpus", False),
    )

    vocab_size = len(vocab)
    net = instantiate_model_by_str_name(cfg.model.name, cfg, vocab_size)
    ckpt_path = exp_dir / "checkpoints" / "classification_transformer_ckpt_best.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Reconcile vocab-size mismatches (off-by-one between training vocab and
    # current vocab file): truncate or zero-pad the checkpoint embedding row.
    ckpt_sd = state["model_state_dict"]
    curr_sd = net.state_dict()
    for key in list(ckpt_sd.keys()):
        if key in curr_sd and ckpt_sd[key].shape != curr_sd[key].shape:
            c_shape, k_shape = curr_sd[key].shape, ckpt_sd[key].shape
            if c_shape[0] != k_shape[0]:
                if k_shape[0] > c_shape[0]:
                    ckpt_sd[key] = ckpt_sd[key][:c_shape[0]]
                else:
                    pad = torch.zeros(c_shape[0] - k_shape[0], *k_shape[1:])
                    ckpt_sd[key] = torch.cat([ckpt_sd[key], pad], dim=0)
    net.load_state_dict(ckpt_sd, strict=False)
    net.eval().to(device)
    transforms = read_transforms_for_input_layer(
        cfg.mdl_common.input_module, cfg, vocab, train=False
    )
    kmer_transform = transforms[0]  # Read2VocabKmer (TrimRead stripped in train=False)
    print(f"  Loaded: {ckpt_path.name}  num_classes={cfg.mdl_common.num_classes}")
    return net, kmer_transform


# ── Batched inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, seqs, kmer_transform, device, batch_size):
    use_amp = torch.cuda.is_available()
    all_logits = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="  Inference", unit="batch"):
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
    parser.add_argument("--exp_dir",       required=True, help="MT experiment dir (contains config.yaml + checkpoints/)")
    parser.add_argument("--val_dir",       required=True, help="Directory of .fa files to evaluate")
    parser.add_argument("--vocab",         required=True, help="vocab_13mer.txt path")
    parser.add_argument("--out",           required=True, help="Output .npz path")
    parser.add_argument("--class_indices", type=int, default=1, help="FASTA header field index for label (1=species, 3=genus)")
    parser.add_argument("--batch_size",    type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading vocabulary: {args.vocab}")
    vocab, _ = load_vocabulary(args.vocab)

    print(f"\nLoading model from: {args.exp_dir}")
    model, kmer_transform = load_mt_model(args.exp_dir, vocab, device)

    print(f"\nParsing reads from: {args.val_dir}  (class_indices={args.class_indices})")
    seqs, labels = parse_fasta_dir(args.val_dir, args.class_indices)
    print(f"  {len(seqs):,} reads  |  {len(np.unique(labels))} unique classes")

    print(f"\nRunning inference (batch_size={args.batch_size})...")
    logits = run_inference(model, seqs, kmer_transform, device, args.batch_size)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy().astype(np.float32)
    preds = logits.argmax(axis=-1).astype(np.int64)

    acc = (preds == labels).mean()
    print(f"  Top-1 accuracy: {acc:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, preds=preds, probs=probs, labels=labels)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
