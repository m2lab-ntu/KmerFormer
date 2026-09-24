#!/usr/bin/env python3
"""
Evaluation script for Token-level GFM Classifier.

Usage:
    python scripts/evaluate.py --config configs/nt_token_genus.yaml
    python scripts/evaluate.py --config configs/nt_token_genus.yaml --rc_tta
    python scripts/evaluate.py --config configs/species.yaml --oracle_genus
    python scripts/evaluate.py --config configs/species.yaml --topk_genus_routing 5 \\
        --genus_config configs/genus.yaml --genus_checkpoint results/genus/best.pt
    python scripts/evaluate.py --config configs/species.yaml --soft_genus_routing \\
        --genus_config configs/genus.yaml --genus_checkpoint results/genus/best.pt
    # Run all three modes together (genus model loaded only once):
    python scripts/evaluate.py --config configs/species.yaml \\
        --oracle_genus --topk_genus_routing 5 --soft_genus_routing \\
        --genus_config configs/genus.yaml --genus_checkpoint results/genus/best.pt

Produces:
  - eval_metrics.json / eval_metrics_rc_tta.json
  - classification_report.csv
  - confusion_matrix.npy / .png
  - top_confusions.csv

Oracle-genus mode (--oracle_genus):
  Uses TRUE genus labels to mask species logits, answering:
  "If genus were perfect, how well can we distinguish species within a genus?"

Top-k genus routing (--topk_genus_routing K):
  Uses a TRAINED genus model to predict top-K genera, then masks species logits
  to only the union of species belonging to those genera.
"""

import argparse
import json
import os
import sys
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd
# seaborn is only used for one confusion-matrix heatmap and is not installed in
# every env (missing from the Taiwania-2 `gfm` env). Import it optionally and
# fall back to matplotlib so evaluation never dies for a plotting dependency.
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import load_data, load_test_data
from .dataset import TokenLevelReadDataset
from .model import TokenLevelGFMClassifier, create_model
from .kmer_tokenizers import build_tokenizer
from .resource_monitor import ResourceMonitor
from .training_utils import checkpoint_weights
from .utils import load_config, save_json
from .splits import split_options


def evaluate_model(model, loader, device, use_amp, num_classes, desc="Evaluating"):
    """Run inference on loader, return preds, probs, labels, logits, timing."""
    model.eval()
    use_amp = use_amp and device.type == "cuda"
    all_logits_list, all_labels = [], []
    total_time, total_reads = 0, 0

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc=desc):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(input_ids, attention_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            total_reads += labels.size(0)

            all_logits_list.append(logits.float().cpu().numpy())
            all_labels.extend(labels.numpy())

    if not all_logits_list:
        raise ValueError("Evaluation dataset is empty")
    all_logits = np.vstack(all_logits_list)
    all_labels = np.array(all_labels)

    # Derive probs and preds from logits
    all_probs = scipy_softmax(all_logits, axis=-1)
    all_preds = all_logits.argmax(axis=-1)

    return all_preds, all_probs, all_labels, all_logits, total_time, total_reads


def resolve_num_classes(labels, detected_classes, override=None):
    """Validate class IDs against the model head without renumbering labels."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("Evaluation dataset is empty")
    num_classes = detected_classes if override is None else override
    if num_classes < 1 or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Label IDs {labels.min()}..{labels.max()} do not fit a {num_classes}-class head; "
            "set --num_classes to the number of classes used during training")
    return num_classes


def compute_metrics(all_labels, all_preds, all_probs, num_classes):
    """Compute comprehensive metrics dict."""
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec_w = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec_w = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    prec_m = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_m = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Top-k accuracy
    k_values = [3, 5, 10]
    topk = {}
    for k in k_values:
        if k < num_classes:
            topk[f"top{k}_accuracy"] = float(
                top_k_accuracy_score(all_labels, all_probs, k=k, labels=range(num_classes))
            )

    # Count classes with F1 = 0
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0,
                            labels=range(num_classes))
    # Only count classes that actually appear in labels
    present_classes = np.unique(all_labels)
    n_f1_zero = int(sum(1 for c in present_classes if per_class_f1[c] == 0.0))

    metrics = {
        "micro_accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "num_classes": int(num_classes),
        "num_classes_present": int(len(present_classes)),
        "num_classes_f1_zero": n_f1_zero,
        **topk,
    }
    return metrics


def save_classification_report(all_labels, all_preds, id2label, num_classes, output_dir):
    """Save per-class classification report as CSV."""
    unique = np.unique(np.concatenate([all_labels, all_preds]))
    names = [id2label.get(int(i), str(i)) for i in unique]
    report = classification_report(
        all_labels, all_preds,
        labels=unique, target_names=names,
        output_dict=True, zero_division=0,
    )
    df = pd.DataFrame(report).T
    path = os.path.join(output_dir, "classification_report.csv")
    df.to_csv(path)
    print(f"  ✅ Classification report saved to {path}")
    return df


def save_confusion_matrix(all_labels, all_preds, id2label, num_classes, output_dir):
    """Save full confusion matrix as .npy and plot top-30 heatmap."""
    # Full confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    npy_path = os.path.join(output_dir, "confusion_matrix.npy")
    np.save(npy_path, cm)
    print(f"  ✅ Confusion matrix ({cm.shape}) saved to {npy_path}")

    # Plot top-30 classes by frequency
    try:
        _plot_confusion_heatmap(
            all_labels, all_preds, id2label,
            os.path.join(output_dir, "confusion_matrix.png"),
            top_n=30,
        )
    except Exception as e:
        print(f"  ⚠️  Could not plot confusion matrix: {e}")

    return cm


def _plot_confusion_heatmap(y_true, y_pred, id_to_label, save_path, top_n=30):
    """Plot normalized confusion matrix heatmap for top-N classes."""
    if plt is None:
        print("  Confusion-matrix image skipped: install kmerformer[plots] to enable plotting.")
        return
    unique, counts = np.unique(y_true, return_counts=True)
    top_classes = unique[np.argsort(counts)[::-1]][:top_n]

    mask = np.isin(y_true, top_classes)
    yt = y_true[mask]
    yp = y_pred[mask]

    cmap = {c: i for i, c in enumerate(top_classes)}
    yt_r = np.array([cmap[y] for y in yt])
    yp_r = np.array([cmap.get(y, -1) for y in yp])

    valid = yp_r >= 0
    cm = confusion_matrix(yt_r[valid], yp_r[valid], labels=range(len(top_classes)))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    labels = [id_to_label.get(int(c), str(c))[:20] for c in top_classes]

    fig, ax = plt.subplots(figsize=(18, 14))
    if sns is not None:
        sns.heatmap(cm_norm, annot=False, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (Top {top_n} Classes, Row-Normalized)", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix plot saved to {save_path}")


def save_top_confusions(cm, id2label, num_classes, output_dir, top_n=50):
    """
    Extract top confused pairs from off-diagonal of confusion matrix.
    Saves both raw count and normalized (by true class support) versions.
    """
    rows = []
    for true_cls in range(num_classes):
        row_total = cm[true_cls].sum()
        if row_total == 0:
            continue
        for pred_cls in range(num_classes):
            if true_cls == pred_cls:
                continue
            count = int(cm[true_cls, pred_cls])
            if count == 0:
                continue
            rows.append({
                "true_id": true_cls,
                "true_name": id2label.get(true_cls, str(true_cls)),
                "pred_id": pred_cls,
                "pred_name": id2label.get(pred_cls, str(pred_cls)),
                "count": count,
                "pct_of_true_class": round(100.0 * count / row_total, 2),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  ⚠️  No off-diagonal confusions found")
        return df

    # Sort by count descending
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    path = os.path.join(output_dir, "top_confusions.csv")
    df.head(top_n * 5).to_csv(path, index=False)  # save more for later analysis
    print(f"  ✅ Top confusions saved to {path} ({len(df)} pairs total, top {min(len(df), top_n*5)} saved)")

    # Print top-N
    print(f"\n  Top-{top_n} confused pairs:")
    print(f"  {'True':<22} {'Predicted':<22} {'Count':>6} {'%True':>7}")
    print(f"  {'-'*22} {'-'*22} {'-'*6} {'-'*7}")
    for _, r in df.head(top_n).iterrows():
        print(f"  {r['true_name']:<22} {r['pred_name']:<22} {r['count']:>6} {r['pct_of_true_class']:>6.1f}%")

    return df


def mask_logits_by_genus(logits, genus_labels, genus_to_species, num_species):
    """Mask species logits to only allow species within the given genus.

    Args:
        logits: [N, num_species] raw logits
        genus_labels: [N] integer genus labels (one per sample)
        genus_to_species: dict mapping genus_id -> list of species_ids
        num_species: total number of species classes

    Returns:
        masked_logits: [N, num_species] with -inf for invalid species
    """
    masked = np.full_like(logits, -1e9)
    for i, g in enumerate(genus_labels):
        valid_sids = genus_to_species.get(int(g), [])
        if valid_sids:
            masked[i, valid_sids] = logits[i, valid_sids]
    return masked


def topk_genus_routing(species_logits, genus_logits, genus_to_species,
                       num_species, k=5):
    """Mask species logits by top-k predicted genera.

    Args:
        species_logits: [N, num_species]
        genus_logits: [N, num_genera]
        genus_to_species: dict genus_id -> list of species_ids
        num_species: total species count
        k: number of top genera to consider

    Returns:
        masked_logits: [N, num_species]
        topk_genera: [N, k] top-k genus predictions
    """
    topk_genera = np.argsort(genus_logits, axis=-1)[:, -k:][:, ::-1]  # [N, k]
    masked = np.full_like(species_logits, -1e9)
    for i in range(len(species_logits)):
        valid_sids = set()
        for g in topk_genera[i]:
            valid_sids.update(genus_to_species.get(int(g), []))
        valid_sids = list(valid_sids)
        if valid_sids:
            masked[i, valid_sids] = species_logits[i, valid_sids]
    return masked, topk_genera


def soft_genus_routing(species_logits, genus_logits, species_to_genus,
                       genus_to_species, num_species):
    """Hierarchical scoring: Pr(genus) x P(species | genus).

    For each species s:
        score(s) = Pr(genus(s)) * P(s | genus(s))

    where P(s | genus(s)) is the softmax of species_logits restricted to
    species within the same genus as s (renormalized within-genus softmax).

    Args:
        species_logits: [N, num_species] raw logits from species model
        genus_logits:   [N, num_genera]  raw logits from genus model
        species_to_genus: dict species_id -> genus_id
        genus_to_species: dict genus_id -> list of species_ids
        num_species: total number of species classes

    Returns:
        scores: [N, num_species] hierarchical scores (valid probability dist
                within each genus, weighted by genus probability)
    """
    N = species_logits.shape[0]

    # Genus probabilities: [N, num_genera]
    genus_probs = scipy_softmax(genus_logits, axis=-1)

    # Precompute species -> genus index array
    s2g = np.array([species_to_genus.get(s, 0) for s in range(num_species)],
                   dtype=np.int64)  # [num_species]

    # Within-genus conditional probabilities P(s | genus(s))
    # For each genus group, run softmax over its species slice
    cond_probs = np.zeros((N, num_species), dtype=np.float64)
    for g_id, sids in genus_to_species.items():
        if not sids:
            continue
        sids_arr = np.array(sids, dtype=np.int64)
        # [N, |genus|]  softmax within this genus
        cond_probs[:, sids_arr] = scipy_softmax(
            species_logits[:, sids_arr].astype(np.float64), axis=-1
        )

    # Genus weight for each species: Pr(genus(s))  ->  [N, num_species]
    genus_weight = genus_probs[:, s2g]

    # Joint score
    scores = genus_weight * cond_probs  # [N, num_species]
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a KmerFormer or baseline checkpoint. Pass --rc_tta "
                    "to match the protocol every number in docs/RESULTS.md uses.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: <output_dir>/best.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: same as config output dir)")
    parser.add_argument("--benchmark_speed", action="store_true",
                        help="Include speed benchmark in output")
    parser.add_argument("--rc_tta", action="store_true",
                        help="Reverse-complement test-time augmentation")
    # Oracle-genus mode
    parser.add_argument("--oracle_genus", action="store_true",
                        help="Mask species logits by TRUE genus label (upper bound)")
    # Top-k genus routing mode
    parser.add_argument("--topk_genus_routing", type=int, default=0,
                        help="Use top-K predicted genera to mask species logits (0=off)")
    parser.add_argument("--soft_genus_routing", action="store_true",
                        help="Pr(genus) x P(species|genus) soft hierarchical scoring")
    parser.add_argument("--genus_config", type=str, default=None,
                        help="Genus model config (for topk/soft routing)")
    parser.add_argument("--genus_checkpoint", type=str, default=None,
                        help="Genus model checkpoint (for topk/soft routing)")
    parser.add_argument("--skip_save_logits", action="store_true",
                        help="Stream predictions and labels to disk without retaining full-dataset logits/probabilities")
    parser.add_argument("--test_fasta", type=str, default=None,
                        help="Independent test FASTA (bypasses train/val split in config)")
    parser.add_argument("--test_labels", type=str, default=None,
                        help="Labels TSV for --test_fasta (required when --test_fasta is set)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override number of output classes (e.g. 120 to match a "
                             "checkpoint's head when the test set covers fewer genera). "
                             "Without it, class count is inferred from --test_labels, which "
                             "breaks load_state_dict if the test set has fewer classes.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]; auto-detected if omitted")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--legacy-eager", action="store_true", help="Use the historical in-memory evaluator")
    args = parser.parse_args()

    if args.test_fasta and not args.test_labels:
        parser.error("--test_labels is required when --test_fasta is set")

    cfg = load_config(args.config)
    if args.test_fasta and not args.legacy_eager and not (args.oracle_genus or args.topk_genus_routing or args.soft_genus_routing):
        from .stream_evaluation import evaluate_stream
        evaluate_stream(args, cfg)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_output_dir = cfg["output"]["dir"]
    output_dir = args.output_dir or train_output_dir
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(train_output_dir, "best.pt")

    # ===== Resource monitoring =====
    monitor = ResourceMonitor(device=str(device), sample_interval=15.0)
    monitor.start()

    print("=" * 70)
    print("KmerFormer — Evaluation")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print()

    # ===== Load data =====
    task = cfg["data"]["task"]
    if args.test_fasta:
        print(f"[Test mode] Using independent test set: {args.test_fasta}")
        data = load_test_data(args.test_fasta, args.test_labels, task=task)
    else:
        data = load_data(
            cfg["data"]["fasta_path"],
            cfg["data"]["labels_path"],
            cfg["data"].get("val_ratio", 0.1),
            cfg["data"].get("seed", 42),
            task=task, **split_options(cfg),
        )

    if task == "genus":
        val_labels = data["val_genus_labels"]
        num_classes = data["num_genera"]
        id2label = data["id2genus"]
    else:
        val_labels = data["val_species_labels"]
        num_classes = data["num_species"]
        id2label = data["id2species"]

    resolved_classes = resolve_num_classes(val_labels, num_classes, args.num_classes)
    if resolved_classes != num_classes:
        print(f"  [override] num_classes {num_classes} -> {resolved_classes} ({task} head)")
    num_classes = resolved_classes

    backbone_name = cfg["model"].get("backbone", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    trust_remote_code = cfg["model"].get("trust_remote_code", True)
    tokenizer = build_tokenizer(cfg)
    print(f"Tokenizer: {type(tokenizer).__name__}  vocab_size={tokenizer.vocab_size:,}")
    if cfg["model"].get("vocab_size") is not None:
        assert cfg["model"]["vocab_size"] == tokenizer.vocab_size, (
            f"config model.vocab_size={cfg['model']['vocab_size']} != "
            f"tokenizer.vocab_size={tokenizer.vocab_size}")

    max_token_length = cfg["data"].get("max_token_length", 128)
    kmer_preprocess = cfg["data"].get("kmer_preprocess", None)
    val_dataset = TokenLevelReadDataset(
        data["val_sequences"].tolist(),
        val_labels,
        tokenizer=tokenizer,
        max_length=max_token_length,
        rc_augment=False,
        kmer_preprocess=kmer_preprocess,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"].get("eval_batch_size",
                                        cfg["training"]["batch_size"] * 2),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\nTask: {task}, Classes: {num_classes}")
    print(f"Val samples: {len(val_labels):,}")
    print(f"max_token_length: {max_token_length}")
    print()

    # ===== Load model =====
    # Must construct model identically to training so state_dict keys match.
    model_cfg = cfg["model"]
    eval_model_cfg = dict(model_cfg)
    eval_model_cfg["gradient_checkpointing"] = False  # not needed for eval
    model = create_model(eval_model_cfg, num_classes).to(device)

    # map_location="cpu", not device: a last.pt for a long-k arm carries the
    # SparseAdam moments as well as the weights (25.7 GiB against 8 GiB for the
    # model alone), and loading all of that straight onto the card OOMs before
    # load_state_dict has a chance to copy only what it needs. Staging on host
    # memory and letting load_state_dict do the transfer halves peak GPU use and
    # makes a mid-run checkpoint scoreable beside a training job.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_weights(model, ckpt["model_state_dict"]))
    del ckpt["model_state_dict"]
    model.eval()
    print(f"\n✅ Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    sys.stdout.flush()

    # ===== Run inference (forward) =====
    use_amp = cfg["training"].get("amp", True)
    fwd_preds, fwd_probs, all_labels, fwd_logits, fwd_time, total_reads = evaluate_model(
        model, val_loader, device, use_amp, num_classes, desc="Eval (fwd)",
    )

    # ===== RC TTA =====
    if args.rc_tta:
        print("\n--- RC Test-Time Augmentation ---")
        rc_sequences = [TokenLevelReadDataset.reverse_complement(s)
                        for s in data["val_sequences"].tolist()]
        rc_dataset = TokenLevelReadDataset(
            rc_sequences, val_labels, tokenizer=tokenizer,
            max_length=max_token_length, rc_augment=False,
            kmer_preprocess=kmer_preprocess,
        )
        rc_loader = DataLoader(
            rc_dataset,
            batch_size=cfg["training"].get("eval_batch_size",
                                            cfg["training"]["batch_size"] * 2),
            shuffle=False, num_workers=4, pin_memory=True,
        )
        _, _, _, rc_logits, rc_time, _ = evaluate_model(
            model, rc_loader, device, use_amp, num_classes, desc="Eval (RC)",
        )

        # Average logits
        avg_logits = (fwd_logits + rc_logits) / 2.0
        avg_probs = scipy_softmax(avg_logits, axis=-1)
        avg_preds = avg_logits.argmax(axis=-1)
        total_time = fwd_time + rc_time

        # Use TTA results as primary
        all_preds, all_probs, all_logits = avg_preds, avg_probs, avg_logits

        # Also compute forward-only metrics for comparison
        fwd_metrics = compute_metrics(all_labels, fwd_preds, fwd_probs, num_classes)
        print(f"  Forward-only → acc={fwd_metrics['micro_accuracy']:.4f}  "
              f"macro_F1={fwd_metrics['f1_macro']:.4f}")
    else:
        all_preds, all_probs, all_logits = fwd_preds, fwd_probs, fwd_logits
        total_time = fwd_time
        fwd_metrics = None

    # ===== Oracle-genus masking (species task only) =====
    oracle_metrics = None
    if args.oracle_genus and task == "species":
        print("\n--- Oracle-Genus Species Evaluation ---")
        genus_to_species = data["genus_to_species"]
        val_genus_labels = data["val_genus_labels"]
        masked_logits = mask_logits_by_genus(
            all_logits, val_genus_labels, genus_to_species, num_classes
        )
        masked_probs = scipy_softmax(masked_logits, axis=-1)
        masked_preds = masked_logits.argmax(axis=-1)
        oracle_metrics = compute_metrics(all_labels, masked_preds, masked_probs, num_classes)
        oracle_metrics["mode"] = "oracle_genus"
        print(f"  Oracle-genus species acc:  {oracle_metrics['micro_accuracy']:.4f}")
        print(f"  Oracle-genus macro F1:     {oracle_metrics['f1_macro']:.4f}")
        for k in [3, 5, 10]:
            key = f"top{k}_accuracy"
            if key in oracle_metrics:
                print(f"  Oracle-genus top-{k}:       {oracle_metrics[key]:.4f}")
        print(f"  Classes with F1=0:         {oracle_metrics['num_classes_f1_zero']}")

    # ===== Top-k genus routing (species task only) =====
    routing_metrics = None
    if args.topk_genus_routing > 0 and task == "species":
        k = args.topk_genus_routing
        print(f"\n--- Top-{k} Genus Routing Evaluation ---")
        assert args.genus_config and args.genus_checkpoint, \
            "--genus_config and --genus_checkpoint required for topk routing"

        genus_cfg = load_config(args.genus_config)
        genus_data = load_data(
            genus_cfg["data"]["fasta_path"],
            genus_cfg["data"]["labels_path"],
            genus_cfg["data"].get("val_ratio", 0.1),
            genus_cfg["data"].get("seed", 42),
            task="genus",
        )
        genus_num_classes = genus_data["num_genera"]

        genus_model_cfg = genus_cfg["model"]
        eval_genus_cfg = dict(genus_model_cfg)
        eval_genus_cfg["gradient_checkpointing"] = False
        genus_model = create_model(eval_genus_cfg, genus_num_classes).to(device)

        genus_ckpt = torch.load(args.genus_checkpoint, map_location=device,
                                weights_only=False)
        genus_model.load_state_dict(genus_ckpt["model_state_dict"])
        genus_model.eval()
        print(f"  Genus model loaded (epoch {genus_ckpt.get('epoch', '?')})")

        # Run genus inference on the SAME val sequences
        genus_max_len = genus_cfg["data"].get("max_token_length", 32)
        genus_kmer_preprocess = genus_cfg["data"].get("kmer_preprocess", None)
        genus_val_dataset = TokenLevelReadDataset(
            data["val_sequences"].tolist(),
            data["val_genus_labels"],
            tokenizer=tokenizer,
            max_length=genus_max_len,
            rc_augment=False,
            kmer_preprocess=genus_kmer_preprocess,
        )
        genus_val_loader = DataLoader(
            genus_val_dataset,
            batch_size=cfg["training"].get("eval_batch_size", 64),
            shuffle=False, num_workers=4, pin_memory=True,
        )

        genus_preds, genus_probs, genus_true, genus_logits, _, _ = evaluate_model(
            genus_model, genus_val_loader, device, use_amp,
            genus_num_classes, desc="Eval (genus)",
        )
        del genus_model
        torch.cuda.empty_cache()

        # Route species by top-k genera
        genus_to_species = data["genus_to_species"]
        masked_logits, topk_genera = topk_genus_routing(
            all_logits, genus_logits, genus_to_species, num_classes, k=k,
        )
        masked_probs = scipy_softmax(masked_logits, axis=-1)
        masked_preds = masked_logits.argmax(axis=-1)
        routing_metrics = compute_metrics(all_labels, masked_preds, masked_probs,
                                          num_classes)

        # Genus accuracy
        genus_acc = float(accuracy_score(genus_true, genus_preds))
        # Conditional species accuracy
        species_to_genus = data["species_to_genus"]
        true_genus_from_species = np.array(
            [species_to_genus.get(int(s), -1) for s in all_labels]
        )
        genus_correct_mask = (genus_preds == true_genus_from_species)
        genus_wrong_mask = ~genus_correct_mask

        sp_acc_genus_correct = float(accuracy_score(
            all_labels[genus_correct_mask], masked_preds[genus_correct_mask]
        )) if genus_correct_mask.sum() > 0 else 0.0
        sp_acc_genus_wrong = float(accuracy_score(
            all_labels[genus_wrong_mask], masked_preds[genus_wrong_mask]
        )) if genus_wrong_mask.sum() > 0 else 0.0

        routing_metrics["mode"] = f"topk_genus_routing_k{k}"
        routing_metrics["genus_accuracy"] = genus_acc
        routing_metrics["species_acc_genus_correct"] = sp_acc_genus_correct
        routing_metrics["species_acc_genus_wrong"] = sp_acc_genus_wrong
        routing_metrics["genus_correct_count"] = int(genus_correct_mask.sum())
        routing_metrics["genus_wrong_count"] = int(genus_wrong_mask.sum())
        routing_metrics["k"] = k

        print(f"  Genus accuracy (top-1):            {genus_acc:.4f}")
        print(f"  Species accuracy (routed):         {routing_metrics['micro_accuracy']:.4f}")
        print(f"  Species acc | genus correct:       {sp_acc_genus_correct:.4f}")
        print(f"  Species acc | genus wrong:          {sp_acc_genus_wrong:.4f}")
        print(f"  Macro F1 (routed):                 {routing_metrics['f1_macro']:.4f}")
        for kk in [3, 5, 10]:
            key = f"top{kk}_accuracy"
            if key in routing_metrics:
                print(f"  Top-{kk} (routed):                  {routing_metrics[key]:.4f}")

    # ===== Soft genus routing (species task only) =====
    # Pr(genus) x P(species | genus)  — runs genus model if not already loaded above
    soft_metrics = None
    if args.soft_genus_routing and task == "species":
        print("\n--- Soft Genus Routing Evaluation ---")
        assert args.genus_config and args.genus_checkpoint, \
            "--genus_config and --genus_checkpoint required for soft routing"

        # Load genus model only if topk routing hasn't already done it
        _genus_logits_available = (args.topk_genus_routing > 0)
        if not _genus_logits_available:
            genus_cfg = load_config(args.genus_config)
            genus_data = load_data(
                genus_cfg["data"]["fasta_path"],
                genus_cfg["data"]["labels_path"],
                genus_cfg["data"].get("val_ratio", 0.1),
                genus_cfg["data"].get("seed", 42),
                task="genus",
            )
            genus_num_classes = genus_data["num_genera"]

            _genus_model_cfg = genus_cfg["model"]
            _eval_genus_cfg = dict(_genus_model_cfg)
            _eval_genus_cfg["gradient_checkpointing"] = False
            genus_model = create_model(_eval_genus_cfg, genus_num_classes).to(device)

            genus_ckpt = torch.load(args.genus_checkpoint, map_location=device,
                                    weights_only=False)
            genus_model.load_state_dict(genus_ckpt["model_state_dict"])
            genus_model.eval()
            print(f"  Genus model loaded (epoch {genus_ckpt.get('epoch', '?')})")

            genus_max_len = genus_cfg["data"].get("max_token_length", 32)
            genus_kmer_preprocess = genus_cfg["data"].get("kmer_preprocess", None)
            genus_val_dataset = TokenLevelReadDataset(
                data["val_sequences"].tolist(),
                data["val_genus_labels"],
                tokenizer=tokenizer,
                max_length=genus_max_len,
                rc_augment=False,
                kmer_preprocess=genus_kmer_preprocess,
            )
            genus_val_loader = DataLoader(
                genus_val_dataset,
                batch_size=cfg["training"].get("eval_batch_size", 64),
                shuffle=False, num_workers=4, pin_memory=True,
            )
            genus_preds, genus_probs, genus_true, genus_logits, _, _ = evaluate_model(
                genus_model, genus_val_loader, device, use_amp,
                genus_num_classes, desc="Eval (genus)",
            )
            del genus_model
            torch.cuda.empty_cache()
        else:
            print("  Reusing genus logits from topk routing pass")

        # Compute soft hierarchical scores
        soft_scores = soft_genus_routing(
            all_logits, genus_logits,
            data["species_to_genus"], data["genus_to_species"],
            num_species=num_classes,
        )
        soft_preds = soft_scores.argmax(axis=-1)
        soft_metrics = compute_metrics(all_labels, soft_preds, soft_scores, num_classes)

        # Conditional accuracy breakdown
        _s2g = data["species_to_genus"]
        _true_g = np.array([_s2g.get(int(s), -1) for s in all_labels])
        _genus_ok = genus_preds == _true_g
        _sp_acc_ok = float(accuracy_score(
            all_labels[_genus_ok], soft_preds[_genus_ok]
        )) if _genus_ok.sum() > 0 else 0.0
        _sp_acc_ng = float(accuracy_score(
            all_labels[~_genus_ok], soft_preds[~_genus_ok]
        )) if (~_genus_ok).sum() > 0 else 0.0

        soft_metrics["mode"] = "soft_genus_routing"
        soft_metrics["genus_accuracy"] = float(accuracy_score(genus_true, genus_preds))
        soft_metrics["species_acc_genus_correct"] = _sp_acc_ok
        soft_metrics["species_acc_genus_wrong"] = _sp_acc_ng

        print(f"  Genus accuracy (top-1):            {soft_metrics['genus_accuracy']:.4f}")
        print(f"  Species accuracy (soft):           {soft_metrics['micro_accuracy']:.4f}")
        print(f"  Species acc | genus correct:       {_sp_acc_ok:.4f}")
        print(f"  Species acc | genus wrong:         {_sp_acc_ng:.4f}")
        print(f"  Macro F1 (soft):                   {soft_metrics['f1_macro']:.4f}")
        for kk in [3, 5, 10]:
            key = f"top{kk}_accuracy"
            if key in soft_metrics:
                print(f"  Top-{kk} (soft):                   {soft_metrics[key]:.4f}")

    # ===== Compute standard (unmasked) metrics =====
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)

    # Speed stats
    if total_time > 0:
        metrics["reads_per_sec"] = float(total_reads / total_time)
        metrics["avg_latency_ms"] = float(total_time / total_reads * 1000)
    metrics["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    metrics["max_token_length"] = max_token_length
    metrics["task"] = task
    metrics["rc_tta"] = args.rc_tta

    # ===== Print summary =====
    mode_label = "RC TTA" if args.rc_tta else "Forward-only"
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results ({mode_label})")
    print(f"{'=' * 60}")
    print(f"  Micro Accuracy:     {metrics['micro_accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    for k in [3, 5, 10]:
        key = f"top{k}_accuracy"
        if key in metrics:
            print(f"  Top-{k} Accuracy:    {metrics[key]:.4f}")
    print(f"  Classes with F1=0:  {metrics['num_classes_f1_zero']}")
    if "reads_per_sec" in metrics:
        print(f"  Reads/sec:          {metrics['reads_per_sec']:,.0f}")
        print(f"  Avg latency:        {metrics['avg_latency_ms']:.2f} ms/read")

    if args.rc_tta and fwd_metrics is not None:
        print(f"\n  --- RC TTA improvement ---")
        for key in ["micro_accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"]:
            delta = metrics[key] - fwd_metrics[key]
            print(f"  {key}: {fwd_metrics[key]:.4f} → {metrics[key]:.4f}  ({delta:+.4f})")
        for k in [3, 5, 10]:
            key = f"top{k}_accuracy"
            if key in metrics and key in fwd_metrics:
                delta = metrics[key] - fwd_metrics[key]
                print(f"  {key}: {fwd_metrics[key]:.4f} → {metrics[key]:.4f}  ({delta:+.4f})")
    sys.stdout.flush()

    # ===== Save all outputs =====
    print(f"\n{'=' * 60}")
    print("Saving outputs...")
    print(f"{'=' * 60}")

    # 1. Metrics JSON
    suffix = "_rc_tta" if args.rc_tta else ""
    save_json(metrics, os.path.join(output_dir, f"eval_metrics{suffix}.json"))
    print(f"  ✅ eval_metrics{suffix}.json saved")

    if args.rc_tta and fwd_metrics is not None:
        save_json(fwd_metrics, os.path.join(output_dir, "eval_metrics_fwd_only.json"))
        print(f"  ✅ eval_metrics_fwd_only.json saved (for comparison)")

    # 2. Classification report
    save_classification_report(all_labels, all_preds, id2label, num_classes, output_dir)

    # 3. Confusion matrix (.npy + .png)
    cm = save_confusion_matrix(all_labels, all_preds, id2label, num_classes, output_dir)

    # 4. Top confused pairs
    save_top_confusions(cm, id2label, num_classes, output_dir, top_n=50)

    # 5. Save raw predictions for further analysis
    if args.skip_save_logits:
        save_dict = dict(preds=all_preds, labels=all_labels)
    else:
        save_dict = dict(preds=all_preds, probs=all_probs, labels=all_labels, logits=all_logits)
        if args.rc_tta:
            save_dict["fwd_logits"] = fwd_logits
            save_dict["rc_logits"] = rc_logits
    np.savez_compressed(os.path.join(output_dir, f"predictions{suffix}.npz"), **save_dict)
    print(f"  ✅ predictions{suffix}.npz saved")

    # 6. Oracle-genus metrics
    if oracle_metrics is not None:
        save_json(oracle_metrics, os.path.join(output_dir, "eval_metrics_oracle_genus.json"))
        print(f"  ✅ eval_metrics_oracle_genus.json saved")

    # 7. Top-k routing metrics
    if routing_metrics is not None:
        k = args.topk_genus_routing
        save_json(routing_metrics,
                  os.path.join(output_dir, f"eval_metrics_topk_routing_k{k}.json"))
        print(f"  ✅ eval_metrics_topk_routing_k{k}.json saved")

    # 8. Soft routing metrics
    if soft_metrics is not None:
        save_json(soft_metrics, os.path.join(output_dir, "eval_metrics_soft_routing.json"))
        print(f"  ✅ eval_metrics_soft_routing.json saved")

    # ===== Resource report =====
    monitor.stop()
    monitor.save_report(os.path.join(output_dir, "resource_report.json"))
    monitor.print_summary()

    print(f"\n{'=' * 60}")
    print(f"All evaluation results saved to {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
