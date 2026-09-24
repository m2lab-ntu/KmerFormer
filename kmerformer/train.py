#!/usr/bin/env python3
"""
Training script for Token-level GFM Classifier.

Usage:
    python -m kmerformer.train --config configs/baselines/ntv2_lora_50M.yaml

Key features:
  - Two-phase training: (1) freeze backbone, train head → (2) LoRA + head
  - Gradient checkpointing + mixed precision for memory efficiency
  - Per-epoch evaluation with comprehensive logging
  - Early stopping on validation accuracy
  - Differential learning rates for backbone vs head
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from .reproducibility import (get_cosine_schedule_with_warmup, seed_training, seed_worker,
                              capture_rng, restore_rng, write_run_record)
from .splits import split_options

from .data_loader import load_data
from .dataset import TokenLevelReadDataset
from .dataset_lazy import LazyFASTADataset
from .kmer_tokenizers import build_tokenizer
from .model import TokenLevelGFMClassifier, create_model
from .resource_monitor import ResourceMonitor
from .training_utils import checkpoint_weights, optimizer_step
from .utils import load_config, save_config, save_json, AverageMeter, EarlyStopping


def _get_amp_dtype(use_amp, cfg_dtype=None):
    """Pick the autocast dtype.

    bf16 ONLY on hardware with bf16 tensor cores (sm_80+: A100/H100/H200).
    torch.cuda.is_bf16_supported() returns True on V100 (sm_70), where bf16 is
    emulated — measured 2.74 it/s vs 11.93 it/s for fp16 on the exact same
    pipeline (4.35x slower, i.e. 267 h vs 61 h for 15 epochs). The old check
    used that function and silently picked the slow path on every V100 run.

    training.amp_dtype: "fp16" | "bf16" in the config forces a choice.
    """
    if cfg_dtype:
        d = str(cfg_dtype).lower()
        if d in ("fp16", "float16", "half"):
            return torch.float16
        if d in ("bf16", "bfloat16"):
            return torch.bfloat16
        raise ValueError(f"unknown training.amp_dtype: {cfg_dtype}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, grad_accum, max_grad_norm, use_amp, amp_dtype=None,
                log_prior=None, rc_consistency=False, lambda_kl=0.1,
                periodic_save_fn=None, periodic_save_interval=5000):
    """Train for one epoch with NaN detection, logit adjustment, and RC consistency.

    Args:
        log_prior: If not None, Tensor [num_classes] with τ·log(π_c) for logit adjustment.
        rc_consistency: If True, loader yields (fwd_ids, fwd_mask, rc_ids, rc_mask, labels).
        lambda_kl: Weight for KL divergence loss in RC consistency mode.
    """
    model.train()
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()
    all_preds, all_labels = [], []
    nan_count = 0

    # optimizer/scheduler may be a single object (all previous experiments) or a
    # list — Job A needs SparseAdam for the 33.5M-row 13-mer embedding plus
    # AdamW for everything else. A list of one behaves exactly as before.
    opts = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
    if scheduler is None:
        scheds = []
    else:
        scheds = list(scheduler) if isinstance(scheduler, (list, tuple)) else [scheduler]

    def _zero_grad():
        for _o in opts:
            _o.zero_grad()

    pending = 0
    _zero_grad()
    pbar = tqdm(loader, desc="  Train", leave=False)

    for step, batch in enumerate(pbar):
        # ---- Unpack batch depending on mode ----
        if rc_consistency:
            fwd_ids, fwd_mask, rc_ids, rc_mask, labels = batch
            fwd_ids = fwd_ids.to(device)
            fwd_mask = fwd_mask.to(device)
            rc_ids = rc_ids.to(device)
            rc_mask = rc_mask.to(device)
        else:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            if rc_consistency:
                # Forward pass for both directions
                logits_fwd = model(fwd_ids, fwd_mask)
                logits_rc = model(rc_ids, rc_mask)

                # Supervised CE on both (with optional logit adjustment)
                logits_fwd_for_loss = logits_fwd + log_prior if log_prior is not None else logits_fwd
                logits_rc_for_loss = logits_rc + log_prior if log_prior is not None else logits_rc

                loss_ce = 0.5 * (criterion(logits_fwd_for_loss, labels)
                                 + criterion(logits_rc_for_loss, labels))

                # Symmetrized KL divergence (stop-gradient on one side each)
                p_fwd = F.softmax(logits_fwd.detach().float(), dim=-1)
                p_rc = F.softmax(logits_rc.detach().float(), dim=-1)
                log_p_fwd = F.log_softmax(logits_fwd.float(), dim=-1)
                log_p_rc = F.log_softmax(logits_rc.float(), dim=-1)
                # KL(p_rc || p_fwd) pushes fwd toward rc, and vice versa
                loss_kl = 0.5 * (F.kl_div(log_p_fwd, p_rc, reduction="batchmean")
                                 + F.kl_div(log_p_rc, p_fwd, reduction="batchmean"))

                loss = (loss_ce + lambda_kl * loss_kl) / grad_accum
                logits = logits_fwd  # use fwd logits for accuracy tracking
            else:
                logits = model(input_ids, attention_mask)
                # Optional logit adjustment for loss only
                logits_for_loss = logits + log_prior if log_prior is not None else logits
                loss = criterion(logits_for_loss, labels) / grad_accum

        # ===== NaN guard: skip corrupted batches =====
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 5:
                print(f"\n  ⚠️  NaN/Inf loss at step {step}, skipping batch (count={nan_count})")
            _zero_grad()
            pending = 0
            continue

        scaler.scale(loss).backward()
        pending += 1
        if pending == grad_accum:
            if not optimizer_step(opts, scheds, scaler, model.parameters(), max_grad_norm):
                nan_count += 1
            pending = 0

        loss_meter.update(loss.item() * grad_accum, labels.size(0))
        if rc_consistency:
            kl_meter.update(loss_kl.item(), labels.size(0))
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        postfix = {"loss": f"{loss_meter.avg:.4f}"}
        if rc_consistency:
            postfix["kl"] = f"{kl_meter.avg:.4f}"
        pbar.set_postfix(postfix)

        if periodic_save_fn and step > 0 and step % periodic_save_interval == 0:
            periodic_save_fn(step)

    if pending:
        if not optimizer_step(opts, scheds, scaler, model.parameters(), max_grad_norm,
                              normalization=grad_accum / pending):
            nan_count += 1

    if not all_labels:
        raise ValueError("Training produced no finite batches; check the data and batch size")
    if nan_count > 0:
        print(f"\n  ⚠️  Total NaN/Inf events this epoch: {nan_count}")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return loss_meter.avg, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, amp_dtype=None):
    """Validate and return metrics."""
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    amp_dtype = amp_dtype or _get_amp_dtype(use_amp)
    for input_ids, attention_mask, labels in tqdm(loader, desc="  Val", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        loss_meter.update(loss.item(), labels.size(0))
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return loss_meter.avg, acc, f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Train a KmerFormer arm, or a LoRA-adapted foundation-model "
                    "baseline, from a YAML config. Single GPU; see train_ddp for "
                    "multi-GPU dense-gradient arms.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (last.pt) to resume SAME experiment")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to checkpoint to initialize weights from (transfer learning, epoch resets to 0)")
    parser.add_argument("--time_limit_sec", type=int, default=None,
                        help="Save last.pt and exit gracefully after this many seconds (for dev partition jobs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_training(cfg["training"].get("seed"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("KmerFormer — Training")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ===== Output directory =====
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(output_dir))
    write_run_record(cfg, output_dir, resume=args.resume)

    # ===== Resource monitoring =====
    monitor = ResourceMonitor(device=str(device), sample_interval=30.0)
    monitor.start()

    # ===== 1. Load data =====
    task = cfg["data"]["task"]
    # Historical configs explicitly retain their original split strategy.
    # The same manifest is consumed by single-process and DDP entry points.
    # Lazy loading changes memory use; split_strategy selects membership.
    use_lazy = bool(cfg["data"].get("lazy", False))
    data = None
    if use_lazy:
        print("\n[lazy] streaming dataset via FASTA byte-offset index")
    else:
        data = load_data(
            cfg["data"]["fasta_path"],
            cfg["data"]["labels_path"],
            cfg["data"].get("val_ratio", 0.1),
            cfg["data"].get("seed", 42),
            task=task, **split_options(cfg),
        )

    if use_lazy:
        pass  # labels/num_classes are taken from the datasets below
    elif task == "genus":
        train_labels = data["train_genus_labels"]
        val_labels = data["val_genus_labels"]
        num_classes = data["num_genera"]
    elif task == "species":
        train_labels = data["train_species_labels"]
        val_labels = data["val_species_labels"]
        num_classes = data["num_species"]
    else:
        raise ValueError(f"Unknown task: {task}")

    if not use_lazy:
        # In the lazy path these are only known after the datasets are built,
        # which needs the tokenizer — printed there instead.
        print(f"\nTask: {task}, Classes: {num_classes}")
        print(f"Train: {len(train_labels):,}, Val: {len(val_labels):,}")

    # ===== 2. Load tokenizer =====
    backbone_name = cfg["model"].get("backbone", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    max_token_length = cfg["data"].get("max_token_length", 128)
    trust_remote_code = cfg["model"].get("trust_remote_code", True)

    tokenizer = build_tokenizer(cfg)
    print(f"\nTokenizer: {type(tokenizer).__name__}  vocab_size={tokenizer.vocab_size:,}")
    if cfg["model"].get("vocab_size") is not None:
        assert cfg["model"]["vocab_size"] == tokenizer.vocab_size, (
            f"config model.vocab_size={cfg['model']['vocab_size']} != "
            f"tokenizer.vocab_size={tokenizer.vocab_size}")

    # ===== 3. Create datasets =====
    rc_augment = cfg["data"].get("rc_augment", True)
    rc_consistency = cfg["data"].get("rc_consistency", False)
    if rc_consistency:
        print(f"\n  RC consistency mode enabled — rc_augment is superseded")
        rc_augment = False  # rc_consistency supersedes rc_augment

    # Optional k-mer preprocessing (required for DNABERT overlapping tokenization).
    # e.g. {"k": 6, "stride": 1} → split seq into overlapping 6-mers joined by spaces.
    kmer_preprocess = cfg["data"].get("kmer_preprocess", None)
    if kmer_preprocess is not None:
        print(f"  kmer_preprocess: k={kmer_preprocess['k']} stride={kmer_preprocess['stride']}")

    if use_lazy:
        _lz = dict(fasta_path=cfg["data"]["fasta_path"],
                   labels_path=cfg["data"]["labels_path"],
                   tokenizer=tokenizer, max_length=max_token_length,
                   val_ratio=cfg["data"].get("val_ratio", 0.1),
                   seed=cfg["data"].get("seed", 42),
                   kmer_preprocess=kmer_preprocess, task=task, **split_options(cfg))
        train_dataset = LazyFASTADataset(split="train", rc_augment=rc_augment, **_lz)
        val_dataset   = LazyFASTADataset(split="val",   rc_augment=False,      **_lz)
        num_classes  = train_dataset.num_genera if task == "genus" else train_dataset.num_species
        train_labels = train_dataset.get_labels()
        val_labels   = val_dataset.get_labels()
        print(f"\nTask: {task}, Classes: {num_classes}")
        print(f"Train: {len(train_labels):,}, Val: {len(val_labels):,}")
    else:
      train_dataset = TokenLevelReadDataset(
        data["train_sequences"].tolist(),
        train_labels,
        tokenizer=tokenizer,
        max_length=max_token_length,
        rc_augment=rc_augment,
        rc_consistency=rc_consistency,
        kmer_preprocess=kmer_preprocess,
      )
      val_dataset = TokenLevelReadDataset(
        data["val_sequences"].tolist(),
        val_labels,
        tokenizer=tokenizer,
        max_length=max_token_length,
        rc_augment=False,
        rc_consistency=False,  # val always uses standard mode
        kmer_preprocess=kmer_preprocess,
      )

    # Quick sanity check
    sample = train_dataset[0]
    print(f"\nTokenization check:")
    print(f"  input_ids shape:      {sample[0].shape}")
    print(f"  attention_mask shape:  {sample[1].shape}")
    print(f"  Non-padding tokens:   {sample[1].sum().item()}")
    if rc_consistency:
        print(f"  RC mode: rc_ids shape: {sample[2].shape}")
        print(f"  RC mode: rc_mask shape: {sample[3].shape}")
        print(f"  RC non-padding tokens: {sample[3].sum().item()}")
        print(f"  label: {sample[4].item()}")
    else:
        print(f"  label: {sample[2].item()}")

    # Validate token length vs max_length (catches padding waste / truncation).
    # LazyFASTADataset has no such helper; the sanity check above already shows
    # the real token count for this config.
    if hasattr(train_dataset, "validate_token_lengths"):
        print("\nToken length validation:")
        train_dataset.validate_token_lengths(n_samples=500)

    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)

    # ===== Optional class-aware sampling (WeightedRandomSampler) =====
    use_weighted_sampler = train_cfg.get("weighted_sampler", False)
    sampler = None
    if use_weighted_sampler:
        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights_s = 1.0 / np.clip(class_counts, 1, None).astype(np.float64)
        sample_weights = class_weights_s[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_labels),
            replacement=True,
        )
        print(f"\nUsing WeightedRandomSampler (class-aware)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers, worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("eval_batch_size", batch_size * 2),
        shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        pin_memory=True,
    )
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Training and validation loaders must contain batches; reduce batch_size or adjust val_ratio")

    # ===== 4. Create model =====
    print()
    sys.stdout.flush()
    model_cfg = cfg["model"]
    model = create_model(model_cfg, num_classes).to(device)
    sys.stdout.flush()

    # ===== Resume / Initialize from checkpoint =====
    resume_epoch = 0
    resume_ckpt = None
    if args.resume:
        print(f"\n--- Resuming from checkpoint: {args.resume} ---")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_weights(model, resume_ckpt["model_state_dict"]))
        resume_epoch = resume_ckpt.get("epoch", 0)
        has_optim = "optimizer_state_dict" in resume_ckpt
        print(f"  Loaded model weights from epoch {resume_epoch}")
        print(f"  Previous val_acc: {resume_ckpt.get('val_acc', 'N/A')}")
        print(f"  Optimizer/scheduler state: {'found (seamless resume)' if has_optim else 'not found (LR will restart)'}")
        sys.stdout.flush()
    elif args.init_from:
        print(f"\n--- Initializing weights from: {args.init_from} ---")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = checkpoint_weights(model, ckpt["model_state_dict"])
        # Allow partial loading (skip mismatched head dimensions)
        model_state = model.state_dict()
        loaded, skipped = [], []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_state)
        print(f"  Loaded {len(loaded)} params, skipped {len(skipped)} (shape mismatch)")
        if skipped:
            print(f"  Skipped keys: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
        print(f"  Source epoch: {ckpt.get('epoch', '?')}, val_acc: {ckpt.get('val_acc', 'N/A')}")
        print(f"  Training starts from epoch 1 (transfer learning mode)")
        sys.stdout.flush()

    # ===== 5. Loss function =====
    use_class_weights = train_cfg.get("class_weights", True)
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    if use_class_weights:
        active_classes = np.unique(train_labels)
        cw = np.zeros(num_classes, dtype=np.float64)
        cw[active_classes] = compute_class_weight("balanced", classes=active_classes, y=train_labels)
        max_weight = train_cfg.get("max_class_weight", 10.0)
        cw = np.clip(cw, a_min=None, a_max=max_weight)
        class_weights = torch.from_numpy(cw).float().to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        print(f"\nUsing class weights (min={cw.min():.4f}, max={cw.max():.4f}, clamped≤{max_weight})")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

    # ===== Logit Adjustment (Menon et al. 2021) =====
    use_logit_adj = train_cfg.get("logit_adjustment", False)
    if use_logit_adj:
        tau = train_cfg.get("logit_adjustment_tau", 1.0)
        unique_cls, counts_cls = np.unique(train_labels, return_counts=True)
        class_prior = np.zeros(num_classes, dtype=np.float64)
        class_prior[unique_cls] = counts_cls / counts_cls.sum()
        class_prior = np.clip(class_prior, 1e-8, None)  # avoid log(0)
        log_prior = torch.from_numpy(tau * np.log(class_prior)).float().to(device)
        print(f"\nLogit adjustment: τ={tau}")
        print(f"  log_prior range: [{log_prior.min().item():.4f}, {log_prior.max().item():.4f}]")
    else:
        log_prior = None

    # ===== RC consistency params =====
    lambda_kl = train_cfg.get("rc_consistency_lambda", 0.1)
    if rc_consistency:
        print(f"  RC consistency λ_KL = {lambda_kl}")

    # ===== Training hyperparams =====
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    num_epochs = train_cfg["num_epochs"]
    patience = train_cfg.get("early_stopping_patience", 5)
    phase1_epochs = train_cfg.get("phase1_epochs", 0)

    # ===== AMP setup =====
    amp_dtype = _get_amp_dtype(use_amp, train_cfg.get("amp_dtype"))
    # GradScaler is only needed for fp16, NOT for bf16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    early_stop = EarlyStopping(patience=patience, mode="max")

    effective_batch = batch_size * grad_accum
    print(f"\nTraining config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  AMP: {use_amp} (dtype={amp_dtype})")
    print(f"  GradScaler: {use_scaler}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Phase 1 (head-only) epochs: {phase1_epochs}")

    # ===== Sanity check: verify backbone output =====
    print("\n--- Sanity check: backbone output quality ---")
    model.eval()
    with torch.no_grad():
        sample_ids, sample_mask, sample_labels = next(iter(val_loader))
        sample_ids = sample_ids.to(device)
        sample_mask = sample_mask.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(sample_ids, sample_mask)
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"  Logits std: {logits.std().item():.4f}")
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-20)).sum(dim=-1).mean()
        print(f"  Avg entropy: {entropy.item():.4f} (uniform={math.log(num_classes):.4f})")
    model.train()
    sys.stdout.flush()

    # ===== 6. Phase 1: Train head only =====
    if phase1_epochs > 0 and not args.resume and not args.init_from:
        print(f"\n{'=' * 70}")
        print(f"Phase 1: Training HEAD only for {phase1_epochs} epochs")
        print(f"{'=' * 70}")
        sys.stdout.flush()

        # Temporarily freeze backbone
        model.freeze_backbone()
        phase1_lr = train_cfg.get("phase1_lr", 5e-4)
        # Phase 1 uses its own grad_accum=1 and a higher grad norm for the head
        phase1_max_grad_norm = 10.0  # Be less aggressive with gradient clipping for head
        phase1_opt = torch.optim.AdamW(
            model.get_head_params(),
            lr=phase1_lr,
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        # Phase 1 LR scheduler: cosine decay over phase1 steps
        phase1_total_steps = len(train_loader) * phase1_epochs
        phase1_warmup = int(phase1_total_steps * 0.05)
        phase1_scheduler = get_cosine_schedule_with_warmup(
            phase1_opt,
            num_warmup_steps=phase1_warmup,
            num_training_steps=phase1_total_steps,
        )
        print(f"  Phase1 LR: {phase1_lr}, grad_norm: {phase1_max_grad_norm}")
        print(f"  Phase1 steps: {phase1_total_steps}, warmup: {phase1_warmup}")
        sys.stdout.flush()

        for ep in range(phase1_epochs):
            tr_loss, tr_acc, tr_f1 = train_epoch(
                model, train_loader, criterion, phase1_opt, phase1_scheduler, scaler,
                device, grad_accum=1, max_grad_norm=phase1_max_grad_norm,
                use_amp=use_amp, amp_dtype=amp_dtype,
                log_prior=log_prior,
                rc_consistency=rc_consistency,
                lambda_kl=lambda_kl,
            )
            va_loss, va_acc, va_f1, _, _ = validate(
                model, val_loader, criterion, device, use_amp, amp_dtype=amp_dtype,
            )
            print(f"  Phase1 Ep {ep + 1}/{phase1_epochs}: "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}")
            sys.stdout.flush()

        # Unfreeze backbone for Phase 2
        model.unfreeze_backbone()
        # Reset GradScaler for Phase 2 (fresh state after backbone unfreeze)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        print("  → Backbone unfrozen for Phase 2\n")
        sys.stdout.flush()

    # ===== 7. Phase 2: Full training (LoRA backbone + head) =====
    remaining_epochs = num_epochs - resume_epoch
    print(f"{'=' * 70}")
    if resume_epoch > 0:
        print(f"Phase 2 (RESUMED): epochs {resume_epoch + 1}→{num_epochs} "
              f"({remaining_epochs} remaining)")
    else:
        print(f"Phase 2: Training backbone + head for {num_epochs} epochs")
    print(f"{'=' * 70}")

    # Differential learning rates
    backbone_lr = train_cfg.get("backbone_lr", 2e-5)
    head_lr = train_cfg["learning_rate"]

    backbone_params = model.get_backbone_params()
    head_params = model.get_head_params()

    # A sparse embedding (Job A: 33.5M exact 13-mers) produces sparse gradients,
    # which AdamW cannot consume — it needs SparseAdam. Split it out and give it
    # its own optimizer + matching LR schedule.
    sparse_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, torch.nn.Embedding) and getattr(m, "sparse", False)
        for p in m.parameters()
    }
    sparse_params = [p for p in model.parameters()
                     if id(p) in sparse_ids and p.requires_grad]
    if sparse_params:
        backbone_params = [p for p in backbone_params if id(p) not in sparse_ids]
        head_params = [p for p in head_params if id(p) not in sparse_ids]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # NOTE: torch.optim.SparseAdam takes no weight_decay, so the sparse
    # embedding is trained undecayed (MetaTransformer uses plain Adam too).
    sparse_optimizer = (torch.optim.SparseAdam(sparse_params, lr=backbone_lr)
                        if sparse_params else None)
    if sparse_optimizer is not None:
        n_sparse = sum(p.numel() for p in sparse_params)
        print(f"  Sparse embedding: {n_sparse:,} params -> SparseAdam "
              f"(lr={backbone_lr}, no weight decay)")

    # Build scheduler over the FULL training duration (not just remaining)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps_full = steps_per_epoch * num_epochs
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(total_steps_full * warmup_ratio)

    # training.lr_schedule: "cosine" (default, every arm so far) or "none".
    # "none" holds the learning rate flat for the whole run, which is what
    # MetaTransformer does -- its config defines a StepLR but leaves
    # lr_scheduler_enabled false, and warmup_lr_scheduler_enabled false with it.
    # Reproducing that is not expressible through warmup_ratio alone: setting it
    # to 0 still leaves the cosine decay to zero in place.
    lr_schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()
    if lr_schedule not in ("cosine", "none"):
        raise ValueError(f"unknown training.lr_schedule: {lr_schedule}")

    def _make_scheduler(opt):
        if opt is None:
            return None
        if lr_schedule == "none":
            return torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)
        return get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps_full)

    scheduler = _make_scheduler(optimizer)
    sparse_scheduler = _make_scheduler(sparse_optimizer)
    print(f"  LR schedule: {lr_schedule}" +
          (f" (warmup {warmup_steps:,} of {total_steps_full:,} steps)"
           if lr_schedule == "cosine" else " (constant)"))

    # Lists handed to train_epoch; also what the checkpoint helpers iterate.
    optimizers = [optimizer] + ([sparse_optimizer] if sparse_optimizer is not None else [])
    schedulers = [scheduler] + ([sparse_scheduler] if sparse_scheduler is not None else [])

    def _sparse_opt_state():
        return sparse_optimizer.state_dict() if sparse_optimizer is not None else None

    def _sparse_sched_state():
        return sparse_scheduler.state_dict() if sparse_scheduler is not None else None

    # Restore optimizer/scheduler states if available (seamless resume)
    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        if sparse_optimizer is not None and resume_ckpt.get("sparse_optimizer_state_dict"):
            sparse_optimizer.load_state_dict(resume_ckpt["sparse_optimizer_state_dict"])
            sparse_scheduler.load_state_dict(resume_ckpt["sparse_scheduler_state_dict"])
            print("  ✅ Restored SparseAdam moments + schedule")
        print(f"\n  ✅ Restored optimizer momentum + LR schedule (seamless)")
        print(f"  Current LR — backbone: {optimizer.param_groups[0]['lr']:.6e}, "
              f"head: {optimizer.param_groups[-1]['lr']:.6e}")
    elif resume_epoch > 0:
        # Legacy checkpoint without optimizer state: fast-forward scheduler
        skip_steps = steps_per_epoch * resume_epoch
        print(f"\n  ⚠️  No optimizer state in checkpoint — fast-forwarding scheduler by {skip_steps} steps")
        for _ in range(skip_steps):
            scheduler.step()
        print(f"  LR after fast-forward — backbone: {optimizer.param_groups[0]['lr']:.6e}, "
              f"head: {optimizer.param_groups[-1]['lr']:.6e}")

    # Restore early stopping state if available
    if resume_ckpt is not None:
        best_val = resume_ckpt.get("best_val_acc")
        if best_val is None:
            best_val = resume_ckpt.get("val_acc")
        if best_val is not None:
            early_stop.best = best_val
            early_stop.counter = resume_ckpt.get("patience_counter") or 0
            print(f"  Restored early stopping: best={early_stop.best:.4f}, "
                  f"patience={early_stop.counter}/{patience}")

    if resume_ckpt is not None:
        restored_rng = restore_rng(resume_ckpt.get("rng_state"))
        print(f"  RNG state restored: {restored_rng}")

    # Free checkpoint memory
    if resume_ckpt is not None:
        del resume_ckpt
        torch.cuda.empty_cache()

    print("\nOptimizer: AdamW" + ("  +  SparseAdam (sparse embedding)" if sparse_optimizer is not None else ""))
    print(f"  Backbone LR: {backbone_lr}")
    print(f"  Head LR:     {head_lr}")
    print(f"  Total steps (full schedule): {total_steps_full}")
    print(f"  Warmup:      {warmup_steps}")
    print(f"  Remaining epochs: {remaining_epochs}")
    print()

    # ===== Training loop =====
    history = []
    if resume_epoch > 0:
        csv_path = output_dir / "training_history.csv"
        if csv_path.exists():
            history = pd.read_csv(csv_path).to_dict("records")
            print(f"  Loaded {len(history)} rows from existing training_history.csv")

    start_time = time.time()
    job_start_time = start_time

    for epoch in range(resume_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        monitor.epoch_start(epoch + 1)

        # Mid-epoch time-limit check: if running close to wall-clock limit,
        # save last.pt and exit before Slurm kills us.
        if args.time_limit_sec is not None:
            def _mid_epoch_save_and_exit(batch_idx):
                elapsed_job = time.time() - job_start_time
                if elapsed_job >= args.time_limit_sec:
                    print(f"\n⏰ Time limit ({args.time_limit_sec}s) reached at batch {batch_idx}."
                          f" Saving last.pt and exiting.")
                    sys.stdout.flush()
                    torch.save(dict(
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        scaler_state_dict=scaler.state_dict(),
                        sparse_optimizer_state_dict=_sparse_opt_state(),
                        sparse_scheduler_state_dict=_sparse_sched_state(),
                        epoch=epoch,         # NOT epoch+1: this epoch is incomplete
                        val_acc=history[-1]["val_acc"] if history else 0.0,
                        best_val_acc=early_stop.best,
                        patience_counter=early_stop.counter,
                        config=cfg, rng_state=capture_rng(), resume_boundary="incomplete_epoch",
                    ), output_dir / "last.pt")
                    print(f"  Saved. Resume with: --resume {output_dir}/last.pt")
                    sys.exit(0)
            periodic_save_fn = _mid_epoch_save_and_exit
        else:
            periodic_save_fn = None

        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizers, schedulers, scaler,
            device, grad_accum, max_grad_norm, use_amp, amp_dtype=amp_dtype,
            log_prior=log_prior,
            rc_consistency=rc_consistency,
            lambda_kl=lambda_kl,
            periodic_save_fn=periodic_save_fn,
            periodic_save_interval=train_cfg.get("periodic_save_interval", 5000),
        )

        va_loss, va_acc, va_f1, va_preds, va_true = validate(
            model, val_loader, criterion, device, use_amp, amp_dtype=amp_dtype,
        )

        monitor.epoch_end(epoch + 1, train_samples=len(train_labels))

        elapsed = time.time() - start_time
        eta = (elapsed / (epoch - resume_epoch + 1)) * (num_epochs - epoch - 1)
        snap = monitor.snapshot()

        print(f"  Train — loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  f1: {tr_f1:.4f}")
        print(f"  Val   — loss: {va_loss:.4f}  acc: {va_acc:.4f}  f1: {va_f1:.4f}")
        print(f"  ⏱️ Elapsed: {timedelta(seconds=int(elapsed))} | "
              f"ETA: {timedelta(seconds=int(eta))}")
        print(f"  📊 GPU: {snap.get('gpu_allocated_gb', '?')}GB alloc, "
              f"{snap.get('gpu_util_pct', '?')}% util | "
              f"RAM: {snap.get('ram_rss_gb', '?')}GB | "
              f"Throughput: {monitor.get_epoch_throughput(epoch + 1):,.0f} reads/sec")
        sys.stdout.flush()

        # Log history
        row = dict(
            epoch=epoch + 1,
            train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1,
            lr_backbone=optimizer.param_groups[0]["lr"] if backbone_params else 0,
            lr_head=optimizer.param_groups[-1]["lr"],
            epoch_time_sec=round(monitor._epoch_durations.get(epoch + 1, 0), 1),
            throughput_reads_per_sec=round(monitor.get_epoch_throughput(epoch + 1), 1),
            gpu_allocated_gb=snap.get("gpu_allocated_gb", None),
            gpu_util_pct=snap.get("gpu_util_pct", None),
            ram_rss_gb=snap.get("ram_rss_gb", None),
            timestamp=datetime.now().isoformat(),
        )
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        # Early stopping / save best
        is_best = early_stop.step(va_acc)
        if is_best:
            ckpt = dict(
                model_state_dict=model.state_dict(),
                epoch=epoch + 1,
                val_acc=va_acc,
                val_f1=va_f1,
                config=cfg, num_classes=num_classes,
            )
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  ⭐ New best! val_acc={va_acc:.4f}")
        else:
            print(f"  (no improvement, patience {early_stop.counter}/{patience})")

        # Save last checkpoint (with optimizer/scheduler for seamless resume)
        torch.save(dict(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            sparse_optimizer_state_dict=_sparse_opt_state(),
            sparse_scheduler_state_dict=_sparse_sched_state(),
            epoch=epoch + 1,
            val_acc=va_acc,
            best_val_acc=early_stop.best,
            patience_counter=early_stop.counter,
            config=cfg, rng_state=capture_rng(), resume_boundary="epoch", num_classes=num_classes,
        ), output_dir / "last.pt")

        if early_stop.should_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
            break

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start_time
    monitor.stop()

    # ===== 8. Final evaluation =====
    print(f"\n{'=' * 70}")
    print("Final evaluation on best model")
    print(f"{'=' * 70}")

    # Load to CPU, not the GPU: the model (and, for a sparse embedding, its
    # optimizer moments) already occupy the device, so map_location=device asks
    # for a second full copy of the weights. With Job A's 8 GiB embedding that
    # OOM'd here *after* all 15 epochs had finished. Free the optimizer state
    # first as well — training is over and nothing below needs it.
    for _o in optimizers:
        _o.zero_grad(set_to_none=True)
        _o.state.clear()
    torch.cuda.empty_cache()
    best_ckpt = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    # Keep the epoch before the checkpoint is dropped: the summary below
    # still wants it, and the 8 GiB dict must not survive until then.
    best_epoch = int(best_ckpt["epoch"])
    del best_ckpt
    gc.collect()
    torch.cuda.empty_cache()

    va_loss, va_acc, va_f1, va_preds, va_true = validate(
        model, val_loader, criterion, device, use_amp, amp_dtype=amp_dtype,
    )

    # Speed benchmark
    print("\nSpeed benchmark...")
    model.eval()
    total_reads = 0
    total_inf_time = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc="  Benchmark"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                _ = model(input_ids, attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_inf_time += time.perf_counter() - t0
            total_reads += labels.size(0)

    reads_per_sec = total_reads / total_inf_time if total_inf_time > 0 else 0
    avg_latency_ms = (total_inf_time / total_reads * 1000) if total_reads > 0 else 0

    # Classification report
    if data is None:
        _lab = pd.read_csv(cfg["data"]["labels_path"], sep="\t",
                           usecols=["genus_class", "genus_name"]).drop_duplicates()
        id2label = dict(zip(_lab["genus_class"].astype(int), _lab["genus_name"]))
    elif task == "genus":
        id2label = data["id2genus"]
    else:
        id2label = data["id2species"]
    unique_labels = np.unique(np.concatenate([va_true, va_preds]))
    target_names = [id2label.get(int(i), str(i)) for i in unique_labels]
    report = classification_report(
        va_true, va_preds,
        labels=unique_labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")

    # Summary metrics
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total_params, trainable_params = (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    metrics = {
        "task": task,
        "backbone": backbone_name,
        "head_type": model_cfg.get("head_type", "transformer"),
        "use_lora": model_cfg.get("use_lora", True),
        "val_accuracy": float(va_acc),
        "val_f1_weighted": float(va_f1),
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "reads_per_sec": float(reads_per_sec),
        "avg_latency_ms": float(avg_latency_ms),
        "peak_gpu_memory_gb": float(peak_mem_gb),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_training_time_sec": float(total_time),
        "training_time_str": str(timedelta(seconds=int(total_time))),
    }
    save_json(metrics, str(output_dir / "metrics.json"))

    # ===== 9. Resource report =====
    monitor.save_report(str(output_dir / "resource_report.json"))
    monitor.print_summary()

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")
    print(f"  Best val accuracy:  {va_acc:.4f}")
    print(f"  Best val F1:        {va_f1:.4f}")
    print(f"  Best epoch:         {best_epoch}")
    print(f"  Total training:     {timedelta(seconds=int(total_time))}")
    print(f"  Reads/sec:          {reads_per_sec:,.0f}")
    print(f"  Avg latency:        {avg_latency_ms:.2f} ms/read")
    print(f"  Peak GPU memory:    {peak_mem_gb:.2f} GB")
    print(f"  Results saved to:   {output_dir}")


if __name__ == "__main__":
    main()
