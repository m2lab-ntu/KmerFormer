#!/usr/bin/env python3
"""
DDP training script for Token-level GFM Classifier — 258M full dataset.

Launch with torchrun (see SLURM script):
  torchrun --nproc_per_node=8 --nnodes=8 \\
      --node_rank=$SLURM_NODEID \\
      --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) \\
      --master_port=29500 \\
      train_ddp.py --config configs/nt_token_genus_v10_258M.yaml

Key differences from train.py:
  - Uses LazyFASTADataset (dataset_lazy.py) → no full-FASTA in RAM
  - DistributedSampler partitions data across ranks
  - Model wrapped with DDP
  - Metrics all-reduced across ranks before logging
  - Checkpointing and logging on rank 0 only
  - Phase 1 (head-only) is skipped in DDP mode (start directly with LoRA+head)
"""

import argparse
import datetime
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.utils.class_weight import compute_class_weight
from .reproducibility import (get_cosine_schedule_with_warmup, seed_training, seed_worker,
                              capture_rng, restore_rng, write_run_record)
from .splits import split_options

from .dataset_lazy import LazyFASTADataset
from .model import create_model
from .kmer_tokenizers import build_tokenizer
from .resource_monitor import ResourceMonitor
from .training_utils import checkpoint_weights, optimizer_step
from .utils import load_config, save_config, save_json, AverageMeter, EarlyStopping


# ── DDP helpers ──────────────────────────────────────────────────────────────

def setup_ddp():
    # Support both srun-python (SLURM_*) and torchrun (LOCAL_RANK) launches
    rank        = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size  = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    slurm_local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    os.environ["RANK"]       = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # If Slurm assigns 1 GPU per task, CUDA_VISIBLE_DEVICES is a single device
    # (e.g. "3") → cuda:0 is the only visible device; otherwise use SLURM_LOCALID.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and "," not in cvd and cvd not in ("-1", "NoDevFiles", ""):
        local_rank = 0  # exactly 1 GPU assigned to this task
    else:
        local_rank = slurm_local
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"),
                                timeout=datetime.timedelta(hours=4))
    else:
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=10))
    return local_rank, rank, world_size


class DistributedEvalSampler(Sampler):
    """Assign each validation read to exactly one rank without padding."""

    def __init__(self, dataset, num_replicas, rank):
        self.indices = range(rank, len(dataset), num_replicas)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def any_rank(flag, device):
    flag = torch.tensor(int(flag), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def optimizer_groups(model, backbone_lr, head_lr, previous_state=None):
    """Use the model's parameter groups, retaining legacy layout for resumes."""
    raw_model = model.module if isinstance(model, DDP) else model
    previous_groups = (previous_state or {}).get("param_groups", [])
    if len(previous_groups) == 2 and not previous_groups[1]["params"]:
        # Historical DDP checkpoints placed every parameter in group zero.
        # Preserve their moments and learning rates when continuing those runs.
        return [{"params": [p for p in raw_model.parameters() if p.requires_grad],
                 "lr": backbone_lr}, {"params": [], "lr": head_lr}]
    return [{"params": raw_model.get_backbone_params(), "lr": backbone_lr},
            {"params": raw_model.get_head_params(), "lr": head_lr}]


def rejects_data_parallel(cfg: dict) -> bool:
    """True when this config cannot be trained under torch's data parallelism.

    The single criterion is ``model.shallow_config.sparse_embedding``. An arm with
    it takes sparse gradients on its embedding table -- required for the
    exact-13-mer arms, where a dense gradient would be table-sized -- and torch's
    data-parallel reducer rejects sparse gradients. Without this check the failure
    is an opaque reducer error tens of minutes in, after the FASTA index and the
    256 MiB vocabulary lookup table have been built on every rank.

    Keyed on the sparse flag and NOT on ``k`` or on the tokenizer type, which is
    the whole point: the hashed 13-mer arms are also long-k, but they update a
    2^22-row table densely and train perfectly well across GPUs. Rejecting them
    would forfeit multi-GPU training for nothing, and it is the kind of mistake
    nobody reports -- it just makes the hashed arms look inherently slow. That is
    why ``tests/test_configs.py`` asserts both directions against the real config
    tree rather than only the rejecting one.
    """
    return bool((cfg.get("model", {}).get("shallow_config") or {})
                .get("sparse_embedding"))


def cleanup_ddp():
    dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> float:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


def all_reduce_sum(tensor: torch.Tensor) -> float:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


# ── AMP dtype ────────────────────────────────────────────────────────────────

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


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, grad_accum, max_grad_norm, use_amp, amp_dtype,
                log_prior=None, rc_consistency=False, lambda_kl=0.0,
                is_main=True, time_limit_sec=None, job_start=None,
                periodic_save_fn=None, periodic_save_interval=5000):
    model.train()
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if rc_consistency:
        raise ValueError("RC consistency training is supported by kmerformer.train only")
    loss_meter = AverageMeter()
    correct = 0
    total   = 0
    nan_count = 0

    optimizer.zero_grad()
    pending = 0
    sync_invalid = lambda invalid: any_rank(invalid, device)
    for step, batch in enumerate(loader):
        # Time-limit check
        expired = (time_limit_sec is not None and job_start is not None
                   and time.time() - job_start >= time_limit_sec)
        if time_limit_sec is not None and any_rank(expired, device):
            if is_main:
                print(f"\n⏰ Time limit reached at step {step}.")
            if periodic_save_fn:
                periodic_save_fn(step)
            return loss_meter.avg, correct / max(total, 1), True  # True = timed out

        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(input_ids, attention_mask)
            logits_for_loss = logits + log_prior if log_prior is not None else logits
            loss = criterion(logits_for_loss, labels) / grad_accum

        if any_rank(not bool(torch.isfinite(loss)), device):
            nan_count += 1
            if isinstance(model, DDP):
                # Finish the reducer cycle on every rank before discarding the
                # group, even when only one rank had a non-finite loss.
                scaler.scale(torch.nan_to_num(logits).sum() * 0).backward()
            optimizer.zero_grad()
            pending = 0
            continue

        scaler.scale(loss).backward()
        pending += 1
        if pending == grad_accum:
            if not optimizer_step([optimizer], [scheduler] if scheduler else [], scaler,
                                  model.parameters(), max_grad_norm,
                                  sync_invalid=sync_invalid):
                nan_count += 1
            pending = 0

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        loss_meter.update(loss.item() * grad_accum, labels.size(0))

        if periodic_save_fn and step > 0 and step % periodic_save_interval == 0:
            periodic_save_fn(step)

    if pending:
        if not optimizer_step([optimizer], [scheduler] if scheduler else [], scaler,
                              model.parameters(), max_grad_norm,
                              normalization=grad_accum / pending,
                              sync_invalid=sync_invalid):
            nan_count += 1
    metrics = torch.tensor([loss_meter.sum, correct, total], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    if metrics[2].item() == 0:
        raise ValueError("Training produced no finite batches; check the data and batch size")
    if nan_count > 0 and is_main:
        print(f"\n  ⚠️  NaN/Inf events this epoch: {nan_count}")

    return (metrics[0] / metrics[2]).item(), (metrics[1] / metrics[2]).item(), False


def validate(model, loader, criterion, device, use_amp, amp_dtype, world_size):
    """Run validation and aggregate metrics across all DDP ranks."""
    model.eval()
    # Unequal rank lengths are intentional. Bypass DDP's forward-time buffer
    # broadcasts; aggregate only after every rank finishes its own reads.
    inference_model = model.module if isinstance(model, DDP) else model
    correct_t = torch.zeros(1, device=device)
    total_t   = torch.zeros(1, device=device)
    loss_sum  = torch.zeros(1, device=device)
    loss_cnt  = torch.zeros(1, device=device)

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = inference_model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            correct_t += (preds == labels).sum()
            total_t   += labels.size(0)
            loss_sum  += loss * labels.size(0)
            loss_cnt  += labels.size(0)

    # Aggregate across ranks
    dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_sum,  op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_cnt,  op=dist.ReduceOp.SUM)

    if total_t.item() == 0:
        raise ValueError("Validation dataset is empty")
    val_loss = (loss_sum / loss_cnt).item()
    val_acc  = (correct_t / total_t).item()
    return val_loss, val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a KmerFormer 6-mer arm across GPUs with torchrun. "
                    "Arms with shallow_config.sparse_embedding (the exact 13-mer "
                    "vocabulary) are rejected here -- train those single-GPU.")
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--time_limit_sec", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_training(cfg["training"].get("seed"))

    # Reject unsupported arms BEFORE touching the distributed world, so this stays
    # a plain config read that needs no GPU, no rendezvous and no launcher -- it
    # fires even for someone who runs this module directly to see what it does.
    if rejects_data_parallel(cfg):
        print("ERROR: this config sets shallow_config.sparse_embedding, which "
              "torch's data-parallel reducer cannot handle.\n"
              "       Train it on a single GPU instead:\n"
              f"         kmerformer-train --config {args.config}\n"
              "       Each rank would hold its own copy of the embedding table "
              "anyway, so DDP buys nothing here.", file=sys.stderr)
        sys.exit(2)
    if cfg["data"].get("task", "genus") != "genus" or cfg["data"].get("rc_consistency"):
        parser.error("DDP supports genus classification without RC consistency; use kmerformer-train for this configuration")

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank, rank, world_size = setup_ddp()
    device   = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main  = (rank == 0)
    job_start = time.time()

    if is_main:
        print("=" * 70)
        print(f"KmerFormer — DDP Training  [world_size={world_size}]")
        print(f"Config: {args.config}")
        print(f"Rank:   {rank}/{world_size}  |  local_rank: {local_rank}")
        if device.type == "cuda":
            print(f"GPU:    {torch.cuda.get_device_name(local_rank)}")
            print(f"VRAM:   {torch.cuda.get_device_properties(local_rank).total_memory/1e9:.1f} GB")
        else:
            print("Device: CPU (Gloo)")
        print("=" * 70)

    output_dir = Path(cfg["output"]["dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, str(output_dir))

    dist.barrier()
    write_run_record(cfg, output_dir, rank=rank, world_size=world_size, resume=args.resume)

    # ── Resource monitor (rank 0 only) ────────────────────────────────────────
    monitor = None
    if is_main:
        monitor = ResourceMonitor(device=str(device), sample_interval=60.0)
        monitor.start()

    # ── Load datasets ─────────────────────────────────────────────────────────
    backbone_name    = cfg["model"].get("backbone", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    trust_remote     = cfg["model"].get("trust_remote_code", True)
    max_token_length = cfg["data"].get("max_token_length", 32)
    rc_augment       = cfg["data"].get("rc_augment", True)
    kmer_preprocess  = cfg["data"].get("kmer_preprocess", None)
    val_ratio        = cfg["data"].get("val_ratio", 0.05)   # use 5% val for 258M

    if is_main:
        print(f"\nLoading tokenizer: {backbone_name}")
    # build_tokenizer returns NT-v2's tokenizer unless data.tokenizer selects a
    # custom k-mer tokenizer (hashed / exact 13-mer — see kmer_tokenizers.py).
    tokenizer = build_tokenizer(cfg)
    if is_main:
        print(f"  Tokenizer: {type(tokenizer).__name__}  vocab_size={tokenizer.vocab_size:,}")
    # Guard: a mismatch here would silently train an embedding of the wrong size.
    if cfg["model"].get("vocab_size") is not None:
        assert cfg["model"]["vocab_size"] == tokenizer.vocab_size, (
            f"config model.vocab_size={cfg['model']['vocab_size']} != "
            f"tokenizer.vocab_size={tokenizer.vocab_size}")

    # Each rank instantiates its own LazyFASTADataset.
    # The index file (.idx.npy) is built once then shared via the filesystem.
    # All ranks build it concurrently on first run, then load from cache.
    if is_main:
        print(f"\nBuilding train/val datasets (lazy loading)...")

    train_dataset = LazyFASTADataset(
        fasta_path=cfg["data"]["fasta_path"],
        labels_path=cfg["data"]["labels_path"],
        tokenizer=tokenizer,
        max_length=max_token_length,
        split="train",
        val_ratio=val_ratio,
        seed=cfg["data"].get("seed", 42),
        rc_augment=rc_augment,
        kmer_preprocess=kmer_preprocess, **split_options(cfg),
    )
    val_dataset = LazyFASTADataset(
        fasta_path=cfg["data"]["fasta_path"],
        labels_path=cfg["data"]["labels_path"],
        tokenizer=tokenizer,
        max_length=max_token_length,
        split="val",
        val_ratio=val_ratio,
        seed=cfg["data"].get("seed", 42),
        rc_augment=False,
        kmer_preprocess=kmer_preprocess, **split_options(cfg),
    )
    num_classes = train_dataset.num_genera

    if is_main:
        print(f"\nTask: genus  |  Classes: {num_classes}")
        print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")
        print(f"Effective train per rank: {len(train_dataset) // world_size:,}")

    # ── Samplers + DataLoaders ────────────────────────────────────────────────
    train_cfg    = cfg["training"]
    batch_size   = train_cfg["batch_size"]         # per-GPU batch size
    num_workers  = train_cfg.get("num_workers", 4)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        seed=int(train_cfg.get("seed") or 0)
    )
    val_sampler = DistributedEvalSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=False, worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("eval_batch_size", batch_size * 2),
        sampler=val_sampler,
        num_workers=0, pin_memory=True,
    )
    if not len(train_loader) or not len(val_dataset):
        raise ValueError("Training requires at least one batch per rank and a non-empty validation set")

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_main:
        print("\nCreating model...")
    model_cfg = cfg["model"]
    model = create_model(model_cfg, num_classes).to(device)
    ckpt = {}

    if args.resume:
        if is_main:
            print(f"\n--- Resuming from: {args.resume} ---")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_weights(model, ckpt["model_state_dict"]))
        resume_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"  Loaded from epoch {resume_epoch}  val_acc={ckpt.get('val_acc','N/A')}")
    elif args.init_from:
        if is_main:
            print(f"\n--- Warm start from: {args.init_from} (epoch resets to 0) ---")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = checkpoint_weights(model, ckpt["model_state_dict"])
        target = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in target and v.shape == target[k].shape}
        target.update(compatible)
        model.load_state_dict(target)
        resume_epoch = 0
        if is_main:
            print(f"  Warm-start weights: {len(compatible)} loaded, {len(state)-len(compatible)} skipped")
    else:
        resume_epoch = 0

    # Wrap with DDP after loading checkpoint (load on raw model, then wrap)
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=False)

    # ── Loss ──────────────────────────────────────────────────────────────────
    use_class_weights = train_cfg.get("class_weights", True)
    label_smoothing   = train_cfg.get("label_smoothing", 0.0)
    if use_class_weights:
        train_genus_labels = train_dataset.get_genus_labels()
        active_classes = np.unique(train_genus_labels)
        cw = np.zeros(num_classes, dtype=np.float64)
        cw[active_classes] = compute_class_weight("balanced", classes=active_classes, y=train_genus_labels)
        max_weight = train_cfg.get("max_class_weight", 10.0)
        cw = np.clip(cw, a_min=None, a_max=max_weight)
        class_weights = torch.from_numpy(cw).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        if is_main:
            print(f"\nClass weights: min={cw.min():.4f} max={cw.max():.4f} (clamped≤{max_weight})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Logit Adjustment ──────────────────────────────────────────────────────
    use_logit_adj = train_cfg.get("logit_adjustment", False)
    log_prior = None
    if use_logit_adj:
        tau = train_cfg.get("logit_adjustment_tau", 1.0)
        train_genus_labels = train_dataset.get_genus_labels()
        unique_cls, counts_cls = np.unique(train_genus_labels, return_counts=True)
        class_prior = np.zeros(num_classes, dtype=np.float64)
        class_prior[unique_cls] = counts_cls / counts_cls.sum()
        class_prior = np.clip(class_prior, 1e-8, None)
        log_prior = torch.from_numpy(tau * np.log(class_prior)).float().to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr       = train_cfg.get("learning_rate", 5e-4)
    bb_lr    = train_cfg.get("backbone_lr", 3e-5)
    wd       = train_cfg.get("weight_decay", 0.01)
    num_epochs    = train_cfg["num_epochs"]
    warmup_ratio  = train_cfg.get("warmup_ratio", 0.05)
    grad_accum    = train_cfg.get("gradient_accumulation_steps", 1)
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    patience      = train_cfg.get("early_stopping_patience", 5)
    use_amp       = train_cfg.get("amp", True) and device.type == "cuda"
    amp_dtype     = _get_amp_dtype(use_amp, train_cfg.get("amp_dtype"))
    use_scaler    = use_amp and (amp_dtype == torch.float16)

    # Differential LR: backbone vs head
    previous_optimizer = ckpt.get("optimizer_state_dict") if args.resume else None
    groups = optimizer_groups(model, bb_lr, lr, previous_optimizer)
    if previous_optimizer and not groups[1]["params"] and is_main:
        print("  Resuming legacy DDP optimizer layout: all parameters retain the checkpoint's backbone LR.")
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=wd,
    )

    total_steps   = math.ceil(len(train_loader) / grad_accum) * num_epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    lr_schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()
    if lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif lr_schedule == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    else:
        raise ValueError(f"unknown training.lr_schedule: {lr_schedule}")
    scaler        = torch.amp.GradScaler("cuda", enabled=use_scaler)
    early_stop    = EarlyStopping(patience=patience, mode="max")

    if is_main:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.2f}%)")
        print(f"Effective batch: {batch_size * grad_accum * world_size} "
              f"({batch_size}/GPU × {grad_accum} accum × {world_size} GPUs)")
        print(f"Total steps: {total_steps:,}  warmup: {warmup_steps:,}")

    # Restore optimiser/scheduler if resuming
    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if is_main:
            print("  Optimiser + scheduler state restored.")

    best_val_acc = ckpt.get("best_val_acc") if args.resume else None
    if args.resume and best_val_acc is None:
        best_val_acc = ckpt.get("val_acc")
    patience_count = (ckpt.get("patience_counter") or 0) if args.resume else 0
    rng_states = ckpt.get("rng_states") if args.resume else None
    if rng_states:
        if len(rng_states) != world_size:
            raise ValueError("RNG resume requires the same world_size; use --init_from for a new run")
        restore_rng(rng_states[rank])
    elif cfg["training"].get("seed") is not None:
        seed_training(int(cfg["training"]["seed"]) + rank)
    del ckpt

    def gather_rng():
        states = [None] * world_size
        dist.all_gather_object(states, capture_rng())
        return states

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(resume_epoch, num_epochs):
        train_sampler.set_epoch(epoch)   # critical for proper shuffling in DDP

        t0 = time.time()

        def _periodic_save(step):
            rng_states = gather_rng()
            if not is_main:
                return
            ckpt_data = {
                "model_state_dict":     model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch, "rng_states": rng_states, "resume_boundary": "incomplete_epoch",
                "val_acc": best_val_acc,
                "best_val_acc": best_val_acc,
                "patience_counter": patience_count,
                "config": cfg,
            }
            torch.save(ckpt_data, output_dir / "last.pt")

        train_loss, train_acc, timed_out = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, grad_accum, max_grad_norm, use_amp, amp_dtype,
            log_prior=log_prior, is_main=is_main,
            time_limit_sec=args.time_limit_sec, job_start=job_start,
            periodic_save_fn=_periodic_save,
        )

        if timed_out:
            if is_main:
                print("\nTime limit hit. last.pt retains the incomplete epoch for resume.")
            dist.barrier()
            break

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_amp, amp_dtype, world_size
        )

        elapsed = time.time() - t0
        rng_states = gather_rng()

        if is_main:
            print(f"\nEpoch {epoch+1:02d}/{num_epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"time={elapsed/60:.1f}m")

            # Save last checkpoint
            is_best = best_val_acc is None or val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_count = 0
            else:
                patience_count += 1
            ckpt_data = {
                "model_state_dict":     model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch + 1, "rng_states": rng_states, "resume_boundary": "epoch",
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "patience_counter": patience_count,
                "config": cfg,
            }
            torch.save(ckpt_data, output_dir / "last.pt")

            # Save best checkpoint
            if is_best:
                torch.save(ckpt_data, output_dir / "best.pt")
                print(f"  *** New best: {best_val_acc:.4f} ***")

            # Append to training history CSV
            history_path = output_dir / "training_history.csv"
            header_needed = not history_path.exists()
            with open(history_path, "a") as hf:
                if header_needed:
                    hf.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
                hf.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},"
                         f"{val_loss:.6f},{val_acc:.6f}\n")

        # Broadcast patience_count so all ranks agree on early stopping
        patience_t = torch.tensor(patience_count, device=device)
        dist.broadcast(patience_t, src=0)
        patience_count = int(patience_t.item())

        if patience_count >= patience:
            if is_main:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

        torch.cuda.empty_cache()
        gc.collect()

    if is_main:
        print(f"\n{'='*70}")
        print(f"Training complete. Best val_acc = {best_val_acc}")
        print(f"Checkpoints: {output_dir}/best.pt  (and last.pt)")
        print(f"{'='*70}")
        if monitor:
            monitor.stop()

    cleanup_ddp()


if __name__ == "__main__":
    main()
