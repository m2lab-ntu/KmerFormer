#!/usr/bin/env python3
"""Shared utilities for Token-level GFM Classifier."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _expand(value):
    """Expand ``$VAR`` / ``${VAR}`` and ``~`` inside every string in a config.

    Every path in ``configs/`` is written against three variables so one config
    runs unchanged on any machine:

        KF_DATA   root of the read pools and evaluation sets
        KF_OUT    where runs write checkpoints and metrics
        KF_VOCAB  directory holding ``vocab_13mer.txt`` (exact 13-mer arms only)

    An unset variable is left verbatim rather than replaced by an empty string,
    so a missing one fails as ``no such file: $KF_DATA/reads_50M.fa`` instead of
    silently reading from the filesystem root.
    """
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(config_path: str) -> dict:
    """Load a YAML config, expanding environment variables in every string."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return _expand(cfg)


def save_config(cfg: dict, output_dir: str):
    """Save config to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def save_json(data: dict, path: str):
    """Save dict as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return True  # is_best

        if self.mode == "max":
            is_best = metric > self.best
        else:
            is_best = metric < self.best

        if is_best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return is_best


class Timer:
    """Simple timer context manager."""

    def __init__(self):
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

