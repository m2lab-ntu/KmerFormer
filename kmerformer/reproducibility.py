"""Training randomness, scheduler and run provenance without baseline imports."""
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
from pathlib import Path
import numpy as np
import torch


def seed_training(seed):
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def capture_rng():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    return True


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                  num_cycles=0.5, last_epoch=-1):
    """Linear warmup followed by the cosine schedule used by historical runs."""
    def scale(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale, last_epoch)


def write_run_record(cfg, output_dir, *, rank=0, world_size=1, resume=None):
    if rank:
        return
    try:
        repo = Path(__file__).resolve().parent.parent
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    dependencies = {}
    for name in ("torch", "numpy", "transformers", "peft", "kmerformer"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {"schema_version": 1, "code_revision": revision, "working_tree_dirty": dirty,
              "python": platform.python_version(), "dependencies": dependencies,
              "training_seed": cfg["training"].get("seed"), "world_size": world_size,
              "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
              "resume_from": str(resume) if resume else None,
              "resume_contract": "Epoch-boundary RNG restoration on the same software/hardware and worker setup; incomplete epochs restart."}
    Path(output_dir, "run_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
