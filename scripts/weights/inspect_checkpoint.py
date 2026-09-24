#!/usr/bin/env python3
"""Print what a KmerFormer checkpoint is, without loading its tensors.

The exact-13-mer checkpoints are 8 GiB because 99.99% of their parameters are the
embedding table, so ``torch.load`` is an expensive way to answer "which arm is
this?". A torch checkpoint is a zip whose ``data.pkl`` holds the object graph with
tensor storages kept out of line, so the config, epoch and validation accuracy can
be read from a few kilobytes.

Use it to confirm a download before trusting it::

    python scripts/weights/inspect_checkpoint.py weights/*.pt

THE ACCURACY IT PRINTS IS NOT THE PAPER'S NUMBER, and should not be "corrected" to
match it. A checkpoint records the best accuracy on the run's own validation split
-- the 10% each arm holds out of its training pool, forward pass only. The paper
reports the separate 100,000-read closed-set test set with reverse-complement
test-time augmentation. Two different pools and two different protocols, so they
land a fraction of a point apart: this prints 0.9111 for the 50M exact-13-mer arm
where docs/RESULTS.md says 91.15%, and 0.6928 for the 29-layer 6-mer arm where it
says 69.52%. Both pairs are right. To reproduce the paper's figure, run
kmerformer-eval against val_100K with --rc_tta; see docs/REPRODUCE.md.

Only explicitly allowed container types are reconstructed; tensor and NumPy
objects become inert placeholders. Other pickle globals are rejected. This avoids
executing arbitrary pickle functions, but is not a sandbox against resource
exhaustion. Inspect only checkpoints from a trusted distribution and verify hashes.
"""

import argparse
import collections
import io
import pickle
import sys
import zipfile
from pathlib import Path

# Names the unpickler is allowed to actually resolve. Everything else -- every
# torch class, every rebuild function -- becomes an inert stub.
_SAFE_GLOBALS = {("collections", "OrderedDict"): collections.OrderedDict,
                 ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
                 ("builtins", "slice"): slice, ("builtins", "complex"): complex}
_NUMPY_STUBS = {("numpy", "dtype"), ("numpy", "ndarray"),
                ("numpy.core.multiarray", "scalar"),
                ("numpy.core.multiarray", "_reconstruct"),
                ("numpy._core.multiarray", "scalar"),
                ("numpy._core.multiarray", "_reconstruct")}


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return "<storage>"

    def __setstate__(self, state):
        pass


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key in _SAFE_GLOBALS:
            return _SAFE_GLOBALS[key]
        if module == "torch" or module.startswith("torch.") or key in _NUMPY_STUBS:
            return _Stub
        raise pickle.UnpicklingError(f"unsupported pickle global: {module}.{name}")

    def persistent_load(self, pid):
        return None  # tensor storages live outside the pickle; we never want them


def read_metadata(path):
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.infolist() if n.filename.endswith("/data.pkl")
                   or n.filename == "data.pkl")
        if pkl.file_size > 16 * 1024 * 1024:
            raise ValueError("checkpoint metadata exceeds the 16 MiB inspection limit")
        return _MetadataUnpickler(io.BytesIO(archive.read(pkl))).load()


def describe(path):
    path = Path(path)
    size_gib = path.stat().st_size / 1024 ** 3
    print(f"{path}  ({size_gib:.2f} GiB)")

    try:
        obj = read_metadata(path)
    except (zipfile.BadZipFile, StopIteration, pickle.UnpicklingError,
            ValueError, EOFError, TypeError, AttributeError) as exc:
        print(f"  unsupported or invalid checkpoint metadata: {exc}")
        return False
    if not isinstance(obj, dict):
        print(f"  unexpected top-level object: {type(obj).__name__}")
        return False

    for key, label in (("epoch", "epoch"),
                       ("val_acc", "internal val"),
                       ("val_f1", "internal F1")):
        if key in obj:
            print(f"  {label:14}: {obj[key]}")

    cfg = obj.get("config")
    if isinstance(cfg, dict):
        model = cfg.get("model") or {}
        data = cfg.get("data") or {}
        shallow = model.get("shallow_config") or {}
        tokenizer = data.get("tokenizer")
        print(f"  {'tokenizer':14}: "
              + (f"{tokenizer.get('type')} k={tokenizer.get('k')} stride={tokenizer.get('stride')}"
                 if tokenizer else "NT-v2 non-overlapping 6-mer"))
        print(f"  {'encoder':14}: d_model={shallow.get('d_model')} layers={shallow.get('num_layers')} "
              f"nhead={shallow.get('nhead')} d_ff={shallow.get('d_ff')}")
        print(f"  {'pooling':14}: {model.get('head_type')}")
        print(f"  {'vocab_size':14}: {model.get('vocab_size')}")
        print(f"  {'trained on':14}: {data.get('fasta_path')}")
        print(f"  {'original run':14}: {(cfg.get('output') or {}).get('dir')}")
    else:
        print("  no config recorded in this checkpoint")
    print("  note          : 'internal val' is the run's own 90/10 split, forward "
          "only --\n"
          "                  NOT the paper's 100K closed-set + RC-TTA figure. "
          "See docs/RESULTS.md.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()

    ok = True
    for i, path in enumerate(args.checkpoints):
        if i:
            print()
        if not path.exists():
            print(f"{path}  MISSING")
            ok = False
            continue
        ok &= describe(path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
