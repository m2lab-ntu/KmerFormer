"""Shared test setup.

WHY THE MKL LINE. tests/test_training_runtime.py spawns DDP ranks with
torch.multiprocessing. In a conda environment carrying Intel MKL, the spawned
child aborts before running anything:

    mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible
    with libgomp-...so.1

The parent sees only `ProcessExitedException: process 0 terminated with exit
code 1`, and the MKL line is buried in captured stderr, so the failure reads as
a bug in the training code rather than as a threading-layer conflict between
MKL and the OpenMP runtime torch was built against. Setting the layer to GNU
before torch is imported resolves it and changes nothing else: these tests
exercise control flow, not numerics.

An explicit value set by the caller wins, so anyone diagnosing the interaction
can still reproduce it with MKL_THREADING_LAYER=INTEL.
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
