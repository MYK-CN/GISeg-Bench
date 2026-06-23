"""
GISeg-Bench  Reproducibility
=============================
Deterministic random seeding — call once at the top of any experiment
script to fix Python, NumPy, and PyTorch RNGs.

Usage::

    from utils.seed import seed_all
    seed_all(42)

Also provides a ``seed_worker`` for DataLoader worker processes and
a ``get_generator`` helper.
"""

import os
import random
import numpy as np
import torch


# ===================================================================
#  Master seed
# ===================================================================

def seed_all(seed=42, deterministic_cudnn=False):
    """Fix random seeds for Python, NumPy, and PyTorch.

    Args:
        seed:               integer seed.
        deterministic_cudnn: if True, enables ``cudnn.deterministic``
                             and ``cudnn.benchmark=False`` (may be slower).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # multi-GPU

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # allow auto-tuning for speed (still deterministic for given GPU arch)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[Seed] Global seed set to {seed}"
          f"{' (cudnn deterministic)' if deterministic_cudnn else ''}")


# ===================================================================
#  DataLoader worker seeding
# ===================================================================

def seed_worker(worker_id):
    """DataLoader worker init function — ensures each worker has a
    unique but reproducible seed.

    Usage::

        loader = DataLoader(ds, ..., worker_init_fn=seed_worker,
                            generator=get_generator())
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed=42):
    """Return a torch Generator for DataLoader reproducibility.

    Usage::

        g = get_generator(42)
        loader = DataLoader(ds, ..., generator=g)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
