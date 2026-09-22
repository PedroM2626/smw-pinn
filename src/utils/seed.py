"""
seed.py
Central global seeding for full experiment reproducibility.

Replaces the previously scattered `torch.manual_seed(seed)` /
`np.random.seed(seed)` calls with a single deterministic entry point
covering Python, NumPy, PyTorch (CPU + CUDA) and DataLoader workers.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs; optionally enforce deterministic cuDNN.

    Args:
        seed: master seed propagated to all libraries.
        deterministic: when True, enables deterministic cuDNN kernels and
            `torch.use_deterministic_algorithms(warn_only=True)`. Disable
            (False) only if you need maximum throughput at the cost of
            bit-wise reproducibility.

    Returns:
        The seed that was applied (convenient for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:  # torch < 1.8 without warn_only support
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Random RNGs inside each DataLoader worker process.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))

    Args:
        worker_id: id assigned by PyTorch to the worker process.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_seed_info(seed: Optional[int] = None) -> dict:
    """Return a small dict with seed + determinism flags for experiment logs."""
    return {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
