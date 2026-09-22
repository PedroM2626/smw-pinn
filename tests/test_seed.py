"""
test_seed.py
Unit tests for the central reproducibility utility (src/utils/seed.py).
"""

import random

import pytest

torch = pytest.importorskip("torch")

from src.utils.seed import get_seed_info, seed_worker, set_global_seed  # noqa: E402


def test_set_global_seed_returns_seed():
    assert set_global_seed(123) == 123


def test_set_global_seed_is_reproducible():
    set_global_seed(42)
    a_torch = torch.randn(4, 4)
    import numpy as np

    a_np = np.random.rand(4)
    a_py = random.random()

    set_global_seed(42)
    assert torch.equal(torch.randn(4, 4), a_torch)
    assert (np.random.rand(4) == a_np).all()
    assert random.random() == a_py


def test_set_global_seed_changes_stream():
    set_global_seed(1)
    first = torch.randn(8)
    set_global_seed(2)
    second = torch.randn(8)
    assert not torch.equal(first, second)


def test_deterministic_flags_enabled():
    set_global_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_deterministic_flags_disabled():
    set_global_seed(0, deterministic=False)
    assert torch.backends.cudnn.benchmark is True
    # Restore deterministic default so later tests are not affected.
    set_global_seed(0, deterministic=True)


def test_seed_worker_runs():
    set_global_seed(7)
    seed_worker(0)
    seed_worker(3)


def test_get_seed_info_reports_seed():
    set_global_seed(99)
    info = get_seed_info(99)
    assert info["seed"] == 99
    assert info["python_hash_seed"] == "99"
    assert "torch_version" in info
