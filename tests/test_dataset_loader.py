"""Tests for episodic splitting, leakage guards and seeded shuffling."""

import numpy as np
import pytest
import torch

from src.environment.dataset_loader import (
    create_dataloaders,
    load_and_preprocess_data,
)


def _write_npz(path, n_episodes=6, per_ep=50):
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 400, size=(n_episodes * per_ep, 8)).astype(np.float32)
    actions = rng.integers(0, 2, size=(n_episodes * per_ep, 6)).astype(np.float32)
    next_states = states + rng.normal(0, 1, size=states.shape).astype(np.float32)
    next_states[:, 1] = np.clip(next_states[:, 1], 0, 500)
    states[:, 1] = np.clip(states[:, 1], 0, 500)
    episodes = np.repeat(np.arange(n_episodes), per_ep).astype(np.int32)
    np.savez_compressed(
        path, states=states, actions=actions, next_states=next_states, episodes=episodes
    )


def test_splits_are_disjoint_and_cover_all(tmp_path):
    p = tmp_path / "data.npz"
    _write_npz(p)
    d = load_and_preprocess_data(dataset_path=str(p), seed=42)
    train_eps = set(np.unique(d["train_episodes"]).tolist())
    val_eps = set(np.unique(d["val_episodes"]).tolist())
    test_eps = set(np.unique(d["test_episodes"]).tolist())
    assert train_eps.isdisjoint(val_eps)
    assert train_eps.isdisjoint(test_eps)
    assert val_eps.isdisjoint(test_eps)
    total = len(d["train_states"]) + len(d["val_states"]) + len(d["test_states"])
    assert total == 6 * 50


def test_invalid_ratios_raise(tmp_path):
    p = tmp_path / "data.npz"
    _write_npz(p)
    with pytest.raises(ValueError):
        load_and_preprocess_data(dataset_path=str(p), train_ratio=0.8, val_ratio=0.3)


def test_low_episode_fallback_has_no_duplication(tmp_path):
    p = tmp_path / "data.npz"
    _write_npz(p, n_episodes=3, per_ep=30)
    d = load_and_preprocess_data(dataset_path=str(p), train_ratio=0.7, val_ratio=0.15, seed=1)
    assert len(d["val_states"]) > 0
    assert len(d["test_states"]) > 0
    assert set(np.unique(d["val_episodes"]).tolist()).isdisjoint(
        set(np.unique(d["test_episodes"]).tolist())
    )
    assert set(np.unique(d["train_episodes"]).tolist()).isdisjoint(
        set(np.unique(d["test_episodes"]).tolist())
    )


def test_too_few_episodes_raises(tmp_path):
    p = tmp_path / "data.npz"
    _write_npz(p, n_episodes=2, per_ep=30)
    with pytest.raises(ValueError):
        load_and_preprocess_data(dataset_path=str(p), train_ratio=0.7, val_ratio=0.15, seed=1)


def test_seeded_loader_is_deterministic(tmp_path):
    p = tmp_path / "data.npz"
    _write_npz(p)
    d = load_and_preprocess_data(dataset_path=str(p), seed=42)
    l1, _, _ = create_dataloaders(d, batch_size=16, seed=123)
    l2, _, _ = create_dataloaders(d, batch_size=16, seed=123)
    b1 = next(iter(l1))[0]
    b2 = next(iter(l2))[0]
    assert torch.equal(b1, b2)
