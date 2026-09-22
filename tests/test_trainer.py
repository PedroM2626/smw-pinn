"""Tests for the unified DynamicsTrainer: convergence, checkpointing, early stop."""

import os

import numpy as np
import torch

from src.environment.dataset_loader import create_dataloaders
from src.models import StatisticalMLPDynamics
from src.training.trainer import DynamicsTrainer


def _toy_data(n=256):
    rng = np.random.default_rng(7)
    states = rng.normal(0, 5, size=(n, 8)).astype(np.float32)
    actions = rng.integers(0, 2, size=(n, 6)).astype(np.float32)
    next_states = (states + 0.1 * rng.normal(size=(n, 8))).astype(np.float32)
    return {
        "train_states": states[:200],
        "train_actions": actions[:200],
        "train_next_states": next_states[:200],
        "val_states": states[200:],
        "val_actions": actions[200:],
        "val_next_states": next_states[200:],
        "test_states": states[200:],
        "test_actions": actions[200:],
        "test_next_states": next_states[200:],
    }


def test_fit_reduces_loss_and_saves_best(tmp_path):
    device = torch.device("cpu")
    data = _toy_data()
    train_loader, val_loader, _ = create_dataloaders(data, batch_size=32, seed=0)
    model = StatisticalMLPDynamics(hidden_dims=[32, 32])
    trainer = DynamicsTrainer(
        model=model,
        model_type="mlp",
        device=device,
        learning_rate=1e-2,
        save_dir=str(tmp_path),
    )
    hist = trainer.fit(train_loader, val_loader, epochs=5, patience=10, verbose=False)
    assert len(hist["train_loss"]) == 5
    assert hist["train_loss"][-1] <= hist["train_loss"][0] + 1e-6
    assert all(np.isfinite(hist["train_loss"]))
    assert os.path.exists(os.path.join(str(tmp_path), "mlp_best.pt"))


def test_early_stopping_triggers(tmp_path):
    device = torch.device("cpu")
    data = _toy_data()
    train_loader, val_loader, _ = create_dataloaders(data, batch_size=32, seed=0)
    model = StatisticalMLPDynamics(hidden_dims=[32, 32])
    trainer = DynamicsTrainer(model=model, model_type="mlp", device=device, save_dir=str(tmp_path))
    # Frozen validation loss never improves -> must stop before max epochs.
    trainer.evaluate = lambda loader: {
        "val_loss_total": 1.0,
        "val_loss_data": 1.0,
        "val_loss_kinematics": 0.0,
    }
    hist = trainer.fit(train_loader, val_loader, epochs=10, patience=1, verbose=False)
    assert len(hist["train_loss"]) < 10


def test_evaluate_returns_finite_metrics(tmp_path):
    device = torch.device("cpu")
    data = _toy_data()
    _, val_loader, _ = create_dataloaders(data, batch_size=32, seed=0)
    trainer = DynamicsTrainer(
        model=StatisticalMLPDynamics(hidden_dims=[16]),
        model_type="mlp",
        device=device,
        save_dir=str(tmp_path),
    )
    metrics = trainer.evaluate(val_loader)
    assert np.isfinite(metrics["val_loss_data"])
    assert np.isfinite(metrics["val_loss_kinematics"])
