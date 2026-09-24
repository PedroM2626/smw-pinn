"""
test_fno.py
Unit tests for the Fourier Neural Operator dynamics baseline (README 10.42):
shapes, gradients, spectral-layer mode handling, interpolated field decoding,
and integration with the unified DynamicsTrainer.
"""

import numpy as np
import torch

from src.environment.dataset_loader import create_dataloaders
from src.models import FNODynamics
from src.models.fno import SpectralConv1d
from src.training.trainer import DynamicsTrainer


def test_spectral_conv1d_shapes_and_modes_clamp():
    spectral = SpectralConv1d(width=8, modes=6)
    v = torch.randn(4, 14, 8)
    out = spectral(v)
    assert out.shape == (4, 14, 8)
    assert not out.is_complex()

    # More modes than the rfft spectrum carries must clamp safely (N=6 grid
    # yields only 4 rfft bins; requesting 6 modes must not raise).
    tiny = SpectralConv1d(width=8, modes=6)
    assert tiny(torch.randn(2, 6, 8)).shape == (2, 6, 8)


def test_fno_forward_and_backward():
    B, state_dim, action_dim = 16, 8, 6
    model = FNODynamics(state_dim=state_dim, action_dim=action_dim, width=16, modes=4, n_layers=2)

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)
    assert pred.shape == (B, state_dim)
    assert torch.isfinite(pred).all()

    pred.sum().backward()
    assert model.output_bias.grad is not None
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_fno_custom_query_coordinates():
    """The decoded field can be queried at arbitrary coordinates in [0, 1]."""
    B, state_dim, action_dim = 4, 8, 6
    model = FNODynamics(width=16, modes=4, n_layers=1)
    model.eval()

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    dense = torch.linspace(0.0, 1.0, 25)
    out = model(state, action, query_coords=dense)
    assert out.shape == (B, 25)

    # Explicit canonical grid differs from the default call only by b0.
    canonical = torch.linspace(0.0, 1.0, state_dim)
    with torch.no_grad():
        explicit = model(state, action, query_coords=canonical)
        default = model(state, action)
        bias = model.output_bias.unsqueeze(0)
    assert torch.allclose(default, explicit + bias, atol=1e-6)


def test_fno_deterministic_in_eval_mode():
    model = FNODynamics(width=16, modes=4)
    model.eval()
    state = torch.randn(4, 8)
    action = torch.randn(4, 6)
    with torch.no_grad():
        first = model(state, action)
        second = model(state, action)
    assert torch.equal(first, second)


def test_fno_integrates_with_dynamics_trainer(tmp_path):
    rng = np.random.default_rng(5)
    states = rng.normal(0, 5, size=(128, 8)).astype(np.float32)
    actions = rng.integers(0, 2, size=(128, 6)).astype(np.float32)
    next_states = (states + 0.1 * rng.normal(size=(128, 8))).astype(np.float32)
    data = {
        "train_states": states[:100],
        "train_actions": actions[:100],
        "train_next_states": next_states[:100],
        "val_states": states[100:],
        "val_actions": actions[100:],
        "val_next_states": next_states[100:],
        "test_states": states[100:],
        "test_actions": actions[100:],
        "test_next_states": next_states[100:],
    }
    train_loader, val_loader, _ = create_dataloaders(data, batch_size=32, seed=0)
    model = FNODynamics(width=16, modes=4, n_layers=1)
    trainer = DynamicsTrainer(
        model=model,
        model_type="fno",
        device=torch.device("cpu"),
        learning_rate=1e-2,
        save_dir=str(tmp_path),
    )
    hist = trainer.fit(train_loader, val_loader, epochs=3, patience=5, verbose=False)
    assert len(hist["train_loss"]) == 3
    assert all(np.isfinite(hist["train_loss"]))
    assert (tmp_path / "fno_best.pt").exists()
