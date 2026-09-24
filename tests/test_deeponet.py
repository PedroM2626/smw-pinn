"""
test_deeponet.py
Unit tests for the DeepONet neural-operator dynamics baseline (README 10.41):
shapes, gradients, branch/trunk operator semantics, the canonical query grid,
and integration with the unified DynamicsTrainer.
"""

import numpy as np
import torch

from src.environment.dataset_loader import create_dataloaders
from src.models import DeepONetDynamics, PhysicsConstrainedDeepONetDynamics
from src.models.deeponet import _build_mlp
from src.training.trainer import DynamicsTrainer


def test_deeponet_forward_and_backward():
    B, state_dim, action_dim, latent = 16, 8, 6, 32
    model = DeepONetDynamics(
        state_dim=state_dim,
        action_dim=action_dim,
        branch_hidden_dims=[32, 32],
        trunk_hidden_dims=[32, 32],
        latent_dim=latent,
    )

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)
    assert pred.shape == (B, state_dim)

    loss = pred.sum()
    loss.backward()

    # Gradients must reach both subnets and the operator bias term.
    assert model.output_bias.grad is not None
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_deeponet_sensor_count_and_query_grid():
    state_dim, action_dim, latent = 8, 6, 16
    model = DeepONetDynamics(state_dim=state_dim, action_dim=action_dim, latent_dim=latent)

    # The input function is sampled at m = state_dim + action_dim sensors.
    assert model.num_sensors == state_dim + action_dim

    # The canonical trunk grid is state_dim coordinates spaced in [-1, 1].
    grid = model.canonical_query_coords
    assert grid.shape == (state_dim, 1)
    assert torch.isclose(grid[0, 0], torch.tensor(-1.0))
    assert torch.isclose(grid[-1, 0], torch.tensor(1.0))


def test_deeponet_custom_query_coordinates():
    """The trunk accepts arbitrary query coordinates (operator evaluation)."""
    B, state_dim, action_dim = 4, 8, 6
    model = DeepONetDynamics(state_dim=state_dim, action_dim=action_dim, latent_dim=16)
    model.eval()

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    # A denser grid than the canonical state_dim channels: interpolated
    # coordinates of the learned output field.
    dense = torch.linspace(-1.0, 1.0, 25).unsqueeze(-1)
    out = model(state, action, query_coords=dense)
    assert out.shape == (B, 25)

    # The canonical grid evaluated explicitly differs from the default call
    # only by the per-channel constant term b0.
    canonical = torch.linspace(-1.0, 1.0, state_dim).unsqueeze(-1)
    with torch.no_grad():
        explicit = model(state, action, query_coords=canonical)
        default = model(state, action)
        bias = model.output_bias.unsqueeze(0)
    assert torch.allclose(default, explicit + bias, atol=1e-6)


def test_deeponet_deterministic_in_eval_mode():
    model = DeepONetDynamics(latent_dim=16)
    model.eval()
    state = torch.randn(4, 8)
    action = torch.randn(4, 6)
    with torch.no_grad():
        first = model(state, action)
        second = model(state, action)
    assert torch.equal(first, second)


def test_build_mlp_structure():
    net = _build_mlp(in_dim=5, hidden_dims=[16, 16], out_dim=3)
    assert isinstance(net, torch.nn.Sequential)
    x = torch.randn(2, 5)
    assert net(x).shape == (2, 3)


def test_physics_constrained_deeponet_exact_kinematic_guarantee():
    """
    Physics-constrained DeepONet must STRICTLY satisfy the Section 4 identity:
        hat_X = X_t + hat_vx / 16.0
        hat_Y = Y_t + hat_vy / 16.0
    The operator learns only forces/contacts; integration is analytical, so the
    kinematic residual is identically zero as in the Hard Residual PINN.
    """
    B, state_dim, action_dim = 16, 8, 6
    model = PhysicsConstrainedDeepONetDynamics(
        state_dim=state_dim, action_dim=action_dim, latent_dim=24
    )

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)
    assert pred.shape == (B, state_dim)

    expected_x = state[:, 0] + (pred[:, 2] / 16.0)
    expected_y = state[:, 1] + (pred[:, 3] / 16.0)
    assert torch.allclose(pred[:, 0], expected_x, atol=1e-6)
    assert torch.allclose(pred[:, 1], expected_y, atol=1e-6)

    # Physical saturation clamping limits (Section 4 engine bounds).
    assert (pred[:, 2] <= 72.0 + 1e-5).all() and (pred[:, 2] >= -72.0 - 1e-5).all()
    assert (pred[:, 3] <= 64.0 + 1e-5).all() and (pred[:, 3] >= -80.0 - 1e-5).all()

    # Gradients flow through the clamp into both operator subnets.
    pred.sum().backward()
    assert model.output_bias.grad is not None
    for name, p in model.named_parameters():
        if "branch" in name or "trunk" in name:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_physics_constrained_deeponet_aux_layout():
    state_dim, action_dim = 8, 6
    model = PhysicsConstrainedDeepONetDynamics(state_dim=state_dim, action_dim=action_dim)
    # Residual outputs = 2 velocity increments + (state_dim - 4) contact flags.
    assert model.aux_dim == state_dim - 2
    assert model.residual_query_coords.shape == (state_dim - 2, 1)


def test_deeponet_integrates_with_dynamics_trainer(tmp_path):
    """The 'deeponet' model_type takes the plain supervised path of the trainer."""
    rng = np.random.default_rng(3)
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
    model = DeepONetDynamics(branch_hidden_dims=[32], trunk_hidden_dims=[32], latent_dim=24)
    trainer = DynamicsTrainer(
        model=model,
        model_type="deeponet",
        device=torch.device("cpu"),
        learning_rate=1e-2,
        save_dir=str(tmp_path),
    )
    hist = trainer.fit(train_loader, val_loader, epochs=3, patience=5, verbose=False)
    assert len(hist["train_loss"]) == 3
    assert all(np.isfinite(hist["train_loss"]))
    assert (tmp_path / "deeponet_best.pt").exists()
