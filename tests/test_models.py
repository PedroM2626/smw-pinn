"""
test_models.py
Automated unit tests for neural architectures:
MLP, LSTM, Soft PINN, and Hard-Residual PINN.
"""

import pytest
import torch
from src.models import (
    StatisticalMLPDynamics,
    StatisticalLSTMDynamics,
    SoftPINNDynamics,
    HardResidualPINNDynamics,
)


def test_statistical_mlp_forward_and_backward():
    B, state_dim, action_dim = 16, 8, 6
    model = StatisticalMLPDynamics(state_dim=state_dim, action_dim=action_dim)

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)
    assert pred.shape == (B, state_dim)

    loss = pred.sum()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_statistical_lstm_forward_and_backward():
    B, T, state_dim, action_dim = 8, 10, 8, 6
    model = StatisticalLSTMDynamics(state_dim=state_dim, action_dim=action_dim)

    state_seq = torch.randn(B, T, state_dim)
    action_seq = torch.randn(B, T, action_dim)

    pred_seq, (hn, cn) = model(state_seq, action_seq)
    assert pred_seq.shape == (B, T, state_dim)

    loss = pred_seq.sum()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_soft_pinn_forward():
    B, state_dim, action_dim = 16, 8, 6
    model = SoftPINNDynamics(state_dim=state_dim, action_dim=action_dim)

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)
    assert pred.shape == (B, state_dim)


def test_hard_residual_pinn_exact_kinematic_guarantee():
    """
    Hard Residual PINN must STRICTLY satisfy:
        hat_X = X_t + hat_vx / 16.0
        hat_Y = Y_t + hat_vy / 16.0
    with zero kinematic integration residual.
    """
    B, state_dim, action_dim = 16, 8, 6
    model = HardResidualPINNDynamics(state_dim=state_dim, action_dim=action_dim)

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)

    # Exact kinematic guarantee by structural construction
    expected_x = state[:, 0] + (pred[:, 2] / 16.0)
    expected_y = state[:, 1] + (pred[:, 3] / 16.0)

    assert torch.allclose(pred[:, 0], expected_x, atol=1e-6)
    assert torch.allclose(pred[:, 1], expected_y, atol=1e-6)

    # Velocity saturation clamping limits
    assert (pred[:, 2] <= 72.0 + 1e-5).all() and (pred[:, 2] >= -72.0 - 1e-5).all()
    assert (pred[:, 3] <= 64.0 + 1e-5).all() and (pred[:, 3] >= -80.0 - 1e-5).all()
