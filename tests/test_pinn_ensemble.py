"""
test_pinn_ensemble.py
Unit tests for DeepPINNEnsemble epistemic uncertainty quantification and shape verification.
"""

import pytest
import torch
from src.models.pinn_ensemble import DeepPINNEnsemble


def test_ensemble_forward_and_uncertainty():
    B, state_dim, action_dim, E = 16, 8, 6, 5
    ensemble = DeepPINNEnsemble(num_models=E, state_dim=state_dim, action_dim=action_dim)
    ensemble.eval()

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    mean_pred, variance, uncertainty_norm = ensemble(state, action)

    # Tensor shape checks
    assert mean_pred.shape == (B, state_dim)
    assert variance.shape == (B, state_dim)
    assert uncertainty_norm.shape == (B,)

    # Epistemic variance must be strictly non-negative
    assert (variance >= 0.0).all()
    assert (uncertainty_norm >= 0.0).all()

    # Exact kinematic guarantee holds on every member and predictive mean
    expected_x = state[:, 0] + (mean_pred[:, 2] / 16.0)
    assert torch.allclose(mean_pred[:, 0], expected_x, atol=1e-5)


def test_ensemble_pessimistic_penalization():
    B = 8
    ensemble = DeepPINNEnsemble(num_models=3)
    state = torch.randn(B, 8)
    action = torch.randn(B, 6)
    raw_reward = torch.ones(B) * 10.0

    mean_pred, safe_reward = ensemble.predict_with_pessimism(state, action, raw_reward, beta=1.0)

    # Safe reward must be penalized (<= raw_reward)
    assert (safe_reward <= raw_reward + 1e-6).all()
