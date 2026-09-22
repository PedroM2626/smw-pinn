"""
test_pinn_invariant.py
Unit tests verifying strict spatial translation equivariance and kinematic guarantees
of TranslationInvariantPINNDynamics.
"""

import torch

from src.models.pinn_invariant import TranslationInvariantPINNDynamics


def test_spatial_translation_equivariance():
    """
    Mathematical Proof Verification:
    If the system is translated by arbitrary displacement C in X:
        hat_f(s + [C, 0, ...], a) must EQUAL hat_f(s, a) + [C, 0, ...]
    exactly to float32 machine epsilon.
    """
    B, state_dim, action_dim = 16, 8, 6
    model = TranslationInvariantPINNDynamics(state_dim=state_dim, action_dim=action_dim)
    model.eval()

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    # Base forward pass
    pred_base = model(state, action)

    # Apply arbitrary spatial shift C = 1500.0 pixels (e.g. middle of stage)
    shift_c = 1500.0
    shifted_state = state.clone()
    shifted_state[:, 0] += shift_c

    pred_shifted = model(shifted_state, action)

    # Check that predicted X shifted by exactly C
    expected_shifted_x = pred_base[:, 0] + shift_c
    assert torch.allclose(pred_shifted[:, 0], expected_shifted_x, atol=1e-5)

    # Check that predicted velocities and contact flags are 100% IDENTICAL
    assert torch.allclose(pred_shifted[:, 1:], pred_base[:, 1:], atol=1e-5)


def test_invariant_pinn_kinematic_guarantee():
    B, state_dim, action_dim = 8, 8, 6
    model = TranslationInvariantPINNDynamics(state_dim=state_dim, action_dim=action_dim)

    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim)

    pred = model(state, action)

    # Exact kinematic integration check
    expected_x = state[:, 0] + (pred[:, 2] / 16.0)
    expected_y = state[:, 1] + (pred[:, 3] / 16.0)

    assert torch.allclose(pred[:, 0], expected_x, atol=1e-6)
    assert torch.allclose(pred[:, 1], expected_y, atol=1e-6)
