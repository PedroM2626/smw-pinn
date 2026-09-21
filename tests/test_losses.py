"""
test_losses.py
Rigorous unit tests for physical PINN loss functions.
"""

import pytest
import torch
from src.losses.physics_losses import (
    DiscreteKinematicsLoss,
    VelocityBoundsLoss,
    GroundContactConsistencyLoss,
    CompositePINNLoss,
)


def test_discrete_kinematics_loss_zero_when_exact():
    """When predicted displacement strictly satisfies dx = vx/16 and dy = vy/16, loss must be zero."""
    kin_loss = DiscreteKinematicsLoss(subpixels_per_pixel=16.0)

    # Current state: X=100.0, Y=200.0, vx=32.0 (2 pixels), vy=-48.0 (-3 pixels)
    curr_state = torch.tensor([[100.0, 200.0, 32.0, -48.0]], dtype=torch.float32)
    # Exact next state: X_next = 100 + 2 = 102.0, Y_next = 200 - 3 = 197.0
    pred_next = torch.tensor([[102.0, 197.0, 32.0, -45.0]], dtype=torch.float32)

    loss = kin_loss(curr_state, pred_next)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_discrete_kinematics_loss_penalizes_drift():
    """When the model predicts coordinates violating velocity, loss must be strictly positive."""
    kin_loss = DiscreteKinematicsLoss(subpixels_per_pixel=16.0)

    curr_state = torch.tensor([[100.0, 200.0, 32.0, 0.0]], dtype=torch.float32)
    # Predicted position with drift (e.g., moved 10 pixels instead of 2)
    pred_next = torch.tensor([[110.0, 200.0, 32.0, 0.0]], dtype=torch.float32)

    loss = kin_loss(curr_state, pred_next)
    # res_x = (110 - 100) - (32/16) = 10 - 2 = 8. Loss = 8^2 = 64
    assert torch.isclose(loss, torch.tensor(64.0), atol=1e-5)


def test_velocity_bounds_loss():
    """Verifies that velocities exceeding SMW structural maximums produce penalties."""
    bounds_loss = VelocityBoundsLoss(max_vx=72.0, terminal_vy=64.0, min_vy=-80.0)

    # Within bounds
    valid_state = torch.tensor([[0.0, 0.0, 48.0, 30.0]], dtype=torch.float32)
    assert torch.isclose(bounds_loss(valid_state), torch.tensor(0.0), atol=1e-6)

    # Out of bounds: vx = 82 (+10 excess), vy = 74 (+10 fall excess)
    invalid_state = torch.tensor([[0.0, 0.0, 82.0, 74.0]], dtype=torch.float32)
    loss = bounds_loss(invalid_state)
    # excess_vx = 10, excess_vy = 10 -> loss = 10^2 + 10^2 = 200
    assert torch.isclose(loss, torch.tensor(200.0), atol=1e-5)


def test_ground_contact_loss():
    """Verifies that downward vertical velocities while grounded without jump are penalized."""
    contact_loss = GroundContactConsistencyLoss()

    # On ground (c_ground=1.0), no jump (jump=0.0), but predicted downward vy = 16.0
    curr_state = torch.tensor([[100.0, 200.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    action = torch.tensor([[0.0]], dtype=torch.float32)
    pred_state = torch.tensor([[100.0, 200.0, 0.0, 16.0]], dtype=torch.float32)

    loss = contact_loss(curr_state, action, pred_state)
    assert torch.isclose(loss, torch.tensor(256.0), atol=1e-5)


def test_composite_pinn_loss_gradients():
    """Verifies that composite PINN loss produces valid non-zero gradients across backpropagation."""
    loss_fn = CompositePINNLoss(lambda_kin=1.0, lambda_bound=1.0, lambda_contact=1.0)

    curr_state = torch.tensor([[100.0, 200.0, 16.0, -32.0, 0.0]], requires_grad=False)
    action = torch.tensor([[1.0]], requires_grad=False)
    target_state = torch.tensor([[101.0, 198.0, 16.0, -29.0, 0.0]], requires_grad=False)

    pred_state = torch.tensor(
        [[103.0, 195.0, 20.0, -25.0, 0.0]],
        requires_grad=True,
    )

    loss, metrics = loss_fn(curr_state, action, pred_state, target_state)
    loss.backward()

    assert pred_state.grad is not None
    assert not torch.isnan(pred_state.grad).any()
    assert metrics["loss_total"] > 0
