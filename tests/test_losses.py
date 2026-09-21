"""
test_losses.py
Testes unitários rigorosos para as funções de perda físicas (PINN).
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
    """Quando o deslocamento predito respeita exatamente dx = vx/16 e dy = vy/16, o loss deve ser zero."""
    kin_loss = DiscreteKinematicsLoss(subpixels_per_pixel=16.0)

    # Estado atual: X=100.0, Y=200.0, vx=32.0 (2 pixels), vy=-48.0 (-3 pixels)
    curr_state = torch.tensor([[100.0, 200.0, 32.0, -48.0]], dtype=torch.float32)
    # Próximo estado exato: X_next = 100 + 2 = 102.0, Y_next = 200 - 3 = 197.0
    pred_next = torch.tensor([[102.0, 197.0, 32.0, -45.0]], dtype=torch.float32)

    loss = kin_loss(curr_state, pred_next)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_discrete_kinematics_loss_penalizes_drift():
    """Quando o modelo prevê posições que violam a velocidade, o loss deve ser estritamente positivo."""
    kin_loss = DiscreteKinematicsLoss(subpixels_per_pixel=16.0)

    curr_state = torch.tensor([[100.0, 200.0, 32.0, 0.0]], dtype=torch.float32)
    # Posição predita com drift (ex: moveu 10 pixels em vez de 2)
    pred_next = torch.tensor([[110.0, 200.0, 32.0, 0.0]], dtype=torch.float32)

    loss = kin_loss(curr_state, pred_next)
    # res_x = (110 - 100) - (32/16) = 10 - 2 = 8. Loss = 8^2 = 64
    assert torch.isclose(loss, torch.tensor(64.0), atol=1e-5)


def test_velocity_bounds_loss():
    """Verifica se velocidades acima dos limites máximos de SMW geram penalidade."""
    bounds_loss = VelocityBoundsLoss(max_vx=72.0, terminal_vy=64.0, min_vy=-80.0)

    # Dentro dos limites
    valid_state = torch.tensor([[0.0, 0.0, 48.0, 30.0]], dtype=torch.float32)
    assert torch.isclose(bounds_loss(valid_state), torch.tensor(0.0), atol=1e-6)

    # Fora dos limites: vx = 82 (+10 excesso), vy = 74 (+10 excesso de queda)
    invalid_state = torch.tensor([[0.0, 0.0, 82.0, 74.0]], dtype=torch.float32)
    loss = bounds_loss(invalid_state)
    # excess_vx = 10, excess_vy = 10 -> loss = 10^2 + 10^2 = 200
    assert torch.isclose(loss, torch.tensor(200.0), atol=1e-5)


def test_ground_contact_loss():
    """Verifica se no chão sem salto, velocidades verticais são penalizadas."""
    contact_loss = GroundContactConsistencyLoss()

    # No chão (c_ground=1.0), sem salto (jump=0.0), mas com vy descendente predita de 16.0
    curr_state = torch.tensor([[100.0, 200.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    action = torch.tensor([[0.0]], dtype=torch.float32)
    pred_state = torch.tensor([[100.0, 200.0, 0.0, 16.0]], dtype=torch.float32)

    loss = contact_loss(curr_state, action, pred_state)
    assert torch.isclose(loss, torch.tensor(256.0), atol=1e-5)


def test_composite_pinn_loss_gradients():
    """Verifica se o loss composto produz gradientes válidos e retropropaga corretamente."""
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
