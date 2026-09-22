"""
test_set_multi_entity.py
Unit tests for SetMultiEntityPINNDynamics architecture:
- Output tensor shapes
- Exact 0.0% analytical kinematic consistency for Mario and dynamic entities
- Permutation invariance / equivariance under sprite slot reordering
- Graceful handling of zero active entities
"""

import pytest
import torch

from src.models.pinn_set_multi_entity import SetMultiEntityPINNDynamics


@pytest.fixture
def set_model():
    torch.manual_seed(42)
    return SetMultiEntityPINNDynamics(max_entities=4, hidden_dim=32, num_heads=2)


def test_set_multi_entity_shapes_and_forward(set_model):
    batch_size = 8
    num_entities = 4
    mario_state = torch.randn(batch_size, 8)
    # [delta_x, delta_y, vx, vy, active]
    entities = torch.randn(batch_size, num_entities, 5)
    entities[..., 4] = (torch.rand(batch_size, num_entities) > 0.3).float()
    actions = torch.randint(0, 2, (batch_size, 6)).float()

    next_mario, next_entities, context = set_model(mario_state, entities, actions)

    assert next_mario.shape == (batch_size, 8)
    assert next_entities.shape == (batch_size, num_entities, 5)
    assert context.shape == (batch_size, 32)


def test_set_multi_entity_exact_kinematic_guarantee(set_model):
    batch_size = 16
    num_entities = 4
    mario_state = torch.randn(batch_size, 8)
    entities = torch.randn(batch_size, num_entities, 5)
    entities[..., 4] = 1.0  # all entities active
    actions = torch.randint(0, 2, (batch_size, 6)).float()

    next_mario, next_entities, _ = set_model(mario_state, entities, actions)

    # Mario analytical check
    expected_mario_x = mario_state[:, 0] + (next_mario[:, 2] / 16.0)
    mario_x_res = torch.max(torch.abs(next_mario[:, 0] - expected_mario_x)).item()
    assert mario_x_res < 1e-5, f"Mario kinematic residual: {mario_x_res}"

    # Entity analytical relative check for active entities
    expected_entity_dx = entities[..., 0] + (next_entities[..., 2] - next_mario[:, 2].unsqueeze(1)) / 16.0
    entity_dx_res = torch.max(torch.abs(next_entities[..., 0] - expected_entity_dx)).item()
    assert entity_dx_res < 1e-5, f"Entity relative kinematic residual: {entity_dx_res}"


def test_set_multi_entity_permutation_invariance(set_model):
    """
    If entity 0 and entity 1 are swapped in the input,
    Mario's next state should be identical (since Mario's physics is decoupled from slot order),
    and the predicted next entities should be swapped correspondingly.
    """
    mario_state = torch.randn(2, 8)
    # 2 active entities
    e0 = torch.tensor([50.0, 10.0, -16.0, 0.0, 1.0])
    e1 = torch.tensor([-30.0, -5.0, 8.0, 0.0, 1.0])
    e_dummy = torch.zeros(5)

    entities_orig = torch.stack([e0, e1, e_dummy, e_dummy]).unsqueeze(0).repeat(2, 1, 1)
    entities_perm = torch.stack([e1, e0, e_dummy, e_dummy]).unsqueeze(0).repeat(2, 1, 1)

    actions = torch.zeros(2, 6)
    actions[:, 5] = 1.0  # pressing RIGHT

    next_mario_orig, next_entities_orig, _ = set_model(mario_state, entities_orig, actions)
    next_mario_perm, next_entities_perm, _ = set_model(mario_state, entities_perm, actions)

    # Mario predictions must be identical
    mario_diff = torch.max(torch.abs(next_mario_orig - next_mario_perm)).item()
    assert mario_diff < 1e-5, f"Mario state changed under entity permutation: {mario_diff}"

    # Entity predictions should be swapped
    diff_slot0_slot1 = torch.max(torch.abs(next_entities_orig[:, 0] - next_entities_perm[:, 1])).item()
    diff_slot1_slot0 = torch.max(torch.abs(next_entities_orig[:, 1] - next_entities_perm[:, 0])).item()
    assert diff_slot0_slot1 < 1e-5
    assert diff_slot1_slot0 < 1e-5


def test_set_multi_entity_zero_active_entities(set_model):
    """Verifies that model runs stably when 0 entities are active without NaN."""
    batch_size = 4
    mario_state = torch.randn(batch_size, 8)
    entities = torch.zeros(batch_size, 4, 5)  # all active_i = 0.0
    actions = torch.randint(0, 2, (batch_size, 6)).float()

    next_mario, next_entities, context = set_model(mario_state, entities, actions)

    assert not torch.isnan(next_mario).any()
    assert not torch.isnan(next_entities).any()
    assert not torch.isnan(context).any()
    assert torch.all(context == 0.0)
