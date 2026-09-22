"""Tests for orphan connections: tilemap wrapper, sprite rows, gravity ID."""

import numpy as np
import pytest
import torch

from src.environment.sprite_sets import sprites_to_entity_rows
from src.evaluation.spatial_holdout_benchmark import far_region_mask
from src.models.pinn_gravity import GravityIdentifiedPINNDynamics
from src.models.pinn_set_multi_entity import SetMultiEntityPINNDynamics
from src.models.pinn_unified_multimodal import UnifiedMultimodalPINNDynamics
from src.planning.tilemap_mpc import TilemapMPCWrapper


def test_wrapper_requires_patch_and_shapes():
    w = TilemapMPCWrapper()
    s = torch.randn(4, 8)
    a = torch.zeros(4, 6)
    with pytest.raises(RuntimeError):
        w(s, a)
    with pytest.raises(ValueError):
        w.set_patch(np.zeros((5, 5), dtype=np.int64))
    w.set_patch(np.zeros((7, 7), dtype=np.int64))
    out = w(s, a)
    assert out.shape == (4, 8)
    assert torch.allclose(out[:, 0], s[:, 0] + out[:, 2] / 16.0, atol=1e-4)


def test_sprites_to_entity_rows():
    sprites = [
        {"slot": 3, "x": 200.0, "y": 300.0, "vx": -16.0, "vy": 0.0},
        {"slot": 1, "x": 100.0, "y": 320.0, "vx": 0.0, "vy": 0.0},
    ]
    rows = sprites_to_entity_rows(sprites, mario_x=150.0, mario_y=310.0, max_entities=12)
    assert rows.shape == (12, 5)
    # Slot-ordered: slot 1 first.
    assert rows[0, 0] == -50.0 and rows[0, 4] == 1.0
    assert rows[1, 0] == 50.0
    assert (rows[2:] == 0.0).all()
    assert sprites_to_entity_rows([], 0.0, 0.0).sum() == 0.0


def test_unified_joint_step_smoke():
    torch.manual_seed(0)
    model = UnifiedMultimodalPINNDynamics()
    B = 4
    s = torch.randn(B, 8)
    a = torch.zeros(B, 6)
    patch = torch.randint(0, 4, (B, 7, 7))
    out = model(s, a, tilemap_patch=patch)
    assert out["next_mario"].shape == (B, 8)
    assert out["contact_logits"].shape == (B, 4)
    hz = torch.tensor([[60.0, 0.0, -16.0, 1.0]] * B)
    out2 = model(s, a, hazard_4d=hz)
    assert out2["next_hazard"].shape == (B, 4)
    loss = out["next_mario"].sum() + out["contact_logits"].sum() + out2["next_hazard"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and not torch.isnan(g).any() for g in grads)


def test_set_model_masked_step_smoke():
    torch.manual_seed(1)
    model = SetMultiEntityPINNDynamics(max_entities=4)
    m = torch.randn(2, 8)
    ent = torch.zeros(2, 4, 5)
    ent[0, 0] = torch.tensor([30.0, 0.0, -16.0, 0.0, 1.0])
    a = torch.zeros(2, 6)
    pred_m, pred_e, _ = model(m, ent, a)
    assert pred_m.shape == (2, 8) and pred_e.shape == (2, 4, 5)
    loss = pred_m.sum() + pred_e.sum()
    loss.backward()
    assert model.entity_encoder[0].weight.grad is not None


def test_gravity_exact_kinematics_and_identifiability():
    torch.manual_seed(0)
    model = GravityIdentifiedPINNDynamics(hidden_dims=[16])
    s = torch.randn(8, 8)
    a = torch.zeros(8, 6)
    a[:, 0] = 1.0
    pred = model(s, a)
    assert torch.allclose(pred[:, 0], s[:, 0] + pred[:, 2] / 16.0, atol=1e-6)
    assert torch.allclose(pred[:, 1], s[:, 1] + pred[:, 3] / 16.0, atol=1e-6)
    assert (pred[:, 2].abs() <= 72.0 + 1e-5).all()
    assert (pred[:, 3] <= 64.0 + 1e-5).all() and (pred[:, 3] >= -80.0 - 1e-5).all()

    # Identifiability: freeze residual to zero, recover g_hold=3 from data.
    model.g_hold.data.fill_(5.0)
    for p in model.force_net.parameters():
        p.requires_grad_(False)
        p.zero_()
    # Synthetic held-jump arc with TRUE g_hold = 3.0.
    vy = torch.arange(0, -60, -6, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
    batch = vy.shape[0]
    states = torch.zeros(batch, 8)
    states[:, 3] = vy[:, 0]
    acts = torch.zeros(batch, 6)
    acts[:, 0] = 1.0
    targets = torch.clamp(vy[:, 0] + 3.0, -80.0, 64.0)
    opt = torch.optim.Adam([model.g_hold], lr=0.2)
    for _ in range(300):
        opt.zero_grad()
        pred = model(states, acts)
        loss = torch.mean((pred[:, 3] - targets) ** 2)
        loss.backward()
        opt.step()
    assert abs(model.g_hold.item() - 3.0) < 0.75


def test_far_region_mask():
    states = np.array([[100.0, 0, 0, 0, 0, 0, 0, 0], [800.0, 0, 0, 0, 0, 0, 0, 0]])
    mask = far_region_mask(states, threshold=700.0)
    assert mask.tolist() == [False, True]
