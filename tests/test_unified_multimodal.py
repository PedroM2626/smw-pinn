"""
test_unified_multimodal.py
Unit tests for the Unified Multimodal PINN architecture.
"""

import torch

from src.models.pinn_unified_multimodal import UnifiedMultimodalPINNDynamics


def test_unified_multimodal_forward_without_patch():
    model = UnifiedMultimodalPINNDynamics()
    model.eval()

    b = 4
    mario_8d = torch.randn(b, 8)
    action = torch.zeros(b, 6)
    action[:, 5] = 1.0  # RIGHT

    out = model(mario_8d, action)
    assert "next_mario" in out
    assert out["next_mario"].shape == (b, 8)


def test_unified_multimodal_forward_with_tilemap_and_hazard():
    model = UnifiedMultimodalPINNDynamics()
    model.eval()

    b = 2
    mario_8d = torch.tensor(
        [
            [16.0, 336.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [50.0, 300.0, 16.0, -20.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    action = torch.zeros(b, 6)
    action[:, 0] = 1.0  # B (Jump)

    hazard_4d = torch.tensor(
        [
            [80.0, 0.0, -16.0, 1.0],
            [999.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    tile_patch = torch.randint(0, 4, (b, 7, 7), dtype=torch.long)

    out = model(mario_8d, action, hazard_4d=hazard_4d, tilemap_patch=tile_patch)

    assert "next_mario" in out
    assert "next_hazard" in out
    assert out["next_mario"].shape == (b, 8)
    assert out["next_hazard"].shape == (b, 4)

    # Exact Kinematic Guarantee verification
    next_mario = out["next_mario"]
    expected_x = mario_8d[:, 0] + next_mario[:, 2] / 16.0
    expected_y = mario_8d[:, 1] + next_mario[:, 3] / 16.0
    assert torch.allclose(next_mario[:, 0], expected_x, atol=1e-5)
    assert torch.allclose(next_mario[:, 1], expected_y, atol=1e-5)

    # Relative hazard kinematics verification for active hazard (b=0)
    next_hazard = out["next_hazard"]
    next_vx_h = next_hazard[0, 2]
    next_vx_m = next_mario[0, 2]
    expected_delta_x = hazard_4d[0, 0] + (next_vx_h - next_vx_m) / 16.0
    assert torch.allclose(next_hazard[0, 0], expected_delta_x, atol=1e-5)
