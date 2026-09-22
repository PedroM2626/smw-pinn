"""
test_tilemap.py
Unit tests verifying WRAM level tilemap patch extraction ($7E:C800)
and the Tilemap-Conditioned Physics-Informed Neural Network (TilemapPINNDynamics).
"""

import numpy as np
import pytest
import torch
from src.environment.snes_emulator import SnesLibretroEmulator
from src.models.tilemap_pinn import TilemapEncoder, TilemapPINNDynamics


def test_wram_tilemap_extraction():
    emu = SnesLibretroEmulator("src/environment/bin/snes9x_libretro.dll")
    emu.load_rom("data/raw/smw_usa.sfc")
    with open("data/raw/smw_yoshi_island_1.state", "rb") as f:
        emu.load_state(f.read())
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()

    s = emu.get_smw_state()
    # Mario is initially at X=16, Y=336 standing on ground
    patch = emu.get_local_tilemap_patch(s["x"], s["y"], radius=3)

    assert patch.shape == (7, 7), f"Expected 7x7 patch, got {patch.shape}"
    assert patch.dtype == np.int64

    # Center row is row index 3 (dy=0, Mario's position)
    # Below Mario (row index 4, 5, 6 where dy > 0) must contain solid ground (class 1)
    ground_below = patch[4:, 3]
    assert (ground_below == 1).any(), f"Expected solid ground below Mario, got {patch}"

    emu.close()


def test_tilemap_encoder_forward():
    encoder = TilemapEncoder(num_tile_classes=4, embed_dim=8, out_features=32)
    dummy_patch = torch.randint(0, 4, (4, 7, 7), dtype=torch.long)
    features = encoder(dummy_patch)

    assert features.shape == (4, 32)
    assert not torch.isnan(features).any()


def test_tilemap_pinn_exact_kinematics_and_gradients():
    model = TilemapPINNDynamics(kinematic_dim=8, action_dim=6, tile_features_dim=32, hidden_dim=64)

    batch_size = 8
    dummy_kinematics = torch.randn(batch_size, 8)
    dummy_kinematics[:, 0] = 100.0  # X
    dummy_kinematics[:, 1] = 300.0  # Y
    dummy_kinematics[:, 2] = 16.0   # vx
    dummy_kinematics[:, 3] = 0.0    # vy

    dummy_patch = torch.randint(0, 4, (batch_size, 7, 7), dtype=torch.long)
    dummy_action = torch.zeros(batch_size, 6)
    dummy_action[:, 5] = 1.0  # RIGHT

    next_kinematics, feat = model(dummy_kinematics, dummy_patch, dummy_action)

    assert next_kinematics.shape == (batch_size, 8)
    assert feat.shape == (batch_size, 32)

    # Mathematical Kinematic Consistency Check:
    # Delta X must strictly equal next_vx / 16.0
    delta_x = next_kinematics[:, 0] - dummy_kinematics[:, 0]
    expected_delta_x = next_kinematics[:, 2] / 16.0
    drift_x = (delta_x - expected_delta_x).abs().max().item()

    delta_y = next_kinematics[:, 1] - dummy_kinematics[:, 1]
    expected_delta_y = next_kinematics[:, 3] / 16.0
    drift_y = (delta_y - expected_delta_y).abs().max().item()

    assert drift_x < 1e-4, f"Kinematic consistency constraint violated in X: drift={drift_x}"
    assert drift_y < 1e-4, f"Kinematic consistency constraint violated in Y: drift={drift_y}"

    # Gradient flow test
    loss = next_kinematics.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
