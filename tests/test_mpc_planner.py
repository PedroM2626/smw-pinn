"""
test_mpc_planner.py
Unit tests for the GPU-vectorized Model Predictive Control (MPC) trajectory planner.
"""

import numpy as np
import pytest
import torch

from src.models import HardResidualPINNDynamics
from src.planning.mpc_planner import (
    ACTION_BUTTONS,
    ACTION_PRIMITIVES,
    ModelPredictiveController,
    TrajectoryObjective,
    action_vector_to_joypad,
)


def test_action_vector_to_joypad_matches_primitive_names():
    # The channel order must agree with the primitive definitions above.
    assert ACTION_BUTTONS == ("B", "Y", "UP", "DOWN", "LEFT", "RIGHT")
    joypad = action_vector_to_joypad(ACTION_PRIMITIVES["RUN_JUMP_RIGHT"])
    assert joypad == {
        "B": True,
        "Y": True,
        "UP": False,
        "DOWN": False,
        "LEFT": False,
        "RIGHT": True,
    }
    # Buttons accepted by the emulator's own mapping (no silent drops, no KeyError).
    assert set(joypad) <= {
        "B",
        "Y",
        "SELECT",
        "START",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "A",
        "X",
        "L",
        "R",
    }


def test_action_vector_to_joypad_thresholds_and_validation():
    assert action_vector_to_joypad([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])["Y"] is True
    assert action_vector_to_joypad([0.49] * 6) == dict.fromkeys(ACTION_BUTTONS, False)

    with pytest.raises(ValueError, match="6-D action vector"):
        action_vector_to_joypad([1.0, 0.0, 0.0])


def test_trajectory_objective_rewards_progress():
    obj = TrajectoryObjective(weight_progress=2.0, pit_penalty=500.0, death_y=400.0)

    # Initial state: X=100.0, Y=200.0, vx=0.0
    init_state = torch.zeros((2, 8))
    init_state[:, 0] = 100.0
    init_state[:, 1] = 200.0

    # Trajectory 0: advances to X=150, Y=200 (progress = +50, alive)
    # Trajectory 1: falls into pit Y=450 (death penalty)
    trajs = torch.zeros((2, 10, 8))
    trajs[0, :, 0] = torch.linspace(100, 150, 10)
    trajs[0, :, 1] = 200.0

    trajs[1, :, 0] = torch.linspace(100, 110, 10)
    trajs[1, :, 1] = torch.linspace(200, 460, 10)  # falls into pit

    rewards = obj.compute_trajectory_rewards(init_state, trajs)
    assert rewards[0] > rewards[1]
    assert rewards[0] > 0
    assert rewards[1] < 0


def test_mpc_planner_outputs_valid_action():
    device = torch.device("cpu")
    model = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    planner = ModelPredictiveController(
        world_model=model,
        device=device,
        horizon=10,
        num_candidates=32,
        cem_iterations=2,
    )

    current_state = np.array([100.0, 200.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action, info = planner.plan(current_state)

    assert action.shape == (6,)
    assert not np.isnan(action).any()
    assert "best_reward" in info
    assert "imagined_trajectory" in info
    assert info["imagined_trajectory"].shape == (10, 8)
