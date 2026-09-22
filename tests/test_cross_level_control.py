"""
test_cross_level_control.py
Unit tests for zero-shot cross-level control benchmark utilities.
"""

import numpy as np
import torch

from src.evaluation.evaluate_cross_level_control import (
    action_vector_to_dict,
    extract_8d_vector,
)
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.planning.mpc_planner import (
    ModelPredictiveController,
    TrajectoryObjective,
)
from src.training.distill_mpc_policy import DistilledActorPolicy


def test_action_vector_to_dict():
    vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    act_dict = action_vector_to_dict(vec)
    assert act_dict["B"] is True
    assert act_dict["Y"] is False
    assert act_dict["RIGHT"] is True
    assert act_dict["LEFT"] is False


def test_extract_8d_vector():
    sample_state = {
        "x": 16.0,
        "y": 336.0,
        "vx": 0.0,
        "vy": 0.0,
        "c_ground": 1,
        "c_ceiling": 0,
        "c_left": 0,
        "c_right": 0,
    }
    vec = extract_8d_vector(sample_state)
    assert vec.shape == (8,)
    assert vec[0] == 16.0
    assert vec[1] == 336.0
    assert vec[4] == 1.0


def test_mpc_hard_residual_planning_step():
    device = torch.device("cpu")
    wm = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    wm.eval()
    objective = TrajectoryObjective(weight_progress=3.0, weight_velocity=0.5)
    mpc = ModelPredictiveController(
        world_model=wm,
        device=device,
        horizon=5,
        num_candidates=16,
        cem_iterations=2,
        objective=objective,
    )
    s8 = np.array([16.0, 336.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action, info = mpc.plan(s8)
    assert action.shape == (6,)
    assert isinstance(info, dict)
    assert "best_reward" in info


def test_dagger_policy_forward_action():
    policy = DistilledActorPolicy(state_dim=12, action_dim=6)
    policy.eval()
    s12 = np.zeros(12, dtype=np.float32)
    action_dict, probs = policy.predict_action(s12, threshold=0.5)
    assert isinstance(action_dict, dict)
    assert "RIGHT" in action_dict
    assert probs.shape == (6,)
