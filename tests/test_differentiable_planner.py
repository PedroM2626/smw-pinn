"""
test_differentiable_planner.py
Unit tests for the Differentiable PINN Planner.
"""

import numpy as np
import torch

from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.planning.differentiable_pinn_planner import DifferentiablePINNPlanner


def test_differentiable_pinn_planner_optimizes_action():
    device = torch.device("cpu")
    wm = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    wm.eval()

    planner = DifferentiablePINNPlanner(
        world_model=wm,
        device=device,
        horizon=6,
        num_iterations=5,
        lr=0.2,
    )

    s0 = np.array([16.0, 336.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action, info = planner.plan(s0)

    assert action.shape == (6,)
    assert set(np.unique(action)).issubset({0.0, 1.0})
    assert "loss_history" in info
    assert len(info["loss_history"]) == 5
    # Loss should decrease or improve over gradient steps
    assert isinstance(info["final_loss"], float)
