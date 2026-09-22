"""
differentiable_pinn_planner.py
First-Order / Differentiable Trajectory Optimization via Analytical Backpropagation
through the Hard Residual Physics-Informed Neural Network (Hard PINN).

Unlike derivative-free / zeroth-order planners (e.g., CEM or Random Shooting),
this planner directly computes the analytical gradient:
    nabla_{u_{0:H-1}} J = nabla_{u_{0:H-1}} sum_{tau=0}^{H-1} R(s_tau, a_tau)
by backpropagating loss signals through the exact kinematic equations of the Hard PINN.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.planning.mpc_planner import TrajectoryObjective


class DifferentiablePINNPlanner:
    """
    Differentiable Model Predictive Controller that optimizes continuous
    action logits directly through the Hard PINN computational graph.
    """

    def __init__(
        self,
        world_model: nn.Module,
        device: torch.device,
        horizon: int = 15,
        num_iterations: int = 15,
        lr: float = 0.25,
        temperature: float = 1.0,
        objective: Optional[TrajectoryObjective] = None,
    ):
        self.world_model = world_model
        self.device = device
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.lr = lr
        self.temperature = temperature
        self.objective = objective or TrajectoryObjective()

        # Freeze world model parameters so autograd only computes gradients w.r.t actions
        for p in self.world_model.parameters():
            p.requires_grad = False

    def plan(
        self,
        initial_state: np.ndarray,
        warm_start_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Performs gradient-based trajectory optimization through the PINN.

        Args:
            initial_state: [state_dim] float32 numpy array
            warm_start_logits: Optional [horizon, action_dim] tensor for warm-starting

        Returns:
            best_action_binary: [action_dim] float32 array in {0.0, 1.0}
            info_dict: Optimization trajectory, final reward, loss history
        """
        state_dim = len(initial_state)
        action_dim = 6

        # Initialize trainable action logits: [horizon, action_dim]
        if warm_start_logits is not None:
            action_logits = warm_start_logits.clone().detach().to(self.device).requires_grad_(True)
        else:
            # Initialize with forward bias (encourage rightward movement initially)
            init_val = torch.zeros(self.horizon, action_dim, device=self.device)
            init_val[:, 5] = 1.5  # Bias towards RIGHT
            action_logits = nn.Parameter(init_val)

        optimizer = optim.Adam([action_logits], lr=self.lr)
        loss_history = []

        curr_s0 = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, state_dim]

        for step in range(self.num_iterations):
            optimizer.zero_grad()

            # Continuous relaxation via Sigmoid: a in (0, 1)
            actions = torch.sigmoid(action_logits / self.temperature)  # [horizon, 6]

            # Unroll trajectory through the differentiable Hard PINN
            s_curr = curr_s0
            traj_states = []

            for h in range(self.horizon):
                a_h = actions[h:h+1]  # [1, 6]
                if state_dim == 8:
                    s_next = self.world_model(s_curr, a_h)
                else:
                    # 12D Multi-Entity or Unified
                    s_next = self.world_model(s_curr, a_h)
                traj_states.append(s_next)
                s_curr = s_next

            traj_tensor = torch.stack(traj_states, dim=1)  # [1, horizon, state_dim]

            # Evaluate differentiable objective:
            # Reward horizontal progress: X_end - X_start
            delta_x = traj_tensor[0, -1, 0] - curr_s0[0, 0]
            mean_vx = torch.mean(traj_tensor[0, :, 2])
            pit_penalty = torch.relu(traj_tensor[0, :, 1] - 420.0).sum() * 50.0

            # Total differentiable objective (maximize progress, minimize pit fall)
            loss = -(delta_x * self.objective.weight_progress + mean_vx * self.objective.weight_velocity) + pit_penalty

            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

        # Discretize optimized action logits at step 0 for hardware execution
        final_actions = torch.sigmoid(action_logits / self.temperature).detach()
        best_action_vec = (final_actions[0] > 0.5).float().cpu().numpy()

        info = {
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "best_sequence": (final_actions > 0.5).float().cpu().numpy(),
            "optimized_logits": action_logits.detach(),
        }

        return best_action_vec, info
