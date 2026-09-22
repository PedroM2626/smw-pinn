"""
pinn_hard_residual.py
Hard Inductive Bias Neural Network / Kinematic Residual PINN.
Embeds discrete kinematics directly into the computational graph, guaranteeing by construction:
    hat_X_{t+1} = X_t + hat_vx_{t+1} / 16.0
    hat_Y_{t+1} = Y_t + hat_vy_{t+1} / 16.0
The neural network only parameterizes non-linear accelerations, friction, and contact forces.
"""

from typing import List
import torch
import torch.nn as nn


class HardResidualPINNDynamics(nn.Module):
    """
    Hybrid Grey-Box Architecture:
    - Exact Discrete Kinematic Consistency Integration (Hard Structural Prior).
    - Dense Residual Network for Forces and Accelerations (Delta Velocity).
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 128],
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
        subpixels_per_pixel: float = 16.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.subpixels_per_pixel = subpixels_per_pixel

        in_dim = state_dim + action_dim
        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            curr_dim = h_dim

        # Network predicts:
        # [delta_vx, delta_vy, hat_c_ground, hat_c_ceiling, hat_c_left, hat_c_right]
        aux_dim = state_dim - 2  # velocities + contact flags
        layers.append(nn.Linear(curr_dim, aux_dim))
        self.force_net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] -> [X_t, Y_t, vx_t, vy_t, c_ground, c_ceil, c_left, c_right]
            action: [B, action_dim]

        Returns:
            next_state: [B, state_dim]
        """
        x_t = state[:, 0]
        y_t = state[:, 1]
        vx_t = state[:, 2]
        vy_t = state[:, 3]

        inp = torch.cat([state, action], dim=-1)
        force_out = self.force_net(inp)

        delta_vx = force_out[:, 0]
        delta_vy = force_out[:, 1]
        aux_pred = force_out[:, 2:]  # contact predictions / flags

        # 1. Velocity integration with physical saturation clamping
        hat_vx_next = torch.clamp(vx_t + delta_vx, -self.max_vx, self.max_vx)
        hat_vy_next = torch.clamp(vy_t + delta_vy, self.min_vy, self.terminal_vy)

        # 2. Exact analytical position integration (Zero Kinematic Residual)
        hat_x_next = x_t + (hat_vx_next / self.subpixels_per_pixel)
        hat_y_next = y_t + (hat_vy_next / self.subpixels_per_pixel)

        # 3. Assemble complete state vector
        next_state = torch.cat(
            [
                hat_x_next.unsqueeze(-1),
                hat_y_next.unsqueeze(-1),
                hat_vx_next.unsqueeze(-1),
                hat_vy_next.unsqueeze(-1),
                aux_pred,
            ],
            dim=-1,
        )

        return next_state
