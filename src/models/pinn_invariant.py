"""
pinn_invariant.py
Translation-Invariant Kinematic Residual PINN (PIML Architecture).

Mathematical Principle:
Newtonian mechanics and 65816 CPU assembly routines for velocity integration,
air drag, acceleration, and jumping are strictly invariant under spatial translation:
    F = m * a  (Independent of global horizontal coordinate X)

In this architecture, absolute coordinates (X_t) are decoupled from the neural force head.
The neural network only parameterizes velocity deltas and contact transitions from:
    [vx_t, vy_t, c_ground, c_ceiling, c_left, c_right, action_t]

Exact spatial equivariance is guaranteed by construction:
    hat_f(s + [C, 0, 0, ...], a) = hat_f(s, a) + [C, 0, 0, ...]
ensuring zero-shot physics generalization across any level without spatial overfitting.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class TranslationInvariantPINNDynamics(nn.Module):
    """
    Spatially Equivariant Grey-Box Dynamics Model:
    - Input to neural layers: Local kinematic state (velocities + contacts + actions).
    - Structural kinematics: Global integration hat_X_{t+1} = X_t + hat_vx / 16.0.
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

        # Invariant input: state[2:] (velocities + contacts = 6 dims) + actions (6 dims) = 12 dims
        self.invariant_state_dim = state_dim - 2  # excludes X and Y
        in_dim = self.invariant_state_dim + action_dim

        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            curr_dim = h_dim

        # Predicts: [delta_vx, delta_vy, hat_c_ground, hat_c_ceiling, hat_c_left, hat_c_right]
        layers.append(nn.Linear(curr_dim, self.invariant_state_dim))
        self.invariant_force_net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, 8] -> [X_t, Y_t, vx_t, vy_t, c_ground, c_ceil, c_left, c_right]
            action: [B, 6] -> joypad action primitives

        Returns:
            next_state: [B, 8] satisfying strict spatial translation equivariance
        """
        x_t = state[:, 0]
        y_t = state[:, 1]
        vx_t = state[:, 2]
        vy_t = state[:, 3]

        # Extract purely local kinematics (translation invariant)
        local_kinematics = state[:, 2:]  # [B, 6]
        inp = torch.cat([local_kinematics, action], dim=-1)  # [B, 12]

        force_out = self.invariant_force_net(inp)
        delta_vx = force_out[:, 0]
        delta_vy = force_out[:, 1]
        aux_pred = force_out[:, 2:]

        # 1. Kinematic velocity integration with engine saturation limits
        hat_vx_next = torch.clamp(vx_t + delta_vx, -self.max_vx, self.max_vx)
        hat_vy_next = torch.clamp(vy_t + delta_vy, self.min_vy, self.terminal_vy)

        # 2. Exact analytical position integration
        hat_x_next = x_t + (hat_vx_next / self.subpixels_per_pixel)
        hat_y_next = y_t + (hat_vy_next / self.subpixels_per_pixel)

        # 3. Assemble complete 8D state vector
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
