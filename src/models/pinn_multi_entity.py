"""
pinn_multi_entity.py
Multi-Entity Physics-Informed Neural Network (Multi-Entity PINN).
Simulates both Mario kinematics (8D) and dynamic stage hazards/sprites (4D)
with joint analytical kinematic consistency:
    hat_X_{mario, t+1} = X_{mario, t} + hat_vx_{mario, t+1} / 16.0
    hat_Delta_X_{hazard, t+1} = Delta_X_{hazard, t} + (hat_vx_{hazard, t+1} - hat_vx_{mario, t+1}) / 16.0
Guarantees 0.0% kinematic violation for both player and relative hazard displacement.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.pinn_hard_residual import HardResidualPINNDynamics


class MultiEntityPINNDynamics(nn.Module):
    """
    Multi-Entity World Model for Super Mario World:
    - Mario State (8D): [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
    - Hazard Relative State (4D): [delta_x_hazard, delta_y_hazard, vx_hazard, hazard_active]
    Total State Dimension: 12D.
    """

    def __init__(
        self,
        base_pinn: Optional[HardResidualPINNDynamics] = None,
        state_dim: int = 12,
        action_dim: int = 6,
        subpixels_per_pixel: float = 16.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subpixels_per_pixel = subpixels_per_pixel

        # Base Mario physics model
        if base_pinn is not None:
            self.mario_pinn = base_pinn
        else:
            self.mario_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=action_dim)

        # Hazard dynamic update head (parameterizes hazard acceleration / bounce if applicable)
        # Hazard input: [delta_x, delta_y, vx_hazard, active, mario_vx, mario_vy]
        self.hazard_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2),  # [delta_vx_hazard, delta_vy_hazard]
        )

    def forward(self, state_12d: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward simulation for combined Mario + Hazard state.

        Args:
            state_12d: [B, 12]
                [0..7]: Mario kinematics
                [8]: delta_x_hazard (X_hazard - X_mario)
                [9]: delta_y_hazard (Y_hazard - Y_mario)
                [10]: vx_hazard
                [11]: hazard_active (1.0 if active, 0.0 otherwise)
            action: [B, action_dim]

        Returns:
            next_state_12d: [B, 12]
        """
        # 1. Forward simulate Mario kinematics via Hard Residual PINN
        mario_state_8d = state_12d[:, :8]
        next_mario_state_8d = self.mario_pinn(mario_state_8d, action)

        mario_vx_next = next_mario_state_8d[:, 2]
        mario_vy_next = next_mario_state_8d[:, 3]

        # 2. Extract hazard telemetry
        dx_h = state_12d[:, 8]
        dy_h = state_12d[:, 9]
        vx_h = state_12d[:, 10]
        active_h = state_12d[:, 11]

        # 3. Predict residual accelerations for hazard
        h_inp = torch.stack([dx_h, dy_h, vx_h, active_h, mario_vx_next, mario_vy_next], dim=-1)
        h_residuals = self.hazard_net(h_inp)
        delta_vx_h = h_residuals[:, 0] * active_h
        delta_vy_h = h_residuals[:, 1] * active_h

        # Update hazard velocities (Rex walks at constant nominal speed or responds to terrain)
        next_vx_h = torch.clamp(vx_h + delta_vx_h, -32.0, 32.0)
        next_vy_h = delta_vy_h  # typically 0 on flat terrain

        # 4. Exact Relative Kinematic Integration:
        # Delta X_{t+1} = Delta X_t + (vx_hazard, t+1 - vx_mario, t+1) / 16.0
        relative_vx = next_vx_h - mario_vx_next
        relative_vy = next_vy_h - mario_vy_next

        next_dx_h = dx_h + (relative_vx / self.subpixels_per_pixel) * active_h
        next_dy_h = dy_h + (relative_vy / self.subpixels_per_pixel) * active_h

        # Inactive hazards remain at horizon default
        next_dx_h = torch.where(active_h > 0.5, next_dx_h, torch.full_like(next_dx_h, 300.0))
        next_dy_h = torch.where(active_h > 0.5, next_dy_h, torch.zeros_like(next_dy_h))
        next_vx_h = torch.where(active_h > 0.5, next_vx_h, torch.zeros_like(next_vx_h))

        # 5. Assemble complete 12D next state
        next_hazard_4d = torch.stack([next_dx_h, next_dy_h, next_vx_h, active_h], dim=-1)
        next_state_12d = torch.cat([next_mario_state_8d, next_hazard_4d], dim=-1)

        return next_state_12d
