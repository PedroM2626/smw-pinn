"""Gravity-identified residual PINN (audit §4.3).

The Hard PINN integrates positions exactly but leaves *both* velocities to a
black-box residual. SMW vertical dynamics, however, are deterministic and
identifiable: gravity is +3 subpx/f² while ascending with jump held and +6
otherwise, with terminal fall +64 and max rise -80. This module keeps the
exact integrator and the residual force head, but routes the vertical update
through *learnable, interpretable* gravity parameters initialized at the
hardware values — system identification inside the graph, not beside it.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GravityIdentifiedPINNDynamics(nn.Module):
    """Residual PINN with identifiable asymmetric gravity.

    v_hat_{t+1} = clamp(v_t + delta_v_nn + g(a_t, v_t)), with
    g = g_hold if jump held and vy < 0 else g_fall.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        hidden_dims: List[int] | None = None,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
        subpixels_per_pixel: float = 16.0,
        g_hold_init: float = 3.0,
        g_fall_init: float = 6.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.subpixels_per_pixel = subpixels_per_pixel

        layers: List[nn.Module] = []
        curr = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            curr = h
        layers.append(nn.Linear(curr, state_dim - 2))
        self.force_net = nn.Sequential(*layers)

        self.g_hold = nn.Parameter(torch.tensor(float(g_hold_init)))
        self.g_fall = nn.Parameter(torch.tensor(float(g_fall_init)))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x_t, y_t, vx_t, vy_t = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        jump = action[:, 0]
        out = self.force_net(torch.cat([state, action], dim=-1))
        delta_vx, delta_vy_res, aux = out[:, 0], out[:, 1], out[:, 2:]

        ascending_held = (jump > 0.5) & (vy_t < 0.0)
        g = torch.where(ascending_held, self.g_hold, self.g_fall)

        hat_vx = torch.clamp(vx_t + delta_vx, -self.max_vx, self.max_vx)
        hat_vy = torch.clamp(vy_t + delta_vy_res + g, self.min_vy, self.terminal_vy)
        hat_x = x_t + hat_vx / self.subpixels_per_pixel
        hat_y = y_t + hat_vy / self.subpixels_per_pixel
        return torch.cat(
            [
                hat_x.unsqueeze(-1),
                hat_y.unsqueeze(-1),
                hat_vx.unsqueeze(-1),
                hat_vy.unsqueeze(-1),
                aux,
            ],
            dim=-1,
        )
