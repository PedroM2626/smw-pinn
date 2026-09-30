"""
residual_dynamics.py
Increment parameterisation with an optional hard kinematic shell (README 10.47).

Sections 10.41/10.42 trained two operator architectures that differ in *both* of
the things a physics injection can change - the target the net predicts and the
kinematics the graph enforces - so neither could be attributed. This module
separates the two axes and makes them configurable independently over any base
network that maps ``(state, action)`` to auxiliary channels:

    target = "residual": the net predicts increments and the graph integrates
        hat_vx = vx_t + delta_vx (optionally clamped to [-max_vx, max_vx])
        hat_x  = x_t  + hat_vx / subpixels_per_pixel        (exact, always)
    target = "state":    the net's own output is the next state and the graph
        does nothing.

``hard=True`` reproduces the Section 4 discrete kinematics exactly as
``PhysicsConstrainedDeepONetDynamics`` and ``HardResidualPINNDynamics`` apply
them, so a shell-constrained FNO or MLP is the same object rather than a
re-derivation; ``hard=False`` keeps the analytic position integration and drops
the velocity saturation, which is the cell that decides whether the published
operator result came from the shell or from the parameterisation.
"""

from typing import Tuple

import torch
import torch.nn as nn

AUX_PREFIX = 2  # residual channels 0 and 1 are delta_vx, delta_vy


class ResidualDynamics(nn.Module):
    """
    Wrap an auxiliary-channel network in the discrete kinematics.

    Args:
        base: module mapping ``(state, action)`` to ``[B, state_dim - 2]``
            auxiliary predictions: ``[delta_vx, delta_vy, contact logits]``.
        state_dim: full state width of the wrapped model (8 for WRAM Mario).
        hard: when true, velocities are saturated by the Section 4 bounds before
            integration; when false, increments are applied unclamped.
        max_vx: horizontal saturation, the same constant the published hard
            shells carry.
        terminal_vy: downward (fall) velocity bound.
        min_vy: upward (jump) velocity bound.
        subpixels_per_pixel: WRAM position scale.

    The position integration ``x_{t+1} = x_t + v_{t+1} / 16`` is exact in both
    modes, so the discrete-kinematics consistency residual is zero by
    construction either way; only the *bounds* depend on ``hard``.
    """

    def __init__(
        self,
        base: nn.Module,
        state_dim: int = 8,
        hard: bool = True,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
        subpixels_per_pixel: float = 16.0,
    ):
        super().__init__()
        self.base = base
        self.state_dim = state_dim
        self.hard = hard
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.subpixels_per_pixel = subpixels_per_pixel
        self.aux_dim = state_dim - AUX_PREFIX

    @property
    def bounds_imposed_by_construction(self) -> bool:
        """True only when the graph, not the net, guarantees the velocity bounds."""
        return self.hard

    def split_residuals(self, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read delta_vx, delta_vy and the contact logits out of an auxiliary prediction."""
        if aux.shape[-1] != self.aux_dim:
            raise ValueError(f"base produced {aux.shape[-1]} channels, expected {self.aux_dim}")
        return aux[:, 0], aux[:, 1], aux[:, AUX_PREFIX:]

    def integrate(
        self,
        state: torch.Tensor,
        delta_vx: torch.Tensor,
        delta_vy: torch.Tensor,
        aux_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Section 4 discrete kinematics to the predicted increments."""
        x_t, y_t, vx_t, vy_t = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        hat_vx = vx_t + delta_vx
        hat_vy = vy_t + delta_vy
        if self.hard:
            hat_vx = torch.clamp(hat_vx, -self.max_vx, self.max_vx)
            hat_vy = torch.clamp(hat_vy, self.min_vy, self.terminal_vy)
        hat_x = x_t + hat_vx / self.subpixels_per_pixel
        hat_y = y_t + hat_vy / self.subpixels_per_pixel
        return torch.cat(
            [
                hat_x.unsqueeze(-1),
                hat_y.unsqueeze(-1),
                hat_vx.unsqueeze(-1),
                hat_vy.unsqueeze(-1),
                aux_pred,
            ],
            dim=-1,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the predicted next state of width ``state_dim``."""
        delta_vx, delta_vy, aux_pred = self.split_residuals(self.base(state, action))
        return self.integrate(state, delta_vx, delta_vy, aux_pred)
