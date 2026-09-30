"""
effective_velocity_dynamics.py
Predicting the velocity the engine integrates with, not the velocity it reports (10.53).

Section 4.1 states the discrete kinematics as a strict identity, and every shell in this
repository has implemented it by advancing position with the velocity the network predicts
*for the next frame*:

    hat_x_{t+1} = x_t + hat_v_{t+1} / 16            (10.27 / 10.42 / 10.47)

Section 10.49 measured the console instead. Over 45,389 WRAM transitions the median of
``|dx - v_{x,t} / 16|`` - the velocity already carried in the state the frame *starts*
from - is 0.0000 px exactly, while the median against ``v_{x,t+1}`` is 0.0625 px, one
sub-pixel of lag on every accelerating frame. The engine's own residual term,
``DiscreteKinematicsLoss``, and the published rollout predicate both already read the
identity with ``v_t``; only the graphs were integrating with the other velocity.

This module makes that quantity the prediction target. It exposes one network with three
parameterisations of the same step, differing only in which velocity advances position:

    mode = "next"     hat_x = x_t + hat_v_{t+1}/16    the published shell, reproduced
                                                      exactly, so the axis can be
                                                      flipped inside one class
    mode = "carried"  hat_x = x_t + v_t/16            the measured console convention:
                                                      position is not predicted at all
    mode = "offset"   hat_x = x_t + (v_t + eps_t)/16  the effective velocity as a free
                                                      head, the only form that can
                                                      represent the frames where the
                                                      console breaks its own rule

In all three modes the velocity channel of the output stays ``hat_v_{t+1} = v_t + delta_v``,
because that is what the WRAM reports after the frame and what the next step must carry.
The two are different quantities on an accelerating frame, and separating them is the
point: a model with a single head for both has to be wrong about one of them.

``hard`` clamps only the reported velocity. The effective velocity is never clamped: on a
wall stop or a screen wrap the displacement the engine applied is legitimately outside the
velocity window - Mario moving at 29 sub-pixels/frame can be displaced by nothing - so
saturating it would rewrite the frame rather than predict it.
"""

from typing import Tuple

import torch
import torch.nn as nn

# Channels the base network emits before the contact logits: two velocity increments, plus,
# in "offset" mode, the two components of the effective-velocity correction.
VELOCITY_CHANNELS = {"next": 2, "carried": 2, "offset": 4}


class EffectiveVelocityDynamics(nn.Module):
    """
    Wrap an auxiliary-channel network in a chosen integration convention.

    Args:
        base: module mapping ``(state, action)`` to ``[B, aux_dim]`` auxiliary channels:
            ``[delta_vx, delta_vy]`` (and ``[eps_x, eps_y]`` in "offset" mode) followed by
            the contact logits.
        state_dim: full state width of the wrapped model (8 for WRAM Mario).
        mode: which velocity advances position - see the module docstring.
        hard: when true, the *reported* next-frame velocity is saturated by the Section 4
            bounds; never applied to the effective velocity.
        max_vx: horizontal saturation of the reported velocity.
        terminal_vy: downward (fall) bound of the reported velocity.
        min_vy: upward (jump) bound of the reported velocity.
        subpixels_per_pixel: WRAM position scale.
    """

    def __init__(
        self,
        base: nn.Module,
        state_dim: int = 8,
        mode: str = "carried",
        hard: bool = False,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
        subpixels_per_pixel: float = 16.0,
    ):
        super().__init__()
        if mode not in VELOCITY_CHANNELS:
            raise ValueError(f"unknown mode {mode!r}, expected one of {sorted(VELOCITY_CHANNELS)}")
        self.base = base
        self.state_dim = state_dim
        self.mode = mode
        self.hard = hard
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.subpixels_per_pixel = subpixels_per_pixel
        self.velocity_channels = VELOCITY_CHANNELS[mode]
        self.aux_dim = state_dim - 2 + self.velocity_channels - 2

    @property
    def bounds_imposed_by_construction(self) -> bool:
        """True only when the graph, not the net, guarantees the velocity bounds."""
        return self.hard

    @property
    def position_is_predicted(self) -> bool:
        """False in "carried" mode, where position is a function of the input alone.

        No weight vector can move the predicted position there, so the whole of that arm's
        position error is a statement about where the console deviates from its own
        integration rule rather than about what the network failed to learn.
        """
        return self.mode != "carried"

    def split_aux(
        self, aux: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the velocity increments, the effective offsets and the contact logits."""
        if aux.shape[-1] != self.aux_dim:
            raise ValueError(f"base produced {aux.shape[-1]} channels, expected {self.aux_dim}")
        delta_vx, delta_vy = aux[:, 0], aux[:, 1]
        contacts = aux[:, self.velocity_channels :]
        if self.mode == "offset":
            return delta_vx, delta_vy, aux[:, 2], aux[:, 3], contacts
        return delta_vx, delta_vy, delta_vx * 0.0, delta_vy * 0.0, contacts

    def integrate(
        self,
        state: torch.Tensor,
        delta_vx: torch.Tensor,
        delta_vy: torch.Tensor,
        offset_vx: torch.Tensor,
        offset_vy: torch.Tensor,
        aux_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the selected convention and return the predicted next state."""
        x_t, y_t, vx_t, vy_t = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        hat_vx = vx_t + delta_vx
        hat_vy = vy_t + delta_vy
        if self.hard:
            hat_vx = torch.clamp(hat_vx, -self.max_vx, self.max_vx)
            hat_vy = torch.clamp(hat_vy, self.min_vy, self.terminal_vy)
        if self.mode == "next":
            used_vx, used_vy = hat_vx, hat_vy
        else:
            used_vx, used_vy = vx_t + offset_vx, vy_t + offset_vy
        hat_x = x_t + used_vx / self.subpixels_per_pixel
        hat_y = y_t + used_vy / self.subpixels_per_pixel
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
        delta_vx, delta_vy, offset_vx, offset_vy, contacts = self.split_aux(
            self.base(state, action)
        )
        return self.integrate(state, delta_vx, delta_vy, offset_vx, offset_vy, contacts)

    @torch.no_grad()
    def effective_velocity(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The velocity this model integrates position with, in sub-pixels per frame."""
        delta_vx, delta_vy, offset_vx, offset_vy, contacts = self.split_aux(
            self.base(state, action)
        )
        if self.mode == "next":
            full = self.integrate(state, delta_vx, delta_vy, offset_vx, offset_vy, contacts)
            return torch.stack([full[:, 2], full[:, 3]], dim=-1)
        return torch.stack([state[:, 2] + offset_vx, state[:, 3] + offset_vy], dim=-1)
