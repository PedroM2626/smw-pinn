"""
neural_ode_dynamics.py
A continuous vector field, integrated by an explicit choice of numerical method (10.55).

Section 10.53 found that the repository's whole "hard kinematics" story turns on one line -
whether position is advanced with the velocity the frame starts with ($v_t$, what the console
does) or with the velocity predicted for the frame after ($\\hat v_{t+1}$, what every shell
since 10.27 does) - and it hard-typed that choice into each graph. This module removes the
typing. It learns a *continuous* vector field

    dx/dt = v_x / 16,     dv_x/dt = a_x(s, a)      (and the two vertical analogues)

and lets a named numerical integrator decide how one frame of it is walked, so the
convention becomes a solver option rather than an edit:

    "euler"       position first, with the velocity the step starts with - forward Euler,
                  bit-identical to the `carried` arm of 10.53;
    "symplectic"  velocity first, then position with the updated velocity - the split the
                  published shells implement, bit-identical to `ResidualDynamics`;
    "midpoint"    the field re-evaluated half a step in;
    "rk4"         four evaluations, classical Runge-Kutta.

The four are not "more and more accurate models of the same physics". They are *different
discrete systems*, ordered by the second-order term each adds to the position increment:
$\\hat x_{t+1} = x_t + v_t/16 + c\\,a/16$ with $c = 0, \\tfrac12, \\tfrac12, 1$ for euler,
midpoint, rk4 and symplectic when the acceleration is constant over the step. Which of them
is the better world model is therefore a question about the SNES engine's own arithmetic, and
10.49 measured the answer already: the console's median residual against $v_t/16$ is exactly
zero, which is what forward Euler does and what no second-order method does. A higher-order
solver here is a more accurate integrator of a system that is not the one being integrated,
and this module exists to let that be measured rather than asserted.

The contact channels are not integrated: they are discrete WRAM flags, read once from the
state the step starts from. `bounded` projects the velocity onto the Section 4 window *inside*
the step, which is where the engine saturates it; the published shells clamp the value the
network reports, after the arithmetic.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

SOLVERS: Tuple[str, ...] = ("euler", "symplectic", "midpoint", "rk4")

# Vector-field evaluations per step, for the artifact's cost report.
FIELD_EVALUATIONS: Dict[str, int] = {"euler": 1, "symplectic": 1, "midpoint": 2, "rk4": 4}


class NeuralODEDynamics(nn.Module):
    """
    Wrap an acceleration-field network in a chosen fixed-step integrator.

    Args:
        base: module mapping ``(state, action)`` to ``[B, state_dim - 2]`` auxiliary
            channels ``[a_x, a_y, contact logits]``. The first argument it receives is the
            state the field is evaluated at, so a mid-step state is passed with the contact
            flags of the step it started from.
        state_dim: full state width (8 for WRAM Mario).
        action_dim: action width (6 for the joypad latch).
        solver: one of `SOLVERS`.
        step_frames: the frame interval integrated over; one WRAM frame by default.
        subpixels_per_pixel: WRAM position scale.
        bounded: clamp the velocity part of the state to the Section 4 window after every
            step, so the constraint lives inside the integration.
        max_vx / terminal_vy / min_vy: the admissible velocity window.
    """

    def __init__(
        self,
        base: nn.Module,
        state_dim: int = 8,
        action_dim: int = 6,
        solver: str = "euler",
        step_frames: float = 1.0,
        subpixels_per_pixel: float = 16.0,
        bounded: bool = False,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
    ):
        super().__init__()
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r}, expected one of {SOLVERS}")
        self.base = base
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver = solver
        self.step_frames = step_frames
        self.subpixels_per_pixel = subpixels_per_pixel
        self.bounded = bounded
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.contact_dim = state_dim - 4
        self.aux_dim = 2 + self.contact_dim

    @property
    def bounds_imposed_by_construction(self) -> bool:
        """True only when the projection inside the step guarantees the velocity bounds."""
        return self.bounded

    @property
    def field_evaluations(self) -> int:
        """Cost of one step, in network calls - what makes rk4 four times the others."""
        return FIELD_EVALUATIONS[self.solver]

    def split_aux(self, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the two acceleration components and the contact logits."""
        if aux.shape[-1] != self.aux_dim:
            raise ValueError(f"base produced {aux.shape[-1]} channels, expected {self.aux_dim}")
        return aux[:, :2], aux[:, 2:]

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        """Saturate the velocity of a state the integration itself produced.

        Only derived states are projected: the velocity the step *starts* from is a WRAM
        reading, and 10.49 measured it outside the documented window on a fifth to a third of
        frames, so rewriting it would replace the console's arithmetic with the model's.
        """
        if not self.bounded:
            return z
        vx = torch.clamp(z[:, 2], -self.max_vx, self.max_vx)
        vy = torch.clamp(z[:, 3], self.min_vy, self.terminal_vy)
        return torch.cat([z[:, :2], vx.unsqueeze(-1), vy.unsqueeze(-1)], dim=-1)

    def _rate(self, z: torch.Tensor, contacts: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The vector field at one state: position with velocity, velocity with acceleration."""
        accel, _logits = self.split_aux(self.base(torch.cat([z, contacts], dim=-1), action))
        return torch.cat([z[:, 2:4] / self.subpixels_per_pixel, accel], dim=-1)

    def _acceleration(
        self, z: torch.Tensor, contacts: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        accel, _logits = self.split_aux(self.base(torch.cat([z, contacts], dim=-1), action))
        return accel

    def _step(self, z: torch.Tensor, contacts: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Walk one frame of the field with the selected method."""
        h = self.step_frames
        if self.solver == "euler":
            return self._project(z + h * self._rate(z, contacts, action))
        if self.solver == "symplectic":
            accel = self._acceleration(z, contacts, action)
            velocity = self._project(torch.cat([z[:, :2], z[:, 2:4] + accel * h], dim=-1))[:, 2:4]
            position = z[:, :2] + velocity * h / self.subpixels_per_pixel
            return torch.cat([position, velocity], dim=-1)
        if self.solver == "midpoint":
            half = self._project(z + 0.5 * h * self._rate(z, contacts, action))
            return self._project(z + h * self._rate(half, contacts, action))
        k1 = self._rate(z, contacts, action)
        k2 = self._rate(self._project(z + 0.5 * h * k1), contacts, action)
        k3 = self._rate(self._project(z + 0.5 * h * k2), contacts, action)
        k4 = self._rate(self._project(z + h * k3), contacts, action)
        return self._project(z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Integrate the continuous channels one frame and emit the predicted next state."""
        z = state[:, :4]
        contacts_in = state[:, 4 : self.state_dim]
        _accel, contacts = self.split_aux(self.base(torch.cat([z, contacts_in], dim=-1), action))
        advanced = self._step(z, contacts_in, action)
        return torch.cat([advanced, contacts], dim=-1)
