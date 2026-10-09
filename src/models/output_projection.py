"""
output_projection.py
Enforcing the engine's velocity bounds on a network that predicts the state (10.51).

Section 10.47 crossed target (state vs increment) with mechanism (none / soft / hard)
and deliberately left one cell unimplemented: ``state`` x ``hard``. The reason is that
a clamp cannot wrap a state-output network the way it wraps an increment - there is
nothing to integrate - but the bounds can still be imposed, by *projecting* the
network's own output: clip the predicted velocities to the engine's admissible set and
re-derive the predicted positions from the clipped velocities. That is a third
mechanism, distinct from both of the ones the repository has used:

* the hard shell (10.27 / 10.42 / 10.47) constrains what the network can express -
  it predicts increments and the graph integrates them, so no output is ever corrected;
* the composite penalty (the Soft PINN, 10.47's ``soft`` cells) constrains what the
  network is rewarded for, and can be violated by the trained model;
* this projection constrains the output after the fact, so the guarantee holds for any
  weights while the network's own loss stays a pure data term.

It is also not the CBF layer of Section 10.16: `src/models/cbf_projection.py` solves a
quadratic program on the *action* to keep the next state inside a safe set, which
changes what the agent commands rather than how a prediction is read.
"""

import torch
import torch.nn as nn

from src.utils.kinematics import CARRIED, NEXT, check_position_velocity


class ProjectedDynamics(nn.Module):
    """
    Wrap a state-output dynamics network in a post-hoc velocity projection.

    Args:
        base: module mapping ``(state, action)`` to ``[B, state_dim]``.
        state_dim: full state width (8 for WRAM Mario).
        max_vx / terminal_vy / min_vy: the engine's admissible velocity window.
        subpixels_per_pixel: WRAM position scale, used to re-derive position from the
            projected velocity so the output stays internally consistent.
        position_velocity: ``"next"`` (the default every published artifact uses) re-derives
            position from the projected next-frame velocity; ``"carried"`` advances it with
            the velocity the frame starts with, which is how README 4.1 states the identity
            and how the telemetry measures the console. Under ``"carried"`` the projection
            still constrains the velocity output, but no weight reaches position at all.

    The contact channels are passed through unchanged: projecting them would mean
    deciding the collision response, which is the engine's business, not the wrapper's.
    """

    def __init__(
        self,
        base: nn.Module,
        state_dim: int = 8,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
        subpixels_per_pixel: float = 16.0,
        position_velocity: str = NEXT,
    ):
        super().__init__()
        self.base = base
        self.state_dim = state_dim
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.subpixels_per_pixel = subpixels_per_pixel
        self.position_velocity = check_position_velocity(position_velocity)

    @property
    def bounds_imposed_by_construction(self) -> bool:
        """True: no weight vector can make the projected output leave the window."""
        return True

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next state, then project it onto the admissible velocities."""
        raw = self.base(state, action)
        vx = torch.clamp(raw[:, 2], -self.max_vx, self.max_vx)
        vy = torch.clamp(raw[:, 3], self.min_vy, self.terminal_vy)
        use_vx, use_vy = (
            (state[:, 2], state[:, 3]) if self.position_velocity == CARRIED else (vx, vy)
        )
        x = state[:, 0] + use_vx / self.subpixels_per_pixel
        y = state[:, 1] + use_vy / self.subpixels_per_pixel
        contacts = raw[:, 4 : self.state_dim]
        return torch.cat(
            [x.unsqueeze(-1), y.unsqueeze(-1), vx.unsqueeze(-1), vy.unsqueeze(-1), contacts], dim=-1
        )
