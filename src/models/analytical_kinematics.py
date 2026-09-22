"""
analytical_kinematics.py
Zero-hidden-unit engine-rules forward model (the missing reference baseline).

Why this exists
---------------
The headline comparison in this repository is Hard Residual PINN vs. MLP/LSTM.
But the Hard PINN *is* the published engine kinematics plus a learned residual
head, so a reviewer's first question is unavoidable: how much of the advantage
comes from the structural physics prior itself, and how much from the network?
Without a no-neural-network reference point that question cannot be answered.

This module implements exactly the rules documented in README section 4 as a
closed-form `nn.Module` (same `(state, action) -> next_state` interface as every
other world model here, so it drops into `RolloutEvaluator` and
`ModelPredictiveController` unchanged):

* exact discrete integration          X_{t+1} = X_t + vx_{t+1}/16   (section 4.1)
* asymmetric, input-modulated gravity g_held = +3, g_fall = +6       (section 4.2)
* hardware velocity saturation        vy in [-80, +64], tiered vx   (sections 4.2, 4.3)
* traction / friction / skid ramps                                 (section 4.3)
* ground boundary condition           vy_{t+1} = 0 when resting      (section 4.3.5)

Parameter honesty
-----------------
Section 4 publishes the *invariant* constants (subpixel ratio, gravity, clamps,
velocity tiers) but deliberately does not state the per-frame traction, friction,
skid or jump-impulse magnitudes. Rather than inventing them, this model carries
exactly six scalars and fits them by deterministic coordinate-wise grid search
restricted to the published ranges (`IDENTIFIABLE` below). Six interpretable
scalars are compared against 9,992 (Hard PINN) and 36,360 (MLP) weights.

Known limitation (stated, not hidden): the four contact flags cannot be derived
from kinematics alone - they require a tile query - so they are propagated
unchanged from the input state. The baseline therefore scores contact channels
as "persistence", which is precisely the residual the learned models must explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.environment import wram

# Published invariants (README section 4). Not fitted.
SUBPIXELS_PER_PIXEL = wram.SUBPIXELS_PER_PIXEL  # 16.0
TERMINAL_VY = 64.0  # section 4.2.5
MIN_VY = -80.0  # section 4.2.1
G_HOLD = 3.0  # section 4.2.3
G_FALL = 6.0  # section 4.2.4
VX_WALK = 20.0  # section 4.3.1
VX_RUN = 48.0  # section 4.3.2
VX_SPRINT = 72.0  # section 4.3.3 (P-meter; not observable in the 8D state)

# The six scalars this model is allowed to identify, with their published search
# ranges. `a_skid` is bounded by the documented "approximately 4 to 6" interval and
# `jump_impulse` by the documented takeoff window [-64, -80]. Section 4.2.1 also
# states the takeoff is "modulated by prior horizontal running momentum"; the
# minimal reading of that sentence is a term linear in |vx|, which is what
# `jump_run_gain` fits (an interpretation, and labelled as one in the artifact).
IDENTIFIABLE: Dict[str, Tuple[float, float, float]] = {
    "a_traction_walk": (0.25, 4.0, 0.25),
    "a_traction_run": (0.5, 8.0, 0.5),
    "a_friction": (0.25, 6.0, 0.25),
    "a_skid": (4.0, 6.0, 0.25),
    "jump_impulse": (-80.0, -64.0, 1.0),
    "jump_run_gain": (0.0, 1.0, 0.05),
}
INITIAL_GUESS: Dict[str, float] = {
    "a_traction_walk": 1.0,
    "a_traction_run": 2.0,
    "a_friction": 1.0,
    "a_skid": 5.0,
    "jump_impulse": -72.0,
    "jump_run_gain": 0.0,
}

# Action layout shared with src.planning.mpc_planner.ACTION_PRIMITIVES:
# [B (jump), Y (run), UP, DOWN, LEFT, RIGHT]
IDX_JUMP, IDX_RUN, IDX_UP, IDX_DOWN, IDX_LEFT, IDX_RIGHT = range(6)


@dataclass
class EngineRuleParameters:
    """The six identifiable scalars, kept together so they are easy to report."""

    a_traction_walk: float = INITIAL_GUESS["a_traction_walk"]
    a_traction_run: float = INITIAL_GUESS["a_traction_run"]
    a_friction: float = INITIAL_GUESS["a_friction"]
    a_skid: float = INITIAL_GUESS["a_skid"]
    jump_impulse: float = INITIAL_GUESS["jump_impulse"]
    jump_run_gain: float = INITIAL_GUESS["jump_run_gain"]

    def as_dict(self) -> Dict[str, float]:
        return {
            "a_traction_walk": self.a_traction_walk,
            "a_traction_run": self.a_traction_run,
            "a_friction": self.a_friction,
            "a_skid": self.a_skid,
            "jump_impulse": self.jump_impulse,
            "jump_run_gain": self.jump_run_gain,
        }

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "EngineRuleParameters":
        return EngineRuleParameters(**{k: float(d[k]) for k in INITIAL_GUESS})


class AnalyticalKinematicsDynamics(nn.Module):
    """Closed-form SMW forward model: published rules + 5 identified scalars.

    State layout (8D): [x, y, vx, vy, c_ground, c_ceiling, c_left, c_right].
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        params: EngineRuleParameters | None = None,
        max_vx: float = VX_SPRINT,
        subpixels_per_pixel: float = SUBPIXELS_PER_PIXEL,
    ):
        super().__init__()
        if state_dim != 8:
            raise ValueError(
                "AnalyticalKinematicsDynamics models the documented 8D player "
                f"state; got state_dim={state_dim}."
            )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_vx = max_vx
        self.subpixels_per_pixel = subpixels_per_pixel
        self.params = params or EngineRuleParameters()

    # --- introspection helpers used by the benchmark / reports ---------------
    @property
    def num_parameters(self) -> int:
        """Trainable tensors are not learnable here: the model is closed-form."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        p = self.params
        x_t, y_t, vx_t, vy_t = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        c_ground = state[:, 4]
        jump = action[:, IDX_JUMP]
        run = action[:, IDX_RUN]
        right = action[:, IDX_RIGHT]
        left = action[:, IDX_LEFT]

        vx_next = self._horizontal_velocity(vx_t, right - left, run, p)
        vy_next = self._vertical_velocity(vx_t, vy_t, c_ground, jump, p)

        # Section 4.1: exact discrete integration, zero kinematic residual.
        x_next = x_t + vx_next / self.subpixels_per_pixel
        y_next = y_t + vy_next / self.subpixels_per_pixel

        # Contact channels: persistence (not derivable from kinematics).
        contacts = state[:, 4:]
        return torch.cat(
            [
                x_next.unsqueeze(-1),
                y_next.unsqueeze(-1),
                vx_next.unsqueeze(-1),
                vy_next.unsqueeze(-1),
                contacts,
            ],
            dim=-1,
        )

    @staticmethod
    def _horizontal_velocity(
        vx: torch.Tensor, direction: torch.Tensor, run: torch.Tensor, p: EngineRuleParameters
    ) -> torch.Tensor:
        """Tiered traction, friction to rest, and the documented skid deceleration."""
        tier = torch.where(run > 0.5, VX_RUN, VX_WALK)  # sprint tier is unobservable
        moving = direction.abs() > 0.5

        # Signed velocity toward the commanded direction.
        signed = vx * direction
        accelerating = moving & (signed < tier)
        # Skid = input opposite to the current motion; else ordinary traction.
        skidding = moving & (signed < 0.0)
        gain = torch.where(
            run > 0.5, torch.full_like(vx, p.a_traction_run), torch.full_like(vx, p.a_traction_walk)
        )
        step = torch.where(skidding, torch.full_like(vx, p.a_skid), gain)
        step = torch.where(accelerating, step, torch.zeros_like(step))
        vx_next = vx + step * direction

        # No directional input: surface friction pulls velocity to zero (section 4.3.4).
        decay = torch.where(moving, torch.zeros_like(vx), torch.full_like(vx, p.a_friction))
        vx_next = torch.where(moving, vx_next, vx - torch.clamp(vx, -decay, decay))

        # A running speed is never shaved off by the walk tier (momentum envelope):
        # the tiers cap *acceleration*, matching how the P-meter retains sprint speed.
        preserving = moving & (vx.abs() > tier) & (vx * direction > 0.0)
        speed = torch.where(preserving, torch.maximum(vx_next.abs(), vx.abs()), vx_next.abs())
        vx_next = torch.sign(vx_next) * speed
        return torch.clamp(vx_next, -VX_SPRINT, VX_SPRINT)

    @staticmethod
    def _vertical_velocity(
        vx: torch.Tensor,
        vy: torch.Tensor,
        c_ground: torch.Tensor,
        jump: torch.Tensor,
        p: EngineRuleParameters,
    ) -> torch.Tensor:
        """Asymmetric gravity, momentum-modulated takeoff, and the resting rule."""
        takeoff = (jump > 0.5) & (c_ground > 0.5) & (vy > -1.0)
        # Section 4.2.1: takeoff speed grows with existing horizontal momentum.
        impulse = p.jump_impulse - p.jump_run_gain * vx.abs()
        ascending_held = (jump > 0.5) & (vy < 0.0)
        g = torch.where(ascending_held, G_HOLD, G_FALL)  # sections 4.2.3 / 4.2.4
        vy_next = torch.where(takeoff, impulse, vy + g)

        # Section 4.3.5: on solid ground, not jumping, downward motion is cancelled.
        resting = (c_ground > 0.5) & (jump <= 0.5) & (vy >= 0.0)
        vy_next = torch.where(resting, torch.zeros_like(vy_next), vy_next)
        return torch.clamp(vy_next, MIN_VY, TERMINAL_VY)

    # --- system identification ------------------------------------------------
    def fit_engine_rules(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        sweeps: int = 4,
        device: torch.device | None = None,
    ) -> List[dict]:
        """Coordinate-wise grid search over the published ranges (deterministic).

        Only the velocity channels enter the loss: positions follow from them by
        the exact integration rule, and the contact flags are not modeled. Returns
        the per-parameter trace so the fit is auditable in the results artifact.
        """
        device = device or states.device
        if isinstance(device, torch.device) and device.type == "cuda":
            # Keep the identification reproducible across machines with/without CUDA.
            device = torch.device("cpu")
        states, actions, next_states = states.to(device), actions.to(device), next_states.to(device)
        trace: List[dict] = []

        def loss() -> float:
            pred = self.forward(states, actions)
            err = pred[:, 2:4] - next_states[:, 2:4]
            return float((err * err).mean().item())

        current = loss()
        for sweep in range(sweeps):
            improved = False
            for name, (lo, hi, step) in IDENTIFIABLE.items():
                best_value, best_loss = getattr(self.params, name), current
                grid = torch.arange(lo, hi + 1e-9, step)
                for candidate in grid.tolist():
                    setattr(self.params, name, float(candidate))
                    candidate_loss = loss()
                    if candidate_loss < best_loss - 1e-12:
                        best_value, best_loss = float(candidate), candidate_loss
                setattr(self.params, name, best_value)
                trace.append(
                    {
                        "sweep": sweep,
                        "parameter": name,
                        "selected": best_value,
                        "velocity_mse": best_loss,
                        "search_range": [lo, hi, step],
                    }
                )
                if best_loss < current - 1e-12:
                    current, improved = best_loss, True
            if not improved:
                break
        return trace


def velocity_prediction_mse(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    device: torch.device,
) -> float:
    """Mean squared error on (vx, vy): the quantity the engine rules actually claim."""
    model.eval()
    with torch.no_grad():
        pred = model(states.to(device), actions.to(device))
    err = pred[:, 2:4].cpu() - next_states[:, 2:4].cpu()
    return float((err * err).mean().item())
