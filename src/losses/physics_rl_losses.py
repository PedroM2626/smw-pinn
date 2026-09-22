"""
physics_rl_losses.py
Physics-Informed Model-Free Reinforcement Learning (PIML-MFRL) losses.

These terms inject the *known discrete engine physics* of Super Mario World into a
genuinely model-free PPO agent (Section 10.39). Crucially, none of them predicts the
next state ``s_{t+1}``: they only score whether the *current* state, the *critic's*
value, or the *chosen action* are consistent with the fixed-point kinematics and the
non-penetration / saturation invariants documented in Sections 4 and 6. That is why
the critic stays model-free while still being physics-aware (README 10.30-1).

Two mechanisms are provided:

* :class:`PhysicsInformedCriticLoss` (Approach A) penalises a critic whose value does
  not decrease along the analytic state-derivative field on danger states, a discrete
  Control-Lyapunov / Hamilton-Jacobi-Bellman style condition
  ``grad_s V(s) . s_dot <= -alpha * V(s)``.
* :class:`ActionPhysicsViolation` (Approach C) penalises actions that *demand* rigid
  contact or over-saturation forces the engine resolves to zero (pressing into a wall
  at speed, or commanding thrust beyond the hardware velocity ceiling).

The shared safety model used by both mechanisms (and by the discrete CBF filter in
``src/models/cbf_projection.py``) is :func:`physics_action_violation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# The eight SNES macro-actions share this exact button layout (see
# src/planning/mpc_planner.ACTION_PRIMITIVES): [B (jump), Y (run), UP, DOWN, LEFT, RIGHT].
B_JUMP = 0
B_RUN = 1
B_UP = 2
B_DOWN = 3
B_LEFT = 4
B_RIGHT = 5


@dataclass(frozen=True)
class TractionFrictionParams:
    """Engine force-budget constants (subpixels/frame), shared by the RL physics terms.

    Values mirror the physical constants documented in README Section 4.3; they are
    exposed as a dataclass so the trainer can forward them and unit tests can pin them.
    """

    max_vx: float = 72.0  # P-meter sprint ceiling (4.5 px/frame).
    walk_accel: float = 0.75  # horizontal traction while walking.
    run_accel: float = 1.50  # horizontal traction while running (Y held).
    subpixels_per_pixel: float = 16.0  # fixed-point integration ratio.
    held_gravity: float = 3.0  # ascent gravity while B is held (Section 4.2.3).
    fall_gravity: float = 6.0  # descent gravity once released (Section 4.2.4).


def horizontal_direction(action: torch.Tensor) -> torch.Tensor:
    """Return the commanded horizontal direction ``{-1, 0, +1}`` from a button vector.

    Args:
        action: ``[..., 6]`` button tensor in ``[B, Y, UP, DOWN, LEFT, RIGHT]`` order.
    """
    return action[..., B_RIGHT] - action[..., B_LEFT]


def commanded_acceleration(action: torch.Tensor, params: TractionFrictionParams) -> torch.Tensor:
    """Signed one-frame horizontal acceleration demanded by ``action`` (subpixels/frame).

    The engine never accelerates a body beyond its traction tier in one frame, so the
    *demand* is bounded by the walk/run traction only while a direction is held.
    """
    direction = horizontal_direction(action)
    tier = torch.where(action[..., B_RUN] > 0.5, params.run_accel, params.walk_accel)
    return direction * tier


def _violation_terms(
    vx: torch.Tensor,
    vy: torch.Tensor,
    c_ground: torch.Tensor,
    c_ceiling: torch.Tensor,
    c_left: torch.Tensor,
    c_right: torch.Tensor,
    ac: torch.Tensor,
    params: TractionFrictionParams,
) -> torch.Tensor:
    """Elementwise non-penetration + velocity-saturation score for matched tensors.

    Shape contract: ``vx``, ``vy`` and the contact flags must broadcast against the
    action buttons in ``ac`` (last dim = 6), which lets callers drive both the
    elementwise [B] case and the broadcast [B, K] action-table case from one place.
    """
    # 1. Demanded rigid-contact force (F.relu keeps one-sided "into the surface").
    into_right_wall = torch.relu(vx) * c_right * ac[..., B_RIGHT]
    into_left_wall = torch.relu(-vx) * c_left * ac[..., B_LEFT]
    into_ceiling = torch.relu(-vy) * c_ceiling * ac[..., B_UP]
    into_ground = torch.relu(vy) * c_ground * ac[..., B_DOWN]
    contact_violation = into_right_wall + into_left_wall + into_ceiling + into_ground

    # 2. Commanded acceleration beyond the velocity saturation ceiling.
    direction = horizontal_direction(ac)
    accel = commanded_acceleration(ac, params)
    overspeed = torch.relu((vx + accel).abs() - params.max_vx) * (direction != 0).to(vx.dtype)

    return torch.relu(contact_violation) + torch.relu(overspeed)


def physics_action_violation(
    state: torch.Tensor,
    action: torch.Tensor,
    params: TractionFrictionParams | None = None,
) -> torch.Tensor:
    """Differentiable, non-negative "physically impossible force" score for an action.

    A violation is charged only when the commanded buttons demand a force the fixed
    point engine resolves to zero, i.e. when a real transition under that action would
    break a hard structural invariant of Section 4:

    1. **Rigid non-penetration** - thrust into a wall you are already pressed against
       (``v_x > 0`` while ``c_right`` and RIGHT held, symmetric for the left), a rise
       into a ceiling already flagged, or a downward press while resting on ground.
    2. **Velocity saturation** - commanded acceleration that would push ``|v_x|`` past
       the hardware ceiling ``max_vx`` in a single frame.

    Args:
        state: ``[B, D]`` with D >= 8 (``[x, y, vx, vy, c_ground, c_ceiling, c_left,
            c_right, ...]``).
        action: ``[B, 6]`` button tensor, elementwise aligned with ``state``.
        params: engine force-budget constants.

    Returns:
        ``[B]`` non-negative violations.
    """
    params = params or TractionFrictionParams()
    return _violation_terms(
        state[:, 2],
        state[:, 3],
        state[:, 4],
        state[:, 5],
        state[:, 6],
        state[:, 7],
        action,
        params,
    )


def physics_action_violation_table(
    state: torch.Tensor,
    action_table: torch.Tensor,
    params: TractionFrictionParams | None = None,
) -> torch.Tensor:
    """Score *every* macro-action for *every* state via broadcasting.

    Args:
        state: ``[B, D]`` states.
        action_table: ``[K, 6]`` shared action primitives (the discrete action space).

    Returns:
        ``[B, K]`` non-negative violation for each (state, action) pair.
    """
    params = params or TractionFrictionParams()
    st = state.unsqueeze(1)  # [B, 1, D]
    ac = action_table.unsqueeze(0)  # [1, K, 6]
    return _violation_terms(
        st[..., 2],
        st[..., 3],
        st[..., 4],
        st[..., 5],
        st[..., 6],
        st[..., 7],
        ac,
        params,
    )


class ActionPhysicsViolation(nn.Module):
    """Approach C - physics-consistency penalty for the *executed* action.

    Added to the PPO surrogate as ``+ lambda_phys * E[violation]`` so the optimiser
    de-prioritises actions that demand impossible contact/over-saturation forces.
    """

    def __init__(self, params: TractionFrictionParams | None = None) -> None:
        super().__init__()
        self.params = params or TractionFrictionParams()

    def forward(self, state: torch.Tensor, action_buttons: torch.Tensor) -> torch.Tensor:
        violation = physics_action_violation(state, action_buttons, self.params)
        return violation.mean()


class PhysicsInformedCriticLoss(nn.Module):
    """Approach A - Control-Lyapunov / HJB consistency regulariser for the critic.

    Penalises a value function that does *not* decay along the analytic discrete
    state-derivative field on danger states. The drift field is the exact kinematics
    of Section 4 (``x_dot = vx / 16``, ``y_dot = vy / 16``, vertical velocity driven by
    asymmetric gravity); it never rolls the state forward, so the critic remains
    model-free. Danger states are those falling into a pit (``y`` near ``death_y``
    with positive vertical velocity), where a well-shaped value must strictly decay.

    The condition imposed is ``relu(dV/dt + alpha * V)`` averaged over danger states;
    ``dV/dt = grad_s V . s_dot`` is computed with a differentiable backward pass, so
    callers must pass a ``state`` with ``requires_grad`` enabled.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        death_y: float = 450.0,
        danger_low: float = 380.0,
        params: TractionFrictionParams | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.death_y = float(death_y)
        self.danger_low = float(danger_low)
        self.params = params or TractionFrictionParams()

    def state_derivative(self, state: torch.Tensor) -> torch.Tensor:
        """Analytic one-frame drift ``s_dot`` implied by the engine kinematics."""
        s_dot = torch.zeros_like(state)
        ratio = 1.0 / self.params.subpixels_per_pixel
        s_dot[..., 0] = state[..., 2] * ratio  # x_dot = vx / 16
        s_dot[..., 1] = state[..., 3] * ratio  # y_dot = vy / 16
        # Vertical velocity is driven by gravity; sign picks the asymmetric constant.
        grav = torch.where(state[..., 3] < 0, self.params.held_gravity, self.params.fall_gravity)
        s_dot[..., 3] = grav
        return s_dot

    def danger_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Falling toward a pit: below ``danger_low`` in screen-Y (approaching death)
        and moving downward, but not already past the death plane."""
        y = state[..., 1]
        vy = state[..., 3]
        grounded = state[..., 4] > 0.5
        falling = (vy > 0) & (~grounded)
        near_pit = (y > self.danger_low) & (y < self.death_y)
        return (falling & near_pit).to(state.dtype)

    def forward(self, state: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if not state.requires_grad:
            raise ValueError(
                "PhysicsInformedCriticLoss needs state.requires_grad=True to form "
                "grad_s V; call state.requires_grad_(True) before the critic forward pass."
            )
        s_dot = self.state_derivative(state)
        grad_v = torch.autograd.grad(value.sum(), state, create_graph=True)[0]
        dV_dt = (grad_v * s_dot).sum(dim=-1)
        decay_condition = torch.relu(dV_dt + self.alpha * value)
        weights = self.danger_mask(state)
        total = (decay_condition * weights).sum()
        return total / (weights.sum() + 1e-8)


__all__: list[str] = [
    "TractionFrictionParams",
    "ActionPhysicsViolation",
    "PhysicsInformedCriticLoss",
    "physics_action_violation",
    "physics_action_violation_table",
    "commanded_acceleration",
    "horizontal_direction",
]
