"""
cbf_projection.py
Physics-Constrained Actor primitives for PIML-MFRL (README Section 10.39, Approach B).

A Control-Barrier-Function safety layer sits between the actor's *candidate* action and
the action that is actually executed. In continuous form it solves the projection

    min_a ||a - a_hat||^2   s.t.   b_i(s) . a >= c_i(s)   (one CBF condition per barrier)

which is the discrete-time analogue of ``h_dot(s, a) >= -gamma * h(s)``. The actor is
free to propose any action; the layer projects it back onto the physically admissible
set before it ever reaches the console, so a dangerous command never leaves the agent.

Two layers are provided:

* :class:`CBFQPLayer` - a genuine, differentiable *continuous* projection (exact in
  closed form for a single active linear constraint, an approximate cyclic/POCS
  projection for several), plus the SMW barrier map :func:`smw_barrier_affine`.
* :class:`DiscreteCBFCategoricalFilter` - the categorical counterpart actually used by
  the 8-button macro-action PPO agent: it re-weights action logits by the shared safety
  model so the policy is projected onto the safe-action set (the limit ``beta -> inf``
  is a hard filter that never samples an infeasible action).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.losses.physics_rl_losses import TractionFrictionParams, physics_action_violation_table


def _clamp_box(a: torch.Tensor, u_max: float | torch.Tensor) -> torch.Tensor:
    """Clamp ``a`` into the symmetric box ``[-u_max, u_max]`` (scalar or per-channel).

    ``torch.Tensor.clamp`` has separate overloads for numeric and tensor bounds, so the
    union type is branched here to keep mypy happy while remaining exact at runtime.
    """
    if isinstance(u_max, torch.Tensor):
        return a.clamp(min=-u_max, max=u_max)
    return a.clamp(min=-u_max, max=u_max)


class CBFQPLayer(nn.Module):
    """Differentiable projection of a continuous action onto linear CBF constraints.

    Constraints are given per sample as ``A @ a >= b`` with ``A`` of shape ``[k, m]``
    and ``b`` of shape ``[k]``; a symmetric box ``|a| <= u_max`` is always enforced.
    For a single active half-space the KKT solution is returned in closed form; for
    several, a fixed number of POCS (projection onto convex sets) sweeps is applied,
    which converges to a feasible point when the constraints are consistent.
    """

    def __init__(self, n_iterations: int = 8, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_iterations = int(n_iterations)
        self.eps = float(eps)

    def forward(
        self,
        a_hat: torch.Tensor,
        affine_rows: torch.Tensor,
        affine_rhs: torch.Tensor,
        u_max: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Project ``a_hat`` onto the CBF-feasible set.

        Args:
            a_hat: ``[B, m]`` candidate (unconstrained) action.
            affine_rows: ``[B, k, m]`` barrier gradients ``b_i(s)``.
            affine_rhs: ``[B, k]`` barrier thresholds ``c_i(s)``.
            u_max: scalar or ``[m]`` box bound for each control channel.

        Returns:
            ``[B, m]`` projected action, differentiable in ``a_hat``.
        """
        a = a_hat
        for _ in range(self.n_iterations):
            a = _clamp_box(a, u_max)
            # Vectorised POCS sweep over the k half-spaces.
            margins = (a.unsqueeze(-2) * affine_rows).sum(-1) - affine_rhs  # [B, k]
            norms = (affine_rows * affine_rows).sum(-1) + self.eps  # [B, k]
            steps = torch.relu(-margins) / norms  # [B, k]
            a = a + (steps.unsqueeze(-1) * affine_rows).sum(-2)  # [B, m]
        return _clamp_box(a, u_max)


def smw_barrier_affine(
    state: torch.Tensor,
    params: TractionFrictionParams | None = None,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the SMW traction / velocity-saturation barriers for a continuous command.

    The continuous action is a single commanded horizontal acceleration ``a_x`` (in
    subpixels/frame). Two first-order barriers keep it physically admissible:

    * velocity saturation ``|v_x + a_x| <= max_vx``  ->  two half-spaces on ``a_x``;
    * the proposed action may not *demand* more speed than the engine can supply, which
      is encoded as the same box the :class:`CBFQPLayer` enforces (``u_max``).

    Args:
        state: ``[B, D]`` with ``D >= 8``; only ``v_x`` (index 2) is consumed here.
        params: engine constants (``max_vx``).
        gamma: CBF decay rate applied to the barrier slack.

    Returns:
        (rows, rhs, u_max) with ``rows`` of shape ``[B, 2, 1]`` and ``rhs`` of shape
        ``[B, 2]`` ready for :class:`CBFQPLayer.forward`.
    """
    params = params or TractionFrictionParams()
    device, dtype = state.device, state.dtype
    vx = state[:, 2]
    batch = vx.shape[0]
    max_vx = params.max_vx

    slack = gamma * (max_vx - vx.abs()).clamp(min=0.0)  # barrier h(s) >= 0
    # a_x <= (max_vx - |vx|) + slack  and  -a_x <= (max_vx - |vx|) + slack
    bound = (max_vx - vx.abs()) + slack  # [B]
    rows = torch.zeros(batch, 2, 1, device=device, dtype=dtype)
    rows[:, 0, 0] = 1.0
    rows[:, 1, 0] = -1.0
    rhs = torch.stack([bound, bound], dim=-1)  # [B, 2]
    u_max = torch.full((1,), float(max_vx), device=device, dtype=dtype)
    return rows, rhs, u_max


class DiscreteCBFCategoricalFilter(nn.Module):
    """Categorical CBF filter: project the policy onto the safe-action set (Approach B).

    The 8-button action space is discrete, so the QP is realised as a differentiable
    re-weighting of the actor logits by the shared physics-violation model:

        logits_safe = logits - beta * violation(state, action)

    ``beta`` controls the projection strength; as ``beta -> inf`` any action with a
    non-zero violation is masked out, recovering a hard safety filter, while ``beta = 0``
    returns the unfiltered policy (an exact identity, unit-tested).
    """

    # Class-level annotation so mypy resolves the `register_buffer`-assigned attribute
    # below as a Tensor instead of falling back to nn.Module.__getattr__ (-> Module).
    action_table: torch.Tensor

    def __init__(
        self,
        action_table: torch.Tensor,
        beta: float = 1.0,
        params: TractionFrictionParams | None = None,
    ) -> None:
        """Store the discrete action primitives as a non-learnable buffer.

        Args:
            action_table: ``[K, 6]`` macro-action button table (rows are actions).
            beta: logit-projection strength (>= 0).
            params: engine constants forwarded to the shared safety model.
        """
        super().__init__()
        self.register_buffer("action_table", torch.as_tensor(action_table, dtype=torch.float32))
        self.beta = float(beta)
        self.params = params or TractionFrictionParams()

    def violation(self, state: torch.Tensor) -> torch.Tensor:
        """Per-(state, action) physics violation of shape ``[B, K]``."""
        return physics_action_violation_table(state, self.action_table, self.params)

    def forward(self, logits: torch.Tensor, state: torch.Tensor) -> Categorical:
        """Return the safety-projected :class:`Categorical` policy over macro-actions."""
        safe_logits = logits - self.beta * self.violation(state)
        return Categorical(logits=safe_logits)
