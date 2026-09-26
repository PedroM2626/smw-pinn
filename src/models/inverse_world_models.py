"""
inverse_world_models.py
The inverse-problem answers of Sections 10.40 and 10.43, wrapped as MPC world models.

Both sections recover physics from data, but neither produces an ``nn.Module`` with the
``(state, action) -> next_state`` signature that
:class:`src.planning.mpc_planner.ModelPredictiveController` consumes: 10.40 returns a
seven-constant vector, 10.43 returns a bag of genetic-program trees. This module supplies
that last mile so the two inverse models can be driven *closed loop on the console* and
compared with the published learned and hand-written models under the identical planner,
objective and budget (README Section 10.44).

State layout (8D), as everywhere in this repository:
``[x, y, vx, vy, c_ground, c_ceiling, c_left, c_right]``.

Contact handling is the honest one for control: the collision byte of the *current*
frame is fed to the map and propagated unchanged into the prediction, because the byte
that describes ``s_{t+1}`` does not exist yet when the planner is querying it. The
closed-form baseline of Section 10.37 does exactly the same for its output channels, so
the three models share the convention; the consequence - a one-frame-stale collision
response - is measured rather than hidden, and it is the same assumption the MPC of
Section 10.36 already makes.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.inverse.parameter_identification import simulate_step
from src.inverse.symbolic_regression import symbolic_step

__all__ = ["IdentifiedKinematicsDynamics", "SymbolicKinematicsDynamics"]


class IdentifiedKinematicsDynamics(nn.Module):
    """The Section 10.40 parametric map with a *recovered* constant vector.

    Unlike :class:`src.models.analytical_kinematics.AnalyticalKinematicsDynamics`, whose
    structure is the hand-written engine rules, this is the extended hybrid kinematic map
    of Section 10.40 - directional traction tiers, Coulomb coast friction, asymmetric
    held/fall gravity and the four-channel rigid collision response - evaluated at whatever
    ``theta`` identification returned. Nothing is trainable: the parameters are frozen into
    a buffer so a checkpoint carries the physics it was planned with.
    """

    theta: torch.Tensor

    def __init__(
        self,
        theta: Sequence[float] | torch.Tensor | np.ndarray,
        state_dim: int = 8,
        action_dim: int = 6,
    ) -> None:
        super().__init__()
        if state_dim != 8:
            raise ValueError(
                f"IdentifiedKinematicsDynamics models the 8D player state; got state_dim={state_dim}."
            )
        vec = torch.as_tensor(
            theta.detach().cpu() if isinstance(theta, torch.Tensor) else np.asarray(theta),
            dtype=torch.float32,
        ).reshape(-1)
        if vec.numel() != 7:
            raise ValueError(f"theta must carry the seven identified constants, got {vec.numel()}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.register_buffer("theta", vec)

    @property
    def num_parameters(self) -> int:
        """Seven physical constants, no weights."""
        return int(self.theta.numel())

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state channels {self.state_dim}, got {state.shape}")
        predicted = simulate_step(state[:, :4], action, self.theta, state[:, 4:8])
        return torch.cat([predicted, state[:, 4:]], dim=-1)


class SymbolicKinematicsDynamics(nn.Module):
    """The Section 10.43 discovered laws as a planable world model.

    Genetic programming evaluates in numpy, so the forward pass crosses the framework
    boundary once per call: the batch is moved to CPU float64, the four discovered
    increment laws are composed by the same ``symbolic_step`` that produced the
    Section 10.43 rollout numbers, and the result is moved back. The model is
    non-differentiable by construction, which is fine for the sampling planner (CEM) used
    here and is stated as a limitation for gradient-based ones.
    """

    def __init__(
        self,
        laws: Dict[str, Any],
        state_dim: int = 8,
        action_dim: int = 6,
    ) -> None:
        super().__init__()
        missing = {"dvx", "dvy", "dx", "dy"} - set(laws)
        if missing:
            raise ValueError(f"the symbolic world model needs every law, missing {sorted(missing)}")
        if state_dim != 8:
            raise ValueError(
                f"SymbolicKinematicsDynamics models the 8D player state; got state_dim={state_dim}."
            )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.laws = laws

    @property
    def num_parameters(self) -> int:
        """Node count is the honest complexity measure for a discovered program."""
        return int(sum(getattr(law, "n_nodes", 0) for law in self.laws.values()))

    def expressions(self) -> Dict[str, str]:
        return {name: str(getattr(law, "expression", repr(law))) for name, law in self.laws.items()}

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state channels {self.state_dim}, got {state.shape}")
        cpu = state.detach().to("cpu").numpy().astype(np.float64)
        acts = action.detach().to("cpu").numpy().astype(np.float64)
        predicted = symbolic_step(cpu[:, :4], acts, cpu[:, 4:8], self.laws)
        out = np.concatenate([predicted, cpu[:, 4:8]], axis=1)
        return torch.as_tensor(out, dtype=state.dtype, device=state.device)
