"""
deeponet.py
Neural Operator Baseline: DeepONet (Lu et al., 2021) for one-step dynamics.

DeepONet learns operators between function spaces rather than functions between
finite-dimensional vectors, under the universal approximation theorem for
operators. The one-step engine map is framed as an operator

    G: u -> v,   v(y_q) = G(u)(y_q),

where the input function u is the instantaneous dynamic regime of the engine
(WRAM kinematic state + latched controller bitmasks) sampled at m sensors, and
v is the next-state field queried at the coordinate y_q of an output channel.

Architecture (unstacked, bias-included formulation):

    G(u)(y_q) = sum_k branch_k(u) * trunk_k(y_q) + b0[q]

- branch: MLP mapping sensor readings [s_t, a_t] in R^14 to p basis
  coefficients. Input coordinate i plays the role of sensor x_i.
- trunk: MLP mapping a query coordinate y_q in [-1, 1] (the identity of the
  predicted state channel) to p basis functions.
- b0: per-channel bias, the constant term of the operator expansion.

Evaluating the model is a basis expansion: the branch learns a dictionary of p
global response patterns indexed by the input regime; the trunk evaluates each
pattern at the requested output coordinate. This factorized rank-p bottleneck
is a different inductive bias from both the monolithic statistical MLP and the
Hard Residual PINN (no analytic kinematics are embedded here on purpose: the
benchmark asks where pure operator learning sits between statistics and
embedded physics).
"""

from typing import List, Optional

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
) -> nn.Sequential:
    """Dense stack with LayerNorm + GELU blocks and a linear projection head."""
    layers: List[nn.Module] = []
    curr_dim = in_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(curr_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.GELU())
        curr_dim = h_dim
    layers.append(nn.Linear(curr_dim, out_dim))
    return nn.Sequential(*layers)


class DeepONetDynamics(nn.Module):
    """
    DeepONet world model for the discrete SMW engine map:

        hat_s_{t+1}[q] = <branch([s_t, a_t]), trunk(y_q)> + b0[q]

    with sensor readings u = [s_t, a_t] in R^(state_dim + action_dim), basis
    dimension p (latent_dim), and query coordinates y_q in [-1, 1] assigned to
    each of the state_dim output channels (linspace-spaced). Because the trunk
    accepts arbitrary coordinates, the learned operator can be queried at
    fractional/interpolating coordinates too; the canonical interface simply
    queries the registered channel grid.
    """

    # Class-level annotation so mypy resolves the register_buffer-assigned
    # attribute as a Tensor instead of falling back to the nn.Module
    # __getattr__ union on newer torch stubs (cf. cbf_projection.py).
    canonical_query_coords: torch.Tensor

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        branch_hidden_dims: Optional[List[int]] = None,
        trunk_hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        if branch_hidden_dims is None:
            branch_hidden_dims = [128, 128]
        if trunk_hidden_dims is None:
            trunk_hidden_dims = [128, 128]

        # Branch net: encodes the input function sampled at its m sensors,
        # i.e. the combined state-action reading z_t in R^(state_dim+action_dim).
        self.num_sensors = state_dim + action_dim
        self.branch = _build_mlp(self.num_sensors, branch_hidden_dims, latent_dim)

        # Trunk net: encodes the query coordinate y_q of the output function.
        self.trunk = _build_mlp(1, trunk_hidden_dims, latent_dim)

        # Constant (bias) term b0 of the operator expansion, one value per
        # canonical output channel.
        self.output_bias = nn.Parameter(torch.zeros(state_dim))

        # Canonical query grid: state channel q owns coordinate y_q in [-1, 1].
        # Registered as a buffer so it follows the module across devices and
        # serialization without being a trainable parameter.
        self.register_buffer(
            "canonical_query_coords",
            torch.linspace(-1.0, 1.0, state_dim).unsqueeze(-1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] kinematic state s_t.
            action: [B, action_dim] controller reading a_t.
            query_coords: optional [Q, 1] trunk coordinates; when omitted the
                canonical per-channel grid is used, yielding state_dim outputs.
                Arity mismatch is guarded below (bias only exists for the
                canonical grid).

        Returns:
            next_state_pred: [B, Q] operator evaluation at the query grid
            ([B, state_dim] on the canonical grid).
        """
        sensors = torch.cat([state, action], dim=-1)
        coefficients = self.branch(sensors)  # [B, p]

        coords: torch.Tensor
        if query_coords is None:
            canonical = True
            coords = self.canonical_query_coords  # [state_dim, 1]
        else:
            canonical = False
            coords = query_coords.to(state.device).reshape(-1, 1)

        basis = self.trunk(coords)  # [Q, p]
        out = torch.einsum("bp,qp->bq", coefficients, basis)

        # The constant term b0 is defined over the canonical channel grid; a
        # custom query set is a pure basis expansion without it.
        if canonical:
            out = out + self.output_bias.unsqueeze(0)
        return out


class PhysicsConstrainedDeepONetDynamics(nn.Module):
    """
    Physics-constrained neural operator (hybrid of DeepONet and the Hard
    Residual PINN, README 10.42).

    The operator branch/trunk predicts only the unmodeled *forces and contacts*
    - [delta_vx, delta_vy, c_ground, c_ceiling, c_left, c_right] as a basis
    expansion over the residual-output grid - while the Section 4 discrete
    kinematics are applied analytically in the computation graph, exactly as in
    HardResidualPINNDynamics:

        hat_vx = clamp(vx_t + delta_vx, -max_vx, max_vx)
        hat_vy = clamp(vy_t + delta_vy, min_vy, terminal_vy)
        hat_X  = X_t + hat_vx / 16.0
        hat_Y  = Y_t + hat_vy / 16.0

    The discrete kinematic consistency residual is therefore identically zero
    by construction: the operator learns forces, the engine rule integrates
    them. This isolates the contribution of the Section 10.41 finding - if the
    DeepONet basis prior helps, embedding it inside the hard kinematic shell
    should preserve the Hard PINN's guarantees while changing the force
    estimator's inductive bias.
    """

    # Class-level annotation: see DeepONetDynamics (buffer resolved as Tensor).
    residual_query_coords: torch.Tensor

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        branch_hidden_dims: Optional[List[int]] = None,
        trunk_hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
        subpixels_per_pixel: float = 16.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy
        self.subpixels_per_pixel = subpixels_per_pixel

        if branch_hidden_dims is None:
            branch_hidden_dims = [128, 128]
        if trunk_hidden_dims is None:
            trunk_hidden_dims = [128, 128]

        # The operator predicts velocities-and-contacts only: aux_dim residual
        # channels (2 velocity residuals + state_dim - 4 contact flags).
        self.aux_dim = state_dim - 2
        self.num_sensors = state_dim + action_dim
        self.branch = _build_mlp(self.num_sensors, branch_hidden_dims, latent_dim)
        self.trunk = _build_mlp(1, trunk_hidden_dims, latent_dim)
        self.output_bias = nn.Parameter(torch.zeros(self.aux_dim))
        self.register_buffer(
            "residual_query_coords",
            torch.linspace(-1.0, 1.0, self.aux_dim).unsqueeze(-1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] -> [X_t, Y_t, vx_t, vy_t, c_ground, ...]
            action: [B, action_dim]

        Returns:
            next_state: [B, state_dim] with exact discrete-kinematic consistency.
        """
        x_t = state[:, 0]
        y_t = state[:, 1]
        vx_t = state[:, 2]
        vy_t = state[:, 3]

        sensors = torch.cat([state, action], dim=-1)
        coefficients = self.branch(sensors)  # [B, p]
        basis = self.trunk(self.residual_query_coords)  # [aux_dim, p]
        residuals = torch.einsum("bp,qp->bq", coefficients, basis) + self.output_bias

        delta_vx = residuals[:, 0]
        delta_vy = residuals[:, 1]
        aux_pred = residuals[:, 2:]  # contact-flag logits

        # Hard Section-4 kinematics: physical saturation, exact integration.
        hat_vx_next = torch.clamp(vx_t + delta_vx, -self.max_vx, self.max_vx)
        hat_vy_next = torch.clamp(vy_t + delta_vy, self.min_vy, self.terminal_vy)
        hat_x_next = x_t + (hat_vx_next / self.subpixels_per_pixel)
        hat_y_next = y_t + (hat_vy_next / self.subpixels_per_pixel)

        return torch.cat(
            [
                hat_x_next.unsqueeze(-1),
                hat_y_next.unsqueeze(-1),
                hat_vx_next.unsqueeze(-1),
                hat_vy_next.unsqueeze(-1),
                aux_pred,
            ],
            dim=-1,
        )
