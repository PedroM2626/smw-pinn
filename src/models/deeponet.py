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
