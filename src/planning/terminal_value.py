"""TD-MPC terminal value: replace hand-coded reflexes with a learned V(s_H).

Finite-horizon MPC (H = 15) cannot see full 30-50-frame leaps, which is why
published runs overlay wall/hazard reflexes. The principled fix (TD-MPC
paradigm) is a terminal value function: reward = base + gamma^H * V(s_H),
where V is fit by Monte-Carlo regression on genuine hardware returns
(progress per frame) from `results/full_level_trajectory_log.json`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.planning.global_planner import WaypointObjective
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

VALUE_STATE_DIM = 4  # [x, y, vx, vy] subset of the 8D state


class TerminalValueNet(nn.Module):
    """Small MLP state-value function V([x, y, vx, vy])."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VALUE_STATE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xyvv: torch.Tensor) -> torch.Tensor:
        return self.net(xyvv).squeeze(-1)


def compute_mc_returns(progress: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Return-to-go G_t = sum_{k>=0} gamma^k * (x_{t+k+1} - x_{t+k})."""
    progress = np.asarray(progress, dtype=np.float64)
    rewards = np.diff(progress, prepend=progress[0])
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def fit_terminal_value(
    states_xyvv: np.ndarray,
    returns: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[TerminalValueNet, Dict[str, float]]:
    """Fits V by MSE regression on standardized returns. Returns (net, stats)."""
    set_global_seed(seed)
    mu, sigma = float(returns.mean()), float(returns.std() or 1.0)
    target = (returns - mu) / sigma
    X = torch.tensor(np.asarray(states_xyvv, dtype=np.float32))
    Y = torch.tensor(target.astype(np.float32))
    net = TerminalValueNet(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        opt.step()
    stats = {"target_mean": mu, "target_std": sigma}
    return net, stats


class TerminalValueObjective(WaypointObjective):
    """Waypoint-steered MPC objective + discounted learned terminal value.

    Combines global waypoint tracking (strategic) with the TD-MPC terminal
    term (tactical myopia fix), replacing hand-coded reflexes.

    Args:
        value_net: fitted TerminalValueNet (eval mode).
        target_mean/std: return standardization stats from fitting.
        terminal_weight: weight of the terminal term.
        gamma: discount; raised to H inside (uses objective horizon at call).
        Remaining kwargs follow WaypointObjective (target_x/y, weights).
    """

    def __init__(
        self,
        value_net: TerminalValueNet,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        terminal_weight: float = 1.0,
        gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_net = value_net
        self.value_net.eval()
        self.target_mean = target_mean
        self.target_std = target_std
        self.terminal_weight = terminal_weight
        self.gamma = gamma

    @torch.no_grad()
    def compute_trajectory_rewards(
        self,
        initial_states: torch.Tensor,
        predicted_trajectories: torch.Tensor,
    ) -> torch.Tensor:
        base = super().compute_trajectory_rewards(initial_states, predicted_trajectories)
        horizon = predicted_trajectories.shape[1]
        final_xyvv = predicted_trajectories[:, -1, :4]
        v_norm = self.value_net(final_xyvv)
        v = v_norm * self.target_std + self.target_mean
        return base + self.terminal_weight * (self.gamma**horizon) * v
