"""
statistical_mlp.py
Statistical Baseline: Multilayer Perceptron (MLP) purely data-driven,
without explicit physical priors or kinematic constraints.
"""

from typing import List
import torch
import torch.nn as nn


class StatisticalMLPDynamics(nn.Module):
    """
    Feedforward Dense Neural Network for direct next-state prediction:
        hat_s_{t+1} = MLP([s_t, a_t])
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 128],
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        in_dim = state_dim + action_dim

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: List[nn.Module] = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        # Output projection layer predicting full next state
        layers.append(nn.Linear(current_dim, state_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
            action: [B, action_dim]

        Returns:
            next_state_pred: [B, state_dim]
        """
        x = torch.cat([state, action], dim=-1)
        next_state_pred = self.network(x)
        return next_state_pred
