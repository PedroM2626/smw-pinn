"""
pinn_soft.py
Physics-Informed Neural Network with Soft Regularization (Soft-Constrained PINN).
Maintains identical architectural capacity to the baseline MLP to ensure fair comparison,
optimized jointly via supervised empirical loss and physical residual losses.
"""

from typing import List

import torch
import torch.nn as nn

from src.models.statistical_mlp import StatisticalMLPDynamics


class SoftPINNDynamics(nn.Module):
    """
    Soft-Constrained PINN.
    Isomorphic architecture to the statistical MLP, isolating the experimental variable
    to determine whether physical regularization terms in the objective function guide
    gradients toward physically grounded representations with lower multi-step rollout drift.
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
        # Employs identical structure to isolate the experimental loss variable
        self.backbone = StatisticalMLPDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.backbone(state, action)
