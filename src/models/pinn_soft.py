"""
pinn_soft.py
Rede Neural Informada pela Física com Regularização Suave (Soft-Constrained PINN).
Possui a mesma capacidade arquitetural da MLP baseline para garantir comparabilidade direta,
sendo otimizada conjuntamente via perda empírica supervisionada e perdas físicas de resíduo.
"""

from typing import List
import torch
import torch.nn as nn
from src.models.statistical_mlp import StatisticalMLPDynamics


class SoftPINNDynamics(nn.Module):
    """
    PINN com Restrição Suave.
    Arquitetura isomórfica à MLP estatística, permitindo testar diretamente a hipótese
    de se os termos de regularização física na função de perda guiam os gradientes para
    representações mais plausíveis e com menor drift multi-passo.
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
        # Utiliza exatamente a mesma estrutura para isolar a variável experimental
        self.backbone = StatisticalMLPDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.backbone(state, action)
