"""
statistical_lstm.py
Temporal Recurrent Baseline: LSTM processing sequence windows of transitions
to model inertia and latent accelerations purely from empirical observations.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class StatisticalLSTMDynamics(nn.Module):
    """
    Recurrent Neural Network (LSTM) for autoregressive next-state forecasting:
        h_t, c_t = LSTM([s_t, a_t], (h_{t-1}, c_{t-1}))
        hat_s_{t+1} = Linear(h_t)
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        in_dim = state_dim + action_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            state_seq: [B, T, state_dim]
            action_seq: [B, T, action_dim]
            hidden: prior hidden state tuple (h_0, c_0)

        Returns:
            next_state_seq: [B, T, state_dim]
            hidden: updated hidden state tuple (h_T, c_T)
        """
        x = torch.cat([state_seq, action_seq], dim=-1)
        out, (h_n, c_n) = self.lstm(x, hidden)
        next_state_pred = self.head(out)
        return next_state_pred, (h_n, c_n)
