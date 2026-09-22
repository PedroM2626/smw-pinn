"""Tilemap-conditioned closed-loop MPC (orphan connection §3).

`TilemapPINNDynamics` predicts `next = f(kinematics, patch, action)`, so the
MPC's `(state, action)` interface needs an adapter. `TilemapMPCWrapper` holds
the latest WRAM patch observed on hardware and reuses it across the H
imagined steps (static-map approximation, stated explicitly: future patches
are unknown without unrolling the global tile buffer).

This gives the planner anticipatory geometry (pipes/ledges *before* contact),
unlike the contact-flag reflexes that only fire once `c_right == 1`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.tilemap_pinn import TilemapPINNDynamics
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TilemapMPCWrapper(nn.Module):
    """Adapts (kinematics, patch, action) dynamics to the MPC interface."""

    def __init__(self, tilemap_model: TilemapPINNDynamics | None = None):
        super().__init__()
        self.model = tilemap_model or TilemapPINNDynamics()
        self._patch: torch.Tensor | None = None

    def set_patch(self, patch: np.ndarray) -> None:
        """Stores the current 7x7 WRAM patch (int classes 0..3)."""
        arr = np.asarray(patch, dtype=np.int64)
        if arr.shape != (7, 7):
            raise ValueError(f"Expected patch [7, 7], got {arr.shape}")
        self._patch = torch.from_numpy(arr).long()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._patch is None:
            raise RuntimeError("set_patch() must be called before forward().")
        batch = state.shape[0]
        device = state.device
        patch_batch = self._patch.to(device).unsqueeze(0).expand(batch, -1, -1)
        next_state, _ = self.model(state, patch_batch, action)
        return next_state
