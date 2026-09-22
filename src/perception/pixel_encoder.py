"""Convolutional pixel-to-state estimator + target normalizer.

Maps native SNES RGB frames [B, 3, H, W] (typically 256x224) directly to the
8D kinematic WRAM vector [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right].
The network is size-agnostic (AdaptiveAvgPool), so hires 512x448 frames work.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class StateNormalizer:
    """Per-dimension standardization fitted on training states.

    Coordinates (X ~ thousands of px) would otherwise dominate the loss;
    contact flags ({0, 1}) would be ignored. The fitted stats are stored in
    the checkpoint so inference denormalizes identically.
    """

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, states: np.ndarray) -> "StateNormalizer":
        states = np.asarray(states, dtype=np.float64)
        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.mean is not None and self.std is not None, "Call fit() first."
        return self.mean, self.std

    def normalize(self, states: np.ndarray | torch.Tensor):
        mean, std = self._stats()
        if isinstance(states, torch.Tensor):
            mean_t = torch.as_tensor(mean, dtype=states.dtype, device=states.device)
            std_t = torch.as_tensor(std, dtype=states.dtype, device=states.device)
            return (states - mean_t) / std_t
        return (np.asarray(states, dtype=np.float64) - mean) / std

    def denormalize(self, states: np.ndarray | torch.Tensor):
        mean, std = self._stats()
        if isinstance(states, torch.Tensor):
            mean_t = torch.as_tensor(mean, dtype=states.dtype, device=states.device)
            std_t = torch.as_tensor(std, dtype=states.dtype, device=states.device)
            return states * std_t + mean_t
        return np.asarray(states, dtype=np.float64) * std + mean

    def to_dict(self) -> Dict[str, List[float]]:
        mean, std = self._stats()
        return {"mean": list(map(float, mean)), "std": list(map(float, std))}

    @classmethod
    def from_dict(cls, payload: Dict[str, List[float]]) -> "StateNormalizer":
        return cls(
            mean=np.array(payload["mean"], dtype=np.float64),
            std=np.array(payload["std"], dtype=np.float64),
        )


class PixelStateEstimator(nn.Module):
    """Small CNN regressor: frames -> 8D state.

    Args:
        state_dim: 8 (WRAM kinematics + contact flags).
        base_channels: width multiplier (16 ≈ 150k params; use 4-8 in tests).
    """

    def __init__(self, state_dim: int = 8, base_channels: int = 16):
        super().__init__()
        b = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, b, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(b, 2 * b, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(2 * b, 4 * b, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * b * 16, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Args: frames [B, 3, H, W] float in [0, 1]. Returns: [B, 8]."""
        return self.head(self.features(frames))


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """uint8 [H, W, 3] -> float32 [1, 3, H, W] in [0, 1]."""
    arr = np.asarray(frame, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
