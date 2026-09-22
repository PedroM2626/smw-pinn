"""Paired (frame, WRAM state) dataset with seeded deterministic loaders."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.perception.pixel_encoder import StateNormalizer
from src.utils.seed import seed_worker


class FrameStateDataset(Dataset):
    """Frames uint8 [N, H, W, 3] + states float32 [N, 8].

    Returns:
        frame: float32 [3, H, W] in [0, 1].
        state: float32 [8] (optionally standardized via `normalizer`).
    """

    def __init__(
        self,
        frames: np.ndarray,
        states: np.ndarray,
        normalizer: Optional[StateNormalizer] = None,
    ):
        frames = np.asarray(frames, dtype=np.uint8)
        states = np.asarray(states, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames [N, H, W, 3], got {frames.shape}")
        if states.ndim != 2 or states.shape[1] != 8 or len(states) != len(frames):
            raise ValueError(f"Expected states [N, 8] matching frames, got {states.shape}")
        self.frames = frames
        self.states = states
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame = torch.from_numpy(self.frames[idx].astype(np.float32) / 255.0)
        frame = frame.permute(2, 0, 1).contiguous()
        state = torch.from_numpy(self.states[idx])
        if self.normalizer is not None:
            state = self.normalizer.normalize(state)
        return frame, state


def create_vision_loaders(
    frames: np.ndarray,
    states: np.ndarray,
    batch_size: int = 64,
    train_ratio: float = 0.85,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, StateNormalizer]:
    """Seeded train/val split; normalizer fitted on train only (no leakage)."""
    n = len(frames)
    n_train = max(1, int(n * train_ratio))
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:] or perm[-1:]

    normalizer = StateNormalizer().fit(np.asarray(states)[train_idx])
    train_ds = FrameStateDataset(frames[train_idx], states[train_idx], normalizer)
    val_ds = FrameStateDataset(frames[val_idx], states[val_idx], normalizer)

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=shuffle_gen,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, normalizer
