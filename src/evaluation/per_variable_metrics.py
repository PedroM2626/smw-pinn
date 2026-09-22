"""Per-variable regression / classification metrics for 8D SMW dynamics.

Aggregate MSE hides scale imbalance (X ~ thousands of px vs. contact flags in
{0, 1}). This module reports, per state channel:

- MSE / MAE / R2 for continuous channels (X, Y, vx, vy)
- Accuracy / F1 for binary contact flags (c_ground, c_ceiling, c_left, c_right)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

STATE_NAMES = ["x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right"]
CONTINUOUS_IDX = [0, 1, 2, 3]
CONTACT_IDX = [4, 5, 6, 7]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return float("nan")
    return 2 * tp / denom


def compute_per_variable_metrics(
    predicted: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    contact_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute per-channel metrics between predicted and target next-states.

    Args:
        predicted: [N, 8] model predictions.
        target: [N, 8] ground-truth next states.
        contact_threshold: binarization threshold for contact-flag heads.

    Returns:
        {"x": {"mse": .., "mae": .., "r2": ..}, ..., "c_ground": {"accuracy": .., "f1": ..}, ...}
    """
    pred = _to_numpy(predicted)
    tgt = _to_numpy(target)
    if pred.shape != tgt.shape or pred.ndim != 2 or pred.shape[1] != 8:
        raise ValueError(f"Expected [N, 8] predictions/targets, got {pred.shape} / {tgt.shape}")

    out: Dict[str, Dict[str, float]] = {}
    for i in CONTINUOUS_IDX:
        name = STATE_NAMES[i]
        err = pred[:, i] - tgt[:, i]
        out[name] = {
            "mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(_r2_score(tgt[:, i], pred[:, i])),
        }
    for i in CONTACT_IDX:
        name = STATE_NAMES[i]
        bin_pred = (pred[:, i] >= contact_threshold).astype(np.int32)
        bin_tgt = (tgt[:, i] >= contact_threshold).astype(np.int32)
        out[name] = {
            "accuracy": float(np.mean(bin_pred == bin_tgt)),
            "f1": float(_binary_f1(bin_tgt, bin_pred)),
        }
    return out
