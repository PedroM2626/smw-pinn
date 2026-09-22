"""
spatial_holdout_benchmark.py
OOD-with-danger evaluation without new savestates (YI2 capture blocked;
see README §10.36): same engine, unseen far-region geometry + live enemies.

Protocol: committed checkpoints (trained on full YI1) are evaluated
zero-shot on far-region slices (Mario X > 700) of the genuine datasets:
  - 8D gameplay data: single-step MSE, violations, 120-frame rollout;
  - 12D multi-entity data: hazard-active frame census + single-step MSE.

No retraining, no emulator needed (pure .npz) — runs in CI.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics
from src.utils.logging import get_logger
from src.utils.paths import (
    CHECKPOINTS_DIR,
    DATASET_GAMEPLAY,
    DATASET_MULTI_ENTITY,
    RESULTS_DIR,
)
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLIT_X = 700.0


def far_region_mask(states: np.ndarray, threshold: float = SPLIT_X) -> np.ndarray:
    """Boolean mask for the held-out far region (never trained on explicitly)."""
    return np.asarray(states[:, 0]) > threshold


def run_spatial_holdout(
    dataset_path: str = DATASET_GAMEPLAY,
    multi_path: str = DATASET_MULTI_ENTITY,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    output_dir: str = RESULTS_DIR,
    seed: int = 42,
) -> dict:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(dataset_path)
    mask = far_region_mask(data["states"])
    far_states = data["states"][mask]
    far_actions = data["actions"][mask]
    far_next = data["next_states"][mask]
    logger.info(f"Far-region 8D transitions: {len(far_states)} / {len(data['states'])}")

    multi = np.load(multi_path)
    multi_mask = far_region_mask(multi["states"])
    hazard_active = multi["states"][multi_mask][:, 11] > 0.5
    logger.info(
        f"Far-region 12D transitions: {int(multi_mask.sum())} "
        f"({int(hazard_active.sum())} with live hazard)"
    )

    models = {
        "Statistical_MLP": (StatisticalMLPDynamics(), "mlp_best.pt"),
        "Hard_Residual_PINN": (HardResidualPINNDynamics(), "pinn_hard_best.pt"),
    }
    crit = nn.MSELoss()
    evaluator = RolloutEvaluator(device=device)
    H = min(120, len(far_states) - 1)

    single, roll = {}, {}
    for name, (model, ckpt) in models.items():
        model = model.to(device)
        model.load_state_dict(
            torch.load(os.path.join(checkpoints_dir, ckpt), map_location=device, weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            s = torch.tensor(far_states, dtype=torch.float32, device=device)
            a = torch.tensor(far_actions, dtype=torch.float32, device=device)
            ns = torch.tensor(far_next, dtype=torch.float32, device=device)
            pred = model(s, a)
            single[name] = {
                "zero_shot_mse": float(crit(pred, ns).item()),
                "kinematic_violation_pct": float(
                    ((pred[:, 0] - s[:, 0] - pred[:, 2] / 16.0).abs() > 0.05).float().mean().item()
                    * 100.0
                ),
            }
        res = evaluator.evaluate_rollout(model, "mlp", far_states[0], far_actions[:H], far_next[:H])
        roll[name] = {
            "mean_drift_px": res["mean_drift"],
            "final_drift_px": res["final_drift"],
            "kinematic_violations": res["kinematic_violations"],
        }
        logger.info(
            f"[{name}] far-MSE={single[name]['zero_shot_mse']:.2f} "
            f"drift={roll[name]['mean_drift_px']:.1f}px"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(H)
    for name, (model, _) in models.items():
        model.eval()
        res = evaluator.evaluate_rollout(model, "mlp", far_states[0], far_actions[:H], far_next[:H])
        ax.plot(xs, res["euclidean_drift"], label=f"{name} (far region)", lw=2.0)
    ax.set_xlabel("Rollout frame")
    ax.set_ylabel("Drift (px)")
    ax.set_title("Zero-shot far-region rollout (X > 700, live hazards nearby)")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "spatial_holdout_comparison.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    payload = {
        "split_x": SPLIT_X,
        "far_8d_transitions": int(mask.sum()),
        "far_12d_transitions": int(multi_mask.sum()),
        "far_hazard_active_transitions": int(hazard_active.sum()),
        "single_step": single,
        "rollout": roll,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "spatial_holdout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    run_spatial_holdout()
