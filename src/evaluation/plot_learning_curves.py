"""
plot_learning_curves.py
Formal Model-Free PPO vs. Dyna-PINN sample-efficiency comparison figure.

Uses only committed artifacts (no new runs):
  - results/model_free_ppo_metrics.json (103-episode return curve, 39,936 steps)
  - results/dyna_ppo_metrics.json (deployment endpoints)
  - results/sample_efficiency_metrics.json (Hard PINN N=200 vs MLP N=5000 MSE)

Honesty note: no per-step Dyna curve was ever logged, so Dyna appears as
annotated operating bands (real frames consumed), never as a fabricated curve.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_plot(output_dir: str = "results") -> dict:
    with open(os.path.join(output_dir, "model_free_ppo_metrics.json"), encoding="utf-8") as f:
        mf = json.load(f)
    with open(os.path.join(output_dir, "dyna_ppo_metrics.json"), encoding="utf-8") as f:
        dyna = json.load(f)
    with open(os.path.join(output_dir, "sample_efficiency_metrics.json"), encoding="utf-8") as f:
        se = json.load(f)

    steps = np.array(mf["step_history"], dtype=float)
    returns = np.array(mf["return_history"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: model-free learning curve + Dyna operating bands.
    ax1.plot(steps, returns, lw=2.0, label="Model-Free PPO (real SNES, 103 eps)")
    ax1.axvspan(200, 8077, color="#10B981", alpha=0.2,
                label="Dyna-PINN operating band (200-8,077 real frames)")
    ax1.axvline(39936, color="black", ls="--", lw=1.0, label="MF convergence (39,936 frames)")
    ax1.set_xlabel("Genuine environment frames")
    ax1.set_ylabel("Mean return")
    ax1.set_title("(A) Learning curve: tabula-rasa PPO vs Dyna-PINN band")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: test MSE at matched data budgets (log scale).
    sizes = se["sample_sizes"]
    for name, color in (("Statistical_MLP", "#d62728"), ("Hard_Residual_PINN", "#2ca02c")):
        ax2.plot(sizes, se["results"][name]["test_mse"], marker="o", lw=2.0, label=name)
    ax2.set_xscale("log")
    ax2.set_xlabel("Training transitions N (log)")
    ax2.set_ylabel("Test MSE")
    ax2.set_title("(B) World-model error vs data budget")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "learning_curve_comparison.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    payload = {
        "model_free_frames_to_converge": int(mf["total_real_steps"]),
        "model_free_final_return": float(mf["mean_final_return"]),
        "dyna_hard_progress_px": float(dyna["Dyna_PPO_Hard_PINN"]["total_progress_pixels"]),
        "dyna_real_frame_band": [200, 8077],
        "figure": fig_path,
    }
    with open(os.path.join(output_dir, "learning_curve_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Learning-curve comparison -> {fig_path}")
    return payload


if __name__ == "__main__":
    run_plot()
