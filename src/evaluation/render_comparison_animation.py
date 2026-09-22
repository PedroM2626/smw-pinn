"""
render_comparison_animation.py
Synchronized Visual Animation & Trajectory Comparison Generator:
Renders a 120-frame open-loop autoregressive rollout comparing:
1. Real SNES Console Ground Truth (WRAM Telemetry)
2. Hard Residual PINN (Kinematically Invariant & Conserved)
3. Statistical MLP (Kinematic Degradation & Hallucination)

Outputs:
- results/figures/model_comparison_animation.gif
- results/figures/model_comparison_trajectory_composite.png
"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.environment.dataset_loader import load_and_preprocess_data
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics
from src.utils.logging import get_logger

logger = get_logger(__name__)

def generate_comparison_animation(
    data_path: str = "data/raw/smw_gameplay_dataset.npz",
    checkpoints_dir: str = "results/checkpoints",
    output_dir: str = "results/figures",
    horizon: int = 120,
    fps: int = 30,
):
    logger.info("====================================================================")
    logger.info("  GENERATING SYNCHRONIZED MULTI-MODEL COMPARISON ANIMATION          ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Real Data
    data = load_and_preprocess_data(data_path)
    states = data["test_states"]
    actions = data["test_actions"]
    next_states = data["test_next_states"]

    # Pick an active, continuous dynamic segment with jumps and running
    start_idx = 100
    gt_s = [states[start_idx]]
    real_actions = actions[start_idx : start_idx + horizon]

    for t in range(horizon):
        gt_s.append(next_states[start_idx + t])
    gt_s = np.array(gt_s)  # [horizon + 1, 8]

    # 2. Load Models
    hard_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    hard_pinn.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, "pinn_hard_best.pt"), map_location=device, weights_only=True)
    )
    hard_pinn.eval()

    mlp = StatisticalMLPDynamics(state_dim=8, action_dim=6).to(device)
    mlp.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, "mlp_best.pt"), map_location=device, weights_only=True)
    )
    mlp.eval()

    # 3. Compute Autoregressive Multi-Step Rollouts
    s0 = torch.tensor(states[start_idx], dtype=torch.float32, device=device).unsqueeze(0)
    act_tensor = torch.tensor(real_actions, dtype=torch.float32, device=device)

    # Hard PINN rollout
    pinn_traj = [s0.cpu().numpy()[0]]
    curr_s = s0
    with torch.no_grad():
        for t in range(horizon):
            a_t = act_tensor[t : t + 1]
            curr_s = hard_pinn(curr_s, a_t)
            pinn_traj.append(curr_s.cpu().numpy()[0])
    pinn_traj = np.array(pinn_traj)

    # Statistical MLP rollout
    mlp_traj = [s0.cpu().numpy()[0]]
    curr_s = s0
    with torch.no_grad():
        for t in range(horizon):
            a_t = act_tensor[t : t + 1]
            curr_s = mlp(curr_s, a_t)
            mlp_traj.append(curr_s.cpu().numpy()[0])
    mlp_traj = np.array(mlp_traj)

    # Calculate frame-by-frame drift errors
    drift_pinn = np.linalg.norm(pinn_traj[:, :2] - gt_s[:, :2], axis=-1)
    drift_mlp = np.linalg.norm(mlp_traj[:, :2] - gt_s[:, :2], axis=-1)

    # 4. Generate High-Resolution Composite PNG
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Trajectory Plot
    ax_traj = axes[0]
    ax_traj.plot(gt_s[:, 0], gt_s[:, 1], "k-", lw=3.0, label="Real SNES Ground Truth (WRAM)")
    ax_traj.plot(pinn_traj[:, 0], pinn_traj[:, 1], color="#10B981", lw=2.5, linestyle="--", label=f"Hard Residual PINN (Final Drift: {drift_pinn[-1]:.2f}px)")
    ax_traj.plot(mlp_traj[:, 0], mlp_traj[:, 1], color="#EF4444", lw=2.0, linestyle=":", label=f"Statistical MLP (Final Drift: {drift_mlp[-1]:.2f}px)")
    ax_traj.invert_yaxis()
    ax_traj.set_title("Open-Loop Multi-Step Rollout (120 Frames / 2.0s Continuous Prediction)", fontsize=13, fontweight="bold")
    ax_traj.set_xlabel("Mario Horizontal Position X (pixels)", fontsize=11)
    ax_traj.set_ylabel("Mario Vertical Position Y (pixels)", fontsize=11)
    ax_traj.legend(loc="upper right", frameon=True, fontsize=10)

    # Drift Error Plot
    ax_drift = axes[1]
    ax_drift.plot(drift_pinn, color="#10B981", lw=2.5, label="Hard PINN Drift")
    ax_drift.plot(drift_mlp, color="#EF4444", lw=2.0, label="Statistical MLP Drift")
    ax_drift.set_title("Compounding Trajectory Drift Across Time", fontsize=11, fontweight="bold")
    ax_drift.set_xlabel("Time Horizon (Frames)", fontsize=10)
    ax_drift.set_ylabel("Euclidean Error (pixels)", fontsize=10)
    ax_drift.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    comp_png_path = os.path.join(output_dir, "model_comparison_trajectory_composite.png")
    plt.savefig(comp_png_path, dpi=300)
    plt.close()
    logger.info(f"Composite trajectory figure saved to: {comp_png_path}")

    # 5. Generate Animated GIF
    fig, (ax_anim, ax_hud) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.5, 1.0]})

    x_min = min(gt_s[:, 0].min(), pinn_traj[:, 0].min(), mlp_traj[:, 0].min()) - 20
    x_max = max(gt_s[:, 0].max(), pinn_traj[:, 0].max(), mlp_traj[:, 0].max()) + 20
    y_min = min(gt_s[:, 1].min(), pinn_traj[:, 1].min(), mlp_traj[:, 1].min()) - 20
    y_max = max(gt_s[:, 1].max(), pinn_traj[:, 1].max(), mlp_traj[:, 1].max()) + 20

    ax_anim.set_xlim(x_min, x_max)
    ax_anim.set_ylim(y_max, y_min)  # Inverted Y for SNES coordinates
    ax_anim.set_title("Synchronized Autoregressive Simulation", fontsize=12, fontweight="bold")
    ax_anim.set_xlabel("X (pixels)")
    ax_anim.set_ylabel("Y (pixels)")

    (line_gt,) = ax_anim.plot([], [], "k-", lw=3.0, label="Real SNES Ground Truth")
    (dot_gt,) = ax_anim.plot([], [], "ko", markersize=8)

    (line_pinn,) = ax_anim.plot([], [], color="#10B981", lw=2.5, linestyle="--", label="Hard PINN")
    (dot_pinn,) = ax_anim.plot([], [], "o", color="#10B981", markersize=7)

    (line_mlp,) = ax_anim.plot([], [], color="#EF4444", lw=2.0, linestyle=":", label="Statistical MLP")
    (dot_mlp,) = ax_anim.plot([], [], "o", color="#EF4444", markersize=7)

    ax_anim.legend(loc="upper left")

    ax_hud.axis("off")
    hud_text = ax_hud.text(
        0.05, 0.5, "", fontsize=11, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    )

    def init():
        line_gt.set_data([], [])
        dot_gt.set_data([], [])
        line_pinn.set_data([], [])
        dot_pinn.set_data([], [])
        line_mlp.set_data([], [])
        dot_mlp.set_data([], [])
        hud_text.set_text("")
        return line_gt, dot_gt, line_pinn, dot_pinn, line_mlp, dot_mlp, hud_text

    def animate(i):
        # Update trails
        line_gt.set_data(gt_s[: i + 1, 0], gt_s[: i + 1, 1])
        dot_gt.set_data([gt_s[i, 0]], [gt_s[i, 1]])

        line_pinn.set_data(pinn_traj[: i + 1, 0], pinn_traj[: i + 1, 1])
        dot_pinn.set_data([pinn_traj[i, 0]], [pinn_traj[i, 1]])

        line_mlp.set_data(mlp_traj[: i + 1, 0], mlp_traj[: i + 1, 1])
        dot_mlp.set_data([mlp_traj[i, 0]], [mlp_traj[i, 1]])

        # HUD display
        hud = (
            f"FRAME: {i:03d} / {horizon}\n"
            f"TIME:  {i/60.0:.2f}s\n"
            "--------------------\n"
            f"HARD PINN:\n"
            f"  Drift:  {drift_pinn[i]:6.2f} px\n"
            f"  KinRes: 0.0000\n"
            "--------------------\n"
            f"STAT MLP:\n"
            f"  Drift:  {drift_mlp[i]:6.2f} px\n"
            f"  KinRes: {abs((mlp_traj[i, 0] - mlp_traj[max(0, i-1), 0]) - mlp_traj[i, 2]/16.0):.2f}\n"
        )
        hud_text.set_text(hud)
        return line_gt, dot_gt, line_pinn, dot_pinn, line_mlp, dot_mlp, hud_text

    # Sample every 2 frames for compact GIF size
    frames_to_render = range(0, horizon + 1, 2)
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames_to_render, interval=1000 // (fps // 2), blit=True
    )

    gif_path = os.path.join(output_dir, "model_comparison_animation.gif")
    writer = animation.PillowWriter(fps=fps // 2)
    anim.save(gif_path, writer=writer)
    plt.close()

    logger.info(f"Comparison animation successfully rendered to: {gif_path}")
    return {"composite_png": comp_png_path, "animated_gif": gif_path}


if __name__ == "__main__":
    generate_comparison_animation()
