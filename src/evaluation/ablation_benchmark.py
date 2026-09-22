"""
ablation_benchmark.py
Comprehensive Academic Ablation Benchmark:
1. Physical Saturation Clamping Ablation: Clamped vs. Unconstrained Hard PINN.
2. Multi-Step Autoregressive Horizon Degradation Curve: H in {1, 5, 15, 30, 60, 120}.
3. CEM MPC Parameter Sensitivity: Candidates N in {32, 64, 128, 256, 512}.

Generates:
- results/ablation_benchmark_metrics.json
- results/figures/ablation_study_comparison.png
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath("."))
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.models.pinn_soft import SoftPINNDynamics
from src.models.statistical_lstm import StatisticalLSTMDynamics
from src.models.statistical_mlp import StatisticalMLPDynamics
from src.planning.mpc_planner import (
    ModelPredictiveController,
    TrajectoryObjective,
)


class UnclampedHardResidualPINNDynamics(HardResidualPINNDynamics):
    """Ablation model: exact kinematic position integration WITHOUT velocity saturation clamping."""

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x_t = state[:, 0]
        y_t = state[:, 1]
        vx_t = state[:, 2]
        vy_t = state[:, 3]

        inp = torch.cat([state, action], dim=-1)
        force_out = self.force_net(inp)

        delta_vx = force_out[:, 0]
        delta_vy = force_out[:, 1]
        aux_pred = force_out[:, 2:]

        # ABLATION: No clamping on velocities
        hat_vx_next = vx_t + delta_vx
        hat_vy_next = vy_t + delta_vy

        # Kinematic integration
        hat_x_next = x_t + (hat_vx_next / self.subpixels_per_pixel)
        hat_y_next = y_t + (hat_vy_next / self.subpixels_per_pixel)

        next_state = torch.cat(
            [
                hat_x_next.unsqueeze(-1),
                hat_y_next.unsqueeze(-1),
                hat_vx_next.unsqueeze(-1),
                hat_vy_next.unsqueeze(-1),
                aux_pred,
            ],
            dim=-1,
        )
        return next_state


def run_clamping_ablation(device: torch.device, dataset_path: str = "data/raw/smw_gameplay_dataset.npz") -> Dict:
    print("\n--- ABLATION 1: PHYSICAL SATURATION CLAMPING ---")
    data = np.load(dataset_path)
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]

    split = int(0.8 * len(states))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(states[:split], dtype=torch.float32),
            torch.tensor(actions[:split], dtype=torch.float32),
            torch.tensor(next_states[:split], dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=True,
    )
    test_s = torch.tensor(states[split:], dtype=torch.float32).to(device)
    test_a = torch.tensor(actions[split:], dtype=torch.float32).to(device)
    test_targets = torch.tensor(next_states[split:], dtype=torch.float32).to(device)

    # Train Unclamped Model
    unclamped_model = UnclampedHardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    opt = torch.optim.AdamW(unclamped_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    for ep in range(15):
        unclamped_model.train()
        for b_s, b_a, b_t in train_loader:
            b_s, b_a, b_t = b_s.to(device), b_a.to(device), b_t.to(device)
            opt.zero_grad()
            pred = unclamped_model(b_s, b_a)
            loss = loss_fn(pred, b_t)
            loss.backward()
            opt.step()

    unclamped_model.eval()
    with torch.no_grad():
        unclamped_pred = unclamped_model(test_s, test_a)
        unclamped_mse = torch.mean((unclamped_pred - test_targets) ** 2).item()
        unclamped_max_vx = torch.max(torch.abs(unclamped_pred[:, 2])).item()

    # Load Clamped Model
    clamped_model = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    if os.path.exists("results/checkpoints/pinn_hard_best.pt"):
        clamped_model.load_state_dict(torch.load("results/checkpoints/pinn_hard_best.pt", map_location=device, weights_only=True))
    clamped_model.eval()
    with torch.no_grad():
        clamped_pred = clamped_model(test_s, test_a)
        clamped_mse = torch.mean((clamped_pred - test_targets) ** 2).item()
        clamped_max_vx = torch.max(torch.abs(clamped_pred[:, 2])).item()

    print(f"Hard Residual PINN (Clamped):   MSE = {clamped_mse:.4f} | Max |vx| = {clamped_max_vx:.1f} subpx")
    print(f"Hard Residual PINN (Unclamped): MSE = {unclamped_mse:.4f} | Max |vx| = {unclamped_max_vx:.1f} subpx")

    return {
        "clamped": {"test_mse": clamped_mse, "max_predicted_vx": clamped_max_vx},
        "unclamped": {"test_mse": unclamped_mse, "max_predicted_vx": unclamped_max_vx},
    }


def run_horizon_degradation_ablation(device: torch.device, dataset_path: str = "data/raw/smw_gameplay_dataset.npz") -> Dict:
    print("\n--- ABLATION 2: MULTI-STEP HORIZON DEGRADATION (H in 1..120) ---")
    data = np.load(dataset_path)
    states = data["states"]
    actions = data["actions"]

    # Pick a continuous rollout episode from test set
    split = int(0.8 * len(states))
    test_states = states[split:]
    test_actions = actions[split:]

    horizons = [1, 5, 15, 30, 60, 120]
    num_eval_rollouts = 10
    rollout_len = 120

    # Load all models
    models = {
        "Statistical MLP": StatisticalMLPDynamics(state_dim=8, action_dim=6).to(device),
        "Statistical LSTM": StatisticalLSTMDynamics(state_dim=8, action_dim=6).to(device),
        "Soft-Constrained PINN": SoftPINNDynamics(state_dim=8, action_dim=6).to(device),
        "Hard Residual PINN": HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device),
    }

    ckpt_map = {
        "Statistical MLP": "results/checkpoints/mlp_best.pt",
        "Statistical LSTM": "results/checkpoints/lstm_best.pt",
        "Soft-Constrained PINN": "results/checkpoints/pinn_soft_best.pt",
        "Hard Residual PINN": "results/checkpoints/pinn_hard_best.pt",
    }

    for name, m in models.items():
        if os.path.exists(ckpt_map[name]):
            m.load_state_dict(torch.load(ckpt_map[name], map_location=device, weights_only=True))
        m.eval()

    horizon_results = {name: {h: [] for h in horizons} for name in models}
    kinematic_violations = {name: 0.0 for name in models}

    # Evaluate across multiple sub-trajectories
    for trial in range(num_eval_rollouts):
        start_idx = trial * 130
        if start_idx + rollout_len >= len(test_states):
            break

        true_traj = test_states[start_idx : start_idx + rollout_len]  # [120, 8]
        acts = test_actions[start_idx : start_idx + rollout_len]      # [120, 6]

        for name, model in models.items():
            curr_s = torch.tensor(true_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
            sim_preds = []
            violations = 0

            # Autoregressive rollout
            for t in range(rollout_len):
                a_t = torch.tensor(acts[t], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    if "LSTM" in name:
                        pred_seq, _ = model(curr_s.unsqueeze(1), a_t.unsqueeze(1))
                        next_s = pred_seq[:, -1, :]
                    else:
                        next_s = model(curr_s, a_t)

                # Check kinematic violation: ||X_{t+1} - (X_t + vx/16)||^2 > 0.05
                dx_pred = next_s[0, 0].item() - curr_s[0, 0].item()
                dx_phys = curr_s[0, 2].item() / 16.0
                if abs(dx_pred - dx_phys) > 0.05:
                    violations += 1

                sim_preds.append(next_s.cpu().numpy()[0])
                curr_s = next_s

            sim_preds = np.array(sim_preds)
            kinematic_violations[name] += (violations / rollout_len) * 100.0

            # Compute error at each horizon
            for h in horizons:
                mse_h = float(np.mean((sim_preds[:h, :2] - true_traj[:h, :2]) ** 2))
                horizon_results[name][h].append(mse_h)

    # Average over trials
    final_horizon_metrics = {}
    for name in models:
        final_horizon_metrics[name] = {
            "drift_by_horizon": {h: float(np.mean(horizon_results[name][h])) for h in horizons},
            "mean_kinematic_violation_pct": float(kinematic_violations[name] / max(1, num_eval_rollouts)),
        }
        print(f"{name:24s} | H=1: {final_horizon_metrics[name]['drift_by_horizon'][1]:6.2f} | "
              f"H=30: {final_horizon_metrics[name]['drift_by_horizon'][30]:7.2f} | "
              f"H=120: {final_horizon_metrics[name]['drift_by_horizon'][120]:8.2f} | "
              f"Violations: {final_horizon_metrics[name]['mean_kinematic_violation_pct']:.1f}%")

    return final_horizon_metrics


def run_mpc_sensitivity_ablation(device: torch.device) -> Dict:
    print("\n--- ABLATION 3: CEM MPC CANDIDATE SENSITIVITY (N in 32..512) ---")
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)
    if os.path.exists("results/checkpoints/pinn_multi_entity_best.pt"):
        world_model.load_state_dict(torch.load("results/checkpoints/pinn_multi_entity_best.pt", map_location=device, weights_only=True))
    world_model.eval()

    candidate_values = [32, 64, 128, 256, 512]
    # Sample state facing hazard
    sample_state = np.array([100.0, 336.0, 24.0, 0.0, 1.0, 0.0, 0.0, 0.0, 60.0, 0.0, -16.0, 1.0], dtype=np.float32)

    cem_results = {}
    for N in candidate_values:
        controller = ModelPredictiveController(
            world_model=world_model,
            device=device,
            horizon=16,
            num_candidates=N,
            cem_iterations=3,
            objective=TrajectoryObjective(weight_progress=3.0, hazard_penalty=800.0, leap_bonus=350.0),
        )

        # Warmup
        for _ in range(5):
            controller.plan(sample_state)

        # Benchmark 20 planning steps
        times = []
        rewards = []
        for _ in range(20):
            t0 = time.perf_counter()
            _, info = controller.plan(sample_state)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
            rewards.append(info["best_reward"])

        mean_ms = float(np.mean(times))
        fps = float(1000.0 / mean_ms)
        mean_rew = float(np.mean(rewards))

        cem_results[N] = {
            "mean_latency_ms": mean_ms,
            "throughput_fps": fps,
            "mean_planning_reward": mean_rew,
        }
        print(f"Candidates N={N:3d} | Latency: {mean_ms:5.2f} ms | Throughput: {fps:5.1f} FPS | Reward: {mean_rew:6.1f}")

    return cem_results


def plot_ablation_results(horizon_data: Dict, cem_data: Dict, output_figure: str):
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Rollout Degradation Curve
    horizons = [1, 5, 15, 30, 60, 120]
    colors = {
        "Statistical MLP": "#d62728",
        "Statistical LSTM": "#ff7f0e",
        "Soft-Constrained PINN": "#9467bd",
        "Hard Residual PINN": "#2ca02c",
    }
    for name, data in horizon_data.items():
        vals = [data["drift_by_horizon"][h] for h in horizons]
        ax1.plot(horizons, vals, marker="o", linewidth=2.2, label=name, color=colors[name])
    ax1.set_yscale("log")
    ax1.set_xlabel("Rollout Horizon H (frames / steps)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Trajectory Drift MSE (log scale)", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Open-Loop Autoregressive Error Growth", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")

    # Panel 2: Kinematic Violations Percentage
    names = list(horizon_data.keys())
    viols = [horizon_data[n]["mean_kinematic_violation_pct"] for n in names]
    bar_colors = [colors[n] for n in names]
    ax2.bar(range(len(names)), viols, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(["MLP", "LSTM", "Soft PINN", "Hard PINN\n(Ours)"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Kinematic Violation Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Physical Constraint Violations (120 Steps)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    for i, v in enumerate(viols):
        ax2.text(i, v + 2.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)

    # Panel 3: CEM Candidates Sensitivity (Latency vs Reward)
    candidates = list(cem_data.keys())
    latencies = [cem_data[c]["mean_latency_ms"] for c in candidates]
    rewards = [cem_data[c]["mean_planning_reward"] for c in candidates]

    ax3_twin = ax3.twinx()
    p1 = ax3.plot(candidates, latencies, color="#1f77b4", marker="s", linewidth=2.0, label="Planning Latency (ms)")
    p2 = ax3_twin.plot(candidates, rewards, color="#e377c2", marker="^", linewidth=2.0, label="Objective Reward")

    ax3.set_xlabel("CEM Candidate Trajectories (N)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latency per Step (ms)", color="#1f77b4", fontsize=11, fontweight="bold")
    ax3_twin.set_ylabel("Trajectory Reward", color="#e377c2", fontsize=11, fontweight="bold")
    ax3.set_title("(C) MPC Planning Trade-Off (N vs. Speed/Reward)", fontsize=12, fontweight="bold")
    ax3.axhline(y=16.67, color="black", linestyle="--", alpha=0.6, label="Real-time 60 Hz Limit (16.7 ms)")

    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    print(f"\nAblation figure saved to: {output_figure}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics = {}
    metrics["ablation_clamping"] = run_clamping_ablation(device)
    metrics["ablation_horizon_degradation"] = run_horizon_degradation_ablation(device)
    metrics["ablation_cem_mpc"] = run_mpc_sensitivity_ablation(device)

    metrics_path = "results/ablation_benchmark_metrics.json"
    figure_path = "results/figures/ablation_study_comparison.png"

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Ablation metrics saved to: {metrics_path}")
    plot_ablation_results(metrics["ablation_horizon_degradation"], metrics["ablation_cem_mpc"], figure_path)


if __name__ == "__main__":
    main()
