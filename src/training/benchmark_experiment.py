"""
benchmark_experiment.py
Orchestrator script for the complete academic comparative benchmark:
Trains all 4 models under identical conditions, evaluates single-step accuracy,
long-horizon stability (rollout drift), and generates comparative analytical figures.
"""

import json
import os
import sys
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, os.path.abspath("."))
from src.environment.dataset_loader import (
    SMWSequenceDataset,
    create_dataloaders,
    load_and_preprocess_data,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalLSTMDynamics,
    StatisticalMLPDynamics,
)
from src.training.trainer import DynamicsTrainer


def run_comprehensive_benchmark(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    epochs: int = 35,
    batch_size: int = 128,
    seed: int = 42,
    output_dir: str = "results",
):
    print("====================================================================")
    print("  ACADEMIC BENCHMARK: STATISTICAL ML VS. PINN ON SUPER MARIO WORLD  ")
    print("====================================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if device.type == "cuda":
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load genuine RAM telemetry dataset
    print("\n[1/5] Loading and partitioning genuine WRAM transitions...")
    data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=batch_size)

    print(
        f"Transitions - Train: {len(data_dict['train_states'])}, "
        f"Validation: {len(data_dict['val_states'])}, "
        f"Test: {len(data_dict['test_states'])}"
    )

    state_dim = data_dict["train_states"].shape[1]
    action_dim = data_dict["train_actions"].shape[1]

    # 2. Instantiate comparative neural architectures
    print("\n[2/5] Initializing comparative neural architectures...")
    models = {
        "Statistical_MLP": StatisticalMLPDynamics(state_dim=state_dim, action_dim=action_dim),
        "Statistical_LSTM": StatisticalLSTMDynamics(state_dim=state_dim, action_dim=action_dim),
        "Soft_PINN": SoftPINNDynamics(state_dim=state_dim, action_dim=action_dim),
        "Hard_Residual_PINN": HardResidualPINNDynamics(state_dim=state_dim, action_dim=action_dim),
    }

    trainers: Dict[str, DynamicsTrainer] = {}
    histories: Dict[str, dict] = {}

    # Sequential dataset specifically for LSTM
    train_seq_ds = SMWSequenceDataset(
        data_dict["train_states"],
        data_dict["train_actions"],
        data_dict["train_next_states"],
        data_dict["train_episodes"],
        seq_len=10,
    )
    val_seq_ds = SMWSequenceDataset(
        data_dict["val_states"],
        data_dict["val_actions"],
        data_dict["val_next_states"],
        data_dict["val_episodes"],
        seq_len=10,
    )
    test_seq_ds = SMWSequenceDataset(
        data_dict["test_states"],
        data_dict["test_actions"],
        data_dict["test_next_states"],
        data_dict["test_episodes"],
        seq_len=10,
    )

    train_seq_loader = torch.utils.data.DataLoader(train_seq_ds, batch_size=batch_size, shuffle=True)
    val_seq_loader = torch.utils.data.DataLoader(val_seq_ds, batch_size=batch_size, shuffle=False)
    test_seq_loader = torch.utils.data.DataLoader(test_seq_ds, batch_size=batch_size, shuffle=False)

    # 3. Comparative Training Loop
    print("\n[3/5] Training models under unified experimental protocol...")
    for name, model in models.items():
        if "lstm" in name.lower():
            m_type = "lstm"
        elif "soft" in name.lower():
            m_type = "pinn_soft"
        elif "hard" in name.lower():
            m_type = "pinn_hard"
        else:
            m_type = "mlp"

        trainer = DynamicsTrainer(
            model=model,
            model_type=m_type,
            device=device,
            learning_rate=1e-3,
            save_dir=os.path.join(output_dir, "checkpoints"),
        )
        trainers[name] = trainer

        cur_train_loader = train_seq_loader if m_type == "lstm" else train_loader
        cur_val_loader = val_seq_loader if m_type == "lstm" else val_loader

        hist = trainer.fit(
            train_loader=cur_train_loader,
            val_loader=cur_val_loader,
            epochs=epochs,
            patience=8,
            verbose=True,
        )
        histories[name] = hist

    # 4. Single-Step Accuracy Evaluation on Independent Test Split
    print("\n[4/5] Evaluating single-step accuracy on test split...")
    single_step_results = {}
    for name, trainer in trainers.items():
        cur_test_loader = test_seq_loader if "lstm" in name.lower() else test_loader
        eval_metrics = trainer.evaluate(cur_test_loader)
        single_step_results[name] = {
            "test_loss_data": eval_metrics["val_loss_data"],
            "test_kinematic_error": eval_metrics["val_loss_kinematics"],
        }
        print(
            f"{name:20s} | Test MSE: {eval_metrics['val_loss_data']:.4f} | "
            f"Kinematic Residual: {eval_metrics['val_loss_kinematics']:.4f}"
        )

    # 5. Long-Horizon Multi-Step Autoregressive Rollout Evaluation
    print("\n[5/5] Executing multi-step autoregressive rollouts (Drift Test)...")
    evaluator = RolloutEvaluator(device=device)

    # Select continuous 120-frame sequence (2 seconds at 60 FPS) from test split
    test_states = data_dict["test_states"]
    test_actions = data_dict["test_actions"]
    test_next_states = data_dict["test_next_states"]

    H = min(120, len(test_actions))
    init_state = test_states[0]
    action_seq = test_actions[:H]
    ground_truth = test_next_states[:H]

    rollout_metrics = {}
    trajectories = {}

    for name, model in models.items():
        m_type = "lstm" if "lstm" in name.lower() else "mlp"
        res = evaluator.evaluate_rollout(
            model=model,
            model_type=m_type,
            initial_state=init_state,
            action_sequence=action_seq,
            ground_truth_states=ground_truth,
        )
        rollout_metrics[name] = {
            "mean_drift_pixels": res["mean_drift"],
            "final_drift_pixels": res["final_drift"],
            "kinematic_violations": res["kinematic_violations"],
            "velocity_violations": res["velocity_violations"],
        }
        trajectories[name] = res["predicted_trajectory"]
        print(
            f"{name:20s} | Mean Drift: {res['mean_drift']:.2f} px | "
            f"Final Drift: {res['final_drift']:.2f} px | "
            f"Kinematic Violations: {res['kinematic_violations']:3d}/{H} | "
            f"Velocity Violations: {res['velocity_violations']:3d}/{H}"
        )

    # 6. Generate High-Resolution Figures
    print("\nGenerating high-resolution comparative figures...")
    sns.set_theme(style="whitegrid")

    # Figure 1: Validation Loss Convergence Curves
    plt.figure(figsize=(10, 5))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=f"{name} (Val MSE)", linewidth=2)
    plt.title("Validation Error Convergence Across Training Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (Smooth L1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_convergence.png"), dpi=300)
    plt.close()

    # Figure 2: Euclidean Rollout Drift Over Horizon H
    plt.figure(figsize=(11, 6))
    time_steps = np.arange(1, H + 1)
    for name, model in models.items():
        m_type = "lstm" if "lstm" in name.lower() else "mlp"
        r = evaluator.evaluate_rollout(model, m_type, init_state, action_seq, ground_truth)
        plt.plot(time_steps, r["euclidean_drift"], label=f"{name}", linewidth=2.5)
    plt.title(
        "Autoregressive Trajectory Drift on Super Mario World (120-Frame / 2s Horizon)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Rollout Frame (t)", fontsize=11)
    plt.ylabel("Euclidean Deviation from Ground Truth (Pixels)", fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rollout_drift_comparison.png"), dpi=300)
    plt.close()

    # Figure 3: 2D Spatial Trajectory Traversal (X vs Y)
    plt.figure(figsize=(12, 6))
    plt.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        "k--",
        label="Ground Truth (Console WRAM)",
        linewidth=3.0,
    )
    for name, traj in trajectories.items():
        plt.plot(traj[:, 0], traj[:, 1], label=f"{name}", linewidth=2.0, alpha=0.85)
    plt.gca().invert_yaxis()  # In SNES coordinate space, Y=0 is top of screen
    plt.title("Mario Trajectory in 2D Space (X vs Y)", fontsize=14, fontweight="bold")
    plt.xlabel("Horizontal Position X (Pixels)")
    plt.ylabel("Vertical Position Y (Pixels - Inverted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "trajectory_2d_space.png"), dpi=300)
    plt.close()

    # Save summary metrics to JSON
    all_summary = {
        "single_step_results": single_step_results,
        "rollout_metrics": rollout_metrics,
    }
    with open(os.path.join(output_dir, "benchmark_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=4)

    print("\nMain benchmark completed successfully!")
    return all_summary


if __name__ == "__main__":
    run_comprehensive_benchmark()
