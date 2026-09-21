"""
sample_efficiency_benchmark.py
Evaluates sample efficiency (Sample Efficiency Pareto):
Measures the empirical impact of known physics across training sample sizes
ranging from extremely scarce (200 transitions) to abundant (5000+ transitions).
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath("."))
from src.environment.dataset_loader import (
    SMWTransitionDataset,
    create_dataloaders,
    load_and_preprocess_data,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalMLPDynamics,
)
from src.training.trainer import DynamicsTrainer


def run_sample_efficiency_study(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    sample_sizes: List[int] = [200, 500, 1000, 2500, 5000],
    epochs_per_run: int = 25,
    seed: int = 42,
    output_dir: str = "results",
):
    print("====================================================================")
    print("  SAMPLE EFFICIENCY STUDY (DATA PARETO FRONTIER BENCHMARK)          ")
    print("====================================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    full_train_ds = SMWTransitionDataset(
        data_dict["train_states"],
        data_dict["train_actions"],
        data_dict["train_next_states"],
    )
    val_loader = DataLoader(
        SMWTransitionDataset(
            data_dict["val_states"],
            data_dict["val_actions"],
            data_dict["val_next_states"],
        ),
        batch_size=128,
        shuffle=False,
    )
    test_loader = DataLoader(
        SMWTransitionDataset(
            data_dict["test_states"],
            data_dict["test_actions"],
            data_dict["test_next_states"],
        ),
        batch_size=128,
        shuffle=False,
    )

    state_dim = data_dict["train_states"].shape[1]
    action_dim = data_dict["train_actions"].shape[1]

    # Models evaluated in sample efficiency study
    model_factories = {
        "Statistical_MLP": lambda: StatisticalMLPDynamics(state_dim, action_dim),
        "Soft_PINN": lambda: SoftPINNDynamics(state_dim, action_dim),
        "Hard_Residual_PINN": lambda: HardResidualPINNDynamics(state_dim, action_dim),
    }

    efficiency_results = {name: {"test_mse": [], "mean_drift": []} for name in model_factories}

    evaluator = RolloutEvaluator(device=device)
    test_states = data_dict["test_states"]
    test_actions = data_dict["test_actions"]
    test_next_states = data_dict["test_next_states"]
    H = min(60, len(test_actions))
    init_state = test_states[0]
    action_seq = test_actions[:H]
    ground_truth = test_next_states[:H]

    for N in sample_sizes:
        print(f"\n>>> Evaluating training regime with N = {N} transitions <<<")
        # Subsampling with fixed seed
        indices = np.random.choice(len(full_train_ds), size=N, replace=False)
        sub_train_ds = Subset(full_train_ds, indices)
        sub_train_loader = DataLoader(sub_train_ds, batch_size=min(64, N), shuffle=True)

        for name, factory in model_factories.items():
            model = factory()
            if "soft" in name.lower():
                m_type = f"pinn_soft_N{N}"
            elif "hard" in name.lower():
                m_type = f"pinn_hard_N{N}"
            else:
                m_type = f"mlp_N{N}"

            trainer = DynamicsTrainer(
                model=model,
                model_type=m_type,
                device=device,
                learning_rate=1e-3,
                save_dir=os.path.join(output_dir, "checkpoints_efficiency"),
            )

            trainer.fit(
                train_loader=sub_train_loader,
                val_loader=val_loader,
                epochs=epochs_per_run,
                patience=6,
                verbose=False,
            )

            # Single-step evaluation
            test_res = trainer.evaluate(test_loader)
            test_mse = test_res["val_loss_data"]

            # Rollout evaluation
            roll_res = evaluator.evaluate_rollout(model, "mlp", init_state, action_seq, ground_truth)
            mean_drift = roll_res["mean_drift"]

            efficiency_results[name]["test_mse"].append(test_mse)
            efficiency_results[name]["mean_drift"].append(mean_drift)

            print(f"  {name:20s} | N={N:4d} | Test MSE: {test_mse:.4f} | Rollout Drift: {mean_drift:.2f} px")

    # Sample Efficiency Plots
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1: Test MSE vs Training Dataset Size
    plt.figure(figsize=(9, 5))
    for name, res in efficiency_results.items():
        plt.plot(sample_sizes, res["test_mse"], marker="o", linewidth=2.5, label=name)
    plt.xscale("log")
    plt.title("Sample Efficiency Curve: Test MSE vs. Training Dataset Volume", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Training Transitions (N - Log Scale)", fontsize=11)
    plt.ylabel("Test Set Mean Squared Error (MSE)", fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sample_efficiency_mse.png"), dpi=300)
    plt.close()

    # Plot 2: Rollout Drift vs Training Dataset Size
    plt.figure(figsize=(9, 5))
    for name, res in efficiency_results.items():
        plt.plot(sample_sizes, res["mean_drift"], marker="s", linewidth=2.5, label=name)
    plt.xscale("log")
    plt.title("Long-Horizon Rollout Stability vs. Training Dataset Volume", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Training Transitions (N - Log Scale)", fontsize=11)
    plt.ylabel("Mean Trajectory Drift (Pixels)", fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sample_efficiency_drift.png"), dpi=300)
    plt.close()

    # Save summary metrics
    with open(os.path.join(output_dir, "sample_efficiency_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_sizes": sample_sizes,
                "results": efficiency_results,
            },
            f,
            indent=4,
        )

    print("\nSample efficiency study completed successfully!")
    return efficiency_results


if __name__ == "__main__":
    run_sample_efficiency_study()
