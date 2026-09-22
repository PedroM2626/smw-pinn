"""
sample_efficiency_benchmark.py
Evaluates sample efficiency (Sample Efficiency Pareto):
Measures the empirical impact of known physics across training sample sizes
ranging from extremely scarce (200 transitions) to abundant (5000+ transitions).
"""

import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset

from src.environment.dataset_loader import (
    SMWTransitionDataset,
    load_and_preprocess_data,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalMLPDynamics,
)
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def run_sample_efficiency_study(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    sample_sizes: List[int] | None = None,
    epochs_per_run: int = 25,
    seed: int = 42,
    output_dir: str = "results",
    rollout_horizon: int = 60,
    num_rollout_starts: int = 5,
):
    logger.info("====================================================================")
    logger.info("  SAMPLE EFFICIENCY STUDY (DATA PARETO FRONTIER BENCHMARK)          ")
    logger.info("====================================================================")

    set_global_seed(seed)
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

    if sample_sizes is None:
        sample_sizes = [200, 500, 1000, 2500, 5000]

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
    H = min(rollout_horizon, len(test_actions))
    init_state = test_states[0]
    action_seq = test_actions[:H]
    ground_truth = test_next_states[:H]

    for N in sample_sizes:
        logger.info(f"\n>>> Evaluating training regime with N = {N} transitions <<<")
        # Subsampling with fixed seed
        indices = np.random.choice(len(full_train_ds), size=N, replace=False)
        sub_train_ds = Subset(full_train_ds, indices)
        sub_gen = torch.Generator()
        sub_gen.manual_seed(seed)
        sub_train_loader = DataLoader(
            sub_train_ds, batch_size=min(64, N), shuffle=True, generator=sub_gen
        )

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

            # Rollout evaluation (single start + multi-start average)
            roll_res = evaluator.evaluate_rollout(model, "mlp", init_state, action_seq, ground_truth)
            mean_drift = roll_res["mean_drift"]
            multi_res = evaluator.evaluate_rollout_multistart(
                model,
                "mlp",
                test_states,
                test_actions,
                test_next_states,
                horizon=H,
                num_starts=num_rollout_starts,
            )

            efficiency_results[name]["test_mse"].append(test_mse)
            efficiency_results[name]["mean_drift"].append(mean_drift)
            efficiency_results[name].setdefault("mean_drift_multistart", []).append(
                multi_res["mean_drift_mean"]
            )

            logger.info(f"  {name:20s} | N={N:4d} | Test MSE: {test_mse:.4f} | Rollout Drift: {mean_drift:.2f} px")

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
                "config": {
                    "dataset_path": dataset_path,
                    "epochs_per_run": epochs_per_run,
                    "seed": seed,
                    "rollout_horizon": H,
                    "num_rollout_starts": num_rollout_starts,
                },
                "results": efficiency_results,
            },
            f,
            indent=4,
        )

    logger.info("\nSample efficiency study completed successfully!")
    return efficiency_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample-efficiency Pareto study.")
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default="data/raw/smw_gameplay_dataset.npz")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--epochs-per-run", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--rollout-horizon", type=int, default=60)
    parser.add_argument("--num-rollout-starts", type=int, default=5)
    args = parse_args_with_config(parser)

    run_sample_efficiency_study(
        dataset_path=args.dataset_path,
        sample_sizes=args.sample_sizes,
        epochs_per_run=args.epochs_per_run,
        seed=args.seed,
        output_dir=args.output_dir,
        rollout_horizon=args.rollout_horizon,
        num_rollout_starts=args.num_rollout_starts,
    )
