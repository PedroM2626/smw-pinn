"""
multiseed_benchmark.py
Rigorous multi-seed statistical significance evaluation across K=5 random seeds.
Evaluates Statistical MLP, Soft PINN, and Hard Residual PINN.
Computes sample mean, sample standard deviation, paired t-tests, and Wilcoxon signed-rank tests.
"""

import json
import os
import sys
import time
from typing import Dict, List
import numpy as np
from scipy import stats
import torch

sys.path.insert(0, os.path.abspath("."))
from src.environment.dataset_loader import (
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
from src.utils.seed import set_global_seed


def run_multiseed_benchmark(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    seeds: List[int] = [42, 43, 44, 45, 46],
    epochs: int = 30,
    batch_size: int = 128,
    output_dir: str = "results",
) -> Dict:
    print("====================================================================")
    print(f"  MULTI-SEED STATISTICAL BENCHMARK (K = {len(seeds)} Seeds: {seeds})")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if device.type == "cuda":
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints_multiseed")
    os.makedirs(checkpoints_dir, exist_ok=True)

    model_classes = {
        "Statistical_MLP": (StatisticalMLPDynamics, "mlp"),
        "Soft_PINN": (SoftPINNDynamics, "pinn_soft"),
        "Hard_Residual_PINN": (HardResidualPINNDynamics, "pinn_hard"),
    }

    # Tracking per seed
    # model -> metric -> list of values
    raw_results: Dict[str, Dict[str, List[float]]] = {
        m: {
            "test_mse": [],
            "kinematic_residual": [],
            "mean_drift_pixels": [],
            "final_drift_pixels": [],
            "kinematic_violations": [],
        }
        for m in model_classes
    }

    t0_total = time.time()

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n>>> [Seed {seed_idx}/{len(seeds)}: seed={seed}] Starting run...")
        set_global_seed(seed)

        data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
        train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=batch_size)

        state_dim = data_dict["train_states"].shape[1]
        action_dim = data_dict["train_actions"].shape[1]

        # Trajectory slice for rollout drift
        test_states = data_dict["test_states"]
        test_actions = data_dict["test_actions"]
        test_next_states = data_dict["test_next_states"]
        H = min(120, len(test_actions))
        init_state = test_states[0]
        action_seq = test_actions[:H]
        ground_truth = test_next_states[:H]

        evaluator = RolloutEvaluator(device=device)

        for model_name, (cls, m_type) in model_classes.items():
            model = cls(state_dim=state_dim, action_dim=action_dim).to(device)

            trainer = DynamicsTrainer(
                model=model,
                model_type=m_type,
                device=device,
                learning_rate=1e-3,
                save_dir=checkpoints_dir,
            )

            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                patience=8,
                verbose=False,
            )

            # Single-step evaluation
            eval_metrics = trainer.evaluate(test_loader)
            test_mse = eval_metrics["val_loss_data"]
            kin_res = eval_metrics["val_loss_kinematics"]

            # Multi-step rollout drift
            rollout_res = evaluator.evaluate_rollout(
                model=model,
                model_type=m_type,
                initial_state=init_state,
                action_sequence=action_seq,
                ground_truth_states=ground_truth,
            )

            raw_results[model_name]["test_mse"].append(float(test_mse))
            raw_results[model_name]["kinematic_residual"].append(float(kin_res))
            raw_results[model_name]["mean_drift_pixels"].append(float(rollout_res["mean_drift"]))
            raw_results[model_name]["final_drift_pixels"].append(float(rollout_res["final_drift"]))
            raw_results[model_name]["kinematic_violations"].append(int(rollout_res["kinematic_violations"]))

            print(
                f"  [{model_name:18s}] Test MSE: {test_mse:.4f} | "
                f"Kin Res: {kin_res:.4f} | "
                f"Mean Drift: {rollout_res['mean_drift']:.2f} px | "
                f"Violations: {rollout_res['kinematic_violations']}/{H}"
            )

    total_time = time.time() - t0_total
    print(f"\nAll {len(seeds)} seeds completed in {total_time:.2f}s!")

    # Compute descriptive statistics (mean and std)
    summary_stats = {}
    for model_name, metrics in raw_results.items():
        summary_stats[model_name] = {}
        for metric_name, values in metrics.items():
            summary_stats[model_name][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }

    # Statistical significance hypothesis testing (Hard PINN vs Baselines)
    hypothesis_tests = {}
    hard_pinn_mse = raw_results["Hard_Residual_PINN"]["test_mse"]
    hard_pinn_drift = raw_results["Hard_Residual_PINN"]["mean_drift_pixels"]

    for baseline in ["Statistical_MLP", "Soft_PINN"]:
        base_mse = raw_results[baseline]["test_mse"]
        base_drift = raw_results[baseline]["mean_drift_pixels"]

        # Paired Student's t-test
        t_stat_mse, p_val_mse = stats.ttest_rel(hard_pinn_mse, base_mse)
        t_stat_drift, p_val_drift = stats.ttest_rel(hard_pinn_drift, base_drift)

        # Wilcoxon signed-rank test
        try:
            w_stat_mse, w_pval_mse = stats.wilcoxon(hard_pinn_mse, base_mse)
            w_stat_drift, w_pval_drift = stats.wilcoxon(hard_pinn_drift, base_drift)
        except Exception:
            w_stat_mse, w_pval_mse = float("nan"), float("nan")
            w_stat_drift, w_pval_drift = float("nan"), float("nan")

        hypothesis_tests[f"Hard_PINN_vs_{baseline}"] = {
            "test_mse": {
                "t_statistic": float(t_stat_mse),
                "p_value_ttest": float(p_val_mse),
                "wilcoxon_stat": float(w_stat_mse),
                "p_value_wilcoxon": float(w_pval_mse),
                "significant_at_p001": bool(p_val_mse < 0.001),
            },
            "mean_drift": {
                "t_statistic": float(t_stat_drift),
                "p_value_ttest": float(p_val_drift),
                "wilcoxon_stat": float(w_stat_drift),
                "p_value_wilcoxon": float(w_pval_drift),
                "significant_at_p005": bool(p_val_drift < 0.05),
            },
        }

    output_payload = {
        "seeds": seeds,
        "descriptive_statistics": summary_stats,
        "hypothesis_testing": hypothesis_tests,
    }

    out_file = os.path.join(output_dir, "multiseed_benchmark_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)

    print(f"\nMulti-seed metrics successfully saved to: {out_file}")

    # Print summary table
    print("\n====================================================================")
    print("  SUMMARY: MULTI-SEED STATISTICAL BENCHMARK (Mean ± Std)")
    print("====================================================================")
    print(f"{'Architecture':20s} | {'Test MSE':20s} | {'Kinematic Residual':22s} | {'Mean Drift (px)':20s}")
    print("-" * 90)
    for model_name, s in summary_stats.items():
        mse_str = f"{s['test_mse']['mean']:.4f} ± {s['test_mse']['std']:.4f}"
        kin_str = f"{s['kinematic_residual']['mean']:.4f} ± {s['kinematic_residual']['std']:.4f}"
        drift_str = f"{s['mean_drift_pixels']['mean']:.2f} ± {s['mean_drift_pixels']['std']:.2f}"
        print(f"{model_name:20s} | {mse_str:20s} | {kin_str:22s} | {drift_str:20s}")
    print("====================================================================")

    for comp, tests in hypothesis_tests.items():
        p_mse = tests["test_mse"]["p_value_ttest"]
        p_drift = tests["mean_drift"]["p_value_ttest"]
        print(f"Hypothesis Test [{comp}]: Test MSE p-value = {p_mse:.4e} | Mean Drift p-value = {p_drift:.4e}")

    return output_payload


if __name__ == "__main__":
    run_multiseed_benchmark()
