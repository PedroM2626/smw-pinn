"""
multiseed_benchmark.py
Rigorous multi-seed statistical significance evaluation across K=10 random seeds.
Evaluates Statistical MLP, Soft PINN, and Hard Residual PINN.
Computes sample mean, sample standard deviation, paired t-tests, Wilcoxon
signed-rank tests, paired effect sizes (Cohen's dz), and multi-start rollout
statistics for adequate power of paired comparisons.
"""

import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

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
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def _cohen_dz(paired_a, paired_b) -> float:
    """Standardized mean difference for paired samples (Cohen's dz)."""
    diffs = np.asarray(paired_a, dtype=np.float64) - np.asarray(paired_b, dtype=np.float64)
    std = float(np.std(diffs, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(diffs) / std)


def run_multiseed_benchmark(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    seeds: List[int] | None = None,
    epochs: int = 30,
    batch_size: int = 128,
    output_dir: str = "results",
    patience: int = 8,
    learning_rate: float = 1e-3,
    rollout_horizon: int = 120,
    num_rollout_starts: int = 10,
) -> Dict:
    logger.info("====================================================================")
    if seeds is None:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    logger.info(f"  MULTI-SEED STATISTICAL BENCHMARK (K = {len(seeds)} Seeds: {seeds})")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"Detected GPU: {torch.cuda.get_device_name(0)}")

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
            "mean_drift_multistart": [],
            "kinematic_violation_rate_multistart": [],
        }
        for m in model_classes
    }

    t0_total = time.time()

    for seed_idx, seed in enumerate(seeds, 1):
        logger.info(f"\n>>> [Seed {seed_idx}/{len(seeds)}: seed={seed}] Starting run...")
        set_global_seed(seed)

        data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dict, batch_size=batch_size, seed=seed
        )

        state_dim = data_dict["train_states"].shape[1]
        action_dim = data_dict["train_actions"].shape[1]

        # Trajectory slice for rollout drift (legacy single-start, kept for
        # comparability with published K=5 numbers)
        test_states = data_dict["test_states"]
        test_actions = data_dict["test_actions"]
        test_next_states = data_dict["test_next_states"]
        H = min(rollout_horizon, len(test_actions))
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
                learning_rate=learning_rate,
                save_dir=checkpoints_dir,
            )

            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                patience=patience,
                verbose=False,
            )

            # Single-step evaluation
            eval_metrics = trainer.evaluate(test_loader)
            test_mse = eval_metrics["val_loss_data"]
            kin_res = eval_metrics["val_loss_kinematics"]

            # Multi-step rollout drift (single start) + multi-start average
            rollout_res = evaluator.evaluate_rollout(
                model=model,
                model_type=m_type,
                initial_state=init_state,
                action_sequence=action_seq,
                ground_truth_states=ground_truth,
            )
            multi_res = evaluator.evaluate_rollout_multistart(
                model=model,
                model_type=m_type,
                states=test_states,
                actions=test_actions,
                next_states=test_next_states,
                horizon=H,
                num_starts=num_rollout_starts,
            )

            raw_results[model_name]["test_mse"].append(float(test_mse))
            raw_results[model_name]["kinematic_residual"].append(float(kin_res))
            raw_results[model_name]["mean_drift_pixels"].append(float(rollout_res["mean_drift"]))
            raw_results[model_name]["final_drift_pixels"].append(float(rollout_res["final_drift"]))
            raw_results[model_name]["kinematic_violations"].append(int(rollout_res["kinematic_violations"]))
            raw_results[model_name]["mean_drift_multistart"].append(float(multi_res["mean_drift_mean"]))
            raw_results[model_name]["kinematic_violation_rate_multistart"].append(
                float(multi_res["kinematic_violation_rate_mean"])
            )

            logger.info(
                f"  [{model_name:18s}] Test MSE: {test_mse:.4f} | "
                f"Kin Res: {kin_res:.4f} | "
                f"Mean Drift: {rollout_res['mean_drift']:.2f} px | "
                f"Multi-start: {multi_res['mean_drift_mean']:.2f}±{multi_res['mean_drift_std']:.2f} px | "
                f"Violations: {rollout_res['kinematic_violations']}/{H}"
            )

    total_time = time.time() - t0_total
    logger.info(f"\nAll {len(seeds)} seeds completed in {total_time:.2f}s!")

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
    hard_pinn_multi = raw_results["Hard_Residual_PINN"]["mean_drift_multistart"]

    for baseline in ["Statistical_MLP", "Soft_PINN"]:
        base_mse = raw_results[baseline]["test_mse"]
        base_drift = raw_results[baseline]["mean_drift_pixels"]
        base_multi = raw_results[baseline]["mean_drift_multistart"]

        # Paired Student's t-test
        t_stat_mse, p_val_mse = stats.ttest_rel(hard_pinn_mse, base_mse)
        t_stat_drift, p_val_drift = stats.ttest_rel(hard_pinn_drift, base_drift)
        t_stat_multi, p_val_multi = stats.ttest_rel(hard_pinn_multi, base_multi)

        # Wilcoxon signed-rank test
        try:
            w_stat_mse, w_pval_mse = stats.wilcoxon(hard_pinn_mse, base_mse)
            w_stat_drift, w_pval_drift = stats.wilcoxon(hard_pinn_drift, base_drift)
            w_stat_multi, w_pval_multi = stats.wilcoxon(hard_pinn_multi, base_multi)
        except Exception:
            w_stat_mse, w_pval_mse = float("nan"), float("nan")
            w_stat_drift, w_pval_drift = float("nan"), float("nan")
            w_stat_multi, w_pval_multi = float("nan"), float("nan")

        hypothesis_tests[f"Hard_PINN_vs_{baseline}"] = {
            "test_mse": {
                "t_statistic": float(t_stat_mse),
                "p_value_ttest": float(p_val_mse),
                "wilcoxon_stat": float(w_stat_mse),
                "p_value_wilcoxon": float(w_pval_mse),
                "cohen_dz": float(_cohen_dz(hard_pinn_mse, base_mse)),
                "significant_at_p001": bool(p_val_mse < 0.001),
            },
            "mean_drift": {
                "t_statistic": float(t_stat_drift),
                "p_value_ttest": float(p_val_drift),
                "wilcoxon_stat": float(w_stat_drift),
                "p_value_wilcoxon": float(w_pval_drift),
                "cohen_dz": float(_cohen_dz(hard_pinn_drift, base_drift)),
                "significant_at_p005": bool(p_val_drift < 0.05),
            },
            "mean_drift_multistart": {
                "t_statistic": float(t_stat_multi),
                "p_value_ttest": float(p_val_multi),
                "wilcoxon_stat": float(w_stat_multi),
                "p_value_wilcoxon": float(w_pval_multi),
                "cohen_dz": float(_cohen_dz(hard_pinn_multi, base_multi)),
                "significant_at_p005": bool(p_val_multi < 0.05),
            },
        }

    output_payload = {
        "seeds": seeds,
        "config": {
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "learning_rate": learning_rate,
            "rollout_horizon": rollout_horizon,
            "num_rollout_starts": num_rollout_starts,
        },
        "descriptive_statistics": summary_stats,
        "hypothesis_testing": hypothesis_tests,
    }

    out_file = os.path.join(output_dir, "multiseed_benchmark_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)

    logger.info(f"\nMulti-seed metrics successfully saved to: {out_file}")

    # Print summary table
    logger.info("\n====================================================================")
    logger.info("  SUMMARY: MULTI-SEED STATISTICAL BENCHMARK (Mean ± Std)")
    logger.info("====================================================================")
    logger.info(f"{'Architecture':20s} | {'Test MSE':20s} | {'Kinematic Residual':22s} | {'Mean Drift (px)':20s}")
    logger.info("-" * 90)
    for model_name, s in summary_stats.items():
        mse_str = f"{s['test_mse']['mean']:.4f} ± {s['test_mse']['std']:.4f}"
        kin_str = f"{s['kinematic_residual']['mean']:.4f} ± {s['kinematic_residual']['std']:.4f}"
        drift_str = f"{s['mean_drift_pixels']['mean']:.2f} ± {s['mean_drift_pixels']['std']:.2f}"
        logger.info(f"{model_name:20s} | {mse_str:20s} | {kin_str:22s} | {drift_str:20s}")
    logger.info("====================================================================")

    for comp, tests in hypothesis_tests.items():
        p_mse = tests["test_mse"]["p_value_ttest"]
        p_drift = tests["mean_drift"]["p_value_ttest"]
        logger.info(f"Hypothesis Test [{comp}]: Test MSE p-value = {p_mse:.4e} | Mean Drift p-value = {p_drift:.4e}")

    return output_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-seed significance benchmark (default K=10).")
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default="data/raw/smw_gameplay_dataset.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rollout-horizon", type=int, default=120)
    parser.add_argument("--num-rollout-starts", type=int, default=10)
    args = parse_args_with_config(parser)

    run_multiseed_benchmark(
        dataset_path=args.dataset_path,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        patience=args.patience,
        learning_rate=args.learning_rate,
        rollout_horizon=args.rollout_horizon,
        num_rollout_starts=args.num_rollout_starts,
    )
