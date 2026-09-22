"""
benchmark_experiment.py
Orchestrator script for the complete academic comparative benchmark:
Trains all 4 models under identical conditions, evaluates single-step accuracy,
long-horizon stability (rollout drift), and generates comparative analytical figures.
"""

import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.environment.dataset_loader import (
    SMWSequenceDataset,
    create_dataloaders,
    load_and_preprocess_data,
)
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    MATCHED_HIDDEN_DIMS,
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalLSTMDynamics,
    StatisticalMLPDynamics,
    build_param_matched_mlp,
)
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.experiment import ExperimentLogger
from src.utils.logging import get_logger
from src.utils.seed import get_seed_info, set_global_seed

log = get_logger(__name__)


@torch.no_grad()
def _collect_test_predictions(trainer, loader, device):
    """Gather (predicted, target) next-states over a test loader (MLP or LSTM)."""
    trainer.model.eval()
    preds, targets = [], []
    for batch in loader:
        if "lstm" in trainer.model_type:
            state_seq, action_seq, target_next = [b.to(device) for b in batch]
            pred_seq, _ = trainer.model(state_seq, action_seq)
            preds.append(pred_seq[:, -1, :].cpu())
            targets.append(target_next.cpu())
        else:
            curr_state, curr_action, target_next = [b.to(device) for b in batch]
            preds.append(trainer.model(curr_state, curr_action).cpu())
            targets.append(target_next.cpu())
    return torch.cat(preds), torch.cat(targets)


def run_comprehensive_benchmark(
    dataset_path: str = "data/raw/smw_gameplay_dataset.npz",
    epochs: int = 35,
    batch_size: int = 128,
    seed: int = 42,
    output_dir: str = "results",
    experiment_name: str = "benchmark_mlp_vs_pinn",
    log_dir: str = "runs",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    deterministic: bool = True,
    patience: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    rollout_horizon: int = 120,
    num_rollout_starts: int = 10,
    matched_baseline: bool = False,
    per_variable_metrics: bool = True,
):
    log.info("====================================================================")
    log.info("  ACADEMIC BENCHMARK: STATISTICAL ML VS. PINN ON SUPER MARIO WORLD  ")
    log.info("====================================================================")

    set_global_seed(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Compute Device: {device}")
    if device.type == "cuda":
        log.info(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    experiment_logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        hparams={
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "deterministic": deterministic,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "rollout_horizon": rollout_horizon,
            "num_rollout_starts": num_rollout_starts,
            "matched_baseline": matched_baseline,
            "device": str(device),
            **get_seed_info(seed),
        },
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
    )

    # 1. Load genuine RAM telemetry dataset
    log.info("\n[1/5] Loading and partitioning genuine WRAM transitions...")
    data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=batch_size, seed=seed)

    log.info(
        f"Transitions - Train: {len(data_dict['train_states'])}, "
        f"Validation: {len(data_dict['val_states'])}, "
        f"Test: {len(data_dict['test_states'])}"
    )

    state_dim = data_dict["train_states"].shape[1]
    action_dim = data_dict["train_actions"].shape[1]

    # 2. Instantiate comparative neural architectures
    log.info("\n[2/5] Initializing comparative neural architectures...")
    models = {
        "Statistical_MLP": StatisticalMLPDynamics(state_dim=state_dim, action_dim=action_dim),
        "Statistical_LSTM": StatisticalLSTMDynamics(state_dim=state_dim, action_dim=action_dim),
        "Soft_PINN": SoftPINNDynamics(state_dim=state_dim, action_dim=action_dim),
        "Hard_Residual_PINN": HardResidualPINNDynamics(state_dim=state_dim, action_dim=action_dim),
    }
    if matched_baseline:
        # Parameter-parity ablation: compact ~10k-param pair (MLP vs Hard PINN).
        models["Statistical_MLP_matched"] = build_param_matched_mlp(
            state_dim=state_dim, action_dim=action_dim
        )
        models["Hard_Residual_PINN_compact"] = HardResidualPINNDynamics(
            state_dim=state_dim, action_dim=action_dim, hidden_dims=list(MATCHED_HIDDEN_DIMS)
        )
        log.info("Parameter-matched compact baselines enabled (~10k params each).")

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

    seq_gen = torch.Generator()
    seq_gen.manual_seed(seed)
    train_seq_loader = torch.utils.data.DataLoader(
        train_seq_ds, batch_size=batch_size, shuffle=True, generator=seq_gen
    )
    val_seq_loader = torch.utils.data.DataLoader(val_seq_ds, batch_size=batch_size, shuffle=False)
    test_seq_loader = torch.utils.data.DataLoader(test_seq_ds, batch_size=batch_size, shuffle=False)

    # 3. Comparative Training Loop
    log.info("\n[3/5] Training models under unified experimental protocol...")
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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_dir=os.path.join(output_dir, "checkpoints"),
        )
        trainers[name] = trainer

        cur_train_loader = train_seq_loader if m_type == "lstm" else train_loader
        cur_val_loader = val_seq_loader if m_type == "lstm" else val_loader

        hist = trainer.fit(
            train_loader=cur_train_loader,
            val_loader=cur_val_loader,
            epochs=epochs,
            patience=patience,
            verbose=True,
            experiment=experiment_logger,
        )
        histories[name] = hist

    # 4. Single-Step Accuracy Evaluation on Independent Test Split
    log.info("\n[4/5] Evaluating single-step accuracy on test split...")
    single_step_results = {}
    per_variable_results = {}
    for name, trainer in trainers.items():
        cur_test_loader = test_seq_loader if "lstm" in name.lower() else test_loader
        eval_metrics = trainer.evaluate(cur_test_loader)
        single_step_results[name] = {
            "test_loss_data": eval_metrics["val_loss_data"],
            "test_kinematic_error": eval_metrics["val_loss_kinematics"],
        }
        if per_variable_metrics:
            preds, targets = _collect_test_predictions(trainer, cur_test_loader, device)
            per_variable_results[name] = compute_per_variable_metrics(preds, targets)
        log.info(
            f"{name:20s} | Test MSE: {eval_metrics['val_loss_data']:.4f} | "
            f"Kinematic Residual: {eval_metrics['val_loss_kinematics']:.4f}"
        )

    # 5. Long-Horizon Multi-Step Autoregressive Rollout Evaluation
    log.info("\n[5/5] Executing multi-step autoregressive rollouts (Drift Test)...")
    evaluator = RolloutEvaluator(device=device)

    # Select continuous rollout sequence (default 120 frames / 2s at 60 FPS) from test split
    test_states = data_dict["test_states"]
    test_actions = data_dict["test_actions"]
    test_next_states = data_dict["test_next_states"]

    H = min(rollout_horizon, len(test_actions))
    init_state = test_states[0]
    action_seq = test_actions[:H]
    ground_truth = test_next_states[:H]

    rollout_metrics = {}
    rollout_multistart = {}
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
        multi = evaluator.evaluate_rollout_multistart(
            model=model,
            model_type=m_type,
            states=test_states,
            actions=test_actions,
            next_states=test_next_states,
            horizon=H,
            num_starts=num_rollout_starts,
        )
        rollout_multistart[name] = {
            k: v for k, v in multi.items() if k not in ("mean_drifts", "final_drifts")
        }
        trajectories[name] = res["predicted_trajectory"]
        log.info(
            f"{name:20s} | Mean Drift: {res['mean_drift']:.2f} px | "
            f"Final Drift: {res['final_drift']:.2f} px | "
            f"Kinematic Violations: {res['kinematic_violations']:3d}/{H} | "
            f"Velocity Violations: {res['velocity_violations']:3d}/{H}"
        )

    # 6. Generate High-Resolution Figures
    log.info("\nGenerating high-resolution comparative figures...")
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

    # Save summary metrics to JSON (legacy keys preserved for reproducibility)
    all_summary = {
        "single_step_results": single_step_results,
        "rollout_metrics": rollout_metrics,
        "rollout_multistart": rollout_multistart,
        "per_variable_metrics": per_variable_results,
        "config": {
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "rollout_horizon": H,
            "num_rollout_starts": num_rollout_starts,
            "matched_baseline": matched_baseline,
        },
    }
    with open(os.path.join(output_dir, "benchmark_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=4)

    # Mirror final per-model metrics into the experiment log (step 0 = summary row)
    try:
        for name in single_step_results:
            experiment_logger.log_metrics(
                {
                    f"{name}/final_test_mse": single_step_results[name]["test_loss_data"],
                    f"{name}/final_test_kin": single_step_results[name]["test_kinematic_error"],
                    f"{name}/final_mean_drift": rollout_metrics[name]["mean_drift_pixels"],
                    f"{name}/final_drift": rollout_metrics[name]["final_drift_pixels"],
                },
                step=epochs,
            )
    finally:
        experiment_logger.close()

    log.info(f"\nExperiment logs: {experiment_logger.dir} (tensorboard --logdir {log_dir})")
    log.info("\nMain benchmark completed successfully!")
    return all_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLP vs PINN benchmark on Super Mario World WRAM telemetry.")
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default="data/raw/smw_gameplay_dataset.npz")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment-name", default="benchmark_mlp_vs_pinn")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rollout-horizon", type=int, default=120)
    parser.add_argument("--num-rollout-starts", type=int, default=10)
    parser.add_argument("--matched-baseline", action="store_true")
    parser.add_argument("--no-per-variable-metrics", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard, keep JSONL logs.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb mirroring (requires wandb install).")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic cuDNN (faster, not bit-reproducible).",
    )
    args = parse_args_with_config(parser)

    run_comprehensive_benchmark(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir,
        use_tensorboard=not args.no_tensorboard,
        use_wandb=args.wandb,
        deterministic=not args.non_deterministic,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rollout_horizon=args.rollout_horizon,
        num_rollout_starts=args.num_rollout_starts,
        matched_baseline=args.matched_baseline,
        per_variable_metrics=not args.no_per_variable_metrics,
    )
