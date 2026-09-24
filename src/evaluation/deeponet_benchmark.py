"""
deeponet_benchmark.py
Neural-operator study for README section 10.41.

Trains the DeepONet baseline (Lu et al., 2021) under the exact unified
protocol of the canonical benchmark (same dataset, episodic split, seed,
optimizer recipe, and SmoothL1 objective), then evaluates single-step accuracy
and long-horizon autoregressive stability. Published comparison rows are read
from the committed ``benchmark_metrics.json`` (no re-training of the canonical
architectures), mirroring the analytical-baseline study.

Writes ``results/deeponet_benchmark_metrics.json`` with provenance through
``src.utils.provenance.write_metrics``.
"""

import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.environment.dataset_loader import create_dataloaders, load_and_preprocess_data
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import DeepONetDynamics
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PUBLISHED_BENCHMARK = "benchmark_metrics.json"
ARTIFACT_NAME = "deeponet_benchmark_metrics.json"


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights, reported for parameter-parity comparisons."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _collect_test_predictions(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targets = [], []
    for curr_state, curr_action, target_next in loader:
        preds.append(model(curr_state.to(device), curr_action.to(device)).cpu())
        targets.append(target_next.cpu())
    return torch.cat(preds), torch.cat(targets)


def _published_reference(output_dir: str) -> Optional[Dict[str, Any]]:
    """Canonical rows of the committed benchmark artifact, when present."""
    path = os.path.join(output_dir, PUBLISHED_BENCHMARK)
    if not os.path.isfile(path):
        logger.warning(
            "%s not found next to the output directory; the published-comparison "
            "block will be omitted.",
            PUBLISHED_BENCHMARK,
        )
        return None
    published = read_metrics(path)
    return {
        "source": f"results/{PUBLISHED_BENCHMARK}",
        "single_step_results": published.get("single_step_results", {}),
        "rollout_metrics": published.get("rollout_metrics", {}),
    }


def run_deeponet_benchmark(
    dataset_path: str = DATASET_GAMEPLAY,
    epochs: int = 35,
    batch_size: int = 128,
    seed: int = 42,
    output_dir: str = RESULTS_DIR,
    patience: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    rollout_horizon: int = 120,
    num_rollout_starts: int = 10,
    latent_dim: int = 64,
    branch_hidden_dims: Optional[List[int]] = None,
    trunk_hidden_dims: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Train and evaluate the DeepONet operator under the unified protocol."""
    logger.info("====================================================================")
    logger.info("  NEURAL OPERATOR STUDY: DEEPONET VS. PUBLISHED SMW BENCHMARK       ")
    logger.info("====================================================================")

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Identical split/partitioning to src.training.benchmark_experiment.
    data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict, batch_size=batch_size, seed=seed
    )
    state_dim = int(data_dict["train_states"].shape[1])
    action_dim = int(data_dict["train_actions"].shape[1])

    model = DeepONetDynamics(
        state_dim=state_dim,
        action_dim=action_dim,
        branch_hidden_dims=branch_hidden_dims,
        trunk_hidden_dims=trunk_hidden_dims,
        latent_dim=latent_dim,
    )
    logger.info(
        "DeepONet: %d sensors (state+action), latent basis p=%d, %d trainable parameters",
        model.num_sensors,
        latent_dim,
        count_parameters(model),
    )

    trainer = DynamicsTrainer(
        model=model,
        model_type="deeponet",
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=os.path.join(output_dir, "checkpoints"),
    )

    t0 = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        verbose=True,
    )
    train_seconds = time.time() - t0

    # Single-step evaluation on the independent test split.
    eval_metrics = trainer.evaluate(test_loader)
    preds, targets = _collect_test_predictions(model, test_loader, device)
    per_variable = compute_per_variable_metrics(preds, targets)

    # Long-horizon autoregressive rollout evaluation (same protocol as the
    # canonical benchmark: single continuous track plus multi-start average).
    evaluator = RolloutEvaluator(device=device)
    test_states = data_dict["test_states"]
    test_actions = data_dict["test_actions"]
    test_next_states = data_dict["test_next_states"]
    H = int(min(rollout_horizon, len(test_actions)))
    res = evaluator.evaluate_rollout(
        model=model,
        model_type="deeponet",
        initial_state=test_states[0],
        action_sequence=test_actions[:H],
        ground_truth_states=test_next_states[:H],
    )
    multi = evaluator.evaluate_rollout_multistart(
        model=model,
        model_type="deeponet",
        states=test_states,
        actions=test_actions,
        next_states=test_next_states,
        horizon=H,
        num_starts=num_rollout_starts,
    )
    multistart = {k: v for k, v in multi.items() if k not in ("mean_drifts", "final_drifts")}

    deeponet_row: Dict[str, Any] = {
        "parameters": count_parameters(model),
        "sensors": model.num_sensors,
        "latent_dim": latent_dim,
        "branch_hidden_dims": branch_hidden_dims or [128, 128],
        "trunk_hidden_dims": trunk_hidden_dims or [128, 128],
        "test_loss_data": eval_metrics["val_loss_data"],
        "test_kinematic_error": eval_metrics["val_loss_kinematics"],
        "training_time_seconds": train_seconds,
        "stop_epochs": len(history["train_loss"]),
        "rollout": {
            "mean_drift_pixels": res["mean_drift"],
            "final_drift_pixels": res["final_drift"],
            "kinematic_violations": res["kinematic_violations"],
            "velocity_violations": res["velocity_violations"],
        },
        "rollout_multistart": multistart,
        "per_variable_metrics": per_variable,
    }

    summary: Dict[str, Any] = {
        "study": (
            "DeepONet neural-operator baseline trained under the unified protocol "
            "of the canonical benchmark (README 10.41)"
        ),
        "architectures": {"DeepONet": deeponet_row},
        "comparison_against_published": _published_reference(output_dir),
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
        },
    }

    out_path = write_metrics(
        os.path.join(output_dir, ARTIFACT_NAME),
        summary,
        seed=seed,
        command="python -m src.evaluation.deeponet_benchmark",
    )

    logger.info(
        "DeepONet | Test MSE: %.4f | Kinematic Residual: %.4f | "
        "Mean Drift: %.2f px | Final Drift: %.2f px | Violations: %d/%d",
        deeponet_row["test_loss_data"],
        deeponet_row["test_kinematic_error"],
        deeponet_row["rollout"]["mean_drift_pixels"],
        deeponet_row["rollout"]["final_drift_pixels"],
        deeponet_row["rollout"]["kinematic_violations"],
        H,
    )
    logger.info("Metrics written to %s", out_path)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepONet neural-operator benchmark on Super Mario World WRAM telemetry."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rollout-horizon", type=int, default=120)
    parser.add_argument("--num-rollout-starts", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parse_args_with_config(parser)

    run_deeponet_benchmark(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rollout_horizon=args.rollout_horizon,
        num_rollout_starts=args.num_rollout_starts,
        latent_dim=args.latent_dim,
    )
