"""
operator_benchmark.py
Neural-operator family study for README section 10.42.

Trains the three operator architectures under the exact unified protocol of
the canonical benchmark (same dataset, episodic split, seed, optimizer recipe,
SmoothL1 objective, rollout evaluation):

1. ``DeepONet`` - the pure branch/trunk operator of section 10.41 (re-run as
   the within-study reference row);
2. ``PhysicsConstrained_DeepONet`` - the operator factorization restricted to
   force/contact residuals, integrated by the Section 4 hard kinematics shell;
3. ``FNO`` - the Fourier Neural Operator (Li et al., 2021): spectral
   convolutions over the sensor lattice with interpolated field decoding.

Every model's data pipeline is re-seeded and rebuilt inside its own loop
iteration, so each row reproduces standalone under the canonical seed.
Published comparison rows are read from the committed
``benchmark_metrics.json`` (no re-training of the canonical architectures).

Writes ``results/operator_benchmark_metrics.json`` with provenance through
``src.utils.provenance.write_metrics``.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.environment.dataset_loader import create_dataloaders, load_and_preprocess_data
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    DeepONetDynamics,
    FNODynamics,
    PhysicsConstrainedDeepONetDynamics,
)
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PUBLISHED_BENCHMARK = "benchmark_metrics.json"
ARTIFACT_NAME = "operator_benchmark_metrics.json"


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights, reported for parameter-parity comparisons."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _collect_test_predictions(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _build_model(
    name: str,
    state_dim: int,
    action_dim: int,
    latent_dim: int,
    fno_width: int,
    fno_modes: int,
    fno_layers: int,
) -> nn.Module:
    """Construct one operator architecture with its section-10.42 defaults."""
    if name == "DeepONet":
        return DeepONetDynamics(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    if name == "PhysicsConstrained_DeepONet":
        return PhysicsConstrainedDeepONetDynamics(
            state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim
        )
    if name == "FNO":
        return FNODynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers,
        )
    raise ValueError(f"unknown operator architecture {name!r}")


def run_operator_benchmark(
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
    fno_width: int = 32,
    fno_modes: int = 6,
    fno_layers: int = 2,
) -> Dict[str, Any]:
    """Train and evaluate the operator family under the unified protocol."""
    logger.info("====================================================================")
    logger.info("  NEURAL OPERATOR FAMILY: DEEPONET / PHYSICS-CONSTRAINED / FNO      ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    evaluator = RolloutEvaluator(device=device)

    architectures: Dict[str, Any] = {}
    for name in ("DeepONet", "PhysicsConstrained_DeepONet", "FNO"):
        logger.info(">>> Operator architecture: %s <<<", name)

        # Rebuild the seeded data pipeline per model so each row of the study
        # reproduces standalone with the canonical seed (README protocol).
        set_global_seed(seed)
        data_dict = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dict, batch_size=batch_size, seed=seed
        )
        state_dim = int(data_dict["train_states"].shape[1])
        action_dim = int(data_dict["train_actions"].shape[1])

        model = _build_model(
            name, state_dim, action_dim, latent_dim, fno_width, fno_modes, fno_layers
        )
        # Distinct per-study checkpoint namespace ("operator_*") so a family
        # re-run never overwrites the §10.41 deeponet_best.pt artifact.
        trainer = DynamicsTrainer(
            model=model,
            model_type=f"operator_{name.lower()}",
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

        eval_metrics = trainer.evaluate(test_loader)
        preds, targets = _collect_test_predictions(model, test_loader, device)
        per_variable = compute_per_variable_metrics(preds, targets)

        test_states = data_dict["test_states"]
        test_actions = data_dict["test_actions"]
        test_next_states = data_dict["test_next_states"]
        H = int(min(rollout_horizon, len(test_actions)))
        # Rollout routing only distinguishes recurrent ("lstm") from stateless
        # models; all three operators are stateless feedforward evaluators.
        res = evaluator.evaluate_rollout(
            model=model,
            model_type="operator_feedforward",
            initial_state=test_states[0],
            action_sequence=test_actions[:H],
            ground_truth_states=test_next_states[:H],
        )
        multi = evaluator.evaluate_rollout_multistart(
            model=model,
            model_type="operator_feedforward",
            states=test_states,
            actions=test_actions,
            next_states=test_next_states,
            horizon=H,
            num_starts=num_rollout_starts,
        )
        multistart = {k: v for k, v in multi.items() if k not in ("mean_drifts", "final_drifts")}

        architectures[name] = {
            "parameters": count_parameters(model),
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
        logger.info(
            "%s | params: %d | Test MSE: %.4f | Kinematic Residual: %.4f | "
            "Mean Drift: %.2f px | Violations: %d/%d",
            name,
            architectures[name]["parameters"],
            architectures[name]["test_loss_data"],
            architectures[name]["test_kinematic_error"],
            architectures[name]["rollout"]["mean_drift_pixels"],
            architectures[name]["rollout"]["kinematic_violations"],
            H,
        )

    summary: Dict[str, Any] = {
        "study": (
            "Neural-operator family under the unified protocol: DeepONet (10.41), "
            "Physics-Constrained DeepONet and FNO (10.42). The DeepONet row re-runs "
            "the exact 10.41 configuration under per-model reseeding."
        ),
        "architectures": architectures,
        "comparison_against_published": _published_reference(output_dir),
        "config": {
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "rollout_horizon": int(rollout_horizon),
            "num_rollout_starts": num_rollout_starts,
            "latent_dim": latent_dim,
            "fno_width": fno_width,
            "fno_modes": fno_modes,
            "fno_layers": fno_layers,
        },
    }

    out_path = write_metrics(
        os.path.join(output_dir, ARTIFACT_NAME),
        summary,
        seed=seed,
        command="python -m src.evaluation.operator_benchmark",
    )
    logger.info("Metrics written to %s", out_path)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural-operator family benchmark on Super Mario World WRAM telemetry."
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
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=6)
    parser.add_argument("--fno-layers", type=int, default=2)
    args = parse_args_with_config(parser)

    run_operator_benchmark(
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
        fno_width=args.fno_width,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
    )
