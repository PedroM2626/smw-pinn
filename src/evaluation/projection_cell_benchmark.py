"""
projection_cell_benchmark.py
The cell 10.47 left out: bounds imposed on a state-output network (README 10.51).

10.47 crossed the target a network predicts against the mechanism that enforces the
engine's bounds, and excluded ``state`` x ``hard`` on the grounds that a clamp has
nothing to integrate there. It does have something to do: project the output. The
velocity the network proposes is clipped to the admissible window and the position is
re-derived from the clipped velocity, so the guarantee holds for any weights while the
training objective stays a pure data term - which is precisely the combination no arm
in this repository has ever been.

Three arms, one per family, trained exactly as 10.47 trains its cells (same dataset,
same split, same optimizer recipe, five seeds), and compared against the two cells that
already exist for the same family: ``state``/``none``, which measures how much the
projection alone buys, and ``residual``/``hard``, which measures whether a guarantee
delivered after the network is worth as much as one built into it. The comparison
numbers are read from ``results/operator_physics_injection_metrics.json``, so the two
sides of every contrast come from the same protocol rather than from prose.

Writes ``results/projection_cell_metrics.json``. Emulator-free.
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.evaluation.operator_physics_injection_benchmark import (
    MAX_VX,
    MIN_VY,
    SUBPIXELS_PER_PIXEL,
    TERMINAL_VY,
    arm_label,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import ProjectedDynamics
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

ARTIFACT_NAME = "projection_cell_metrics.json"
FIGURE_NAME = "projection_cell.png"

FAMILIES: Sequence[str] = ("MLP", "DeepONet", "FNO")
DEFAULT_SEEDS: Sequence[int] = (42, 43, 44, 45, 46)


def projection_checkpoint_name(family: str) -> str:
    """The weight file the primary seed publishes for one projected arm."""
    return f"proj_{family.lower()}_best.pt"


def build_projected(
    family: str,
    state_dim: int = 8,
    action_dim: int = 6,
    latent_dim: int = 64,
    fno_width: int = 32,
    fno_modes: int = 6,
    fno_layers: int = 2,
) -> torch.nn.Module:
    """One state-output network of the given family behind the output projection."""
    from src.models import DeepONetDynamics, FNODynamics, StatisticalMLPDynamics

    if family == "MLP":
        base: torch.nn.Module = StatisticalMLPDynamics(state_dim=state_dim, action_dim=action_dim)
    elif family == "DeepONet":
        base = DeepONetDynamics(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    elif family == "FNO":
        base = FNODynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers,
        )
    else:
        raise ValueError(f"unknown family {family!r}")
    return ProjectedDynamics(
        base=base,
        state_dim=state_dim,
        max_vx=MAX_VX,
        terminal_vy=TERMINAL_VY,
        min_vy=MIN_VY,
        subpixels_per_pixel=SUBPIXELS_PER_PIXEL,
    )


def _paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, Optional[float]]:
    """Paired statistics over seeds for the contrast ``b - a``."""
    from scipy import stats

    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = y - x
    if d.size < 2 or np.allclose(d, 0.0):
        return {
            "n_pairs": int(d.size),
            "mean_difference": float(d.mean()) if d.size else None,
            "difference_std": float(d.std(ddof=1)) if d.size > 1 else None,
            "ttest_p": None,
            "cohen_dz": None,
        }
    std = float(d.std(ddof=1))
    return {
        "n_pairs": int(d.size),
        "mean_difference": float(d.mean()),
        "difference_std": std,
        "ttest_p": float(stats.ttest_rel(y, x).pvalue) if std > 0 else None,
        "cohen_dz": float(d.mean() / std) if std > 0 else None,
    }


def _series(artifact: Dict[str, Any], arm: str, key: str, seeds: Sequence[int]) -> List[float]:
    per_seed = artifact.get("per_seed", {})
    return [
        float(per_seed[str(seed)][arm][key]) for seed in seeds if arm in per_seed.get(str(seed), {})
    ]


def run_projection_cell_benchmark(
    dataset_path: str = DATASET_GAMEPLAY,
    output_dir: Optional[str] = None,
    epochs: int = 35,
    batch_size: int = 128,
    patience: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    rollout_horizon: int = 120,
    num_rollout_starts: int = 10,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    latent_dim: int = 64,
    fno_width: int = 32,
    fno_modes: int = 6,
    fno_layers: int = 2,
) -> Dict[str, Any]:
    """Train the three projected arms and contrast them with the two mechanisms 10.47 has."""
    out_dir = output_dir or RESULTS_DIR
    from src.environment.dataset_loader import create_dataloaders, load_and_preprocess_data
    from src.training.trainer import DynamicsTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = RolloutEvaluator(device=device)
    scratch = os.path.join(tempfile.gettempdir(), f"mworld_projection_scratch_{os.getpid()}")
    os.makedirs(scratch, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, "checkpoints")

    reference_path = os.path.join(RESULTS_DIR, "operator_physics_injection_metrics.json")
    reference = read_metrics(reference_path) if os.path.isfile(reference_path) else {"per_seed": {}}

    runs: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for seed in seeds:
        runs[seed] = {}
        for family in FAMILIES:
            set_global_seed(seed)
            data = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
            train_loader, val_loader, test_loader = create_dataloaders(
                data, batch_size=batch_size, seed=seed
            )
            model = build_projected(
                family,
                latent_dim=latent_dim,
                fno_width=fno_width,
                fno_modes=fno_modes,
                fno_layers=fno_layers,
            ).to(device)
            trainer = DynamicsTrainer(
                model=model,
                model_type=projection_checkpoint_name(family).removesuffix("_best.pt")
                if seed == seeds[0]
                else f"{projection_checkpoint_name(family).removesuffix('_best.pt')}_s{seed}",
                device=device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                save_dir=scratch if seed != seeds[0] else checkpoint_dir,
            )
            t0 = time.time()
            trainer.fit(
                train_loader=train_loader, val_loader=val_loader, epochs=epochs, patience=patience
            )
            metrics = trainer.evaluate(test_loader)
            H = int(min(rollout_horizon, len(data["test_actions"])))
            runs[seed][f"{family}/state/projected"] = {
                "test_loss_data": metrics["val_loss_data"],
                "test_kinematic_error": metrics["val_loss_kinematics"],
                "drift_multistart_mean_px": float(
                    evaluator.evaluate_rollout_multistart(
                        model=model,
                        model_type="operator_feedforward",
                        states=data["test_states"],
                        actions=data["test_actions"],
                        next_states=data["test_next_states"],
                        horizon=H,
                        num_starts=num_rollout_starts,
                    )["mean_drift_mean"]
                ),
                "vx_mae_px": _vx_mae(model, data, device),
                "seconds": time.time() - t0,
            }
            logger.info(
                "seed %d | %-22s | loss %.4f | drift %.1f px",
                seed,
                f"{family}/state/projected",
                runs[seed][f"{family}/state/projected"]["test_loss_data"],
                runs[seed][f"{family}/state/projected"]["drift_multistart_mean_px"],
            )
    shutil.rmtree(scratch, ignore_errors=True)

    projected = {f"{family}/state/projected" for family in FAMILIES}
    summary: Dict[str, Any] = {}
    for label in projected:
        family = label.split("/")[0]
        values = {
            key: [runs[s][label][key] for s in seeds]
            for key in ("test_loss_data", "drift_multistart_mean_px", "vx_mae_px")
        }
        summary[label] = {
            "test_loss_data": _mean(values["test_loss_data"]),
            "drift_multistart_mean_px": _mean(values["drift_multistart_mean_px"]),
            "vx_mae_px": _mean(values["vx_mae_px"]),
            "compared_with_state_none": _paired(
                _series(
                    reference, arm_label(family, "state", "none"), "drift_multistart_mean_px", seeds
                ),
                values["drift_multistart_mean_px"],
            ),
            "compared_with_residual_hard": _paired(
                _series(
                    reference,
                    arm_label(family, "residual", "hard"),
                    "drift_multistart_mean_px",
                    seeds,
                ),
                values["drift_multistart_mean_px"],
            ),
            "per_seed": {str(s): runs[s][label] for s in seeds},
        }

    payload: Dict[str, Any] = {
        "study": (
            "The 10.47 cell that was excluded as unimplementable, implemented as an output "
            "projection: bounds imposed after the network, on a network that predicts the state."
        ),
        "protocol": {
            "dataset_path": dataset_path,
            "seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "rollout_horizon": int(rollout_horizon),
            "num_rollout_starts": num_rollout_starts,
            "projection_window": {
                "max_vx": MAX_VX,
                "terminal_vy": TERMINAL_VY,
                "min_vy": MIN_VY,
                "subpixels_per_pixel": SUBPIXELS_PER_PIXEL,
            },
            "reference_artifact": "results/operator_physics_injection_metrics.json",
            "primary_seed_checkpoints": [projection_checkpoint_name(f) for f in FAMILIES],
        },
        "summary": summary,
        "verdict": _verdict(summary),
    }

    if _render_figure(summary, os.path.join(out_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=int(seeds[0]),
        command=f"python -m src.evaluation.projection_cell_benchmark --seeds {','.join(str(s) for s in seeds)}",
    )
    logger.info("Metrics written to %s", artifact)
    return payload


@torch.no_grad()
def _vx_mae(model: torch.nn.Module, data: Dict[str, Any], device: torch.device) -> float:
    model.eval()
    states = torch.as_tensor(data["test_states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(data["test_actions"], dtype=torch.float32, device=device)
    target = torch.as_tensor(data["test_next_states"], dtype=torch.float32)
    pred = model(states, actions).cpu()
    return float((pred[:, 2] - target[:, 2]).abs().mean())


def _mean(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_seeds": int(arr.size),
    }


def _verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
    lines = {
        label: {
            "gain_over_state_none_px": block["compared_with_state_none"]["mean_difference"],
            "gain_vs_residual_hard_px": block["compared_with_residual_hard"]["mean_difference"],
        }
        for label, block in summary.items()
    }
    beats_shell = [
        label for label, block in lines.items() if (block["gain_vs_residual_hard_px"] or 0) > 0
    ]
    return {
        "per_family": lines,
        "families_where_projection_beats_the_shell": beats_shell,
        "reading": (
            "Imposing the bounds on the output changes drift by "
            + ", ".join(
                f"{label} {block['gain_over_state_none_px']:+.1f} px against the same network "
                f"with nothing and {block['gain_vs_residual_hard_px']:+.1f} px against the shell"
                for label, block in lines.items()
                if block["gain_over_state_none_px"] is not None
                and block["gain_vs_residual_hard_px"] is not None
            )
            + f"; the projection beats the in-graph shell for {beats_shell or 'no family'}."
        ),
    }


def _render_figure(summary: Dict[str, Any], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(summary)
    if not labels:
        return False
    means = [summary[k]["drift_multistart_mean_px"]["mean"] for k in labels]
    stds = [summary[k]["drift_multistart_mean_px"]["std"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(labels)), means, yerr=stds, color="#f28e2b", capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("multi-start drift (px, lower is better)")
    ax.set_title("State target with the bounds projected onto the output (README 10.51)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=6)
    parser.add_argument("--fno-layers", type=int, default=2)
    args = parse_args_with_config(parser)

    run_projection_cell_benchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seeds=tuple(int(s) for s in str(args.seeds).split(",") if s.strip()),
        latent_dim=args.latent_dim,
        fno_width=args.fno_width,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
    )
