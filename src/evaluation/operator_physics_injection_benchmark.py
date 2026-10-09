"""
operator_physics_injection_benchmark.py
The physics-injection grid: target x mechanism x family (README 10.47).

Sections 10.27 (MLP family) and 10.41/10.42 (operator family) each published two
cells of the same 2x2 - a network that predicts the next state directly, and a
network whose output is consumed by the engine's own discrete kinematics - but
the two cells differ in *two* things at once: the target the net is trained on
and the kinematics the graph enforces. Section 10.44.1 therefore found that the
Physics-Constrained DeepONet controls while the Hard Residual PINN does not, and
could only attribute the difference to "the residual the operator learns on top
of the shell", because no arm separated shell from parameterisation and no soft
operator existed at all.

This study fills the grid under one protocol. Axes:

- family: Statistical MLP, DeepONet, FNO.
- target: ``state`` (the net outputs the next state) or ``residual`` (the net
  outputs ``[delta_vx, delta_vy, contact logits]`` and the graph integrates).
- mechanism: ``none`` (SmoothL1 data term only), ``soft`` (the published Soft
  PINN composite penalty: kinematic + bound + contact residuals added to the
  loss), ``hard`` (Section 4 clamps and exact position integration in the graph).

``state`` x ``hard`` is not implemented: clamping a state-output network's own
velocity and re-integrating its position is an output *projection*, which is a
third mechanism and is already the subject of the CBF layer of Section 10.16.
The two published operator shells are kept as reference arms built from their
original classes, so the study also shows whether the shared
``ResidualDynamics`` integration reproduces them.

Writes ``results/operator_physics_injection_metrics.json`` and
``results/figures/operator_physics_injection.png``, and saves one checkpoint per
arm per seed under ``results/checkpoints/`` for the closed-loop and structural
follow-ups.
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.environment.dataset_loader import create_dataloaders, load_and_preprocess_data
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.losses.physics_losses import CompositePINNLoss
from src.models import (
    DeepONetDynamics,
    FNODynamics,
    HardResidualPINNDynamics,
    PhysicsConstrainedDeepONetDynamics,
    ResidualDynamics,
    StatisticalMLPDynamics,
)
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.kinematics import CONTACT_ZERO_VELOCITY, NEXT
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

ARTIFACT_NAME = "operator_physics_injection_metrics.json"
FIGURE_NAME = "operator_physics_injection.png"

MAX_VX = 72.0
TERMINAL_VY = 64.0
MIN_VY = -80.0
SUBPIXELS_PER_PIXEL = 16.0

FAMILIES: Tuple[str, ...] = ("MLP", "DeepONet", "FNO")
TARGETS: Tuple[str, ...] = ("state", "residual")
MECHANISMS: Tuple[str, ...] = ("none", "soft", "hard")

# Soft penalty weights, taken verbatim from the published Soft PINN configuration
# (src/training/trainer.py) so "soft" means the same thing here as in 10.27.
SOFT_LAMBDAS: Dict[str, float] = {"lambda_kin": 1.0, "lambda_bound": 0.5, "lambda_contact": 0.5}

REFERENCE_ARMS: Tuple[Tuple[str, str], ...] = (
    ("Published_Hard_PINN_10_27", "hard_pinn"),
    ("Published_PC_DeepONet_10_42", "pc_deeponet"),
)


def arm_label(family: str, target: str, mechanism: str) -> str:
    """Stable identifier for one grid cell."""
    return f"{family}/{target}/{mechanism}"


def checkpoint_name(label: str) -> str:
    """The weight file the primary seed publishes for one arm."""
    return f"physinj_{label.replace('/', '_').lower()}_best.pt"


def probe_registry(
    state_dim: int = 8,
    action_dim: int = 6,
    latent_dim: int = 64,
    fno_width: int = 32,
    fno_modes: int = 6,
    fno_layers: int = 2,
) -> Tuple[Tuple[str, str, Any, bool, float], ...]:
    """(label, checkpoint, factory, clamp-imposed, shell constant) per grid arm.

    Consumed by the 10.46 structural probes so the new shells are attributed the
    same way the published ones were: a hard arm's fixed point could be the
    constructor's number rather than the network's.
    """
    rows = []
    for family, target, mechanism in grid_cells():
        label = arm_label(family, target, mechanism)
        imposed = target == "residual" and mechanism == "hard"
        rows.append(
            (
                label,
                checkpoint_name(label),
                lambda f=family, t=target, m=mechanism: build_arm(
                    f, t, m, state_dim, action_dim, latent_dim, fno_width, fno_modes, fno_layers
                ),
                imposed,
                MAX_VX if imposed else float("nan"),
            )
        )
    return tuple(rows)


def grid_cells() -> List[Tuple[str, str, str]]:
    """Every grid cell except ``state x hard``, which is a projection, not a shell."""
    cells = []
    for family in FAMILIES:
        for target in TARGETS:
            for mechanism in MECHANISMS:
                if target == "state" and mechanism == "hard":
                    continue
                cells.append((family, target, mechanism))
    return cells


def training_plan() -> List[Tuple[str, str, Optional[str], str, str]]:
    """(label, kind, family, target, mechanism) for every arm the study trains.

    The two ``reference`` items are the published shell classes, re-trained under
    this protocol, so the study can state whether the shared ``ResidualDynamics``
    integration reproduces them rather than asserting it.
    """
    items = [
        (arm_label(family, target, mechanism), "cell", family, target, mechanism)
        for family, target, mechanism in grid_cells()
    ]
    items += [(name, "reference", None, "residual", "hard") for name, _ in REFERENCE_ARMS]
    return items


def _build_reference_arm(name: str, state_dim: int, action_dim: int, latent_dim: int) -> nn.Module:
    """Construct one published shell model from its original class."""
    if name == "hard_pinn":
        return HardResidualPINNDynamics(state_dim=state_dim, action_dim=action_dim)
    if name == "pc_deeponet":
        return PhysicsConstrainedDeepONetDynamics(
            state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim
        )
    raise ValueError(f"unknown reference arm {name!r}")


def build_arm(
    family: str,
    target: str,
    mechanism: str,
    state_dim: int,
    action_dim: int,
    latent_dim: int,
    fno_width: int,
    fno_modes: int,
    fno_layers: int,
    position_velocity: str = NEXT,
) -> nn.Module:
    """Construct one grid cell: the family's net, optionally wrapped in the shell."""
    sensor_dim = state_dim + action_dim
    aux_dim = state_dim - 2
    if target == "state":
        if family == "MLP":
            return StatisticalMLPDynamics(state_dim=state_dim, action_dim=action_dim)
        if family == "DeepONet":
            return DeepONetDynamics(
                state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim
            )
        return FNODynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers,
        )
    if family == "MLP":
        base: nn.Module = StatisticalMLPDynamics(
            state_dim=aux_dim, action_dim=action_dim, sensor_dim=sensor_dim
        )
    elif family == "DeepONet":
        base = DeepONetDynamics(
            state_dim=aux_dim, action_dim=action_dim, latent_dim=latent_dim, sensor_dim=sensor_dim
        )
    else:
        base = FNODynamics(
            state_dim=aux_dim,
            action_dim=action_dim,
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers,
            sensor_dim=sensor_dim,
        )
    return ResidualDynamics(
        base=base,
        state_dim=state_dim,
        hard=(mechanism == "hard"),
        max_vx=MAX_VX,
        terminal_vy=TERMINAL_VY,
        min_vy=MIN_VY,
        subpixels_per_pixel=SUBPIXELS_PER_PIXEL,
        position_velocity=position_velocity,
    )


def build_loss(mechanism: str, contact_rule: str = CONTACT_ZERO_VELOCITY) -> nn.Module:
    """One of the three training objectives; ``hard`` is a graph constraint, not a loss."""
    if mechanism == "soft":
        return CompositePINNLoss(**SOFT_LAMBDAS, contact_rule=contact_rule)
    return nn.SmoothL1Loss()


@torch.no_grad()
def _velocity_compliance(
    model: nn.Module,
    states: Any,
    actions: Any,
    device: torch.device,
    batch: int = 512,
) -> Dict[str, float]:
    """How much of the model's own output the engine's bounds would have to clip."""
    model.eval()
    states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    violations: List[int] = []
    maxima: List[float] = []
    for start in range(0, states.shape[0], batch):
        s = states[start : start + batch].to(device)
        a = actions[start : start + batch].to(device)
        pred = model(s, a)
        vx = pred[:, 2].abs()
        vy = pred[:, 3]
        violations.append(int(((vx > MAX_VX) | (vy > TERMINAL_VY) | (vy < MIN_VY)).sum().item()))
        maxima.append(float(vx.max().item()))
    total = int(states.shape[0])
    out_of_bounds = sum(violations)
    return {
        "out_of_bounds_rate": out_of_bounds / total,
        "max_abs_vx_predicted": max(maxima),
    }


def _paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, Optional[float]]:
    """Paired statistics over seeds for one contrast (b - a)."""
    from scipy import stats

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = y - x
    if len(d) < 3 or np.allclose(d, 0.0):
        return {
            "n_pairs": int(len(d)),
            "mean_difference": float(d.mean()) if len(d) else None,
            "difference_std": float(d.std(ddof=1)) if len(d) > 1 else None,
            "ttest_p": None,
            "cohen_dz": None,
        }
    t_p = float(stats.ttest_rel(y, x).pvalue)
    dz = float(d.mean() / d.std(ddof=1))
    return {
        "n_pairs": int(len(d)),
        "mean_difference": float(d.mean()),
        "difference_std": float(d.std(ddof=1)),
        "ttest_p": t_p,
        "cohen_dz": dz,
    }


def _aggregate(runs: Dict[int, Dict[str, Dict[str, Any]]], arm: str, key: str) -> Dict[str, float]:
    """Mean and sample standard deviation of one metric across the seeds that ran."""
    values = [
        runs[s][arm][key] for s in sorted(runs) if arm in runs[s] and runs[s][arm][key] is not None
    ]
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_seeds": int(arr.size),
    }


def _series(runs: Dict[int, Dict[str, Dict[str, Any]]], arm: str, key: str) -> List[float]:
    return [float(runs[s][arm][key]) for s in sorted(runs) if arm in runs[s]]


def _contrast_table(runs: Dict[int, Dict[str, Dict[str, Any]]], metric: str) -> Dict[str, Any]:
    """The three single-axis contrasts, paired over seeds, for every family."""
    contrasts: Dict[str, Any] = {}
    for family in FAMILIES:
        cells = {
            "state/none": arm_label(family, "state", "none"),
            "state/soft": arm_label(family, "state", "soft"),
            "residual/none": arm_label(family, "residual", "none"),
            "residual/soft": arm_label(family, "residual", "soft"),
            "residual/hard": arm_label(family, "residual", "hard"),
        }
        blocks: Dict[str, Any] = {}
        for name, (left, right) in {
            "parameterisation_only": ("state/none", "residual/none"),
            "shell_at_fixed_parameterisation": ("residual/none", "residual/hard"),
            "soft_penalty_at_fixed_parameterisation": ("state/none", "state/soft"),
            "hard_vs_soft_on_residual": ("residual/soft", "residual/hard"),
        }.items():
            blocks[name] = _paired(
                _series(runs, cells[left], metric),
                _series(runs, cells[right], metric),
            )
        contrasts[family] = blocks
    return contrasts


def _verdict(
    summary: Dict[str, Any],
    drift: Dict[str, Any],
    bounds: Dict[str, Any],
    contrasts: Dict[str, Any],
) -> Dict[str, Any]:
    """Prose computed only from measured cells, so no claim outruns the artifact."""
    best = min(drift, key=lambda k: drift[k]["mean"])
    worst = max(drift, key=lambda k: drift[k]["mean"])
    shell_wins = {
        family: blocks["shell_at_fixed_parameterisation"]["mean_difference"]
        for family, blocks in contrasts.items()
        if blocks["shell_at_fixed_parameterisation"]["mean_difference"] is not None
    }
    helps = [f for f, d in shell_wins.items() if d < 0.0]
    hurts = [f for f, d in shell_wins.items() if d > 0.0]
    return {
        "best_drift_arm": best,
        "worst_drift_arm": worst,
        "spread_best_to_worst_px": drift[worst]["mean"] - drift[best]["mean"],
        "shell_reduces_drift_for": helps,
        "shell_increases_drift_for": hurts,
        "bounds_violated_by_soft_cells": {
            arm: value
            for arm, value in bounds.items()
            if value and value["out_of_bounds_rate"] > 0.0
        },
        "reading": (
            f"{len(summary)} cells of the target x mechanism x family grid were trained under one "
            f"protocol. The best multi-start drift is {drift[best]['mean']:.1f} px "
            f"({best}) and the worst is {drift[worst]['mean']:.1f} px ({worst}). The hard shell moves "
            f"drift the way the 10.42 account predicts for {helps or 'no family'} and the opposite way "
            f"for {hurts or 'no family'}, so the shell's effect on prediction is "
            + (
                "not uniform across families."
                if helps and hurts
                else "uniform in sign across the families that completed."
            )
        ),
    }


def _render_figure(summary: Dict[str, Any], out_path: str) -> bool:
    """Grouped bar chart of multi-start drift by family and mechanism."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in summary if "/residual/" in a or a.endswith("state/none")]
    if not arms:
        return False
    means = [summary[a]["drift_multistart_mean_px"]["mean"] for a in arms]
    stds = [summary[a]["drift_multistart_mean_px"]["std"] for a in arms]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"none": "#8ab4f8", "soft": "#f6c445", "hard": "#41d19a", "state": "#c0c0c0"}
    face = [colors["state"] if "/state/" in a else colors[a.split("/")[-1]] for a in arms]
    ax.bar(range(len(arms)), means, yerr=stds, color=face, capsize=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("multi-start mean drift (px, lower is better)")
    ax.set_title("Physics injection: target x mechanism x family (README 10.47)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def run_operator_physics_injection_benchmark(
    dataset_path: str = DATASET_GAMEPLAY,
    epochs: int = 35,
    batch_size: int = 128,
    seed: int = 42,
    seeds: Optional[Sequence[int]] = None,
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
    save_checkpoints: bool = True,
    only_arms: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train and score every cell of the physics-injection grid."""
    draws = list(seeds) if seeds else [seed]
    plan = training_plan()
    if only_arms:
        wanted = {str(name).strip() for name in only_arms}
        plan = [item for item in plan if item[0] in wanted or item[2] in wanted]
    logger.info("=== Physics-injection grid: %d arms x %d seeds ===", len(plan), len(draws))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    evaluator = RolloutEvaluator(device=device)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    scratch_dir = os.path.join(tempfile.gettempdir(), f"mworld_physinj_scratch_{os.getpid()}")
    os.makedirs(scratch_dir, exist_ok=True)

    runs: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for draw in draws:
        runs[draw] = {}
        for label, kind, family, target, mechanism in plan:
            set_global_seed(draw)
            data = load_and_preprocess_data(dataset_path=dataset_path, seed=draw)
            train_loader, val_loader, test_loader = create_dataloaders(
                data, batch_size=batch_size, seed=draw
            )
            state_dim = int(data["train_states"].shape[1])
            action_dim = int(data["train_actions"].shape[1])

            if kind == "reference":
                reference_key = dict(REFERENCE_ARMS)[label]
                model = _build_reference_arm(reference_key, state_dim, action_dim, latent_dim)
            else:
                assert family is not None
                model = build_arm(
                    family,
                    target,
                    mechanism,
                    state_dim,
                    action_dim,
                    latent_dim,
                    fno_width,
                    fno_modes,
                    fno_layers,
                )
            slug = f"physinj_{label.replace('/', '_').lower()}"
            # Only the first seed keeps committed weights: the closed-loop and
            # structural follow-ups load one published checkpoint per arm, exactly
            # as 10.42/10.46 do. The remaining seeds train into a scratch directory
            # that the study deletes, so five replicates do not multiply the LFS blob.
            primary = draw == draws[0]
            save_dir = checkpoint_dir if (primary and save_checkpoints) else scratch_dir
            trainer = DynamicsTrainer(
                model=model,
                model_type=slug if primary else f"{slug}_s{draw}",
                device=device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                loss_fn=build_loss(mechanism if kind == "cell" else "none"),
                save_dir=save_dir,
            )
            t0 = time.time()
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                patience=patience,
                verbose=False,
            )
            metrics = trainer.evaluate(test_loader)
            H = int(min(rollout_horizon, len(data["test_actions"])))
            single = evaluator.evaluate_rollout(
                model=model,
                model_type="operator_feedforward",
                initial_state=data["test_states"][0],
                action_sequence=data["test_actions"][:H],
                ground_truth_states=data["test_next_states"][:H],
            )
            multi = evaluator.evaluate_rollout_multistart(
                model=model,
                model_type="operator_feedforward",
                states=data["test_states"],
                actions=data["test_actions"],
                next_states=data["test_next_states"],
                horizon=H,
                num_starts=num_rollout_starts,
            )
            preds, targets = _collect(data, model, device)
            per_variable = compute_per_variable_metrics(preds, targets)
            compliance = _velocity_compliance(
                model,
                data["test_states"],
                data["test_actions"],
                device,
            )
            runs[draw][label] = {
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "test_loss_data": metrics["val_loss_data"],
                "test_kinematic_error": metrics["val_loss_kinematics"],
                "drift_single_px": float(single["mean_drift"]),
                "drift_multistart_mean_px": float(multi["mean_drift_mean"]),
                "drift_multistart_final_px": float(multi["final_drift_mean"]),
                "kinematic_violation_rate": float(multi["kinematic_violation_rate_mean"]),
                "velocity_violation_rate": float(multi["velocity_violation_rate_mean"]),
                "out_of_bounds_rate": compliance["out_of_bounds_rate"],
                "max_abs_vx_predicted": compliance["max_abs_vx_predicted"],
                "test_seconds": time.time() - t0,
                "x_mae_px": float(per_variable["x"]["mae"]),
                "vx_mae_px": float(per_variable["vx"]["mae"]),
                "vy_mae_px": float(per_variable["vy"]["mae"]),
                "c_ground_accuracy": float(per_variable["c_ground"]["accuracy"]),
            }
            logger.info(
                "seed %d | %-26s | loss %.4f | drift %.1f px | viol %.3f",
                draw,
                label,
                runs[draw][label]["test_loss_data"],
                runs[draw][label]["drift_multistart_mean_px"],
                runs[draw][label]["kinematic_violation_rate"],
            )
    shutil.rmtree(scratch_dir, ignore_errors=True)
    if save_checkpoints:
        logger.info("primary-seed checkpoints kept in %s", checkpoint_dir)

    labels = [label for label, *_ in plan]
    drift = {
        arm: _aggregate(runs, arm, "drift_multistart_mean_px")
        for arm in labels
        if _series(runs, arm, "drift_multistart_mean_px")
    }
    bounds = {
        a: {
            "out_of_bounds_rate": _aggregate(runs, a, "out_of_bounds_rate")["mean"],
            "max_abs_vx_predicted": _aggregate(runs, a, "max_abs_vx_predicted")["mean"],
        }
        for a in drift
    }
    summary = {
        a: {
            "test_loss_data": _aggregate(runs, a, "test_loss_data"),
            "test_kinematic_error": _aggregate(runs, a, "test_kinematic_error"),
            "drift_multistart_mean_px": _aggregate(runs, a, "drift_multistart_mean_px"),
            "drift_multistart_final_px": _aggregate(runs, a, "drift_multistart_final_px"),
            "kinematic_violation_rate": _aggregate(runs, a, "kinematic_violation_rate"),
            "velocity_violation_rate": _aggregate(runs, a, "velocity_violation_rate"),
            "drift_single_px": _aggregate(runs, a, "drift_single_px"),
            "x_mae_px": _aggregate(runs, a, "x_mae_px"),
            "vx_mae_px": _aggregate(runs, a, "vx_mae_px"),
            "vy_mae_px": _aggregate(runs, a, "vy_mae_px"),
            "c_ground_accuracy": _aggregate(runs, a, "c_ground_accuracy"),
            "out_of_bounds_rate": _aggregate(runs, a, "out_of_bounds_rate"),
            "parameters": _aggregate(runs, a, "parameters"),
        }
        for a in drift
    }
    contrasts = _contrast_table(runs, "drift_multistart_mean_px")
    payload: Dict[str, Any] = {
        "study": (
            "Physics-injection grid: family x target x mechanism under the unified "
            "protocol of 10.42, over "
            f"{len(draws)} seeds."
        ),
        "protocol": {
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "seeds": list(draws),
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "rollout_horizon": int(rollout_horizon),
            "num_rollout_starts": num_rollout_starts,
            "latent_dim": latent_dim,
            "fno_width": fno_width,
            "fno_modes": fno_modes,
            "fno_layers": fno_layers,
            "soft_lambdas": SOFT_LAMBDAS,
            "shell_constants": {
                "max_vx": MAX_VX,
                "terminal_vy": TERMINAL_VY,
                "min_vy": MIN_VY,
                "subpixels_per_pixel": SUBPIXELS_PER_PIXEL,
            },
            "excluded_cell": "state x hard (an output projection, not a shell)",
        },
        "per_seed": {str(s): runs[s] for s in sorted(runs)},
        "summary": summary,
        "bounds": bounds,
        "contrasts_drift_px": contrasts,
        "verdict": _verdict(summary, drift, bounds, contrasts),
    }

    figure_ok = _render_figure(summary, os.path.join(output_dir, "figures", FIGURE_NAME))
    if figure_ok:
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    command = (
        "python -m src.evaluation.operator_physics_injection_benchmark"
        f" --seeds {','.join(str(s) for s in draws)}"
    )
    if only_arms:
        command += f" --arms {','.join(only_arms)}"
    out_path = write_metrics(
        os.path.join(output_dir, ARTIFACT_NAME),
        payload,
        seed=draws[0],
        command=command,
    )
    logger.info("Metrics written to %s", out_path)
    return payload


@torch.no_grad()
def _collect(
    data: Dict[str, Any], model: nn.Module, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict every held-out transition in one batch."""
    model.eval()
    states = torch.as_tensor(data["test_states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(data["test_actions"], dtype=torch.float32, device=device)
    preds = model(states, actions).cpu()
    return preds, torch.as_tensor(data["test_next_states"], dtype=torch.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Physics-injection grid (target x mechanism x family) on SMW WRAM telemetry."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seed list.")
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
    parser.add_argument(
        "--no-checkpoints", action="store_true", help="Skip saving per-arm weights."
    )
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated arm labels or families to run (default: the whole grid).",
    )
    args = parse_args_with_config(parser)

    run_operator_physics_injection_benchmark(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        seeds=[int(s) for s in str(args.seeds).split(",") if s.strip()],
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
        save_checkpoints=not args.no_checkpoints,
        only_arms=[a.strip() for a in args.arms.split(",") if a.strip()],
    )
