r"""
effective_velocity_benchmark.py
Changing what the network is asked to predict: the frame's effective velocity (README 10.53).

Every dynamics model in this repository has predicted the velocity the WRAM *reports*
after the frame and then used that same number to advance position,
$\hat x_{t+1} = x_t + \hat v_{t+1}/16$. Section 10.49 read the console and found the
opposite convention: over 45,389 transitions the median of $|\Delta x - v_{x,t}/16|$ is
0.0000 px exactly, while the same statistic against $v_{x,t+1}$ is 0.0625 px - one
sub-pixel of lag on every accelerating frame. Position is advanced with the velocity the
frame *starts* with; the velocity the frame ends with is a different quantity, and the
two coincide only on a frame of constant speed. The repository's own penalty term
(`DiscreteKinematicsLoss`) and its rollout predicate both already read the identity with
$v_t$; only the graphs used the other one.

So there is a prediction target that has never been trained here: the **effective**
velocity, the displacement the engine actually applied, expressed in sub-pixels per
frame. `EffectiveVelocityDynamics` implements it in three modes that differ in one line:

- ``next``: position advanced by $\hat v_{t+1}$ - the published shell, rebuilt inside this
  class so the convention axis can be flipped without touching any other code path, and
  then checked against 10.47's artifact cell for cell;
- ``carried``: position advanced by $v_t$, the console's convention, which makes position
  a deterministic function of the input state. The network has no influence at all on the
  position it outputs, so that output's error measures where the engine departs from its
  own rule rather than what a network failed to learn;
- ``offset``: position advanced by $v_t + \varepsilon_t$ with $\hat\varepsilon$ a free
  head - the only form that can represent those departing frames.

Three families x three modes x three mechanisms (none / the published soft composite
penalty / the Section 4 hard bounds), five seeds, the 10.42 protocol. Two figures the
earlier studies could not produce: the published predicate scored **at a ladder of
tolerances** and **against both velocity conventions** at once, which separates "exact by
construction" from "inside the tolerance", and the **console's own score** on the same
grid, so a model more regular than Mario is recognisable as such.

Writes ``results/effective_velocity_metrics.json`` and a primary-seed checkpoint for each
of the nine unconstrained cells, which the closed-loop follow-up flies. Emulator-free.
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
from src.evaluation.kinematic_metric_decomposition_benchmark import (
    _real_reference,
    _rollout_trajectories,
)
from src.evaluation.operator_physics_injection_benchmark import (
    MAX_VX,
    MIN_VY,
    SOFT_LAMBDAS,
    SUBPIXELS_PER_PIXEL,
    TERMINAL_VY,
    _aggregate,
    _paired,
    _series,
    _velocity_compliance,
    build_loss,
)
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_diagnostics import (
    PUBLISHED_TOLERANCE_PX,
    TOLERANCE_LADDER,
    aggregate,
    carried_integration_vs_published,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import (
    DeepONetDynamics,
    EffectiveVelocityDynamics,
    FNODynamics,
    StatisticalMLPDynamics,
)
from src.models.effective_velocity_dynamics import VELOCITY_CHANNELS
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed
from src.utils.typography import demath_typographic

logger = get_logger(__name__)

ARTIFACT_NAME = "effective_velocity_metrics.json"
FIGURE_NAME = "effective_velocity.png"
GRID_ARTIFACT_NAME = "operator_physics_injection_metrics.json"

FAMILIES: Tuple[str, ...] = ("MLP", "DeepONet", "FNO")
MODES: Tuple[str, ...] = ("next", "carried", "offset")
MECHANISMS: Tuple[str, ...] = ("none", "soft", "hard")

# Only the arms the closed loop flies publish weights, and the loop flies the unconstrained
# cell of each mode; the constrained cells are scored inside this study.
PUBLISHED_MECHANISM = "none"

# The tightest tolerance tried, which is the field the headline claims are read from.
TIGHTEST_TOLERANCE = min(TOLERANCE_LADDER)

# The engine's displacement is a whole number of sub-pixels, so "recovered exactly" means
# recovered to the same integer, and half a sub-pixel is the tightest meaningful threshold.
EXACT_SUBPIXELS = 0.5

DEFAULT_SEEDS: Sequence[int] = (42, 43, 44, 45, 46)

# artifact block -> the label section 10.53 gives that metric in the convention-contrast table
CONTRAST_LABELS: Dict[str, str] = {
    "contrasts_drift_px": "multi-start drift (px)",
    "contrasts_x_mae_px": "held-out position error (px)",
    "contrasts_effective_vx_mae_subpx": "effective-velocity error (sub-px)",
    "contrasts_published_violation_at_tightest": "violation at 0.002 px",
}


def published_violation_key(tolerance_px: float) -> str:
    """Record name for the published predicate, written against the carried velocity."""
    return f"published_violation_at_{tolerance_px:g}px"


def next_velocity_violation_key(tolerance_px: float) -> str:
    """The same predicate written against the model's own next-frame velocity."""
    return f"next_velocity_violation_at_{tolerance_px:g}px"


def violation_keys() -> Tuple[str, ...]:
    """Every tolerance x convention field an arm record carries."""
    return tuple(
        key
        for tolerance in TOLERANCE_LADDER
        for key in (published_violation_key(tolerance), next_velocity_violation_key(tolerance))
    )


def arm_label(family: str, mode: str, mechanism: str) -> str:
    """Stable identifier for one cell of the convention grid."""
    return f"{family}/{mode}/{mechanism}"


def checkpoint_name(label: str) -> str:
    """The weight file the primary seed publishes for one arm."""
    return f"effvel_{label.replace('/', '_').lower()}_best.pt"


def publishes(label: str) -> bool:
    """Whether one cell contributes a committed checkpoint."""
    return label.endswith(f"/{PUBLISHED_MECHANISM}")


def grid_cells() -> List[Tuple[str, str, str]]:
    """Every family x mode x mechanism cell."""
    return [
        (family, mode, mechanism)
        for family in FAMILIES
        for mode in MODES
        for mechanism in MECHANISMS
    ]


def training_plan() -> List[Tuple[str, str, str, str]]:
    """(label, family, mode, mechanism) for every arm the study trains."""
    return [
        (arm_label(family, mode, mechanism), family, mode, mechanism)
        for family, mode, mechanism in grid_cells()
    ]


def checkpoint_registry() -> Dict[str, str]:
    """closed-loop row name -> published checkpoint, for the arms the loop flies."""
    return {
        f"{family.lower()}_{mode}_{PUBLISHED_MECHANISM}": checkpoint_name(
            arm_label(family, mode, PUBLISHED_MECHANISM)
        )
        for family in FAMILIES
        for mode in MODES
    }


def aux_dim_for(mode: str, state_dim: int = 8) -> int:
    """Channels the base net emits: increments, effective offsets, contact logits."""
    return state_dim - 2 + VELOCITY_CHANNELS[mode] - 2


def build_arm(
    family: str,
    mode: str,
    mechanism: str,
    state_dim: int,
    action_dim: int,
    latent_dim: int = 64,
    fno_width: int = 32,
    fno_modes: int = 6,
    fno_layers: int = 2,
) -> EffectiveVelocityDynamics:
    """Construct one cell: the family's net inside the chosen integration convention."""
    sensor_dim = state_dim + action_dim
    aux_dim = aux_dim_for(mode, state_dim)
    base: nn.Module
    if family == "MLP":
        base = StatisticalMLPDynamics(
            state_dim=aux_dim, action_dim=action_dim, sensor_dim=sensor_dim
        )
    elif family == "DeepONet":
        base = DeepONetDynamics(
            state_dim=aux_dim, action_dim=action_dim, latent_dim=latent_dim, sensor_dim=sensor_dim
        )
    elif family == "FNO":
        base = FNODynamics(
            state_dim=aux_dim,
            action_dim=action_dim,
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers,
            sensor_dim=sensor_dim,
        )
    else:
        raise ValueError(f"unknown family {family!r}")
    return EffectiveVelocityDynamics(
        base=base,
        state_dim=state_dim,
        mode=mode,
        hard=(mechanism == "hard"),
        max_vx=MAX_VX,
        terminal_vy=TERMINAL_VY,
        min_vy=MIN_VY,
        subpixels_per_pixel=SUBPIXELS_PER_PIXEL,
    )


def telemetry_reference(states: np.ndarray, next_states: np.ndarray) -> Dict[str, float]:
    """The console's own integration, measured on the held-out transitions.

    This is the row every model is compared against: it says how much of a model's
    position error is a property of Mario rather than of the network, and what the engine
    itself scores on the predicate the repository uses to judge kinematic consistency.
    """
    s = np.asarray(states, dtype=np.float64)
    ns = np.asarray(next_states, dtype=np.float64)
    out: Dict[str, float] = {"n_transitions": float(s.shape[0])}
    for pos, vel, name in ((0, 2, "x"), (1, 3, "y")):
        d = ns[:, pos] - s[:, pos]
        carried = d - s[:, vel] / SUBPIXELS_PER_PIXEL
        following = d - ns[:, vel] / SUBPIXELS_PER_PIXEL
        offset = SUBPIXELS_PER_PIXEL * d - s[:, vel]
        out[f"{name}_carried_residual_mean_px"] = float(np.abs(carried).mean())
        out[f"{name}_carried_residual_median_px"] = float(np.median(np.abs(carried)))
        out[f"{name}_next_residual_mean_px"] = float(np.abs(following).mean())
        out[f"{name}_next_residual_median_px"] = float(np.median(np.abs(following)))
        out[f"{name}_effective_exact_rate"] = float(np.mean(np.abs(offset) < EXACT_SUBPIXELS))
        out[f"{name}_effective_offset_mean_abs_subpx"] = float(np.abs(offset).mean())
        out[f"{name}_effective_offset_p95_abs_subpx"] = float(np.quantile(np.abs(offset), 0.95))
    return out


def prediction_metrics(
    pred: np.ndarray, s: np.ndarray, ns: np.ndarray, per_variable: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Held-out errors that any next-state predictor has, independent of its parameterisation.

    Shared with 10.55 because the two studies must score the same quantity the same way: an
    integrator arm and a convention arm are only comparable if "position error" and
    "effective-velocity error" are the same arithmetic in both artifacts.
    """
    moved_true = SUBPIXELS_PER_PIXEL * (ns[:, :2] - s[:, :2])
    moved_pred = SUBPIXELS_PER_PIXEL * (np.asarray(pred)[:, :2] - s[:, :2])
    position_error = np.abs(moved_pred - moved_true)
    reported_error = SUBPIXELS_PER_PIXEL * np.abs(np.asarray(pred)[:, 2:4] - ns[:, 2:4])
    return {
        "x_mae_px": float(per_variable["x"]["mae"]),
        "y_mae_px": float(per_variable["y"]["mae"]),
        "vx_mae_px": float(per_variable["vx"]["mae"]),
        "vy_mae_px": float(per_variable["vy"]["mae"]),
        "c_ground_accuracy": float(per_variable["c_ground"]["accuracy"]),
        "effective_vx_mae_subpx": float(position_error[:, 0].mean()),
        "effective_vy_mae_subpx": float(position_error[:, 1].mean()),
        "x_effective_exact_rate": float(np.mean(position_error[:, 0] < EXACT_SUBPIXELS)),
        "y_effective_exact_rate": float(np.mean(position_error[:, 1] < EXACT_SUBPIXELS)),
        "metrics_position_subpx": float(position_error.mean()),
        "metrics_velocity_subpx": float(reported_error.mean()),
    }


@torch.no_grad()
def one_step_metrics(
    model: EffectiveVelocityDynamics, data: Dict[str, Any], device: torch.device
) -> Dict[str, float]:
    """`prediction_metrics` plus the effective-velocity offset this target owns."""
    model.eval()
    states = torch.as_tensor(data["test_states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(data["test_actions"], dtype=torch.float32, device=device)
    targets = torch.as_tensor(data["test_next_states"], dtype=torch.float32)
    pred = model(states, actions).cpu()
    s = np.asarray(data["test_states"], dtype=np.float64)
    ns = np.asarray(data["test_next_states"], dtype=np.float64)
    out = prediction_metrics(pred.numpy(), s, ns, compute_per_variable_metrics(pred, targets))
    offsets = model.effective_velocity(states, actions).cpu().numpy() - s[:, 2:4]
    out["offset_vx_mean_abs_subpx"] = float(np.abs(offsets[:, 0]).mean())
    out["offset_vy_mean_abs_subpx"] = float(np.abs(offsets[:, 1]).mean())
    return out


def rollout_violations(runs: np.ndarray) -> Dict[str, float]:
    """The published predicate at each tolerance, under both velocity conventions."""
    parts = [carried_integration_vs_published(runs[i]) for i in range(runs.shape[0])]
    return {key: float(np.mean([part[key] for part in parts])) for key in parts[0]}


def _mean_of_dicts(blocks: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a whole metrics dict across seeds, key by key."""
    keys = list(blocks[0])
    return {key: float(np.mean([block[key] for block in blocks])) for key in keys}


def _contrasts(runs: Dict[int, Dict[str, Dict[str, Any]]], metric: str) -> Dict[str, Any]:
    """Single-axis contrasts over the convention grid, paired by seed, per family."""
    named = {
        "convention_only": ("next/none", "carried/none"),
        "offset_head_on_carried": ("carried/none", "offset/none"),
        "offset_head_on_next": ("next/none", "offset/none"),
        "soft_penalty_on_next": ("next/none", "next/soft"),
        "soft_penalty_on_carried": ("carried/none", "carried/soft"),
        "soft_penalty_on_offset": ("offset/none", "offset/soft"),
        "bounds_on_next": ("next/none", "next/hard"),
        "bounds_on_carried": ("carried/none", "carried/hard"),
        "bounds_on_offset": ("offset/none", "offset/hard"),
    }
    return {
        family: {
            name: _paired(
                _series(runs, arm_label(family, *left.split("/")), metric),
                _series(runs, arm_label(family, *right.split("/")), metric),
            )
            for name, (left, right) in named.items()
        }
        for family in FAMILIES
    }


def _parity_with_published_grid(runs: Dict[int, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Does mode ``next`` reproduce the 10.47 ``residual`` cells it is coded to equal?

    The claim that the integration convention is the only difference between ``next`` and
    ``carried`` is only checkable if ``next`` *is* the published arm, so every one of the
    nine ``next`` cells is compared, per seed, against the drift 10.47 recorded for the
    matching cell under the same protocol.
    """
    path = os.path.join(RESULTS_DIR, GRID_ARTIFACT_NAME)
    if not os.path.isfile(path):
        return {"available": False, "reason": f"{GRID_ARTIFACT_NAME} not found"}
    published = read_metrics(path).get("per_seed", {})
    gaps: List[float] = []
    rows: Dict[str, Any] = {}
    for family in FAMILIES:
        for mechanism in MECHANISMS:
            mine, theirs = [], []
            for seed, record in sorted(runs.items()):
                cell = record.get(arm_label(family, "next", mechanism))
                peer = published.get(str(seed), {}).get(arm_label(family, "residual", mechanism))
                if cell is None or peer is None:
                    continue
                mine.append(cell["drift_multistart_mean_px"])
                theirs.append(peer["drift_multistart_mean_px"])
            if not mine:
                continue
            gap = float(np.max(np.abs(np.asarray(mine) - np.asarray(theirs))))
            gaps.append(gap)
            rows[f"{family}/{mechanism}"] = {
                "published_cell": arm_label(family, "residual", mechanism),
                "n_seeds_compared": len(mine),
                "max_abs_drift_difference_px": gap,
            }
    return {
        "available": True,
        "published_artifact": f"results/{GRID_ARTIFACT_NAME}",
        "per_cell": rows,
        "worst_abs_drift_difference_px": max(gaps) if gaps else None,
        "identical": bool(gaps) and max(gaps) == 0.0,
    }


def _mode_spread(summary: Dict[str, Any], key: str, mode: str) -> Dict[str, Any]:
    """Mean and spread of one metric over every cell that shares a convention."""
    values = [block[key]["mean"] for arm, block in summary.items() if f"/{mode}/" in arm]
    if not values:
        return {"n_cells": 0, "mean": None, "spread": None}
    return {
        "n_cells": len(values),
        "mean": float(np.mean(values)),
        "spread": float(max(values) - min(values)),
    }


def _verdict(summary: Dict[str, Any], reference: Dict[str, float]) -> Dict[str, Any]:
    """Prose built only from cells that were actually measured."""
    unconstrained = {arm: block for arm, block in summary.items() if arm.endswith("/none")}
    drift = {arm: block["drift_multistart_mean_px"]["mean"] for arm, block in unconstrained.items()}
    tight = published_violation_key(TIGHTEST_TOLERANCE)
    exact_zero = [arm for arm, block in unconstrained.items() if block[tight]["mean"] == 0.0]
    carried = _mode_spread(summary, "x_mae_px", "carried")
    best = min(drift, key=lambda arm: drift[arm]) if drift else None
    return {
        "drift_unconstrained_cells_px": drift,
        "best_drift_arm": best,
        "carried_position_error_px": carried["mean"],
        "carried_position_error_spread_px": carried["spread"],
        "n_carried_cells": carried["n_cells"],
        "next_position_error_px": _mode_spread(summary, "x_mae_px", "next")["mean"],
        "offset_position_error_px": _mode_spread(summary, "x_mae_px", "offset")["mean"],
        "console_position_error_under_carried_convention_px": reference[
            "x_carried_residual_mean_px"
        ],
        "console_effective_exact_rate": reference["x_effective_exact_rate"],
        "violation_rate_by_cell": {
            arm: {
                f"{tolerance:g}px": {
                    "published": block[published_violation_key(tolerance)]["mean"],
                    "against_next_velocity": block[next_velocity_violation_key(tolerance)]["mean"],
                }
                for tolerance in TOLERANCE_LADDER
            }
            for arm, block in summary.items()
        },
        "console_violation_rate": {
            f"{tolerance:g}px": {
                "published": reference[published_violation_key(tolerance)],
                "against_next_velocity": reference[next_velocity_violation_key(tolerance)],
            }
            for tolerance in TOLERANCE_LADDER
        },
        "arms_at_zero_for_the_published_predicate": exact_zero,
        "reading": (
            f"{len(summary)} cells of the convention grid were trained under one protocol. The "
            f"{carried['n_cells']} cells that advance position with the velocity the frame starts "
            f"with share a held-out position error of {carried['mean']:.4f} px (spread "
            f"{carried['spread']:.6f} px) because no weight vector touches position in them; the "
            f"console's own mean residual under that convention is "
            f"{reference['x_carried_residual_mean_px']:.4f} px and it integrates with the carried "
            f"velocity exactly on {reference['x_effective_exact_rate'] * 100:.1f}% of held-out "
            f"frames. {len(exact_zero)} of {len(unconstrained)} unconstrained arms hold the "
            f"published predicate at 0.0000 down to {TIGHTEST_TOLERANCE:g} px, while the console "
            f"is flagged on "
            f"{reference[published_violation_key(PUBLISHED_TOLERANCE_PX)] * 100:.1f}% of its own "
            f"frames."
        ),
    }


def _render_figure(summary: Dict[str, Any], reference: Dict[str, float], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"next": "#8ab4f8", "carried": "#41d19a", "offset": "#f6c445"}
    unconstrained = sorted(arm for arm in summary if arm.endswith("/none"))
    if not unconstrained:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(
        range(len(unconstrained)),
        [summary[arm]["x_mae_px"]["mean"] for arm in unconstrained],
        yerr=[summary[arm]["x_mae_px"]["std"] for arm in unconstrained],
        color=[colors[arm.split("/")[1]] for arm in unconstrained],
        capsize=3,
    )
    axes[0].axhline(
        reference["x_carried_residual_mean_px"], ls="--", color="#333333", label="console"
    )
    axes[0].set_xticks(range(len(unconstrained)))
    axes[0].set_xticklabels(unconstrained, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("held-out position error (px)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    labels = [f"{tolerance:g}px" for tolerance in TOLERANCE_LADDER]
    for mode in MODES:
        chosen = [arm for arm in unconstrained if f"/{mode}/" in arm]
        if not chosen:
            continue
        axes[1].plot(
            labels,
            [
                float(
                    np.mean(
                        [summary[arm][published_violation_key(tolerance)]["mean"] for arm in chosen]
                    )
                )
                for tolerance in TOLERANCE_LADDER
            ],
            marker="o",
            color=colors[mode],
            label=mode,
        )
    axes[1].axhline(
        reference[published_violation_key(PUBLISHED_TOLERANCE_PX)], ls="--", color="#333333"
    )
    axes[1].set_ylabel("violation rate of the published predicate")
    axes[1].set_xlabel("integration tolerance")
    axes[1].set_title("Tolerance ladder (mean over families, mechanism none)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Which velocity advances position (README 10.53)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def render_convention_table(payload: Dict[str, Any]) -> List[str]:
    r"""The README's main result table, generated from the artifact.

    Section 10.53 quotes one row per family and convention plus the console itself. Emitting
    the rows here means the citation gate and the README are produced by the same code: a
    re-run that moves a number moves the table text with it, and the gate fails until the
    README is regenerated, so no cell of that table can be typed in from a neighbouring row.
    """
    summary, reference = payload["summary"], payload["console_reference"]

    def cell(arm: str, key: str) -> float:
        return summary[arm][key]["mean"]

    rows: List[str] = []
    for mode in MODES:
        for family in FAMILIES:
            arm = arm_label(family, mode, "none")
            rows.append(
                rf"| {family} | `{mode}` | {cell(arm, 'x_mae_px'):.4f}"
                rf" | {cell(arm, 'effective_vx_mae_subpx'):.2f}"
                rf" | {cell(arm, 'x_effective_exact_rate') * 100:.1f}%"
                rf" | {cell(arm, 'drift_multistart_mean_px'):.1f}"
                rf" | {cell(arm, published_violation_key(0.2)):.4f}"
                rf" | {cell(arm, published_violation_key(0.002)):.4f}"
                rf" | {cell(arm, next_velocity_violation_key(0.002)):.4f} |"
            )
    rows.append(
        rf"| console | `carried` | {reference['x_carried_residual_mean_px']:.4f}"
        rf" | {reference['x_effective_offset_mean_abs_subpx']:.2f}"
        rf" | {reference['x_effective_exact_rate'] * 100:.1f}% | -"
        rf" | {reference[published_violation_key(0.2)]:.4f}"
        rf" | {reference[published_violation_key(0.002)]:.4f}"
        rf" | {reference[next_velocity_violation_key(0.002)]:.4f} |"
    )
    return [demath_typographic(row) for row in rows]


def render_ladder_table(payload: Dict[str, Any]) -> List[str]:
    r"""The tolerance-ladder rows: one per convention, plus the console.

    Averaged over families because the ladder's message is about the convention, not the
    net: every `carried` cell is 0.0000 at every tolerance and every `next` cell collapses
    toward one as the tolerance tightens.
    """
    summary, reference = payload["summary"], payload["console_reference"]
    rows: List[str] = []
    for mode in MODES:
        cells = [
            float(
                np.mean(
                    [
                        summary[arm_label(family, mode, "none")][
                            published_violation_key(tolerance)
                        ]["mean"]
                        for family in FAMILIES
                    ]
                )
            )
            for tolerance in TOLERANCE_LADDER
        ]
        against = float(
            np.mean(
                [
                    summary[arm_label(family, mode, "none")][
                        next_velocity_violation_key(TIGHTEST_TOLERANCE)
                    ]["mean"]
                    for family in FAMILIES
                ]
            )
        )
        rows.append(
            f"| `{mode}` | " + " | ".join(f"{value:.4f}" for value in cells) + f" | {against:.4f} |"
        )
    rows.append(
        "| console | "
        + " | ".join(
            f"{reference[published_violation_key(tolerance)]:.4f}" for tolerance in TOLERANCE_LADDER
        )
        + f" | {reference[next_velocity_violation_key(TIGHTEST_TOLERANCE)]:.4f} |"
    )
    return [demath_typographic(row) for row in rows]


def render_mechanism_table(payload: Dict[str, Any]) -> List[str]:
    """Every convention under every mechanism, averaged over families."""
    summary = payload["summary"]
    rows: List[str] = []
    for mode in MODES:
        for mechanism in MECHANISMS:
            arms = [arm_label(family, mode, mechanism) for family in FAMILIES]
            rows.append(
                f"| `{mode}` | `{mechanism}`"
                f" | {float(np.mean([summary[a]['x_mae_px']['mean'] for a in arms])):.4f}"
                f" | {float(np.mean([summary[a]['drift_multistart_mean_px']['mean'] for a in arms])):.1f}"
                f" | {float(np.mean([summary[a]['test_kinematic_error']['mean'] for a in arms])):.4f}"
                f" | {float(np.mean([summary[a]['out_of_bounds_rate']['mean'] for a in arms])):.3f} |"
            )
    return [demath_typographic(row) for row in rows]


def render_convention_contrast_table(payload: Dict[str, Any]) -> List[str]:
    """The single-axis convention contrast (next/none -> carried/none) per family."""
    rows: List[str] = []
    for block, label in CONTRAST_LABELS.items():
        if block not in payload:
            continue
        for family, contrasts in payload[block].items():
            value = contrasts["convention_only"]
            if value["mean_difference"] is None:
                continue
            rows.append(
                rf"| {family} | {label} | {value['mean_difference']:+.4f}"
                rf" | $d_z$ {value['cohen_dz']:+.2f} | $p$ {value['ttest_p']:.3f} |"
            )
    return [demath_typographic(row) for row in rows]


def run_effective_velocity_benchmark(
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
    save_checkpoints: bool = True,
    only_arms: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train and score every cell of the integration-convention grid."""
    out_dir = output_dir or RESULTS_DIR
    draws = list(seeds)
    plan = training_plan()
    if only_arms:
        wanted = {str(name).strip() for name in only_arms}
        plan = [item for item in plan if item[0] in wanted or item[1] in wanted]
    logger.info("=== Effective-velocity grid: %d arms x %d seeds ===", len(plan), len(draws))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    evaluator = RolloutEvaluator(device=device)
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    scratch = os.path.join(tempfile.gettempdir(), f"mworld_effvel_scratch_{os.getpid()}")
    os.makedirs(scratch, exist_ok=True)

    runs: Dict[int, Dict[str, Dict[str, Any]]] = {}
    telemetry: Dict[int, Dict[str, float]] = {}
    real_reference: Optional[np.ndarray] = None
    for draw in draws:
        runs[draw] = {}
        reference_row: Optional[Dict[str, float]] = None
        for label, family, mode, mechanism in plan:
            set_global_seed(draw)
            data = load_and_preprocess_data(dataset_path=dataset_path, seed=draw)
            train_loader, val_loader, test_loader = create_dataloaders(
                data, batch_size=batch_size, seed=draw
            )
            state_dim = int(data["train_states"].shape[1])
            action_dim = int(data["train_actions"].shape[1])
            reference_row = telemetry_reference(data["test_states"], data["test_next_states"])
            H = int(min(rollout_horizon, len(data["test_actions"])))
            if real_reference is None:
                real_reference = _real_reference(
                    data["test_states"],
                    data["test_actions"],
                    data["test_next_states"],
                    H,
                    num_rollout_starts,
                )

            model = build_arm(
                family,
                mode,
                mechanism,
                state_dim,
                action_dim,
                latent_dim,
                fno_width,
                fno_modes,
                fno_layers,
            )
            slug = f"effvel_{label.replace('/', '_').lower()}"
            primary = draw == draws[0] and save_checkpoints and publishes(label)
            trainer = DynamicsTrainer(
                model=model,
                model_type=slug if primary else f"{slug}_s{draw}",
                device=device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                loss_fn=build_loss(mechanism),
                save_dir=checkpoint_dir if primary else scratch,
            )
            t0 = time.time()
            trainer.fit(
                train_loader=train_loader, val_loader=val_loader, epochs=epochs, patience=patience
            )
            metrics = trainer.evaluate(test_loader)
            multi = evaluator.evaluate_rollout_multistart(
                model=model,
                model_type="operator_feedforward",
                states=data["test_states"],
                actions=data["test_actions"],
                next_states=data["test_next_states"],
                horizon=H,
                num_starts=num_rollout_starts,
            )
            single = evaluator.evaluate_rollout(
                model=model,
                model_type="operator_feedforward",
                initial_state=data["test_states"][0],
                action_sequence=data["test_actions"][:H],
                ground_truth_states=data["test_next_states"][:H],
            )
            trajectories = _rollout_trajectories(
                model,
                evaluator,
                data["test_states"],
                data["test_actions"],
                data["test_next_states"],
                H,
                num_rollout_starts,
            )
            record: Dict[str, Any] = {
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "test_loss_data": metrics["val_loss_data"],
                "test_kinematic_error": metrics["val_loss_kinematics"],
                "drift_single_px": float(single["mean_drift"]),
                "drift_multistart_mean_px": float(multi["mean_drift_mean"]),
                "drift_multistart_final_px": float(multi["final_drift_mean"]),
                "kinematic_violation_rate": float(multi["kinematic_violation_rate_mean"]),
                "velocity_violation_rate": float(multi["velocity_violation_rate_mean"]),
            }
            record.update(one_step_metrics(model, data, device))
            record.update(
                _velocity_compliance(model, data["test_states"], data["test_actions"], device)
            )
            record.update(rollout_violations(trajectories))
            decomposition = aggregate(trajectories)
            decomposition.pop("published_violation_rate", None)
            record["decomposition"] = decomposition
            record["test_seconds"] = time.time() - t0
            runs[draw][label] = record
            logger.info(
                "seed %d | %-24s | loss %.4f | xerr %.4f px | drift %.1f px | tight viol %.4f",
                draw,
                label,
                record["test_loss_data"],
                record["x_mae_px"],
                record["drift_multistart_mean_px"],
                record[published_violation_key(TIGHTEST_TOLERANCE)],
            )
        if reference_row is not None:
            telemetry[draw] = reference_row
    shutil.rmtree(scratch, ignore_errors=True)

    flat_keys = (
        "parameters",
        "test_loss_data",
        "test_kinematic_error",
        "drift_single_px",
        "drift_multistart_mean_px",
        "drift_multistart_final_px",
        "kinematic_violation_rate",
        "velocity_violation_rate",
        "out_of_bounds_rate",
        "max_abs_vx_predicted",
        "test_seconds",
        "x_mae_px",
        "y_mae_px",
        "vx_mae_px",
        "vy_mae_px",
        "c_ground_accuracy",
        "effective_vx_mae_subpx",
        "effective_vy_mae_subpx",
        "x_effective_exact_rate",
        "y_effective_exact_rate",
        "offset_vx_mean_abs_subpx",
        "offset_vy_mean_abs_subpx",
        "metrics_position_subpx",
        "metrics_velocity_subpx",
        *violation_keys(),
    )
    summary: Dict[str, Any] = {}
    for label, *_ in plan:
        if not _series(runs, label, "drift_multistart_mean_px"):
            continue
        block: Dict[str, Any] = {key: _aggregate(runs, label, key) for key in flat_keys}
        block["decomposition"] = _mean_of_dicts(
            [runs[s][label]["decomposition"] for s in sorted(runs) if label in runs[s]]
        )
        summary[label] = block

    reference = _mean_of_dicts([telemetry[s] for s in sorted(telemetry)])
    if real_reference is not None:
        reference.update(rollout_violations(real_reference))
        reference.update(aggregate(real_reference))

    payload: Dict[str, Any] = {
        "study": (
            "Prediction target, not architecture: the velocity the engine integrates with. "
            "family x convention x mechanism under the protocol of 10.42/10.47, over "
            f"{len(draws)} seeds."
        ),
        "protocol": {
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "seeds": draws,
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
            "modes": {
                "next": "hat_x = x + hat_v_{t+1}/16 (the published shell)",
                "carried": "hat_x = x + v_t/16 (the convention 10.49 measured on the console)",
                "offset": "hat_x = x + (v_t + eps_t)/16 (free effective-velocity head)",
            },
            "tolerances_px": list(TOLERANCE_LADDER),
            "published_tolerance_px": PUBLISHED_TOLERANCE_PX,
            "exact_subpixels": EXACT_SUBPIXELS,
            "published_checkpoints": [
                checkpoint_name(label) for label, *_ in plan if publishes(label)
            ],
        },
        "per_seed": {str(s): runs[s] for s in sorted(runs)},
        "telemetry_per_seed": {str(s): telemetry[s] for s in sorted(telemetry)},
        "console_reference": reference,
        "summary": summary,
        "contrasts_drift_px": _contrasts(runs, "drift_multistart_mean_px"),
        "contrasts_x_mae_px": _contrasts(runs, "x_mae_px"),
        "contrasts_effective_vx_mae_subpx": _contrasts(runs, "effective_vx_mae_subpx"),
        "contrasts_metrics_position_subpx": _contrasts(runs, "metrics_position_subpx"),
        "contrasts_metrics_velocity_subpx": _contrasts(runs, "metrics_velocity_subpx"),
        "contrasts_published_violation_at_tightest": _contrasts(
            runs, published_violation_key(TIGHTEST_TOLERANCE)
        ),
        "next_mode_vs_published_grid": _parity_with_published_grid(runs),
        "verdict": _verdict(summary, reference),
    }

    if _render_figure(summary, reference, os.path.join(out_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    command = (
        "python -m src.evaluation.effective_velocity_benchmark"
        f" --seeds {','.join(str(s) for s in draws)}"
    )
    if only_arms:
        command += f" --arms {','.join(only_arms)}"
    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(artifact, payload, seed=draws[0], command=command)
    logger.info("Metrics written to %s", artifact)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Integration-convention grid on SMW WRAM telemetry (README 10.53)."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--rollout-horizon", type=int, default=120)
    parser.add_argument("--num-rollout-starts", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=6)
    parser.add_argument("--fno-layers", type=int, default=2)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--arms", default="", help="Comma-separated arm labels or families.")
    args = parse_args_with_config(parser)

    run_effective_velocity_benchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seeds=tuple(int(s) for s in str(args.seeds).split(",") if s.strip()),
        rollout_horizon=args.rollout_horizon,
        num_rollout_starts=args.num_rollout_starts,
        latent_dim=args.latent_dim,
        fno_width=args.fno_width,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
        save_checkpoints=not args.no_checkpoints,
        only_arms=[a.strip() for a in args.arms.split(",") if a.strip()],
    )
