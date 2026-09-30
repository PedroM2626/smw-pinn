"""
kinematic_metric_decomposition_benchmark.py
What the published violation rate actually measures (README 10.48).

`RolloutEvaluator` has reported one kinematic figure since Section 8: a frame is a
violation when the position the model advanced differs by more than 0.2 px from the
velocity it reported on the **previous** frame. Because the model rolls itself
forward, that predicate is not a pure consistency test - it also fires when two
consecutive velocity estimates differ by more than 3.2 sub-pixels/frame, which is a
statement about smoothness. Section 10.47 met the consequence head-on: an arm whose
graph integrates position exactly by construction still posted a 0.1618 violation
rate, and the section had to explain the number instead of reporting it.

This study separates the quantities and re-scores every committed model with them,
including the console's own recorded transitions as the reference row:

* ``published_violation_rate`` - the 10.27 predicate, unchanged;
* ``integration_error_mean_px`` - the residual against the velocity the model
  actually advanced by (what a shell guarantees to be zero);
* ``jump_rate`` - how often consecutive velocities move more than the tolerance;
* ``bound_exceedance_rate`` - how often the velocity leaves the engine's bounds.

It also re-classifies every probed ceiling of 10.46/10.47 under four traction
thresholds instead of the chosen 2.5 px/frame, so the acceptance rule's one
free parameter is shown to be non-load-bearing or not.

Writes ``results/kinematic_metric_decomposition_metrics.json``. Emulator-free.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.evaluation.learned_structure_probe_benchmark import MODEL_REGISTRY
from src.evaluation.operator_physics_injection_benchmark import probe_registry
from src.evaluation.rollout_diagnostics import (
    PUBLISHED_TOLERANCE_PX,
    TOLERANCE_AS_VELOCITY_JUMP,
    aggregate,
    traction_sensitivity,
)
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

ARTIFACT_NAME = "kinematic_metric_decomposition_metrics.json"
FIGURE_NAME = "kinematic_metric_decomposition.png"

TRACTION_THRESHOLDS = (1.0, 1.5, 2.5, 3.5)
NUM_ROLLOUT_STARTS = 10
ROLLOUT_HORIZON = 120


def _load_models(device: torch.device) -> Dict[str, Tuple[Any, bool]]:
    """Every committed learned model: label -> (module, is_a_hard_shell)."""
    from src.evaluation.learned_structure_probe_benchmark import load_model

    models: Dict[str, Tuple[Any, bool]] = {}
    for registry, is_grid in ((MODEL_REGISTRY, False), (probe_registry(), True)):
        for row in registry:
            label, checkpoint, factory = row[0], row[1], row[2]
            imposed = bool(row[3])
            name = f"grid:{label}" if is_grid else label
            model = load_model(checkpoint, factory, device)
            if model is not None:
                models[name] = (model, imposed)
    return models


def _rollout_trajectories(
    model: Any,
    evaluator: RolloutEvaluator,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    horizon: int,
    num_starts: int,
) -> np.ndarray:
    """Collect the predicted trajectories behind the published multi-start figure."""
    n = len(actions)
    horizon = min(horizon, n - 1)
    stride = max(1, (n - horizon) // num_starts)
    starts: List[int] = []
    idx = 0
    while len(starts) < num_starts and idx + horizon <= n:
        starts.append(idx)
        idx += stride
    runs = np.zeros((len(starts), horizon, states.shape[1]), dtype=np.float32)
    for i, start in enumerate(starts):
        res = evaluator.evaluate_rollout(
            model=model,
            model_type="operator_feedforward",
            initial_state=states[start],
            action_sequence=actions[start : start + horizon],
            ground_truth_states=next_states[start : start + horizon],
        )
        runs[i] = res["predicted_trajectory"]
    return runs


def _real_reference(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, horizon: int, num_starts: int
) -> np.ndarray:
    """The console's own transitions, walked as if they were a model's rollout.

    Each start contributes the recorded state sequence; velocity at $t$ is taken from
    the recorded state, exactly as a rollout would carry it forward.
    """
    n = len(actions)
    horizon = min(horizon, n - 1)
    stride = max(1, (n - horizon) // num_starts)
    starts: List[int] = []
    idx = 0
    while len(starts) < num_starts and idx + horizon <= n:
        starts.append(idx)
        idx += stride
    runs = np.zeros((len(starts), horizon, states.shape[1]), dtype=np.float32)
    for i, start in enumerate(starts):
        runs[i] = next_states[start : start + horizon]
    return runs


def _render_figure(models: Dict[str, Any], reference: Dict[str, float], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shells = [n for n, b in models.items() if b["imposed_by_construction"]]
    free = [n for n in models if n not in shells]
    fig, ax = plt.subplots(figsize=(9, 6))
    for group, names, color in (("hard shell", shells, "#41d19a"), ("no shell", free, "#8ab4f8")):
        if not names:
            continue
        ax.scatter(
            [models[n]["decomposition"]["jump_rate"] for n in names],
            [models[n]["decomposition"]["published_violation_rate"] for n in names],
            s=28,
            color=color,
            label=group,
        )
    ax.scatter(
        [reference["jump_rate"]],
        [reference["published_violation_rate"]],
        marker="*",
        s=180,
        color="#e06666",
        label="recorded console telemetry",
    )
    ax.set_xlabel("velocity jump rate (fraction of frames moving more than 3.2 sub-px)")
    ax.set_ylabel("published kinematic-violation rate")
    ax.set_title("What the published violation rate measures (README 10.48)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _verdict(
    models: Dict[str, Any], reference: Dict[str, float], sensitivity: Dict[str, Any]
) -> Dict[str, Any]:
    exact = [n for n, b in models.items() if b["decomposition"]["integration_error_mean_px"] < 1e-4]
    mislabeled = [
        n
        for n, b in models.items()
        if b["decomposition"]["integration_error_mean_px"] < 1e-4
        and b["decomposition"]["published_violation_rate"] > 0.05
    ]
    counts = {k: v["accepted"] for k, v in sensitivity.items()}
    stable = len(set(counts.values())) == 1
    published_key = "learned_structure_probe_metrics.json"
    needed = [
        float(threshold)
        for threshold, block in sensitivity.items()
        if block["by_artifact"].get(published_key, 0) > 0
    ]
    verb = "is" if len(mislabeled) == 1 else "are"
    return {
        "models_with_exact_integration": len(exact),
        "exact_but_flagged_by_the_published_predicate": mislabeled,
        "console_jump_rate": reference["jump_rate"],
        "console_published_violation_rate": reference["published_violation_rate"],
        "traction_threshold_accept_counts": counts,
        "classification_is_threshold_independent": stable,
        "smallest_threshold_admitting_a_10_46_acceptance": min(needed) if needed else None,
        "reading": (
            f"{len(exact)} of {len(models)} committed models integrate position exactly "
            f"(<1e-4 px residual against the velocity they advanced by), and "
            f"{len(mislabeled)} of those {verb} still called kinematically inconsistent by the "
            f"published predicate at a rate above 0.05. Recorded telemetry itself trips the "
            f"predicate on {reference['published_violation_rate']:.3f} of frames because its "
            f"velocity jumps on {reference['jump_rate']:.3f} of them, so the figure is "
            f"partly a smoothness measurement and cannot be read as a consistency test alone. "
            + (
                f"The ceiling classification is identical for every traction threshold tried "
                f"({counts})."
                if stable
                else (
                    f"The ceiling classification is load-bearing in the chosen threshold: it "
                    f"accepts {counts} records at {sorted(counts)} px/frame respectively, and the "
                    + (
                        f"10.46 acceptances only appear at {min(needed):.1f} px/frame or looser."
                        if needed
                        else "10.46 acceptances appear at none of the thresholds tried."
                    )
                )
            )
        ),
    }


def run_kinematic_metric_decomposition(
    dataset_path: str = DATASET_GAMEPLAY,
    results_dir: str = RESULTS_DIR,
    output_dir: Optional[str] = None,
    horizon: int = ROLLOUT_HORIZON,
    num_starts: int = NUM_ROLLOUT_STARTS,
    thresholds: Tuple[float, ...] = TRACTION_THRESHOLDS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Decompose the published metric for every committed learned model."""
    out_dir = output_dir or results_dir
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.environment.dataset_loader import load_and_preprocess_data

    data = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    states = data["test_states"]
    actions = data["test_actions"]
    next_states = data["test_next_states"]

    evaluator = RolloutEvaluator(device=device)
    models = _load_models(device)
    logger.info("=== Decomposing the kinematic metric on %d models ===", len(models))

    scored: Dict[str, Any] = {}
    for name, (model, imposed) in models.items():
        runs = _rollout_trajectories(
            model, evaluator, states, actions, next_states, horizon, num_starts
        )
        scored[name] = {
            "imposed_by_construction": imposed,
            "decomposition": aggregate(runs),
        }
        logger.info(
            "  %-34s published %.4f | exact integration %.2e | jumps %.4f | bounds %.4f",
            name,
            scored[name]["decomposition"]["published_violation_rate"],
            scored[name]["decomposition"]["integration_error_mean_px"],
            scored[name]["decomposition"]["jump_rate"],
            scored[name]["decomposition"]["bound_exceedance_rate"],
        )

    reference = aggregate(_real_reference(states, actions, next_states, horizon, num_starts))
    logger.info(
        "  %-34s published %.4f | exact integration %.2e | jumps %.4f | bounds %.4f",
        "RECORDED TELEMETRY",
        reference["published_violation_rate"],
        reference["integration_error_mean_px"],
        reference["jump_rate"],
        reference["bound_exceedance_rate"],
    )

    rows: Dict[str, Any] = {}
    for artifact_name in (
        "learned_structure_probe_metrics.json",
        "physics_injection_structure_probe_metrics.json",
    ):
        path = os.path.join(results_dir, artifact_name)
        if os.path.isfile(path):
            rows[artifact_name] = read_metrics(path).get("structural_probes", {})
    sensitivity = traction_sensitivity(rows, thresholds)

    payload: Dict[str, Any] = {
        "study": (
            "The published rollout kinematic-violation figure decomposed into the three "
            "properties it mixes, scored for every committed learned model and for the "
            "recorded telemetry itself."
        ),
        "protocol": {
            "dataset_path": dataset_path,
            "seed": seed,
            "horizon": horizon,
            "num_starts": num_starts,
            "published_tolerance_px": PUBLISHED_TOLERANCE_PX,
            "tolerance_as_velocity_jump_subpixels": TOLERANCE_AS_VELOCITY_JUMP,
            "traction_thresholds_px_per_frame": list(thresholds),
            "emulator_required": False,
        },
        "models": scored,
        "recorded_telemetry_reference": reference,
        "traction_threshold_sensitivity": sensitivity,
        "verdict": _verdict(scored, reference, sensitivity),
    }

    if _render_figure(scored, reference, os.path.join(out_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.kinematic_metric_decomposition_benchmark",
        extra_meta={"models": list(scored)},
    )
    logger.info("Metrics written to %s", artifact)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizon", type=int, default=ROLLOUT_HORIZON)
    parser.add_argument("--num-starts", type=int, default=NUM_ROLLOUT_STARTS)
    parser.add_argument("--seed", type=int, default=42)
    args = parse_args_with_config(parser)

    run_kinematic_metric_decomposition(
        dataset_path=args.dataset_path,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        horizon=args.horizon,
        num_starts=args.num_starts,
        seed=args.seed,
    )
