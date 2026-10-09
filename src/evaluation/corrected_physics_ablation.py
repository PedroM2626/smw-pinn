r"""
corrected_physics_ablation.py
Price the three corrections README section 10.54 records as prose-and-code disagreement (10.57).

Section 4 now states what the telemetry measures: position advances with the velocity the frame
*starts* with (§4.1), the ground flag suppresses the gravity *step* rather than stopping the body
(§4.3.5, which measured `v_y = 0` on 0.00% of the grounded un-jumping frames), and 72/64 are a
speed class and an unenforced parameter rather than bounds the console respects (§4.2.5/§4.3.3,
left by real recordings at 49.0 and 70.0). Four classes can now be built that way -
`ResidualDynamics`, `ProjectedDynamics`, `HardResidualPINNDynamics`,
`AnalyticalKinematicsDynamics` - behind flags, with the published form still the default, because
every artifact in this repository was recorded with it. This study is the measurement that choice
needs: what does implementing §4 literally cost or buy, per family, paired over seeds?

Three contrasts, one changed thing each:

1. **Integration convention** - the same class, the same clamps, the same objective, position
   advanced with `v_t` instead of the predicted `v_{t+1}`. Also replayed on *identical weights*
   (flip the attribute and re-score), which takes training noise out of the comparison entirely.
2. **Ground rule** - the composite penalty asking `v_y = 0` on a grounded frame versus asking the
   gravity step to be skipped, which is the rule §4.3.5 now carries.
3. **The ruler** - every trajectory scored twice: the published predicate at 72.0, and the same
   rollouts against the run class the recordings reach, 48.0. Nothing about the model changes, so
   the published violation column can be read for what it measures.

A control runs alongside all three: the default-convention arms must reproduce Section 10.47's
published cells, which is what proves the flags did not move the path the paper stands on. No
checkpoint is published here and no committed artifact is rewritten - this is an addition, and the
closed-loop side is deliberately out of scope.

Writes ``results/corrected_physics_ablation_metrics.json``. Emulator-free.

Run:  python -m src.evaluation.corrected_physics_ablation
      python -m src.evaluation.corrected_physics_ablation --epochs 2 --seeds 42 --arms "Hard PINN/next,Hard PINN/carried"
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
from src.evaluation.operator_physics_injection_benchmark import (
    MAX_VX,
    MIN_VY,
    TERMINAL_VY,
    _collect,
    _paired,
    _velocity_compliance,
    build_arm,
    build_loss,
)
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models.analytical_kinematics import AnalyticalKinematicsDynamics
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.training.trainer import DynamicsTrainer
from src.utils.config import parse_args_with_config
from src.utils.kinematics import (
    CARRIED,
    CONTACT_ZERO_INCREMENT,
    CONTACT_ZERO_VELOCITY,
    NEXT,
    set_position_velocity,
)
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed
from src.utils.typography import demath_typographic

logger = get_logger(__name__)

# The run class 10.49 establishes as the bound this data can reach - used here as a ruler only.
RUN_CLASS_VX = 48.0
PUBLISHED_TOLERANCE_PX = 0.2
# The tight end of 10.53's ladder: two thousandths of a pixel, i.e. 1/800 of a sub-pixel.
TIGHT_TOLERANCE_PX = 0.002
LATENT_DIM = 64
FNO_WIDTH, FNO_MODES, FNO_LAYERS = 32, 6, 2

SUMMARY_KEYS: Tuple[str, ...] = (
    "test_loss_data",
    "x_mae_px",
    "drift_multistart_mean_px",
    "kinematic_violation_rate",
    "kinematic_violation_rate_tight",
    "velocity_violation_rate",
    "velocity_violation_rate_run_class",
    "out_of_bounds_rate",
    "max_abs_vx_predicted",
    "ground_median_abs_vy_model",
    "ground_median_abs_step_model",
    "ground_rate_vy_exactly_zero_model",
)

# (label, kind, family, target, mechanism, mode_or_rule)
#   "reference" - a published class built directly, SmoothL1 objective
#   "cell"      - 10.47's build_arm, so the object is literally the same one
#   "soft"      - a state-output net trained with the composite penalty
#   "closed"    - the engine rules, no training, scored under both conventions
ARM_PLAN: List[Tuple[str, str, Optional[str], str, str, str]] = [
    ("Hard PINN/next", "reference", None, "state", "none", NEXT),
    ("Hard PINN/carried", "reference", None, "state", "none", CARRIED),
    ("MLP/residual/hard/next", "cell", "MLP", "residual", "hard", NEXT),
    ("MLP/residual/hard/carried", "cell", "MLP", "residual", "hard", CARRIED),
    ("DeepONet/residual/hard/next", "cell", "DeepONet", "residual", "hard", NEXT),
    ("DeepONet/residual/hard/carried", "cell", "DeepONet", "residual", "hard", CARRIED),
    ("FNO/residual/hard/next", "cell", "FNO", "residual", "hard", NEXT),
    ("FNO/residual/hard/carried", "cell", "FNO", "residual", "hard", CARRIED),
    ("MLP/state/soft/published", "soft", "MLP", "state", "soft", CONTACT_ZERO_VELOCITY),
    ("MLP/state/soft/increment", "soft", "MLP", "state", "soft", CONTACT_ZERO_INCREMENT),
    ("DeepONet/state/soft/published", "soft", "DeepONet", "state", "soft", CONTACT_ZERO_VELOCITY),
    ("DeepONet/state/soft/increment", "soft", "DeepONet", "state", "soft", CONTACT_ZERO_INCREMENT),
    ("FNO/state/soft/published", "soft", "FNO", "state", "soft", CONTACT_ZERO_VELOCITY),
    ("FNO/state/soft/increment", "soft", "FNO", "state", "soft", CONTACT_ZERO_INCREMENT),
    ("engine rules/next", "closed", None, "state", "none", NEXT),
    ("engine rules/carried", "closed", None, "state", "none", CARRIED),
]


PAIR_SUFFIXES = (("/next", "/carried"), ("/published", "/increment"))

# The arms the console flies. Only these publish weights, and only when the study is asked
# to: the default run trains into a scratch directory so the committed checkpoints - which are
# what every earlier artifact is a measurement of - keep their lineage.
FLYABLE_LABELS: Tuple[str, ...] = (
    "Hard PINN/next",
    "Hard PINN/carried",
    "MLP/residual/hard/next",
    "MLP/residual/hard/carried",
    "DeepONet/residual/hard/next",
    "DeepONet/residual/hard/carried",
    "FNO/residual/hard/next",
    "FNO/residual/hard/carried",
)


def publishes(label: str) -> bool:
    """True for the eight arms 10.57.1 flies, which are the only weights this study can publish."""
    return label in FLYABLE_LABELS


def arm_slug(label: str) -> str:
    """The console-runner key for one arm: a family prefix the paired contrasts can group on."""
    return label.replace("/", "_").replace(" ", "_").lower()


def checkpoint_name(label: str) -> str:
    """The weight file the primary seed publishes for one arm."""
    return f"corrphys_{arm_slug(label)}_best.pt"


def build_arm_for(label: str, state_dim: int = 8, action_dim: int = 6) -> nn.Module:
    """Rebuild one published arm untrained, so a checkpoint load is a shape check too."""
    for plan_label, kind, family, target, mechanism, mode in ARM_PLAN:
        if plan_label == label:
            return _build(
                kind, family, target, mechanism, mode, (state_dim, action_dim, LATENT_DIM)
            )
    raise ValueError(f"{label!r} is not an arm of this study")


def pair_of(label: str) -> Optional[Tuple[str, str]]:
    """The (published, corrected) arm a label belongs to, or None if it stands alone."""
    for published_suffix, corrected_suffix in PAIR_SUFFIXES:
        if label.endswith(published_suffix):
            return label, label[: -len(published_suffix)] + corrected_suffix
        if label.endswith(corrected_suffix):
            return label[: -len(corrected_suffix)] + published_suffix, label
    return None


def _build(
    kind: str,
    family: Optional[str],
    target: str,
    mechanism: str,
    mode: str,
    dims: Tuple[int, int, int],
) -> nn.Module:
    """Construct one arm. The default convention is never passed explicitly, so "next" is the
    same object the published grid built."""
    state_dim, action_dim, latent_dim = dims
    if kind == "closed":
        return AnalyticalKinematicsDynamics(
            state_dim=state_dim, action_dim=action_dim, position_velocity=mode
        )
    if kind == "reference":
        model = HardResidualPINNDynamics(state_dim=state_dim, action_dim=action_dim)
        model.position_velocity = mode
        return model
    assert family is not None
    if kind == "soft":
        return build_arm(
            family,
            "state",
            "state",
            state_dim,
            action_dim,
            latent_dim,
            FNO_WIDTH,
            FNO_MODES,
            FNO_LAYERS,
        )
    return build_arm(
        family,
        target,
        mechanism,
        state_dim,
        action_dim,
        latent_dim,
        FNO_WIDTH,
        FNO_MODES,
        FNO_LAYERS,
        position_velocity=mode,
    )


def _ground_stratum(
    model: nn.Module, data: Dict[str, Any], device: torch.device, batch: int = 512
) -> Dict[str, float]:
    """Read the model's own output on the frames §4.3.5 is about: grounded, no jump commanded.

    Both readings of that rule are reported - the vertical velocity itself, and the step taken -
    because that difference is the whole disagreement between the published penalty and the
    corrected one. The recorded frames are measured identically so the two are comparable.
    """
    states = np.asarray(data["test_states"], dtype=np.float64)
    actions = np.asarray(data["test_actions"], dtype=np.float64)
    next_states = np.asarray(data["test_next_states"], dtype=np.float64)
    mask = (states[:, 4] > 0.5) & (actions[:, 0] < 0.5)
    if not mask.any():
        return {}

    model.eval()
    predicted = np.zeros_like(next_states)
    with torch.no_grad():
        source = torch.as_tensor(data["test_states"], dtype=torch.float32)
        commands = torch.as_tensor(data["test_actions"], dtype=torch.float32)
        for start in range(0, source.shape[0], batch):
            chunk = model(
                source[start : start + batch].to(device), commands[start : start + batch].to(device)
            )
            predicted[start : start + batch] = chunk.cpu().numpy()

    def describe(vy: np.ndarray, step: np.ndarray) -> Dict[str, float]:
        return {
            "n_frames": int(vy.size),
            "median_abs_vy_model": float(np.median(np.abs(vy))),
            "median_abs_step_model": float(np.median(np.abs(step))),
            "rate_vy_exactly_zero_model": float(np.mean(vy == 0.0)),
        }

    out = describe(predicted[mask, 3], predicted[mask, 3] - states[mask, 3])
    out["median_abs_vy_recorded"] = float(np.median(np.abs(next_states[mask, 3])))
    out["median_abs_step_recorded"] = float(
        np.median(np.abs(next_states[mask, 3] - states[mask, 3]))
    )
    return {"ground_" + key: value for key, value in out.items()}


def _score(
    model: nn.Module, data: Dict[str, Any], device: torch.device, horizon: int, num_starts: int
) -> Dict[str, float]:
    """Every rollout metric, on the same trajectories, under both rulers and both tolerances.

    The published predicate is scored at its own 0.2 px, and again at 0.002 px: §10.48 and §10.53
    established that the kinematic-violation column is a tolerance statement, so an arm that
    changes the convention has to be read against that as well, inside this study rather than by
    borrowing the number from another section.
    """
    published = RolloutEvaluator(device=device)
    run_class = RolloutEvaluator(device=device, max_vx=RUN_CLASS_VX)
    tight = RolloutEvaluator(device=device, tolerance_px=TIGHT_TOLERANCE_PX)
    single = published.evaluate_rollout(
        model=model,
        model_type="operator_feedforward",
        initial_state=data["test_states"][0],
        action_sequence=data["test_actions"][:horizon],
        ground_truth_states=data["test_next_states"][:horizon],
    )
    kwargs = dict(
        model=model,
        model_type="operator_feedforward",
        states=data["test_states"],
        actions=data["test_actions"],
        next_states=data["test_next_states"],
        horizon=horizon,
        num_starts=num_starts,
    )
    multi = published.evaluate_rollout_multistart(**kwargs)
    multi_class = run_class.evaluate_rollout_multistart(**kwargs)
    multi_tight = tight.evaluate_rollout_multistart(**kwargs)
    preds, targets = _collect(data, model, device)
    per_variable = compute_per_variable_metrics(preds, targets)
    compliance = _velocity_compliance(model, data["test_states"], data["test_actions"], device)
    return {
        "drift_single_px": float(single["mean_drift"]),
        "drift_multistart_mean_px": float(multi["mean_drift_mean"]),
        "drift_multistart_final_px": float(multi["final_drift_mean"]),
        "kinematic_violation_rate": float(multi["kinematic_violation_rate_mean"]),
        "kinematic_violation_rate_tight": float(multi_tight["kinematic_violation_rate_mean"]),
        "velocity_violation_rate": float(multi["velocity_violation_rate_mean"]),
        "velocity_violation_rate_run_class": float(multi_class["velocity_violation_rate_mean"]),
        "out_of_bounds_rate": compliance["out_of_bounds_rate"],
        "max_abs_vx_predicted": compliance["max_abs_vx_predicted"],
        "x_mae_px": float(per_variable["x"]["mae"]),
        "y_mae_px": float(per_variable["y"]["mae"]),
        "vx_mae_px": float(per_variable["vx"]["mae"]),
    }


def _mean(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_seeds": int(arr.size),
    }


def _series(runs: Dict[int, Dict[str, Dict[str, Any]]], arm: str, key: str) -> List[float]:
    return [
        float(runs[seed][arm][key])
        for seed in sorted(runs)
        if arm in runs[seed] and runs[seed][arm].get(key) is not None
    ]


def _contrasts(runs: Dict[int, Dict[str, Dict[str, Any]]], metric: str) -> Dict[str, Any]:
    """Paired over seeds, published arm against corrected arm, one row per family."""
    out: Dict[str, Any] = {}
    seen = set()
    for label, *_rest in ARM_PLAN:
        pair = pair_of(label)
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        published, corrected = pair
        a, b = _series(runs, published, metric), _series(runs, corrected, metric)
        if a and len(a) == len(b):
            out[f"{published} -> {corrected}"] = _paired(a, b)
    return out


def _replay(
    model: nn.Module,
    data: Dict[str, Any],
    device: torch.device,
    horizon: int,
    num_starts: int,
    trained_mode: str,
) -> Dict[str, Any]:
    """The same weights, the other convention: a contrast with no training noise in it."""
    other = CARRIED if trained_mode == NEXT else NEXT
    previous = set_position_velocity(model, other)
    scored = _score(model, data, device, horizon, num_starts)
    set_position_velocity(model, previous)
    return {
        "as_trained": trained_mode,
        "replayed_as": other,
        "drift_multistart_mean_px": scored["drift_multistart_mean_px"],
        "x_mae_px": scored["x_mae_px"],
        "kinematic_violation_rate": scored["kinematic_violation_rate"],
        "velocity_violation_rate_run_class": scored["velocity_violation_rate_run_class"],
    }


def _parity_with_10_47(runs: Dict[int, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Do the default-convention arms reproduce the published grid cells, digit for digit?

    The flags must not move the path the paper stands on; this is that claim tested on the three
    metrics 10.47 publishes for these same objects.
    """
    path = os.path.join(RESULTS_DIR, "operator_physics_injection_metrics.json")
    if not os.path.isfile(path):
        return {"available": False, "reason": "the 10.47 grid artifact is not present"}
    grid = read_metrics(path)["per_seed"]
    mapping = {
        "MLP/residual/hard/next": "MLP/residual/hard",
        "DeepONet/residual/hard/next": "DeepONet/residual/hard",
        "FNO/residual/hard/next": "FNO/residual/hard",
        "MLP/state/soft/published": "MLP/state/soft",
        "DeepONet/state/soft/published": "DeepONet/state/soft",
        "FNO/state/soft/published": "FNO/state/soft",
    }
    differences: Dict[str, float] = {}
    compared = 0
    for arm, published_arm in mapping.items():
        for seed, cells in sorted(grid.items()):
            here = runs.get(int(seed), {}).get(arm)
            if here is None or published_arm not in cells:
                continue
            for key in ("drift_multistart_mean_px", "x_mae_px", "test_loss_data"):
                if key in here and key in cells[published_arm]:
                    differences[key] = max(
                        differences.get(key, 0.0), abs(here[key] - float(cells[published_arm][key]))
                    )
            compared += 1
    return {
        "available": compared > 0,
        "cell_seed_pairs_compared": compared,
        "max_abs_difference_by_metric": differences,
        "identical": bool(differences) and max(differences.values()) == 0.0,
    }


def run_corrected_physics_ablation(
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
    latent_dim: int = LATENT_DIM,
    only_arms: Optional[Sequence[str]] = None,
    save_checkpoints: bool = False,
) -> Dict[str, Any]:
    """Train and score every arm of the correction ablation, then write its artifact."""
    draws = list(seeds) if seeds else [seed]
    plan = list(ARM_PLAN)
    if only_arms:
        wanted = {str(name).strip() for name in only_arms}
        plan = [item for item in plan if item[0] in wanted]
    logger.info("=== Corrected-physics ablation: %d arms x %d seeds ===", len(plan), len(draws))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    # By default no weight is published: the committed checkpoints are what every earlier
    # artifact is a measurement *of*. `--save-checkpoints` opts in, and only for the eight arms
    # 10.57.1 flies, under a `corrphys_` prefix no other study reads.
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    scratch_dir = os.path.join(tempfile.gettempdir(), f"mworld_corrphys_scratch_{os.getpid()}")
    os.makedirs(scratch_dir, exist_ok=True)
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    runs: Dict[int, Dict[str, Dict[str, Any]]] = {draw: {} for draw in draws}
    replays: Dict[str, Dict[str, Any]] = {}
    for draw in draws:
        for label, kind, family, target, mechanism, mode_or_rule in plan:
            set_global_seed(draw)
            data = load_and_preprocess_data(dataset_path=dataset_path, seed=draw)
            train_loader, val_loader, test_loader = create_dataloaders(
                data, batch_size=batch_size, seed=draw
            )
            dims = (
                int(data["train_states"].shape[1]),
                int(data["train_actions"].shape[1]),
                latent_dim,
            )
            model = _build(kind, family, target, mechanism, mode_or_rule, dims)
            record: Dict[str, Any] = {
                "mode": mode_or_rule,
                "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            }
            if kind != "closed":
                loss_rule = mode_or_rule if kind == "soft" else CONTACT_ZERO_VELOCITY
                publishes_this = save_checkpoints and draw == draws[0] and publishes(label)
                trainer = DynamicsTrainer(
                    model=model,
                    model_type=(
                        f"corrphys_{arm_slug(label)}"
                        if publishes_this
                        else f"corrphys_scratch_{arm_slug(label)}_s{draw}"
                    ),
                    device=device,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    loss_fn=build_loss("soft" if kind == "soft" else "none", loss_rule),
                    save_dir=checkpoint_dir if publishes_this else scratch_dir,
                )
                started = time.time()
                trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    patience=patience,
                    verbose=False,
                )
                metrics = trainer.evaluate(test_loader)
                record["test_loss_data"] = float(metrics["val_loss_data"])
                record["test_kinematic_error"] = float(metrics["val_loss_kinematics"])
                record["train_seconds"] = time.time() - started
            horizon = int(min(rollout_horizon, len(data["test_actions"])))
            record.update(_score(model, data, device, horizon, num_rollout_starts))
            record.update(_ground_stratum(model, data, device))
            if draw == draws[0] and kind in ("reference", "cell", "closed"):
                replays[label] = _replay(
                    model, data, device, horizon, num_rollout_starts, mode_or_rule
                )
            runs[draw][label] = record
            logger.info(
                "seed %d | %-32s | drift %.1f px | viol@72 %.4f | viol@48 %.4f",
                draw,
                label,
                record["drift_multistart_mean_px"],
                record["velocity_violation_rate"],
                record["velocity_violation_rate_run_class"],
            )
    shutil.rmtree(scratch_dir, ignore_errors=True)

    labels = [item[0] for item in plan]
    payload: Dict[str, Any] = {
        "study": (
            "What the three corrections README 10.54 records as prose-and-code disagreement cost "
            "or buy: the 4.1 integration convention, the 4.3.5 ground rule, and the predicate's "
            "velocity bound, each changed one at a time and paired over seeds."
        ),
        "protocol": {
            "dataset": dataset_path,
            "seeds": [int(draw) for draw in draws],
            "epochs": epochs,
            "batch_size": batch_size,
            "rollout_horizon": rollout_horizon,
            "num_rollout_starts": num_rollout_starts,
            "published_predicate": {
                "max_vx": MAX_VX,
                "terminal_vy": TERMINAL_VY,
                "min_vy": MIN_VY,
                "tolerance_px": PUBLISHED_TOLERANCE_PX,
            },
            "alternative_ruler_max_vx": RUN_CLASS_VX,
            "arms_built_with": "10.47 build_arm/build_loss, same constructor defaults",
            "checkpoints_published": save_checkpoints,
            "emulator_required": False,
            "closed_loop_measured": False,
        },
        "arms": {label: {"mode": runs[draws[0]][label]["mode"]} for label in labels},
        "per_seed": {str(draw): runs[draw] for draw in draws},
        "summary": {
            label: {
                key: _mean(_series(runs, label, key))
                for key in SUMMARY_KEYS
                if _series(runs, label, key)
            }
            for label in labels
        },
        "same_weights_replay": replays,
        "ruler_within_arm": _ruler_within_arm(runs),
        "contrasts": {
            "drift_multistart_mean_px": _contrasts(runs, "drift_multistart_mean_px"),
            "x_mae_px": _contrasts(runs, "x_mae_px"),
            "velocity_violation_rate_run_class": _contrasts(
                runs, "velocity_violation_rate_run_class"
            ),
            "ground_median_abs_vy_model": _contrasts(runs, "ground_median_abs_vy_model"),
            "ground_median_abs_step_model": _contrasts(runs, "ground_median_abs_step_model"),
        },
        "parity_with_10_47": _parity_with_10_47(runs),
    }
    payload["verdict"] = _verdict(payload)
    artifact = os.path.join(output_dir, "corrected_physics_ablation_metrics.json")
    write_metrics(
        artifact,
        payload,
        seed=draws[0],
        command="python -m src.evaluation.corrected_physics_ablation",
    )
    return payload


def _ruler_within_arm(runs: Dict[int, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """For one arm, what raising the ruler from 72.0 to the reachable 48.0 costs it.

    The per-arm difference of the two violation columns, not a contrast between arms: the
    published predicate flags nothing at all on these rollouts, so the only way to see what that
    column measures is to score the same trajectory against the bound the data can reach.
    """
    out: Dict[str, Any] = {}
    for label in next(iter(runs.values())):
        published = _series(runs, label, "velocity_violation_rate")
        reachable = _series(runs, label, "velocity_violation_rate_run_class")
        if published and len(published) == len(reachable):
            out[label] = {
                "at_published_bound": _mean(published)["mean"],
                "at_run_class_bound": _mean(reachable)["mean"],
                "difference": _mean(reachable)["mean"] - _mean(published)["mean"],
            }
    return out


def _verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The readings, computed here so the README cannot restate them wrongly."""
    drift = payload["contrasts"]["drift_multistart_mean_px"]
    convention = {k: v for k, v in drift.items() if "/hard" in k or "Hard PINN" in k}
    ground = {k: v for k, v in drift.items() if "/soft" in k}
    sized = {k: v["mean_difference"] for k, v in drift.items() if v["mean_difference"] is not None}
    biggest = max(sized, key=lambda k: abs(sized[k])) if sized else None
    ruler = payload["ruler_within_arm"]
    moved = {k: v["difference"] for k, v in ruler.items() if v["difference"] != 0.0}
    replay = payload["same_weights_replay"]
    replay_moves = {
        label: block["drift_multistart_mean_px"]
        - payload["summary"]
        .get(label, {})
        .get("drift_multistart_mean_px", {})
        .get("mean", float("nan"))  # noqa: E501
        for label, block in replay.items()
    }
    return {
        "arms": len(payload["summary"]),
        "parity_with_10_47": payload["parity_with_10_47"],
        "convention_pairs": len(convention),
        "ground_rule_pairs": len(ground),
        "largest_drift_difference_px": (
            {"pair": biggest, "mean_difference_px": sized[biggest]} if biggest else None
        ),
        "ruler_flags_nothing_at_the_published_bound": all(
            v["at_published_bound"] == 0.0 for v in ruler.values()
        ),
        "arms_the_run_class_ruler_moves": len(moved),
        "largest_ruler_difference": (max(moved, key=lambda k: abs(moved[k])) if moved else None),
        "same_weights_replay_drift_difference_px": replay_moves,
        "closed_loop_measured_here": False,
    }


def _cell(block: Optional[Dict[str, Any]], spec: str = "{:.4f}") -> str:
    return spec.format(block["mean"]) if block else "-"


def render_convention_table(payload: Dict[str, Any]) -> List[str]:
    """One row per convention arm: accuracy, drift, and the violation rate under both rulers."""
    rows: List[str] = []
    for label, cells in payload["summary"].items():
        if not label.endswith(("/next", "/carried")):
            continue
        rows.append(
            f"| {label} | {_cell(cells.get('test_loss_data'))} | {_cell(cells.get('x_mae_px'))} | "
            f"{_cell(cells.get('drift_multistart_mean_px'), '{:.2f}')} | "
            f"{_cell(cells.get('kinematic_violation_rate'))} | "
            f"{_cell(cells.get('kinematic_violation_rate_tight'))} | "
            f"{_cell(cells.get('velocity_violation_rate'))} | "
            f"{_cell(cells.get('velocity_violation_rate_run_class'))} |"
        )
    return [demath_typographic(row) for row in rows]


def render_ground_rule_table(payload: Dict[str, Any]) -> List[str]:
    """What each contact rule leaves on the grounded frames, model against the recording."""
    rows: List[str] = []
    for label, cells in payload["summary"].items():
        if not label.endswith(("/published", "/increment")):
            continue
        rows.append(
            f"| {label} | {_cell(cells.get('ground_median_abs_vy_model'), '{:.3f}')} | "
            f"{_cell(cells.get('ground_median_abs_step_model'), '{:.3f}')} | "
            f"{_cell(cells.get('ground_rate_vy_exactly_zero_model'), '{:.2%}')} | "
            f"{_cell(cells.get('test_loss_data'))} | "
            f"{_cell(cells.get('drift_multistart_mean_px'), '{:.2f}')} |"
        )
    return [demath_typographic(row) for row in rows]


def render_replay_table(payload: Dict[str, Any]) -> List[str]:
    """Identical weights, both conventions - the same contrast with the training noise removed."""
    rows: List[str] = []
    for label, block in payload["same_weights_replay"].items():
        trained = payload["summary"].get(label, {})
        as_trained = trained.get("drift_multistart_mean_px", {}).get("mean")
        drift = block["drift_multistart_mean_px"]
        rows.append(
            f"| {label} | {block['as_trained']} | {block['replayed_as']} | "
            f"{'-' if as_trained is None else format(as_trained, '.2f')} | "
            f"{drift:.2f} | "
            f"{'-' if as_trained is None else format(drift - as_trained, '+.2f')} |"
        )
    return [demath_typographic(row) for row in rows]


def render_ruler_table(payload: Dict[str, Any]) -> List[str]:
    """| arm | violation rate at 72.0 | at 48.0 | the difference |, per arm."""
    rows: List[str] = []
    for label, block in payload["ruler_within_arm"].items():
        rows.append(
            f"| {label} | {block['at_published_bound']:.4f} | "
            f"{block['at_run_class_bound']:.4f} | {block['difference']:+.4f} |"
        )
    return [demath_typographic(row) for row in rows]


def main() -> Dict[str, Any]:
    """CLI entry: every flag is also a key in ``configs/corrected_physics_ablation.yaml``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--arms", default=None, help="comma-separated arm-label filter")
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="publish the eight flown arms' weights under results/checkpoints (off by default)",
    )
    args = parse_args_with_config(parser)
    payload = run_corrected_physics_ablation(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        seeds=[int(s) for s in str(args.seeds).split(",") if s.strip()],
        output_dir=args.output_dir or RESULTS_DIR,
        patience=args.patience,
        rollout_horizon=args.horizon,
        num_rollout_starts=args.num_starts,
        latent_dim=args.latent_dim,
        only_arms=None if not args.arms else str(args.arms).split(","),
        save_checkpoints=args.save_checkpoints,
    )
    logger.info("verdict written: %d arms", payload["verdict"]["arms"])
    return payload


if __name__ == "__main__":
    main()
