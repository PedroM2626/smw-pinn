"""
analytical_baselines.py
Two missing reference points for the central claims of this repository.

Baseline A - engine rules, no network (always runnable, CI-safe)
    `AnalyticalKinematicsDynamics` implements README section 4 as closed-form
    arithmetic with five identified scalars. It answers: how much of the Hard
    PINN's advantage is the structural physics prior (which this baseline also
    has) versus the learned residual head (which it does not)?

Baseline B - oracle-model MPC upper bound (requires the emulator)
    Same CEM planner, same objective, same savestate - only the world model
    differs: the learned Hard PINN versus the closed-form engine rules. It
    answers: how much of the published control performance is *model error*
    rather than planner/objective error?

Run:
    python -m src.evaluation.analytical_baselines
    python -m src.evaluation.analytical_baselines --no-hardware --frames 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import numpy as np
import torch

from src.environment.dataset_loader import load_and_preprocess_data
from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models.analytical_kinematics import (
    AnalyticalKinematicsDynamics,
    EngineRuleParameters,
)
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.planning.mpc_planner import (
    ACTION_PRIMITIVES,
    ModelPredictiveController,
    TrajectoryObjective,
    action_vector_to_joypad,
)
from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_GAMEPLAY,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    checkpoint_file,
    hardware_available,
    results_file,
)
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PUBLISHED_BENCHMARK = results_file("benchmark_metrics.json")
HARD_PINN_CKPT = checkpoint_file("pinn_hard_best.pt")
STATE_KEYS_8D = ["x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right"]


def _to_tensors(split: Dict[str, np.ndarray], keys: str, device: torch.device):
    return (
        torch.tensor(split[f"{keys}_states"], dtype=torch.float32).to(device),
        torch.tensor(split[f"{keys}_actions"], dtype=torch.float32).to(device),
        torch.tensor(split[f"{keys}_next_states"], dtype=torch.float32).to(device),
    )


def evaluate_engine_rules(
    dataset_path: str = DATASET_GAMEPLAY,
    seed: int = 42,
    sweeps: int = 3,
    rollout_horizon: int = 120,
    num_rollout_starts: int = 10,
) -> Dict[str, Any]:
    """Fit the five engine-rule scalars and score the model like any other here."""
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_preprocess_data(dataset_path, seed=seed)
    train = _to_tensors(data, "train", device)
    test = _to_tensors(data, "test", device)
    test_states, test_actions, test_next = test

    model = AnalyticalKinematicsDynamics().to(device)
    prior_guess = EngineRuleParameters.from_dict(model.params.as_dict())  # snapshot before fitting
    trace = model.fit_engine_rules(*train, sweeps=sweeps, device=device)

    with torch.no_grad():
        pred = model(test_states, test_actions)
    test_mse = float(((pred - test_next) ** 2).mean().item())
    velocity_mse = float(((pred[:, 2:4] - test_next[:, 2:4]) ** 2).mean().item())
    position_mse = float(((pred[:, :2] - test_next[:, :2]) ** 2).mean().item())
    kin_residual = float(
        ((pred[:, 0] - test_states[:, 0]) - pred[:, 2] / model.subpixels_per_pixel)
        .pow(2)
        .mean()
        .item()
    )

    evaluator = RolloutEvaluator(device)
    multistart = evaluator.evaluate_rollout_multistart(
        model=model,
        model_type="analytical",
        states=data["test_states"],
        actions=data["test_actions"],
        next_states=data["test_next_states"],
        horizon=rollout_horizon,
        num_starts=num_rollout_starts,
    )

    per_variable = compute_per_variable_metrics(pred.cpu().numpy(), test_next.cpu().numpy())

    payload: Dict[str, Any] = {
        "engine_rule_parameters_identified": model.params.as_dict(),
        "engine_rule_parameters_prior_guess": prior_guess.as_dict(),
        "identification_trace": trace,
        "single_step": {
            "test_mse_all_channels": test_mse,
            "test_mse_velocity_only": velocity_mse,
            "test_mse_position_only": position_mse,
            "kinematic_residual": kin_residual,
            "learned_parameters": model.num_parameters,
        },
        "per_variable_metrics": per_variable,
        "vertical_dynamics_diagnostics": vertical_diagnostics(
            test_states.cpu().numpy(), test_actions.cpu().numpy(), test_next.cpu().numpy()
        ),
        "rollout_multistart": {
            k: v for k, v in multistart.items() if not isinstance(v, list) or k in ("starts",)
        },
        "notes": [
            "Contact channels are propagated by persistence: they are not derivable "
            "from kinematics and are exactly what a learned residual head must explain.",
            "The P-meter sprint tier is unobservable in the 8D state, so the run tier "
            "caps commanded acceleration while existing momentum is preserved.",
            "`jump_run_gain` is the minimal linear reading of section 4.2.1's \"modulated by "
            'prior horizontal running momentum"; observed takeoff magnitudes exceed the '
            "documented [-64, -80] window, which is reported rather than fitted away.",
        ],
    }
    if os.path.exists(PUBLISHED_BENCHMARK):
        published = read_metrics(PUBLISHED_BENCHMARK)["single_step_results"]
        payload["comparison_against_published"] = {
            name: published[name]["test_loss_data"]
            for name in ("Statistical_MLP", "Statistical_LSTM", "Soft_PINN", "Hard_Residual_PINN")
            if name in published
        }
    logger.info(
        "Engine rules: velocity MSE %.4f | all-channel MSE %.4f | kinematic residual %.6f",
        velocity_mse,
        test_mse,
        kin_residual,
    )
    return payload


def vertical_diagnostics(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> Dict[str, Any]:
    """Evidence for *why* the closed-form vertical rule cannot reach the residual net.

    The dataset records velocity after the engine has applied its jump machinery, so
    a strongly negative vy is already present in s_t: an impulse is therefore not a
    single-frame event, and `(s_t, a_t)` alone does not say whether one is being
    applied. These counts make that claim checkable instead of rhetorical.
    """
    jump_held = actions[:, 0] > 0.5
    grounded = states[:, 4] > 0.5
    vy, vy_next = states[:, 3], next_states[:, 3]
    impulse_frames = vy_next <= -50.0
    ascending_held = (~grounded) & jump_held & (vy < 0.0)
    dvy = vy_next - vy
    return {
        "frames": int(len(states)),
        "impulse_frames": int(impulse_frames.sum()),
        "impulse_frames_already_rising_share": (
            float((vy[impulse_frames] < 0.0).mean()) if impulse_frames.any() else None
        ),
        "impulse_frames_with_jump_held_share": (
            float(jump_held[impulse_frames].mean()) if impulse_frames.any() else None
        ),
        "observed_vy_next_min": float(vy_next.min()),
        "documented_takeoff_window": [-80.0, -64.0],
        "held_ascent_mean_dvy": (
            float(dvy[ascending_held].mean()) if ascending_held.any() else None
        ),
        "held_ascent_dvy_std": (float(dvy[ascending_held].std()) if ascending_held.any() else None),
        "reading": (
            "Held-ascent dvy is deterministic at +3 (g_held, section 4.2.3) and is reproduced "
            "exactly; the loss concentrates in the impulse frames, which are not identifiable "
            "from a single transition because the recorded velocity is already post-impulse."
        ),
    }


def score_like_dynamics_trainer(
    model: torch.nn.Module, dataloader, device: torch.device
) -> Dict[str, float]:
    """Score a model with exactly `DynamicsTrainer.evaluate`'s math.

    `DynamicsTrainer` cannot be reused directly: its constructor builds an
    optimizer, and a parameter-free model hands it an empty list. The numbers
    below are therefore still comparable to the MLP/PINN rows of
    `benchmark_metrics.json` (same SmoothL1 data loss, same kinematic residual).
    """
    loss_fn = torch.nn.SmoothL1Loss()
    model.eval()
    data_loss_sum = kin_sum = 0.0
    batches = 0
    with torch.no_grad():
        for curr_state, curr_action, target_next in dataloader:
            curr_state = curr_state.to(device)
            pred = model(curr_state, curr_action.to(device))
            data_loss_sum += float(loss_fn(pred, target_next.to(device)).item())
            dx_pred = pred[:, 0] - curr_state[:, 0]
            kin_sum += float(((dx_pred - curr_state[:, 2] / 16.0) ** 2).mean().item())
            batches += 1
    n = max(1, batches)
    return {
        "val_loss_total": data_loss_sum / n,
        "val_loss_data": data_loss_sum / n,
        "val_loss_kinematics": kin_sum / n,
    }


def _extract_8d(state_dict: Dict[str, float]) -> np.ndarray:
    return np.array([state_dict[k] for k in STATE_KEYS_8D], dtype=np.float32)


def run_oracle_mpc_comparison(
    frames: int = 300,
    horizon: int = 15,
    num_candidates: int = 256,
    cem_iterations: int = 3,
    seed: int = 42,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    pinn_ckpt: str = HARD_PINN_CKPT,
) -> Dict[str, Any]:
    """Closed-loop control on real hardware: learned model vs. exact engine rules.

    Both controllers share the savestate, objective, horizon and candidate budget;
    the only difference is the dynamics they plan with, so the gap isolates model
    error. Action agreement is reported per frame as a direct diagnostic.
    """
    from src.environment.snes_emulator import SnesLibretroEmulator

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(pinn_ckpt):
        raise FileNotFoundError(
            f"{pinn_ckpt} missing: run `python -m src.training.benchmark_experiment` first."
        )

    hard_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    hard_pinn.load_state_dict(torch.load(pinn_ckpt, map_location=device, weights_only=True))
    hard_pinn.eval()
    analytical = AnalyticalKinematicsDynamics().to(device)
    data = load_and_preprocess_data(seed=seed)
    analytical.fit_engine_rules(*_to_tensors(data, "train", device), device=device)

    # Weights, horizon and CEM budget copied verbatim from the published
    # closed-loop protocol (src/evaluation/mbrl_mpc_benchmark.py, README 10.6),
    # so the learned row is a reproduction check and the two rows are directly
    # comparable with every other MPC entry in the master table (10.27).
    objective = lambda: TrajectoryObjective(  # noqa: E731 - identical budget per controller
        weight_progress=2.0, weight_velocity=0.5, pit_penalty=1000.0, death_y=450.0
    )
    controllers = {
        "MPC + Hard Residual PINN (learned)": ModelPredictiveController(
            world_model=hard_pinn,
            device=device,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=cem_iterations,
            objective=objective(),
        ),
        "MPC + Analytical Engine Rules (oracle-like)": ModelPredictiveController(
            world_model=analytical,
            device=device,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=cem_iterations,
            objective=objective(),
        ),
    }

    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()

    emu = SnesLibretroEmulator()
    emu.load_rom(rom_path)
    outcomes: Dict[str, Any] = {}
    try:
        for name, controller in controllers.items():
            set_global_seed(seed)  # identical CEM sampling for both controllers
            start = emu.start_episode(initial_savestate)
            x0 = start["x"]
            survived = 0
            t0 = time.time()
            chosen: list[str] = []
            snapshot = start
            state = _extract_8d(snapshot)
            for _ in range(frames):
                action, _info = controller.plan(state)
                primitive = _nearest_primitive(action)
                chosen.append(primitive)
                emu.set_input(action_vector_to_joypad(action))
                emu.step_frame()
                snapshot = emu.get_smw_state()
                state = _extract_8d(snapshot)
                survived += 1
                if snapshot["y"] > 450.0 or snapshot["y"] < 0.0:
                    break
            elapsed = max(time.time() - t0, 1e-9)
            fell = snapshot["y"] > 450.0 or snapshot["y"] < 0.0
            outcomes[name] = {
                "progress_px": round(float(snapshot["x"] - x0), 2),
                "frames_survived": survived,
                "control_fps": round(survived / elapsed, 2),
                "termination": "pit/death" if fell else "timeout",
                "action_sequence": "".join(_primitive_letter(p) for p in chosen),
            }
            logger.info("%s -> %.2f px in %d frames", name, snapshot["x"] - x0, survived)
    finally:
        emu.close()

    agreement = _sequence_agreement(outcomes)
    return {
        "configuration": {
            "frames_budget": frames,
            "horizon": horizon,
            "num_candidates": num_candidates,
            "cem_iterations": cem_iterations,
            "objective_weights": {"progress": 2.0, "velocity": 0.5, "pit_penalty": 1000.0},
            "protocol_source": "mbrl_mpc_benchmark.py (README 10.6)",
        },
        "per_controller": outcomes,
        "first_action_agreement": agreement["first"],
        "full_sequence_agreement": agreement["overall"],
        "interpretation": (
            "The gap between the two rows is attributable to world-model error only: "
            "planner, objective, savestate and CEM sampling are identical."
        ),
    }


_PRIMITIVE_LETTERS = {
    "NOOP": ".",
    "WALK_RIGHT": "w",
    "RUN_RIGHT": "r",
    "WALK_JUMP_RIGHT": "j",
    "RUN_JUMP_RIGHT": "R",
    "JUMP_UP": "u",
    "WALK_LEFT": "l",
    "RUN_LEFT": "L",
}


def _primitive_letter(name: str) -> str:
    return _PRIMITIVE_LETTERS.get(name, "?")


def _nearest_primitive(action: np.ndarray) -> str:
    """Map a planner action vector back to its discrete primitive name."""
    diffs = {
        name: float(
            np.abs(np.array(vec, dtype=np.float32) - np.asarray(action, dtype=np.float32)).sum()
        )
        for name, vec in ACTION_PRIMITIVES.items()
    }
    return min(diffs, key=lambda name: diffs[name])


def _sequence_agreement(outcomes: Dict[str, Any]) -> Dict[str, float]:
    sequences = [info["action_sequence"] for info in outcomes.values()]
    if len(sequences) < 2:
        return {"first": None, "overall": None}
    reference, other = sequences[0], sequences[1]
    shared = min(len(reference), len(other))
    overall = sum(a == b for a, b in zip(reference, other)) / shared if shared else 0.0
    first = 1.0 if shared and reference[0] == other[0] else 0.0
    return {"first": first, "overall": round(overall, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine-rule and oracle-MPC reference baselines.")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=300, help="hardware budget per controller")
    parser.add_argument(
        "--hardware",
        dest="hardware",
        action="store_true",
        default=None,
        help="force the closed-loop oracle comparison",
    )
    parser.add_argument(
        "--no-hardware",
        dest="hardware",
        action="store_false",
        help="skip the closed-loop comparison (CI mode)",
    )
    args = parser.parse_args()

    payload = evaluate_engine_rules(dataset_path=args.dataset_path, seed=args.seed)
    written = write_metrics(
        results_file("analytical_baseline_metrics.json"),
        payload,
        seed=args.seed,
        command="python -m src.evaluation.analytical_baselines",
    )
    logger.info("wrote %s", written)

    wants_hardware = args.hardware
    if wants_hardware is None:
        wants_hardware = hardware_available()
    if wants_hardware:
        oracle = run_oracle_mpc_comparison(frames=args.frames, seed=args.seed)
        logger.info(
            "wrote %s",
            write_metrics(
                results_file("oracle_mpc_metrics.json"),
                oracle,
                seed=args.seed,
                command="python -m src.evaluation.analytical_baselines (oracle MPC)",
            ),
        )
    else:
        logger.info("oracle-model MPC comparison skipped (no Libretro core/ROM available).")
    print(json.dumps(payload["single_step"], indent=2))


if __name__ == "__main__":
    main()
