"""
inverse_model_mpc_benchmark.py
Do the recovered physics models actually control the console? (README Section 10.44)

Sections 10.40 and 10.43 both end on predictive numbers: the first recovers seven engine
constants to sub-percent accuracy and scores a synthetic held-out control battery, the
second discovers the update laws and inherits the opposite bias. Neither has ever closed a
loop on the hardware, which is where this repository's headline metrics live. This study
does exactly that: the same sampling MPC, the same objective, the same savestate, the same
candidate budget and the same CEM seed, driven by four different world models:

* **Established WRAM physics** - the closed-form engine rules of Section 10.37, whose six
  scalars are grid-searched on the recorded telemetry. This is the "known physics" reference.
* **Parametrically identified constants (10.40)** - the same structure, but with the seven
  constants *recovered* from multi-step rollouts by Adam, i.e. the inverse answer instead of
  the hand-measured one.
* **Symbolically discovered laws (10.43)** - the genetic-programming laws, with no posited
  structure at all, evaluated through the identical planner.
* **Hard Residual PINN (published)** - the learned world model of Section 8, reloaded from
  its committed checkpoint, so the row is a reproduction check against Section 10.37.

Everything else is shared, so a difference in progress is attributable to the dynamics model
alone. Action agreement against the established-physics controller is reported per frame: two
models can disagree and still drive equally well, and that disagreement is the operational
meaning of "the physics differs".

Requires the Libretro core and a ROM dump (see README 11.2); without them the study is
skipped with a diagnostic rather than writing fabricated numbers. Writes
``results/inverse_model_mpc_metrics.json`` with ``_meta``.

Run:  python -m src.evaluation.inverse_model_mpc_benchmark
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.environment.dataset_loader import load_and_preprocess_data
from src.evaluation.analytical_baselines import (
    _extract_8d,
    _nearest_primitive,
    _primitive_letter,
    _to_tensors,
)
from src.evaluation.inverse_transfer_benchmark import PRIOR
from src.inverse.parameter_identification import identify_params, make_windows, theta_tensor
from src.inverse.symbolic_regression import (
    bank_from_transitions,
    fit_law_bank,
)
from src.models.analytical_kinematics import AnalyticalKinematicsDynamics
from src.models.inverse_world_models import (
    IdentifiedKinematicsDynamics,
    SymbolicKinematicsDynamics,
)
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import (
    PINN_HARD_CKPT,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    hardware_present,
)
from src.utils.provenance import write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

ARTIFACT_NAME = "inverse_model_mpc_metrics.json"
FIGURE_NAME = "inverse_model_mpc_progress.png"
# The published closed-loop budget (src/evaluation/mbrl_mpc_benchmark.py, README 10.6),
# copied so these rows are comparable with the master table of Section 10.27.
OBJECTIVE_WEIGHTS = {
    "weight_progress": 2.0,
    "weight_velocity": 0.5,
    "pit_penalty": 1000.0,
    "death_y": 450.0,
}


def _agreement(outcomes: Dict[str, Any], reference: str) -> Dict[str, float]:
    """Per-controller agreement with the established-physics controller."""
    base = outcomes[reference]["action_sequence"]
    out: Dict[str, float] = {}
    for name, info in outcomes.items():
        seq = info["action_sequence"]
        shared = min(len(base), len(seq))
        out[name] = {
            "first_action_agreement": 1.0 if shared and base[0] == seq[0] else 0.0,
            "sequence_agreement": round(sum(a == b for a, b in zip(base, seq)) / shared, 4)
            if shared
            else 0.0,
        }
    return out


def build_world_models(
    device: torch.device,
    seed: int,
    gp_seeds: Sequence[int],
    budget: Dict[str, Any],
    id_steps: int,
    pinn_ckpt: str,
) -> Dict[str, Any]:
    """Fit/reload the four candidate world models on the canonical training split."""
    data = load_and_preprocess_data(seed=seed)
    models: Dict[str, Any] = {}

    analytical = AnalyticalKinematicsDynamics().to(device)
    analytical.fit_engine_rules(*_to_tensors(data, "train", device), device=device)
    models["established_wram_engine_rules"] = analytical

    windows = make_windows(
        data["train_states"],
        data["train_actions"],
        data["train_next_states"],
        data["train_episodes"],
        rollout_len=8,
    )
    theta_hat, _ = identify_params(windows, theta_tensor(PRIOR), steps=id_steps, lr=0.05, seed=seed)
    models["parametrically_identified_10_40"] = IdentifiedKinematicsDynamics(theta_hat.numpy()).to(
        device
    )

    bank = bank_from_transitions(
        data["train_states"], data["train_actions"], data["train_next_states"]
    )
    laws = fit_law_bank(bank, "real", seeds=tuple(gp_seeds), **budget)
    models["symbolically_discovered_10_43"] = SymbolicKinematicsDynamics(laws.bagged()).to(device)

    if os.path.isfile(pinn_ckpt):
        pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
        pinn.load_state_dict(torch.load(pinn_ckpt, map_location=device, weights_only=True))
        pinn.eval()
        models["learned_hard_residual_pinn"] = pinn
    else:
        logger.warning("%s missing; the learned reference row is omitted.", pinn_ckpt)
    diagnostics = {
        "identified_theta": {
            k: float(v)
            for k, v in zip(
                (
                    "max_vx",
                    "walk_accel",
                    "run_accel",
                    "subpixels_per_pixel",
                    "held_gravity",
                    "fall_gravity",
                    "decel",
                ),
                theta_hat.tolist(),
            )
        },
        # ``expressions()`` is the first GP seed; the controller that is actually driven
        # below averages every seed, so both are recorded.
        "discovered_expressions": laws.expressions(),
        "discovered_expressions_per_seed": {
            name: [law.expression for law in members] for name, members in laws.per_seed.items()
        },
        "discovered_nodes_mean": laws.mean_nodes(),
        "discovered_gp_seconds": laws.total_seconds(),
    }
    return {"models": models, "diagnostics": diagnostics}


def _render_figure(outcomes: Dict[str, Any], agreement: Dict[str, Any], path: str) -> str:
    """Progress and divergence-from-established-physics, one panel each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DISPLAY = {
        "established_wram_engine_rules": "Established WRAM\nengine rules (10.37)",
        "parametrically_identified_10_40": "Identified\nconstants (10.40)",
        "symbolically_discovered_10_43": "Discovered\nlaws (10.43)",
        "learned_hard_residual_pinn": "Hard Residual\nPINN (published)",
    }
    names = list(outcomes)
    progress = [float(outcomes[n]["progress_px"]) for n in names]
    shared = [float(agreement[n]["sequence_agreement"]) for n in names]
    fell = [outcomes[n]["termination"] != "timeout" for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    y = np.arange(len(names))[::-1]
    colors = ["#d62728" if f else "#1f77b4" for f in fell]
    ax1.barh(y, progress, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels([DISPLAY.get(n, n) for n in names], fontsize=8)
    ax1.set_xlabel("closed-loop distance (px, 300-frame budget)")
    ax1.set_title("Progress on the real console", fontweight="bold")
    ax1.set_xlim(0, max(progress) * 1.28)
    for yi, value, flag in zip(y, progress, fell):
        ax1.text(
            value + 8,
            yi,
            f"{value:.2f}" + ("  (pit/death)" if flag else ""),
            va="center",
            fontsize=8,
        )
    ax2.barh(y, shared, color="#2ca02c")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("frames choosing the same action as the established-physics planner")
    ax2.set_title("Divergence of the control program", fontweight="bold")
    for yi, value in zip(y, shared):
        ax2.text(value + 0.02, yi, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle(
        "Section 10.44 - the same CEM-MPC planner driven by four inverse world models",
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)
    return path


def run_closed_loop(
    frames: int = 300,
    horizon: int = 15,
    num_candidates: int = 256,
    cem_iterations: int = 3,
    seed: int = 42,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    pinn_ckpt: str = PINN_HARD_CKPT,
    gp_seeds: Sequence[int] = (0, 1, 2),
    id_steps: int = 900,
    budget: Optional[Dict[str, Any]] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Drive the console once per world model with an identical planner and budget."""
    if not hardware_present():
        raise RuntimeError(
            "The inverse-model MPC comparison needs the Libretro core and a ROM dump "
            f"(core: {ROM_PATH}, rom: {ROM_PATH}). See README section 11.2."
        )
    from src.environment.snes_emulator import SnesLibretroEmulator
    from src.planning.mpc_planner import action_vector_to_joypad

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = build_world_models(
        device,
        seed,
        gp_seeds,
        budget or {"population_size": 500, "generations": 25, "max_train": 4000},
        id_steps,
        pinn_ckpt,
    )
    models: Dict[str, Any] = built["models"]

    def make_controller(model: Any) -> ModelPredictiveController:
        return ModelPredictiveController(
            world_model=model,
            device=device,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=cem_iterations,
            objective=TrajectoryObjective(**OBJECTIVE_WEIGHTS),
        )

    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()

    emu = SnesLibretroEmulator()
    emu.load_rom(rom_path)
    outcomes: Dict[str, Any] = {}
    try:
        for name, model in models.items():
            set_global_seed(seed)  # identical CEM sampling for every controller
            controller = make_controller(model)
            start = emu.start_episode(initial_savestate)
            x0 = start["x"]
            survived = 0
            t0 = time.time()
            chosen: List[str] = []
            snapshot = start
            state = _extract_8d(snapshot)
            for _ in range(frames):
                action, _info = controller.plan(state)
                chosen.append(_primitive_letter(_nearest_primitive(action)))
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
                "action_sequence": "".join(chosen),
            }
            logger.info(
                "  %-32s -> %.2f px in %d frames (%.1f FPS)",
                name,
                outcomes[name]["progress_px"],
                survived,
                outcomes[name]["control_fps"],
            )
    finally:
        emu.close()

    reference = "established_wram_engine_rules"
    payload: Dict[str, Any] = {
        "study": (
            "Closed-loop control on the real console with world models obtained from the "
            "inverse problem: established WRAM engine rules, the parametrically identified "
            "constants of 10.40, the symbolically discovered laws of 10.43, and the published "
            "Hard Residual PINN - all driven by the same CEM-MPC planner, objective, budget "
            "and savestate."
        ),
        "configuration": {
            "frames_budget": frames,
            "horizon": horizon,
            "num_candidates": num_candidates,
            "cem_iterations": cem_iterations,
            "objective_weights": OBJECTIVE_WEIGHTS,
            "protocol_source": "mbrl_mpc_benchmark.py (README 10.6) / analytical_baselines.py (10.37)",
            "seed": seed,
            "parametric_id_steps": id_steps,
            "gp_seeds": list(gp_seeds),
        },
        "per_controller": outcomes,
        "agreement_with_established_physics": _agreement(outcomes, reference),
        "model_diagnostics": built["diagnostics"],
        "interpretation": (
            "Planner, objective, savestate, CEM seed and frame budget are identical across "
            "rows, so the progress gap is attributable to the dynamics model alone. The "
            "action-agreement column is the operational reading of 'the physics differs': a "
            "model can be accurate on recorded transitions and still refuse to command the "
            "console the way the engine rules do."
        ),
    }
    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.inverse_model_mpc_benchmark",
        extra_meta={"controllers": list(outcomes)},
    )
    logger.info("Artifact written to: %s", artifact)
    figure = _render_figure(
        outcomes,
        payload["agreement_with_established_physics"],
        os.path.join(output_dir, "figures", FIGURE_NAME),
    )
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-candidates", dest="num_candidates", type=int, default=256)
    parser.add_argument("--cem-iterations", dest="cem_iterations", type=int, default=3)
    parser.add_argument("--gp-seeds", dest="gp_seeds", default="0,1,2")
    parser.add_argument("--population-size", dest="population_size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--max-train", dest="max_train", type=int, default=4000)
    parser.add_argument("--id-steps", dest="id_steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    if not hardware_present():
        logger.error(
            "Libretro core or ROM not found - the closed-loop inverse comparison cannot run. "
            "See README section 11.2 (git lfs pull, or set SMW_ROM / SMW_CORE)."
        )
        return 1
    seeds = tuple(int(s) for s in str(args.gp_seeds).split(",") if s.strip())
    run_closed_loop(
        frames=args.frames,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        seed=args.seed,
        gp_seeds=seeds,
        id_steps=args.id_steps,
        budget={
            "population_size": args.population_size,
            "generations": args.generations,
            "max_train": args.max_train,
            "parsimony_coefficient": 1e-3,
            "n_jobs": 1,
        },
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
