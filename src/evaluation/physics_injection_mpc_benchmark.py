"""
physics_injection_mpc_benchmark.py
Driving the 10.47 grid cells on the real console (README 10.47).

Section 10.44 established that no prediction metric orders closed-loop control:
the FNO is the most accurate physics-free single-step model in the repository and
loses 247 px to the hand reverse-engineered rules with $d_z = -12.14$, while the
Physics-Constrained DeepONet beats them. It also found that two models sharing
the same hard shell - the Hard Residual PINN and the PC-DeepONet - differ by more
than 300 px, which left the attribution open: is a good controller the shell, the
increment target, or the quality of the net inside them? Section 10.47 built the
missing cells, and this study flies them with the published planner so the
question is answered where it was posed.

Rows: the established WRAM engine rules (reference), the two published shell
models of 10.27/10.42, and the grid arms that carry a checkpoint - the hard FNO,
the same FNO without the shell, the soft-trained FNO, the DeepONet predicting
increments without any physics, the DeepONet trained with the soft composite
penalty, and the MLP predicting increments with neither.

Writes ``results/physics_injection_mpc_metrics.json``; it needs the Libretro core
and ROM of Section 11.2 and writes nothing without hardware.
"""

import argparse
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.evaluation.analytical_baselines import (
    _extract_8d,
    _nearest_primitive,
    _primitive_letter,
    _to_tensors,
)
from src.evaluation.inverse_model_mpc_benchmark import (
    OBJECTIVE_WEIGHTS,
    _agreement,
    _multi_seed_summary,
)
from src.evaluation.operator_physics_injection_benchmark import (
    MAX_VX,
    MIN_VY,
    SUBPIXELS_PER_PIXEL,
    TERMINAL_VY,
    arm_label,
    checkpoint_name,
)
from src.models.analytical_kinematics import AnalyticalKinematicsDynamics
from src.models.deeponet import DeepONetDynamics, PhysicsConstrainedDeepONetDynamics
from src.models.fno import FNODynamics
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.residual_dynamics import ResidualDynamics
from src.models.statistical_mlp import StatisticalMLPDynamics
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import PINN_HARD_CKPT, ROM_PATH, STATE_YOSHI_ISLAND_1, hardware_present
from src.utils.provenance import write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

ARTIFACT_NAME = "physics_injection_mpc_metrics.json"
FIGURE_NAME = "physics_injection_mpc_progress.png"
CHECKPOINT_DIR = os.path.join("results", "checkpoints")

STATE_DIM = 8
ACTION_DIM = 6
AUX_DIM = STATE_DIM - 2
SENSOR_DIM = STATE_DIM + ACTION_DIM


def _residual_fno(hard: bool) -> nn.Module:
    return ResidualDynamics(
        base=FNODynamics(
            state_dim=AUX_DIM,
            action_dim=ACTION_DIM,
            width=32,
            modes=6,
            n_layers=2,
            sensor_dim=SENSOR_DIM,
        ),
        state_dim=STATE_DIM,
        hard=hard,
        max_vx=MAX_VX,
        terminal_vy=TERMINAL_VY,
        min_vy=MIN_VY,
        subpixels_per_pixel=SUBPIXELS_PER_PIXEL,
    )


def _residual_deeponet() -> nn.Module:
    return ResidualDynamics(
        base=DeepONetDynamics(
            state_dim=AUX_DIM,
            action_dim=ACTION_DIM,
            latent_dim=64,
            sensor_dim=SENSOR_DIM,
        ),
        state_dim=STATE_DIM,
        hard=False,
    )


def _residual_mlp() -> nn.Module:
    return ResidualDynamics(
        base=StatisticalMLPDynamics(
            state_dim=AUX_DIM, action_dim=ACTION_DIM, sensor_dim=SENSOR_DIM
        ),
        state_dim=STATE_DIM,
        hard=False,
    )


# label -> (checkpoint file published by 10.47, factory with the published hyperparameters)
GRID_CHECKPOINTS: Dict[str, Tuple[str, Callable[[], nn.Module]]] = {
    "fno_residual_hard": (
        checkpoint_name(arm_label("FNO", "residual", "hard")),
        lambda: _residual_fno(hard=True),
    ),
    "fno_residual_none": (
        checkpoint_name(arm_label("FNO", "residual", "none")),
        lambda: _residual_fno(hard=False),
    ),
    "fno_state_soft": (
        checkpoint_name(arm_label("FNO", "state", "soft")),
        lambda: FNODynamics(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, width=32, modes=6, n_layers=2
        ),
    ),
    "deeponet_residual_none": (
        checkpoint_name(arm_label("DeepONet", "residual", "none")),
        _residual_deeponet,
    ),
    "deeponet_state_soft": (
        checkpoint_name(arm_label("DeepONet", "state", "soft")),
        lambda: DeepONetDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=64),
    ),
    "mlp_residual_none": (
        checkpoint_name(arm_label("MLP", "residual", "none")),
        _residual_mlp,
    ),
}

PUBLISHED_CHECKPOINTS: Dict[str, Tuple[str, Callable[[], nn.Module]]] = {
    "published_pc_deeponet_10_42": (
        "operator_physicsconstrained_deeponet_best.pt",
        lambda: PhysicsConstrainedDeepONetDynamics(state_dim=8, action_dim=6, latent_dim=64),
    ),
    "published_hard_pinn_10_27": (
        os.path.basename(PINN_HARD_CKPT),
        lambda: HardResidualPINNDynamics(state_dim=8, action_dim=6),
    ),
}

REFERENCE = "established_wram_engine_rules"


def _load(
    checkpoint: str, factory: Callable[[], nn.Module], device: torch.device
) -> Optional[nn.Module]:
    path = checkpoint if os.path.isabs(checkpoint) else os.path.join(CHECKPOINT_DIR, checkpoint)
    if not os.path.isfile(path):
        logger.warning("%s missing; the row is omitted.", path)
        return None
    model = factory().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def build_contenders(device: torch.device, seed: int) -> Dict[str, nn.Module]:
    """Fit the reference rules and reload every grid/published arm that exists."""
    from src.environment.dataset_loader import load_and_preprocess_data

    data = load_and_preprocess_data(seed=seed)
    models: Dict[str, nn.Module] = {}

    analytical = AnalyticalKinematicsDynamics().to(device)
    analytical.fit_engine_rules(*_to_tensors(data, "train", device), device=device)
    models[REFERENCE] = analytical

    registry: Dict[str, Tuple[str, Callable[[], nn.Module]]] = {
        **PUBLISHED_CHECKPOINTS,
        **GRID_CHECKPOINTS,
    }
    for label, (checkpoint, factory) in registry.items():
        model = _load(checkpoint, factory, device)
        if model is not None:
            models[label] = model
    return models


def run_physics_injection_closed_loop(
    frames: int = 300,
    horizon: int = 15,
    num_candidates: int = 256,
    cem_iterations: int = 3,
    seed: int = 42,
    seeds: Optional[Sequence[int]] = None,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Drive every grid arm that has a checkpoint with the published planner."""
    if not hardware_present():
        raise RuntimeError(
            "The physics-injection closed loop needs the Libretro core and a ROM dump "
            f"(core: {ROM_PATH}, rom: {ROM_PATH}). See README section 11.2."
        )
    from src.environment.snes_emulator import SnesLibretroEmulator
    from src.planning.mpc_planner import action_vector_to_joypad

    draws = list(seeds) if seeds else [seed]
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_contenders(device, seed)
    logger.info("=== Physics-injection closed loop: %d controllers ===", len(models))

    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()

    emu = SnesLibretroEmulator()
    emu.load_rom(rom_path)
    runs: Dict[int, Dict[str, Any]] = {}
    try:
        for draw in draws:
            runs[draw] = {}
            for name, model in models.items():
                set_global_seed(draw)
                controller = ModelPredictiveController(
                    world_model=model,
                    device=device,
                    horizon=horizon,
                    num_candidates=num_candidates,
                    cem_iterations=cem_iterations,
                    objective=TrajectoryObjective(**OBJECTIVE_WEIGHTS),
                )
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
                runs[draw][name] = {
                    "progress_px": round(float(snapshot["x"] - x0), 2),
                    "frames_survived": survived,
                    "control_fps": round(survived / elapsed, 2),
                    "termination": "pit/death" if fell else "timeout",
                    "action_sequence": "".join(chosen),
                }
                logger.info(
                    "  seed %d | %-30s -> %.2f px in %d frames (%.1f FPS)",
                    draw,
                    name,
                    runs[draw][name]["progress_px"],
                    survived,
                    runs[draw][name]["control_fps"],
                )
    finally:
        emu.close()

    outcomes = runs[draws[0]]
    payload: Dict[str, Any] = {
        "study": (
            "Closed-loop control of the physics-injection grid of 10.47 on the real console, "
            "with the published planner, objective, savestate and frame budget of 10.44, so the "
            "shell-versus-parameterisation question left open by 10.44.1 is answered where it "
            "was posed."
        ),
        "configuration": {
            "frames_budget": frames,
            "horizon": horizon,
            "num_candidates": num_candidates,
            "cem_iterations": cem_iterations,
            "objective_weights": OBJECTIVE_WEIGHTS,
            "protocol_source": "inverse_model_mpc_benchmark.py (README 10.44)",
            "seed": seed,
            "cem_seeds": draws,
        },
        "per_controller": outcomes,
        "agreement_with_established_physics": _agreement(outcomes, REFERENCE),
        "multi_seed": _multi_seed_summary(runs, REFERENCE, frames)
        if len(draws) > 1
        else {"available": False, "reason": "a single CEM seed was run"},
    }

    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command=(
            "python -m src.evaluation.physics_injection_mpc_benchmark"
            f" --seeds {','.join(str(s) for s in draws)}"
        ),
        extra_meta={"controllers": list(outcomes)},
    )
    logger.info("Artifact written to: %s", artifact)
    payload["_artifact"] = artifact
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop control of the 10.47 physics-injection grid on real SMW."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--rom-path", default=ROM_PATH)
    parser.add_argument("--state-path", default=STATE_YOSHI_ISLAND_1)
    parser.add_argument("--output-dir", default="results")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    run_physics_injection_closed_loop(
        frames=args.frames,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        seed=args.seed,
        seeds=[int(s) for s in str(args.seeds).split(",") if s.strip()],
        rom_path=args.rom_path,
        state_path=args.state_path,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
