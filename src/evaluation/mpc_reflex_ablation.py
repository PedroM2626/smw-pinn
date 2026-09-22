"""
mpc_reflex_ablation.py
Honesty ablation: pure World-Model MPC vs. reflexive MPC on real hardware.

Published full-level runs overlay three hand-coded reflexes on top of the
MPC plan (wall vault, hazard vault, B edge-pulse). This script measures both
conditions head-to-head from the same savestate so the contribution of the
heuristics — vs. emergent MPC behavior — is explicit:

  - pure:    raw CEM-MPC action, no overrides (apart from joypad dict cast).
  - reflex:  MPC + wall vault (c_right & grounded) + hazard vault
             (0 < dx_enemy < 75 & grounded) + B edge-pulse.

Finite-horizon myopia (H = 16 frames ≈ 0.27 s) is the mathematical reason the
reflexes help: a full running leap spans 30-50 frames, so a 16-frame planner
cannot see landing; the reflexes inject the missing launch impulse.

Outputs `results/mpc_reflex_ablation.json` + grouped-bar figure.
"""

import json
import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models import HardResidualPINNDynamics, MultiEntityPINNDynamics
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

BUTTONS = ["B", "Y", "UP", "DOWN", "LEFT", "RIGHT"]


def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {b: bool(vec[i] > 0.5) for i, b in enumerate(BUTTONS)}


def apply_reflexes(
    action_dict: Dict[str, bool],
    curr_state: dict,
    frame: int,
    prev_b: bool,
) -> Tuple[Dict[str, bool], bool, Dict[str, int]]:
    """The three hand-coded reflexes, extracted verbatim for ablation.

    Returns (possibly overridden action_dict, updated prev_b, intervention counts).
    Pure function — unit-testable without the emulator.
    """
    action_dict = dict(action_dict)
    hits = {"hazard_vault": 0, "wall_vault": 0, "b_pulse": 0}

    dx_enemy = curr_state["delta_x_enemy"]
    is_hazard_active = curr_state["hazard_active"] > 0.5
    if is_hazard_active and 0.0 < dx_enemy < 75.0 and curr_state["c_ground"] > 0.5:
        action_dict["B"] = True
        action_dict["Y"] = True
        action_dict["RIGHT"] = True
        hits["hazard_vault"] = 1

    if curr_state["c_right"] > 0.5 and curr_state["c_ground"] > 0.5:
        action_dict["B"] = True
        action_dict["RIGHT"] = True
        action_dict["Y"] = True
        hits["wall_vault"] = 1

    curr_b = action_dict.get("B", False)
    if curr_b and prev_b and curr_state["c_ground"] > 0.5:
        if (frame % 2) == 0:
            action_dict["B"] = False
            hits["b_pulse"] = 1
    prev_b = curr_b
    return action_dict, prev_b, hits


def _extract_12d(curr_state: dict) -> np.ndarray:
    return np.array(
        [
            curr_state["x"], curr_state["y"], curr_state["vx"], curr_state["vy"],
            curr_state["c_ground"], curr_state["c_ceiling"],
            curr_state["c_left"], curr_state["c_right"],
            curr_state["delta_x_enemy"], curr_state["delta_y_enemy"],
            curr_state["vx_enemy"], curr_state["hazard_active"],
        ],
        dtype=np.float32,
    )


def run_condition(
    controller: ModelPredictiveController,
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    use_reflex: bool,
    max_frames: int,
) -> Dict:
    emu.load_state(initial_savestate)
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()
    start_x = emu.get_smw_state()["x"]

    prev_b, prev_x = False, start_x
    stuck_frames, no_progress_streak = 0, 0
    hits_total = {"hazard_vault": 0, "wall_vault": 0, "b_pulse": 0}
    progress_log = []
    termination = "timeout"
    survived = 0

    for frame in range(max_frames):
        curr_state = emu.get_smw_extended_state()
        curr_x = curr_state["x"]
        if abs(curr_x - prev_x) < 0.2:
            stuck_frames += 1
            no_progress_streak += 1
        else:
            no_progress_streak = 0
        prev_x = curr_x

        act_vec, _ = controller.plan(_extract_12d(curr_state))
        action_dict = action_vector_to_dict(act_vec)
        if use_reflex:
            action_dict, prev_b, hits = apply_reflexes(action_dict, curr_state, frame, prev_b)
            for k, v in hits.items():
                hits_total[k] += v

        emu.set_input(action_dict)
        emu.step_frame()
        progress_log.append(curr_x - start_x)
        survived += 1

        if curr_state["y"] > 450.0:
            termination = "pit_fall"
            break
        if no_progress_streak >= 150:
            termination = "stuck_150f"
            break

    return {
        "use_reflex": use_reflex,
        "survived_frames": survived,
        "progress_px": float(progress_log[-1]) if progress_log else 0.0,
        "termination": termination,
        "stuck_frames": stuck_frames,
        "reflex_interventions": hits_total,
        "progress_log": [float(v) for v in progress_log[::10]],
    }


def run_ablation(
    model_checkpoint: str = "results/checkpoints/pinn_multi_entity_best.pt",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    rom_path: str = "data/raw/smw_usa.sfc",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    max_frames: int = 600,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)
    world_model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
    world_model.eval()
    controller = ModelPredictiveController(
        world_model=world_model,
        device=device,
        horizon=16,
        num_candidates=256,
        cem_iterations=3,
        objective=TrajectoryObjective(
            weight_progress=3.5, weight_velocity=0.6, pit_penalty=1200.0,
            death_y=450.0, hazard_penalty=850.0, leap_bonus=400.0,
        ),
    )

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    results = {}
    for use_reflex in (False, True):
        name = "reflex" if use_reflex else "pure"
        logger.info(f"Running condition: {name} ...")
        t0 = time.time()
        results[name] = run_condition(controller, emu, initial_savestate, use_reflex, max_frames)
        results[name]["wall_time_s"] = time.time() - t0
        logger.info(
            f"[{name}] progress={results[name]['progress_px']:.1f}px "
            f"survived={results[name]['survived_frames']} "
            f"termination={results[name]['termination']}"
        )
    emu.close()

    with open(os.path.join(output_dir, "mpc_reflex_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    names = ["pure", "reflex"]
    ax1.bar(names, [results[n]["progress_px"] for n in names], color=["#64748B", "#10B981"])
    ax1.set_ylabel("Progress (px)")
    ax1.set_title("MPC pure vs reflexive: hardware progress")
    for i, n in enumerate(names):
        ax1.text(i, results[n]["progress_px"] + 5, f"{results[n]['progress_px']:.0f}px", ha="center")
    hits = results["reflex"]["reflex_interventions"]
    ax2.bar(list(hits), list(hits.values()), color="#F59E0B")
    ax2.set_ylabel("Interventions (frames)")
    ax2.set_title("Reflex rule firings (reflex condition)")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "mpc_reflex_ablation.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Ablation saved -> {fig_path}")
    return results


if __name__ == "__main__":
    run_ablation()
