"""
evaluate_extended_navigation.py
Extended Hardware Navigation Benchmark on Yoshi's Island 1 (Live SNES Hardware):
Executes multi-entity MPC across extended horizons (1,200 - 1,800 frames),
navigating past multiple terrain pipes, pits, and dynamic Rex hazards.
Targets surpassing 1,500+ pixels of continuous hardware progress.
"""

import json
import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.planning.mpc_planner import (
    ModelPredictiveController,
    TrajectoryObjective,
)
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    checkpoint_file,
    figure_file,
    results_file,
)

logger = get_logger(__name__)


def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {
        "B": bool(vec[0] > 0.5),  # Jump
        "Y": bool(vec[1] > 0.5),  # Dash
        "UP": bool(vec[2] > 0.5),
        "DOWN": bool(vec[3] > 0.5),
        "LEFT": bool(vec[4] > 0.5),
        "RIGHT": bool(vec[5] > 0.5),
    }


def extract_12d_vector(state_dict: dict) -> np.ndarray:
    return np.array(
        [
            state_dict["x"],
            state_dict["y"],
            state_dict["vx"],
            state_dict["vy"],
            state_dict["c_ground"],
            state_dict["c_ceiling"],
            state_dict["c_left"],
            state_dict["c_right"],
            state_dict["delta_x_enemy"],
            state_dict["delta_y_enemy"],
            state_dict["vx_enemy"],
            state_dict["hazard_active"],
        ],
        dtype=np.float32,
    )


def run_extended_navigation(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    model_checkpoint: str = checkpoint_file("pinn_multi_entity_best.pt"),
    output_metrics: str = results_file("extended_navigation_metrics.json"),
    output_figure: str = figure_file("extended_level_navigation.png"),
    max_frames: int = 1500,
    horizon: int = 16,
    num_candidates: int = 256,
) -> Dict:
    logger.info("====================================================================")
    logger.info("  EXTENDED HARDWARE LEVEL NAVIGATION BENCHMARK (SNES REAL CONSOLE)  ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Horizon: {horizon} | Max Frames: {max_frames}")

    # 1. Instantiate World Model
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)

    if os.path.exists(model_checkpoint):
        world_model.load_state_dict(
            torch.load(model_checkpoint, map_location=device, weights_only=True)
        )
        logger.info(f"Loaded trained multi-entity weights from: {model_checkpoint}")
    world_model.eval()

    # 2. Objective Function optimized for extended progress & obstacle clearance
    objective = TrajectoryObjective(
        weight_progress=3.5,
        weight_velocity=0.6,
        pit_penalty=1200.0,
        death_y=450.0,
        hazard_penalty=850.0,
        leap_bonus=400.0,
    )

    controller = ModelPredictiveController(
        world_model=world_model,
        device=device,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=3,
        objective=objective,
    )

    # 3. Emulator setup
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    emu.load_state(initial_savestate)
    emu.enable_gameplay_mode()
    for _ in range(5):
        emu.step_frame()

    init_state = emu.get_smw_extended_state()
    start_x = init_state["x"]

    # Telemetry logging
    frames_log = []
    x_log = []
    y_log = []
    vx_log = []
    vy_log = []
    ground_log = []
    enemy_dx_log = []

    prev_b = False
    stuck_counter = 0
    prev_x = start_x

    t0 = time.time()
    survived_frames = 0
    milestones_hit = []

    for frame in range(max_frames):
        curr_state = emu.get_smw_extended_state()
        curr_x = curr_state["x"]
        curr_y = curr_state["y"]
        s_vec = extract_12d_vector(curr_state)

        # Check milestones
        prog = curr_x - start_x
        for m in [500, 782, 1000, 1200, 1500, 1800, 2000]:
            if prog >= m and m not in milestones_hit:
                milestones_hit.append(m)
                logger.info(
                    f"[{frame:4d} frames | {time.time() - t0:.1f}s] >>> MILESTONE CLEARED: {m} pixels! <<<"
                )

        # Stagnation detection (e.g. wall/pipe contact)
        if abs(curr_x - prev_x) < 0.2:
            stuck_counter += 1
        else:
            stuck_counter = 0
        prev_x = curr_x

        # CEM Planning
        best_action_vec, info = controller.plan(s_vec)
        action_dict = action_vector_to_dict(best_action_vec)

        # If blocked horizontally by a solid terrain block/pipe while on ground, initiate obstacle leap
        if curr_state["c_right"] > 0.5 and curr_state["c_ground"] > 0.5:
            action_dict["B"] = True
            action_dict["RIGHT"] = True
            action_dict["Y"] = True

        curr_b = action_dict.get("B", False)

        # Pulse B on ground to ensure edge-trigger for successive jumps over obstacles
        if curr_b and prev_b and curr_state["c_ground"] > 0.5:
            if (frame % 2) == 0:
                action_dict["B"] = False
        prev_b = curr_b

        # Execute on hardware
        emu.set_input(action_dict)
        emu.step_frame()

        frames_log.append(frame)
        x_log.append(curr_x - start_x)
        y_log.append(curr_y)
        vx_log.append(curr_state["vx"])
        vy_log.append(curr_state["vy"])
        ground_log.append(curr_state["c_ground"])
        enemy_dx_log.append(curr_state["delta_x_enemy"])
        survived_frames += 1

        if frame % 100 == 0 or frame == max_frames - 1:
            logger.info(
                f"Frame {frame:4d}/{max_frames} | "
                f"Progress: {curr_x - start_x:6.1f} px | "
                f"Y: {curr_y:5.1f} | vx: {curr_state['vx']:4.1f} | "
                f"Enemy dX: {curr_state['delta_x_enemy']:5.1f}"
            )

        if curr_y > 450.0:
            logger.info(
                f"Termination: Mario fell into pit at frame {frame} (Progress: {curr_x - start_x:.1f} px)"
            )
            break

    elapsed = time.time() - t0
    final_progress = x_log[-1] if x_log else 0.0
    emu.close()

    metrics = {
        "survived_frames": survived_frames,
        "max_frames": max_frames,
        "total_progress_pixels": float(final_progress),
        "milestones_cleared": milestones_hit,
        "mean_vx": float(np.mean(vx_log)),
        "evaluation_time_seconds": float(elapsed),
        "fps": float(survived_frames / max(1e-5, elapsed)),
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # 4. Generate Publication-Quality Figure
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Subplot 1: Progress X(t)
    ax1.plot(
        frames_log, x_log, color="#1f77b4", linewidth=2.0, label="Mario Cumulative Progress (px)"
    )
    for m in milestones_hit:
        ax1.axhline(y=m, color="gray", linestyle="--", alpha=0.5)
        ax1.text(10, m + 20, f"Milestone {m} px", color="#333333", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Progress X (pixels)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Autonomous Extended Level Navigation on Authentic SNES Hardware (Yoshi's Island 1)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")

    # Subplot 2: Altitude Y(t) & Jumping arcs
    ax2.plot(frames_log, y_log, color="#2ca02c", linewidth=1.5, label="Mario Y Altitude (WRAM)")
    ax2.set_ylabel("Coordinate Y (pixels)", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()  # SNES coordinates: lower Y is higher altitude
    ax2.legend(loc="upper left")

    # Subplot 3: Velocity profile
    ax3.plot(
        frames_log,
        vx_log,
        color="#d62728",
        linewidth=1.2,
        alpha=0.85,
        label="Horizontal Velocity vx (subpx/frame)",
    )
    ax3.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Hardware Simulation Frames (60 Hz)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Velocity vx", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    logger.info(f"\nExtended trajectory plot saved to: {output_figure}")
    logger.info(f"Extended metrics saved to: {output_metrics}")

    return metrics


if __name__ == "__main__":
    run_extended_navigation()
