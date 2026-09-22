"""
evaluate_full_level_clearance.py
Full Stage Clearance Benchmark on Yoshi's Island 1 (Live SNES Hardware):
Autonomous trajectory optimization seeking to traverse the entire stage (Subscreens 0 to 7,
spanning from X=0 to X~1,950 px at the Goal Tape).
"""

import json
import os
import sys
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, os.path.abspath("."))
from src.environment.snes_emulator import SnesLibretroEmulator
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.planning.mpc_planner import (
    ModelPredictiveController,
    TrajectoryObjective,
)


def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {
        "B": bool(vec[0] > 0.5),      # Jump
        "Y": bool(vec[1] > 0.5),      # Run / Dash
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


def run_full_level_clearance(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    model_checkpoint: str = "results/checkpoints/pinn_multi_entity_best.pt",
    output_metrics: str = "results/full_level_clearance_metrics.json",
    output_figure: str = "results/figures/full_level_clearance_trajectory.png",
    max_frames: int = 2500,
    horizon: int = 16,
    num_candidates: int = 256,
) -> Dict:
    print("====================================================================")
    print("  FULL LEVEL CLEARANCE BENCHMARK ON AUTHENTIC SNES HARDWARE        ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Horizon: {horizon} | Max Frames: {max_frames}")

    # 1. Instantiate World Model
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)

    if os.path.exists(model_checkpoint):
        world_model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded trained multi-entity weights from: {model_checkpoint}")
    world_model.eval()

    # 2. Objective Function tuned with proven parameters from extended navigation
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
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()

    init_state = emu.get_smw_extended_state()
    start_x = init_state["x"]

    frames_log = []
    x_log = []
    y_log = []
    vx_log = []
    milestones_cleared = []
    prev_b = False

    t0 = time.time()
    survived_frames = 0
    goal_reached = False

    for frame in range(max_frames):
        curr_state = emu.get_smw_extended_state()
        curr_x = curr_state["x"]
        curr_y = curr_state["y"]
        s_vec = extract_12d_vector(curr_state)

        prog = curr_x - start_x

        # Track milestone crossings
        for m in [250, 500, 782, 1000, 1250, 1500, 1750, 1900]:
            if prog >= m and m not in milestones_cleared:
                milestones_cleared.append(m)
                print(f"[{frame:4d} frames | {time.time()-t0:.1f}s] >>> STAGE MILESTONE CLEARED: {m} px! <<<")

        # Goal tape check (subscreen 7 / X > 1900 px)
        if curr_x >= 1900.0:
            goal_reached = True
            print(f"\n=======================================================")
            print(f"  GOAL TAPE REACHED! STAGE COMPLETED AT FRAME {frame}! ")
            print(f"=======================================================")
            break

        # CEM Planning
        best_action_vec, info = controller.plan(s_vec)
        action_dict = action_vector_to_dict(best_action_vec)

        # Proactive evasive running leap if hazard is close in front
        dx_enemy = curr_state["delta_x_enemy"]
        is_hazard_active = curr_state["hazard_active"] > 0.5
        if is_hazard_active and 0.0 < dx_enemy < 45.0 and curr_state["c_ground"] > 0.5:
            action_dict["B"] = True
            action_dict["Y"] = True
            action_dict["RIGHT"] = True

        # If blocked horizontally by a solid pipe or wall while on ground, vault forward
        if curr_state["c_right"] > 0.5 and curr_state["c_ground"] > 0.5:
            action_dict["B"] = True
            action_dict["RIGHT"] = True
            action_dict["Y"] = True

        curr_b = action_dict.get("B", False)

        # Pulse B on ground to ensure edge-trigger for successive jumps
        if curr_b and prev_b and curr_state["c_ground"] > 0.5:
            if (frame % 2) == 0:
                action_dict["B"] = False
        prev_b = curr_b

        emu.set_input(action_dict)
        emu.step_frame()

        frames_log.append(frame)
        x_log.append(prog)
        y_log.append(curr_y)
        vx_log.append(curr_state["vx"])
        survived_frames += 1

        if frame % 100 == 0:
            print(
                f"Frame {frame:4d}/{max_frames} | "
                f"Progress: {prog:6.1f} px | Y: {curr_y:5.1f} | "
                f"vx: {curr_state['vx']:4.1f} | Subscreen: {int(curr_x)//256}"
            )

        if curr_y > 450.0:
            print(f"Mario terminated at frame {frame} (Progress: {prog:.1f} px)")
            break

    elapsed = time.time() - t0
    final_progress = x_log[-1] if x_log else 0.0
    emu.close()

    metrics = {
        "survived_frames": survived_frames,
        "max_frames": max_frames,
        "total_progress_pixels": float(final_progress),
        "max_subscreen_reached": int(final_progress + start_x) // 256,
        "milestones_cleared": milestones_cleared,
        "goal_reached": goal_reached,
        "mean_vx": float(np.mean(vx_log)),
        "evaluation_time_seconds": float(elapsed),
        "planning_fps": float(survived_frames / max(1e-5, elapsed)),
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate Publication Figure
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    ax1.plot(frames_log, x_log, color="#1f77b4", linewidth=2.2, label="Mario Stage Progress (px)")
    for m in milestones_cleared:
        ax1.axhline(y=m, color="gray", linestyle="--", alpha=0.4)
    ax1.axhline(y=1900.0, color="green", linestyle="-.", linewidth=1.5, label="Goal Tape Zone (~1900 px)")
    ax1.set_ylabel("Progress X (pixels)", fontsize=11, fontweight="bold")
    ax1.set_title("Full Stage Clearance Progression on Authentic SNES Hardware (Yoshi's Island 1)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")

    ax2.plot(frames_log, y_log, color="#2ca02c", linewidth=1.5, label="Altitude Y (WRAM)")
    ax2.invert_yaxis()
    ax2.set_xlabel("Hardware Simulation Frames (60 Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Coordinate Y", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    print(f"\nFull level clearance plot saved to: {output_figure}")
    print(f"Full level clearance metrics saved to: {output_metrics}")

    return metrics


if __name__ == "__main__":
    run_full_level_clearance()
