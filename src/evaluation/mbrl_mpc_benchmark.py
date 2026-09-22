"""
mbrl_mpc_benchmark.py
Empirical Model-Based Reinforcement Learning (MBRL) Benchmark:
Compares closed-loop Model Predictive Control (MPC) trajectory planning in Super Mario World
using the Hard Residual PINN World Model vs. Statistical MLP vs. Soft PINN vs. Random Actions.
Executes directly on the real headless SNES emulator (Snes9x core).
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
from src.models import (
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalMLPDynamics,
)
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.seed import set_global_seed


def convert_action_vector_to_dict(action_vec: np.ndarray) -> Dict[str, bool]:
    """
    Converts 6D continuous/binary action vector [B, Y, UP, DOWN, LEFT, RIGHT]
    into SNES joypad button dictionary.
    """
    return {
        "B": bool(action_vec[0] > 0.5),      # Jump
        "Y": bool(action_vec[1] > 0.5),      # Run / Dash
        "UP": bool(action_vec[2] > 0.5),
        "DOWN": bool(action_vec[3] > 0.5),
        "LEFT": bool(action_vec[4] > 0.5),
        "RIGHT": bool(action_vec[5] > 0.5),
    }


def extract_state_vector(s_dict: dict) -> np.ndarray:
    return np.array(
        [
            s_dict["x"],
            s_dict["y"],
            s_dict["vx"],
            s_dict["vy"],
            s_dict["c_ground"],
            s_dict["c_ceiling"],
            s_dict["c_left"],
            s_dict["c_right"],
        ],
        dtype=np.float32,
    )


def run_mbrl_closed_loop_trial(
    controller: ModelPredictiveController,
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    max_frames: int = 300,
    policy_name: str = "MPC_Agent",
) -> Dict:
    """
    Executes a closed-loop MPC control episode in the real SNES emulator.
    """
    emu.load_state(initial_savestate)

    trajectory_x = []
    trajectory_y = []
    trajectory_vx = []
    alignment_errors = []

    init_state_dict = emu.get_smw_state()
    x_init = init_state_dict["x"]

    t0 = time.time()
    survived_frames = 0

    for frame in range(max_frames):
        current_state_dict = emu.get_smw_state()
        curr_vec = extract_state_vector(current_state_dict)

        trajectory_x.append(curr_vec[0])
        trajectory_y.append(curr_vec[1])
        trajectory_vx.append(curr_vec[2])

        # Terminal conditions: fallen into pit or out of bounds
        if curr_vec[1] > 450.0 or curr_vec[1] < 0.0:
            break

        survived_frames += 1

        # 1. Plan optimal action via MPC using the World Model
        action_vec, plan_info = controller.plan(curr_vec)

        # 2. Inject action into SNES joypad
        action_dict = convert_action_vector_to_dict(action_vec)
        emu.set_input(action_dict)

        # 3. Step real SNES hardware
        emu.step_frame()

        # 4. Measure alignment between World Model's 1-step imagination and reality
        imagined_next_s = plan_info["imagined_trajectory"][0]
        actual_next_s = extract_state_vector(emu.get_smw_state())

        spatial_alignment_err = np.sqrt(
            (imagined_next_s[0] - actual_next_s[0]) ** 2
            + (imagined_next_s[1] - actual_next_s[1]) ** 2
        )
        alignment_errors.append(float(spatial_alignment_err))

    elapsed = time.time() - t0
    final_x = trajectory_x[-1] if trajectory_x else x_init
    max_x = max(trajectory_x) if trajectory_x else x_init
    total_progress = float(final_x - x_init)
    max_progress = float(max_x - x_init)
    mean_vx = float(np.mean(trajectory_vx)) if trajectory_vx else 0.0
    mean_alignment_error = float(np.mean(alignment_errors)) if alignment_errors else 0.0

    print(
        f"[{policy_name:20s}] Survived: {survived_frames:3d}/{max_frames} frames | "
        f"Progress: {total_progress:+6.1f} px (Max: {max_progress:+6.1f} px) | "
        f"Mean vx: {mean_vx:+5.1f} | Alignment Error: {mean_alignment_error:5.2f} px | "
        f"Time: {elapsed:4.1f}s ({survived_frames/elapsed:4.1f} FPS)"
    )

    return {
        "policy_name": policy_name,
        "survived_frames": survived_frames,
        "total_progress": total_progress,
        "max_progress": max_progress,
        "mean_vx": mean_vx,
        "mean_alignment_error": mean_alignment_error,
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "trajectory_vx": trajectory_vx,
        "alignment_errors": alignment_errors,
    }


def run_random_baseline(
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    max_frames: int = 300,
    seed: int = 42,
) -> Dict:
    """Executes a stochastic exploration baseline (Random Actions)."""
    set_global_seed(seed)
    emu.load_state(initial_savestate)

    trajectory_x = []
    trajectory_y = []
    trajectory_vx = []

    init_state_dict = emu.get_smw_state()
    x_init = init_state_dict["x"]
    survived_frames = 0

    actions_pool = [
        {"RIGHT": True, "Y": True},
        {"RIGHT": True, "B": True},
        {"RIGHT": True},
        {"LEFT": True},
        {"B": True},
        {},
    ]

    for frame in range(max_frames):
        current_state_dict = emu.get_smw_state()
        curr_vec = extract_state_vector(current_state_dict)

        trajectory_x.append(curr_vec[0])
        trajectory_y.append(curr_vec[1])
        trajectory_vx.append(curr_vec[2])

        if curr_vec[1] > 450.0 or curr_vec[1] < 0.0:
            break

        survived_frames += 1
        act = np.random.choice(actions_pool)
        emu.set_input(act)
        emu.step_frame()

    final_x = trajectory_x[-1] if trajectory_x else x_init
    max_x = max(trajectory_x) if trajectory_x else x_init
    total_progress = float(final_x - x_init)
    max_progress = float(max_x - x_init)

    print(
        f"[{'Random_Baseline':20s}] Survived: {survived_frames:3d}/{max_frames} frames | "
        f"Progress: {total_progress:+6.1f} px (Max: {max_progress:+6.1f} px) | "
        f"Mean vx: {np.mean(trajectory_vx):+5.1f} | Alignment Error: N/A"
    )

    return {
        "policy_name": "Random_Baseline",
        "survived_frames": survived_frames,
        "total_progress": total_progress,
        "max_progress": max_progress,
        "mean_vx": float(np.mean(trajectory_vx)),
        "mean_alignment_error": 0.0,
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "trajectory_vx": trajectory_vx,
        "alignment_errors": [],
    }


def run_mbrl_mpc_benchmark(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    checkpoints_dir: str = "results/checkpoints",
    output_dir: str = "results",
    max_frames: int = 300,
):
    print("====================================================================")
    print("  MODEL-BASED REINFORCEMENT LEARNING (MBRL): MPC WORLD MODEL BENCHMARK")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Planning Compute Device: {device}")
    if device.type == "cuda":
        print(f"Planning GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Initialize emulator
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    # Load World Models
    models = {
        "Hard_PINN_World_Model": (
            HardResidualPINNDynamics(state_dim=8, action_dim=6),
            os.path.join(checkpoints_dir, "pinn_hard_best.pt"),
        ),
        "Soft_PINN_World_Model": (
            SoftPINNDynamics(state_dim=8, action_dim=6),
            os.path.join(checkpoints_dir, "pinn_soft_best.pt"),
        ),
        "Statistical_MLP_World_Model": (
            StatisticalMLPDynamics(state_dim=8, action_dim=6),
            os.path.join(checkpoints_dir, "mlp_best.pt"),
        ),
    }

    trials_results = {}

    for name, (model, ckpt_path) in models.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found for {name} at {ckpt_path}, skipping.")
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.to(device)

        planner = ModelPredictiveController(
            world_model=model,
            device=device,
            horizon=15,
            num_candidates=256,
            cem_iterations=3,
            elite_ratio=0.1,
            objective=TrajectoryObjective(
                weight_progress=2.0,
                weight_velocity=0.5,
                pit_penalty=1000.0,
                death_y=450.0,
            ),
        )

        res = run_mbrl_closed_loop_trial(
            controller=planner,
            emu=emu,
            initial_savestate=initial_savestate,
            max_frames=max_frames,
            policy_name=name,
        )
        trials_results[name] = res

    # Random exploration baseline
    rand_res = run_random_baseline(
        emu=emu,
        initial_savestate=initial_savestate,
        max_frames=max_frames,
        seed=42,
    )
    trials_results["Random_Baseline"] = rand_res

    emu.close()

    # Save summary metrics to JSON
    summary_metrics = {
        name: {
            "survived_frames": r["survived_frames"],
            "total_progress_pixels": r["total_progress"],
            "max_progress_pixels": r["max_progress"],
            "mean_vx": r["mean_vx"],
            "mean_alignment_error_pixels": r["mean_alignment_error"],
        }
        for name, r in trials_results.items()
    }

    out_json = os.path.join(output_dir, "mbrl_mpc_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"\nMBRL metrics saved to: {out_json}")

    # Generate comparative trajectory figure
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    colors = {
        "Hard_PINN_World_Model": "#2ecc71",       # Emerald Green
        "Statistical_MLP_World_Model": "#e74c3c",  # Alizarin Red
        "Soft_PINN_World_Model": "#f39c12",        # Orange
        "Random_Baseline": "#7f8c8d",              # Gray
    }

    # Plot 1: 2D World Trajectory (X vs Y) in SNES Level Space
    for name, r in trials_results.items():
        c = colors.get(name, "#3498db")
        ax1.plot(r["trajectory_x"], r["trajectory_y"], label=name, color=c, linewidth=2.5, alpha=0.9)
    ax1.invert_yaxis()
    ax1.set_title("MBRL Closed-Loop Trajectory Execution in Super Mario World (SNES Console)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Level Horizontal Coordinate X (Pixels)")
    ax1.set_ylabel("Level Vertical Coordinate Y (Pixels - Inverted)")
    ax1.legend(loc="best", fontsize=10)

    # Plot 2: Forward Progress Over Time (Frames)
    for name, r in trials_results.items():
        c = colors.get(name, "#3498db")
        frames = np.arange(len(r["trajectory_x"]))
        prog = np.array(r["trajectory_x"]) - r["trajectory_x"][0]
        ax2.plot(frames, prog, label=name, color=c, linewidth=2.5)
    ax2.set_title("Cumulative Forward Progress vs. Execution Frame", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Simulation Frame (60 Hz)")
    ax2.set_ylabel("Forward Displacement ΔX (Pixels)")
    ax2.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "mbrl_mpc_trajectories.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Trajectory figure saved to: {fig_path}")

    # Summary table
    print("\n====================================================================")
    print("  MBRL MPC BENCHMARK SUMMARY")
    print("====================================================================")
    print(f"{'World Model Controller':30s} | {'Progress (px)':15s} | {'Alignment Err':15s} | {'Frames Alive':12s}")
    print("-" * 80)
    for name, s in summary_metrics.items():
        prog_str = f"{s['total_progress_pixels']:+6.1f} px"
        err_str = f"{s['mean_alignment_error_pixels']:5.2f} px" if s['mean_alignment_error_pixels'] > 0 else "N/A"
        print(f"{name:30s} | {prog_str:15s} | {err_str:15s} | {s['survived_frames']:4d}/{max_frames}")
    print("====================================================================")

    return summary_metrics


if __name__ == "__main__":
    run_mbrl_mpc_benchmark()
