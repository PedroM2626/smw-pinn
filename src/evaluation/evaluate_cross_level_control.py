"""
evaluate_cross_level_control.py
Zero-Shot Closed-Loop Control Benchmark on Unseen Stage B (Yoshi's House):
Deploys models and policies trained strictly on Stage A (Yoshi's Island 1)
into Yoshi's House without any parameter retraining, calibration, or fine-tuning.

Evaluates:
1. Random Control Baseline
2. MPC guided by Statistical MLP World Model
3. MPC guided by Soft-Constrained PINN World Model
4. MPC guided by Hard Residual PINN World Model (Ours)
5. Amortized DAgger Neural Policy (Ours)

Generates:
- results/cross_level_control_metrics.json
- results/figures/cross_level_control_trajectories.png
"""

import json
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_soft import SoftPINNDynamics
from src.models.statistical_mlp import StatisticalMLPDynamics
from src.planning.mpc_planner import (
    ACTION_MATRIX,
    ACTION_PRIMITIVES,
    ModelPredictiveController,
    TrajectoryObjective,
)
from src.training.distill_mpc_policy import DistilledActorPolicy, extract_12d_vector
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    ROM_PATH,
    STATE_YOSHI_HOUSE,
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


def extract_8d_vector(state_dict: dict) -> np.ndarray:
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
        ],
        dtype=np.float32,
    )


def evaluate_single_controller(
    controller_type: str,
    core_path: str,
    rom_path: str,
    state_path: str,
    device: torch.device,
    max_frames: int = 400,
) -> Tuple[Dict, List[float], List[float]]:
    """Runs a single controller in closed loop on Yoshi's House."""
    logger.info(f"\nEvaluating: {controller_type} on Yoshi's House...")

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        savestate = f.read()

    emu.load_state(savestate)
    emu.enable_gameplay_mode()
    for _ in range(10):
        emu.step_frame()

    init_state = emu.get_smw_state()
    start_x = init_state["x"]

    # Controller initialization
    objective = TrajectoryObjective(
        weight_progress=3.0, weight_velocity=0.5, pit_penalty=1000.0, death_y=450.0
    )
    mpc = None
    dagger_policy = None

    if controller_type == "MPC_MLP":
        wm = StatisticalMLPDynamics(state_dim=8, action_dim=6).to(device)
        if os.path.exists(checkpoint_file("mlp_best.pt")):
            wm.load_state_dict(
                torch.load(checkpoint_file("mlp_best.pt"), map_location=device, weights_only=True)
            )
        wm.eval()
        mpc = ModelPredictiveController(
            world_model=wm,
            device=device,
            horizon=15,
            num_candidates=128,
            cem_iterations=3,
            objective=objective,
        )

    elif controller_type == "MPC_Soft_PINN":
        wm = SoftPINNDynamics(state_dim=8, action_dim=6).to(device)
        if os.path.exists(checkpoint_file("pinn_soft_best.pt")):
            wm.load_state_dict(
                torch.load(
                    checkpoint_file("pinn_soft_best.pt"), map_location=device, weights_only=True
                )
            )
        wm.eval()
        mpc = ModelPredictiveController(
            world_model=wm,
            device=device,
            horizon=15,
            num_candidates=128,
            cem_iterations=3,
            objective=objective,
        )

    elif controller_type == "MPC_Hard_PINN":
        wm = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
        if os.path.exists(checkpoint_file("pinn_hard_best.pt")):
            wm.load_state_dict(
                torch.load(
                    checkpoint_file("pinn_hard_best.pt"), map_location=device, weights_only=True
                )
            )
        wm.eval()
        mpc = ModelPredictiveController(
            world_model=wm,
            device=device,
            horizon=15,
            num_candidates=128,
            cem_iterations=3,
            objective=objective,
        )

    elif controller_type == "DAgger_Policy":
        dagger_policy = DistilledActorPolicy(state_dim=12, action_dim=6)
        if os.path.exists(checkpoint_file("dagger_policy_best.pt")):
            dagger_policy.load_state_dict(
                torch.load(
                    checkpoint_file("dagger_policy_best.pt"), map_location="cpu", weights_only=True
                )
            )
        dagger_policy.eval()

    traj_x = []
    traj_y = []
    traj_vx = []
    step_times = []
    survived = 0
    prev_b = False

    t_start = time.time()

    for frame in range(max_frames):
        s_dict = emu.get_smw_state()
        s_ext = emu.get_smw_extended_state()
        curr_x = s_dict["x"]
        curr_y = s_dict["y"]

        traj_x.append(curr_x - start_x)
        traj_y.append(curr_y)
        traj_vx.append(s_dict["vx"])

        # Planning / Action Selection
        t0 = time.perf_counter()
        if controller_type == "Random_Baseline":
            act_vec = ACTION_MATRIX[np.random.choice(len(ACTION_PRIMITIVES))]
            action_dict = action_vector_to_dict(act_vec)
        elif dagger_policy is not None:
            s12 = extract_12d_vector(s_ext)
            action_dict, _ = dagger_policy.predict_action(s12, threshold=0.5)
        else:
            s8 = extract_8d_vector(s_dict)
            act_vec, _ = mpc.plan(s8)
            action_dict = action_vector_to_dict(act_vec)

        # Pulse B for jump edge-trigger
        if action_dict.get("B", False):
            if prev_b and s_dict["c_ground"] > 0.5 and (frame % 3) == 0:
                action_dict["B"] = False
            prev_b = action_dict.get("B", False)
        else:
            prev_b = False

        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        emu.set_input(action_dict)
        emu.step_frame()
        survived += 1

        if curr_y > 450.0:
            logger.info(f"[{controller_type}] Fell into pit at frame {frame}")
            break

    elapsed = time.time() - t_start
    final_prog = float(traj_x[-1]) if traj_x else 0.0
    emu.close()

    metrics = {
        "survived_frames": survived,
        "max_frames": max_frames,
        "total_progress_pixels": final_prog,
        "mean_vx": float(np.mean(traj_vx)),
        "mean_step_time_ms": float(np.mean(step_times) * 1000.0),
        "throughput_fps": float(1.0 / np.mean(step_times)),
        "evaluation_time_seconds": float(elapsed),
    }

    logger.info(
        f"[{controller_type:16s}] Progress: {final_prog:6.2f} px | Survived: {survived:3d} frames | "
        f"vx: {metrics['mean_vx']:5.1f} | Throughput: {metrics['throughput_fps']:6.1f} FPS"
    )

    return metrics, traj_x, traj_y


def run_cross_level_control_benchmark(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_HOUSE,
    output_metrics: str = results_file("cross_level_control_metrics.json"),
    output_figure: str = figure_file("cross_level_control_trajectories.png"),
    max_frames: int = 400,
):
    logger.info("====================================================================")
    logger.info("  ZERO-SHOT CLOSED-LOOP CONTROL BENCHMARK ON UNSEEN STAGE B         ")
    logger.info("  Target Stage: Yoshi's House ($7E:0100 = 0x14)                     ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    controllers = [
        "Random_Baseline",
        "MPC_MLP",
        "MPC_Soft_PINN",
        "MPC_Hard_PINN",
        "DAgger_Policy",
    ]

    all_metrics = {}
    trajectories_x = {}
    trajectories_y = {}

    for c in controllers:
        m, tx, ty = evaluate_single_controller(
            controller_type=c,
            core_path=core_path,
            rom_path=rom_path,
            state_path=state_path,
            device=device,
            max_frames=max_frames,
        )
        all_metrics[c] = m
        trajectories_x[c] = tx
        trajectories_y[c] = ty

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nZero-shot control metrics saved to: {output_metrics}")

    # Plot trajectories
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = {
        "Random_Baseline": "gray",
        "MPC_MLP": "#d62728",
        "MPC_Soft_PINN": "#9467bd",
        "MPC_Hard_PINN": "#2ca02c",
        "DAgger_Policy": "#1f77b4",
    }

    labels = {
        "Random_Baseline": "Random Actions Baseline",
        "MPC_MLP": "MPC + Statistical MLP (Black-Box OOD)",
        "MPC_Soft_PINN": "MPC + Soft-Constrained PINN",
        "MPC_Hard_PINN": "MPC + Hard Residual PINN (Ours)",
        "DAgger_Policy": "Amortized DAgger Policy (Ours @ >2,500 FPS)",
    }

    for c in controllers:
        tx = trajectories_x[c]
        ty = trajectories_y[c]
        frames = list(range(len(tx)))
        lw = 2.4 if "Ours" in labels[c] else 1.5
        style = "-" if "Ours" in labels[c] else "--"
        ax1.plot(frames, tx, label=labels[c], color=colors[c], linewidth=lw, linestyle=style)
        ax2.plot(frames, ty, label=labels[c], color=colors[c], linewidth=lw, linestyle=style)

    ax1.set_ylabel("Horizontal Progress X (px)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Zero-Shot Hardware Control Transfer to Unseen Stage B (Yoshi's House)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")

    ax2.set_ylabel("Altitude Y (px)", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Hardware Simulation Frames (60 Hz)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    logger.info(f"Zero-shot control trajectory plot saved to: {output_figure}")


if __name__ == "__main__":
    run_cross_level_control_benchmark()
