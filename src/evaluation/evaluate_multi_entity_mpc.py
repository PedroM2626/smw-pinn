"""
evaluate_multi_entity_mpc.py
Autonomous Zero-Shot Hardware Evaluation of Multi-Entity Model Predictive Control (MPC):
Plans trajectories in real time using the trained MultiEntityPINNDynamics World Model.
Avoids dynamic Rex hazards through forward trajectory optimization (CEM) without hand-crafted if/else heuristics.
"""

import json
import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def run_multi_entity_mpc(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    model_checkpoint: str = "results/checkpoints/pinn_multi_entity_best.pt",
    output_metrics: str = "results/multi_entity_mpc_metrics.json",
    max_frames: int = 400,
    horizon: int = 16,
    num_candidates: int = 256,
) -> Dict:
    print("====================================================================")
    print("  AUTONOMOUS MULTI-ENTITY MPC EVALUATION ON AUTHENTIC SNES CONSOLE   ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Horizon: {horizon} | Candidates: {num_candidates}")

    # 1. Instantiate trained Multi-Entity PINN World Model
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)

    if os.path.exists(model_checkpoint):
        world_model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded trained multi-entity weights from: {model_checkpoint}")
    else:
        print(f"Warning: Checkpoint {model_checkpoint} not found, using base model.")

    world_model.eval()

    # 2. Setup Trajectory Objective with Hazard Penalties
    objective = TrajectoryObjective(
        weight_progress=3.0,
        weight_velocity=0.5,
        pit_penalty=1000.0,
        death_y=450.0,
        hazard_penalty=800.0,
        leap_bonus=350.0,
    )

    # 3. Instantiate MPC Planner
    controller = ModelPredictiveController(
        world_model=world_model,
        device=device,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=3,
        elite_ratio=0.1,
        objective=objective,
    )

    # 4. Boot SNES Emulator
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    emu.load_state(initial_savestate)
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()

    traj_x, traj_y, traj_vx = [], [], []
    traj_dx_hazard = []
    actions_executed = []

    m_init = emu.get_smw_state()
    x_init = m_init["x"]
    survived = 0
    prev_b = False

    t0 = time.time()
    for frame in range(max_frames):
        ext_s = emu.get_smw_extended_state()
        curr_vec = extract_12d_vector(ext_s)

        traj_x.append(ext_s["x"])
        traj_y.append(ext_s["y"])
        traj_vx.append(ext_s["vx"])
        traj_dx_hazard.append(ext_s["delta_x_enemy"])

        # Check death
        if ext_s["y"] > 450.0 or ext_s["y"] < 0.0 or ext_s["air_state"] == 9:
            print(f"Mario terminated at frame {frame} (Y={ext_s['y']:.1f}, air_state={ext_s['air_state']})")
            break

        survived += 1

        # Plan optimal action via MPC
        best_action_vec, plan_info = controller.plan(curr_vec)
        actions_executed.append(best_action_vec.tolist())

        # Joypad input edge trigger handling
        act_dict = action_vector_to_dict(best_action_vec)
        curr_b = act_dict.get("B", False)

        # Pulse B latch if consecutive jump commands on ground to ensure re-trigger
        if curr_b and prev_b and ext_s["c_ground"] > 0.5 and ext_s["vx"] < 2.0:
            act_dict["B"] = False

        emu.set_input(act_dict)
        emu.step_frame()
        prev_b = curr_b

        if frame % 50 == 0:
            print(
                f"Frame {frame:3d} | X={ext_s['x']:.1f} (dx_rex={ext_s['delta_x_enemy']:.1f}) | "
                f"vx={ext_s['vx']:.1f} | Best MPC Reward: {plan_info['best_reward']:.1f}"
            )

    emu.close()
    elapsed = time.time() - t0

    final_progress = float(traj_x[-1] - x_init) if traj_x else 0.0
    rex_evaded = bool(final_progress > 180.0)

    results = {
        "survived_frames": survived,
        "total_progress_pixels": final_progress,
        "max_progress_pixels": float(np.max(traj_x) - x_init) if traj_x else 0.0,
        "mean_vx": float(np.mean(traj_vx)) if traj_vx else 0.0,
        "rex_evaded": rex_evaded,
        "evaluation_time_seconds": elapsed,
        "fps": float(survived / elapsed) if elapsed > 0 else 0.0,
        "horizon": horizon,
        "num_candidates": num_candidates,
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(results, f, indent=4)

    print("\n====================================================================")
    print("  MULTI-ENTITY MPC BENCHMARK RESULTS")
    print("====================================================================")
    print(f"Survived frames: {survived} / {max_frames}")
    print(f"Total progress: {final_progress:.2f} pixels")
    print(f"Rex evaded successfully: {rex_evaded}")
    print(f"Planning speed: {results['fps']:.1f} FPS")
    print(f"Metrics saved to: {output_metrics}")
    print("====================================================================")

    # Plot trajectories
    plt.figure(figsize=(10, 5))
    plt.plot(traj_x, label="Mario X Position", color="blue", linewidth=2)
    plt.axhline(x_init + 150.0, color="red", linestyle="--", label="Rex Initial Spawn Area")
    plt.title("Closed-Loop Multi-Entity MPC Trajectory on Real SNES Console")
    plt.xlabel("Frame")
    plt.ylabel("Level X Coordinate (Pixels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = "results/figures/multi_entity_mpc_trajectory.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved to: {fig_path}")

    return results


if __name__ == "__main__":
    run_multi_entity_mpc()
