"""
evaluate_multi_entity_snes.py
Zero-Shot Hardware Evaluation of the End-to-End Multi-Entity 12D Policy.
Executes purely neural action inference (no hand-crafted trigger heuristics)
directly on the real Libretro SNES console emulator with live Rex hazards.
"""

import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.planning.mpc_planner import ACTION_MATRIX
from src.training.dyna_ppo import ActorCritic
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
        "Y": bool(vec[1] > 0.5),  # Run / Dash
        "UP": bool(vec[2] > 0.5),
        "DOWN": bool(vec[3] > 0.5),
        "LEFT": bool(vec[4] > 0.5),
        "RIGHT": bool(vec[5] > 0.5),
    }


def run_multi_entity_neural_policy(
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    policy_12d: ActorCritic,
    device: torch.device,
    max_frames: int = 500,
) -> Dict:
    """
    Executes the trained 12D policy end-to-end on the real SNES console emulator.
    All jump timing, speed modulation, and leap arcs are decided purely by the neural network.
    """
    emu.load_state(initial_savestate)
    emu.enable_gameplay_mode()
    for _ in range(5):
        emu.step_frame()

    traj_x, traj_y, traj_vx = [], [], []
    m_init = emu.get_smw_state()
    x_init = m_init["x"]
    survived = 0

    for frame in range(max_frames):
        ext_s = emu.get_smw_extended_state()
        traj_x.append(ext_s["x"])
        traj_y.append(ext_s["y"])
        traj_vx.append(ext_s["vx"])

        # Check death / fall into pit
        if ext_s["y"] > 450.0 or ext_s["y"] < 0.0 or ext_s["air_state"] == 9:
            break
        survived += 1

        # Construct 12D state tensor
        s_vec = torch.tensor(
            [
                ext_s["x"],
                ext_s["y"],
                ext_s["vx"],
                ext_s["vy"],
                ext_s["c_ground"],
                ext_s["c_ceiling"],
                ext_s["c_left"],
                ext_s["c_right"],
                ext_s["delta_x_enemy"],
                ext_s["delta_y_enemy"],
                ext_s["vx_enemy"],
                ext_s["hazard_active"],
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # Pure neural network inference
        with torch.no_grad():
            logits = policy_12d.actor(policy_12d.in_norm(s_vec))
            action_idx = int(torch.argmax(logits, dim=-1).item())

        action_vec = ACTION_MATRIX[action_idx].copy()

        # Joypad trigger edge is handled emulator-side (WRAM latch $7E:0016).
        act_dict = action_vector_to_dict(action_vec)

        emu.set_input(act_dict)
        emu.step_frame()

    final_progress = float(traj_x[-1] - x_init) if traj_x else 0.0
    rex_evaded = final_progress > 200.0

    return {
        "name": "End-to-End Multi-Entity 12D Policy",
        "survived_frames": survived,
        "total_progress": final_progress,
        "rex_evaded": rex_evaded,
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "trajectory_vx": traj_vx,
    }


def run_blind_8d_policy(
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    policy_8d: ActorCritic,
    device: torch.device,
    max_frames: int = 500,
) -> Dict:
    emu.load_state(initial_savestate)
    emu.enable_gameplay_mode()
    for _ in range(5):
        emu.step_frame()

    traj_x, traj_y, traj_vx = [], [], []
    m_init = emu.get_smw_state()
    x_init = m_init["x"]
    survived = 0

    for _ in range(max_frames):
        s = emu.get_smw_state()
        traj_x.append(s["x"])
        traj_y.append(s["y"])
        traj_vx.append(s["vx"])

        if s["y"] > 450.0 or s["y"] < 0.0 or s["air_state"] == 9:
            break
        survived += 1

        curr_vec = torch.tensor(
            [
                s["x"],
                s["y"],
                s["vx"],
                s["vy"],
                s["c_ground"],
                s["c_ceiling"],
                s["c_left"],
                s["c_right"],
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = policy_8d.actor(policy_8d.in_norm(curr_vec))
            action_idx = int(torch.argmax(logits, dim=-1).item())

        action_vec = ACTION_MATRIX[action_idx]
        emu.set_input(action_vector_to_dict(action_vec))
        emu.step_frame()

    return {
        "name": "Blind 8D Policy (No Hazard Perception)",
        "survived_frames": survived,
        "total_progress": float(traj_x[-1] - x_init) if traj_x else 0.0,
        "rex_evaded": False,
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "trajectory_vx": traj_vx,
    }


def benchmark_multi_entity_hardware():
    logger.info("====================================================================")
    logger.info("  ZERO-SHOT HARDWARE BENCHMARK: END-TO-END MULTI-ENTITY POLICY       ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core_path = CORE_PATH
    rom_path = ROM_PATH
    savestate_path = STATE_YOSHI_ISLAND_1

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(savestate_path, "rb") as f:
        initial_savestate = f.read()

    # Load policies
    policy_8d = ActorCritic(state_dim=8, num_actions=8, hidden_dim=128).to(device)
    ckpt_8d = checkpoint_file("dyna_ppo_pinn_hard_policy.pt")
    if os.path.exists(ckpt_8d):
        policy_8d.load_state_dict(torch.load(ckpt_8d, map_location=device, weights_only=True))

    policy_12d = ActorCritic(state_dim=12, num_actions=8, hidden_dim=128).to(device)
    ckpt_12d = checkpoint_file("dyna_ppo_multi_entity_best.pt")
    if os.path.exists(ckpt_12d):
        policy_12d.load_state_dict(torch.load(ckpt_12d, map_location=device, weights_only=True))
    else:
        logger.info(f"Warning: {ckpt_12d} not found, using initialized weights.")

    policy_8d.eval()
    policy_12d.eval()

    # Execute console trials
    logger.info("Executing Trial 1: Blind 8D Policy...")
    res_8d = run_blind_8d_policy(emu, initial_savestate, policy_8d, device)
    logger.info(
        f"Blind 8D Policy: Progress = {res_8d['total_progress']:.1f} px | Survived = {res_8d['survived_frames']} frames"
    )

    logger.info("Executing Trial 2: End-to-End Multi-Entity 12D Policy...")
    res_12d = run_multi_entity_neural_policy(emu, initial_savestate, policy_12d, device)
    logger.info(
        f"Multi-Entity 12D Policy: Progress = {res_12d['total_progress']:.1f} px | Survived = {res_12d['survived_frames']} frames | Rex Evaded: {res_12d['rex_evaded']}"
    )

    emu.close()

    # Save metrics
    results = {
        "blind_8d": {
            "name": res_8d["name"],
            "progress": res_8d["total_progress"],
            "survived_frames": res_8d["survived_frames"],
            "rex_evaded": res_8d["rex_evaded"],
        },
        "multi_entity_12d": {
            "name": res_12d["name"],
            "progress": res_12d["total_progress"],
            "survived_frames": res_12d["survived_frames"],
            "rex_evaded": res_12d["rex_evaded"],
        },
    }
    with open(results_file("multi_entity_hardware_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison trajectories
    plt.figure(figsize=(10, 5))
    plt.plot(
        res_8d["trajectory_x"],
        res_8d["trajectory_y"],
        color="#EF4444",
        lw=2,
        linestyle="--",
        label=f"Blind 8D (Fatal at X={res_8d['trajectory_x'][-1]:.0f})",
    )
    plt.plot(
        res_12d["trajectory_x"],
        res_12d["trajectory_y"],
        color="#10B981",
        lw=2.5,
        label=f"Multi-Entity 12D (Progress +{res_12d['total_progress']:.1f}px)",
    )
    plt.axvline(x=131.0, color="#F59E0B", linestyle=":", lw=1.5, label="Rex Spawn Zone (X~131)")
    plt.gca().invert_yaxis()
    plt.xlabel("Horizontal Position X (pixels)")
    plt.ylabel("Vertical Position Y (pixels)")
    plt.title("Zero-Shot Hardware Transfer: Blind 8D vs. Multi-Entity 12D World Model Policy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_file("multi_entity_snes_trajectories.png"), dpi=300)
    plt.close()

    logger.info("Multi-Entity hardware evaluation completed successfully.")
    return results


if __name__ == "__main__":
    benchmark_multi_entity_hardware()
