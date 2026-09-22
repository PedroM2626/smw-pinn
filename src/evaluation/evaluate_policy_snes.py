"""
evaluate_policy_snes.py
Zero-Shot Model-to-Real Transfer Benchmark on authentic Super Mario World (SNES).
Evaluates policies trained entirely inside the Hard PINN World Model vs. Statistical MLP World Model.
Executes directly in the headless Libretro Snes9x emulator core on stage Yoshi's Island 1.
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
from src.planning.mpc_planner import ACTION_MATRIX
from src.training.dyna_ppo import ActorCritic
from src.utils.seed import set_global_seed


# Mapping from 6D action vector [B, Y, UP, DOWN, LEFT, RIGHT] to emulator joypad dict
def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {
        "B": bool(vec[0] > 0.5),      # Jump
        "Y": bool(vec[1] > 0.5),      # Run / Dash
        "UP": bool(vec[2] > 0.5),
        "DOWN": bool(vec[3] > 0.5),
        "LEFT": bool(vec[4] > 0.5),
        "RIGHT": bool(vec[5] > 0.5),
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


def run_policy_evaluation_trial(
    policy_net: ActorCritic,
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    device: torch.device,
    max_frames: int = 600,
    policy_name: str = "Dyna_PPO_Agent",
    deterministic: bool = True,
) -> Dict:
    """
    Runs a closed-loop deployment trial of the amortized policy inside the SNES console.
    """
    emu.load_state(initial_savestate)
    # Enable Interactive Gameplay Mode (0x14) directly in WRAM
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()

    trajectory_x = []
    trajectory_y = []
    trajectory_vx = []
    actions_taken = []

    init_state_dict = emu.get_smw_state()
    x_init = init_state_dict["x"]

    policy_net.eval()
    survived_frames = 0
    t0 = time.time()

    for frame in range(max_frames):
        current_state_dict = emu.get_smw_state()
        curr_vec = extract_state_vector(current_state_dict)

        trajectory_x.append(curr_vec[0])
        trajectory_y.append(curr_vec[1])
        trajectory_vx.append(curr_vec[2])

        # Terminal conditions: fallen into pit or dead
        if curr_vec[1] > 450.0 or curr_vec[1] < 0.0 or current_state_dict["air_state"] == 9:
            break

        survived_frames += 1

        # 1. Forward pass through policy network (<0.1 ms inference latency)
        state_tensor = torch.tensor(curr_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                # Argmax over actor logits for deterministic execution
                logits = policy_net.actor(policy_net.in_norm(state_tensor))
                action_idx = int(torch.argmax(logits, dim=-1).item())
            else:
                action_idx_tensor, _, _, _ = policy_net.get_action_and_value(state_tensor)
                action_idx = int(action_idx_tensor.item())

        actions_taken.append(action_idx)

        # 2. Inject action into SNES joypad
        action_vec = ACTION_MATRIX[action_idx]
        action_dict = action_vector_to_dict(action_vec)
        emu.set_input(action_dict)

        # 3. Advance hardware simulation by 1 frame (1/60s)
        emu.step_frame()

    elapsed = time.time() - t0
    final_x = trajectory_x[-1] if trajectory_x else x_init
    max_x = max(trajectory_x) if trajectory_x else x_init
    total_progress = float(final_x - x_init)
    max_progress = float(max_x - x_init)
    mean_vx = float(np.mean(trajectory_vx)) if trajectory_vx else 0.0

    print(
        f"[{policy_name:25s}] Survived: {survived_frames:4d}/{max_frames} frames | "
        f"Progress: {total_progress:+7.1f} px (Max: {max_progress:+7.1f} px) | "
        f"Mean vx: {mean_vx:+5.1f} | "
        f"Latency: {1000.0 * elapsed / max(1, survived_frames):.2f} ms/frame ({survived_frames/max(1e-3, elapsed):.0f} FPS)"
    )

    return {
        "policy_name": policy_name,
        "survived_frames": survived_frames,
        "total_progress": total_progress,
        "max_progress": max_progress,
        "mean_vx": mean_vx,
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "trajectory_vx": trajectory_vx,
        "actions_taken": actions_taken,
    }


def run_random_control_baseline(
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    max_frames: int = 600,
    seed: int = 42,
) -> Dict:
    """Stochastic random action baseline."""
    set_global_seed(seed)
    emu.load_state(initial_savestate)
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()

    trajectory_x = []
    trajectory_y = []
    trajectory_vx = []

    init_state_dict = emu.get_smw_state()
    x_init = init_state_dict["x"]
    survived_frames = 0

    for frame in range(max_frames):
        current_state_dict = emu.get_smw_state()
        curr_vec = extract_state_vector(current_state_dict)

        trajectory_x.append(curr_vec[0])
        trajectory_y.append(curr_vec[1])
        trajectory_vx.append(curr_vec[2])

        if curr_vec[1] > 450.0 or curr_vec[1] < 0.0:
            break

        survived_frames += 1
        rand_idx = np.random.randint(0, len(ACTION_MATRIX))
        action_vec = ACTION_MATRIX[rand_idx]
        emu.set_input(action_vector_to_dict(action_vec))
        emu.step_frame()

    final_x = trajectory_x[-1] if trajectory_x else x_init
    max_x = max(trajectory_x) if trajectory_x else x_init
    total_progress = float(final_x - x_init)
    max_progress = float(max_x - x_init)

    print(
        f"[{'Random_Baseline':25s}] Survived: {survived_frames:4d}/{max_frames} frames | "
        f"Progress: {total_progress:+7.1f} px (Max: {max_progress:+7.1f} px) | "
        f"Mean vx: {np.mean(trajectory_vx):+5.1f}"
    )

    return {
        "policy_name": "Random_Baseline",
        "survived_frames": survived_frames,
        "total_progress": total_progress,
        "max_progress": max_progress,
        "mean_vx": float(np.mean(trajectory_vx)),
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "trajectory_vx": trajectory_vx,
        "actions_taken": [],
    }


def run_zero_shot_model_to_real_benchmark(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    checkpoints_dir: str = "results/checkpoints",
    output_dir: str = "results",
    max_frames: int = 600,
):
    print("====================================================================")
    print("  ZERO-SHOT MODEL-TO-REAL TRANSFER BENCHMARK ON SUPER MARIO WORLD   ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Policy Evaluation Device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    policies_to_evaluate = [
        ("Dyna_PPO_Hard_PINN", "dyna_ppo_pinn_hard_policy.pt"),
        ("Dyna_PPO_Statistical_MLP", "dyna_ppo_mlp_policy.pt"),
    ]

    benchmark_results = {}

    for display_name, ckpt_name in policies_to_evaluate:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Warning: Policy checkpoint {ckpt_path} not found. Skipping.")
            continue

        agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=128).to(device)
        agent.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        res = run_policy_evaluation_trial(
            policy_net=agent,
            emu=emu,
            initial_savestate=initial_savestate,
            device=device,
            max_frames=max_frames,
            policy_name=display_name,
            deterministic=True,
        )
        benchmark_results[display_name] = res

    # Random baseline
    rand_res = run_random_control_baseline(
        emu=emu,
        initial_savestate=initial_savestate,
        max_frames=max_frames,
        seed=42,
    )
    benchmark_results["Random_Baseline"] = rand_res

    emu.close()

    # Save summary JSON
    summary_metrics = {
        name: {
            "survived_frames": r["survived_frames"],
            "total_progress_pixels": r["total_progress"],
            "max_progress_pixels": r["max_progress"],
            "mean_vx": r["mean_vx"],
        }
        for name, r in benchmark_results.items()
    }

    out_json = os.path.join(output_dir, "dyna_ppo_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"\nBenchmark metrics saved to: {out_json}")

    # Generate comparative trajectory figure
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = {
        "Dyna_PPO_Hard_PINN": "#2ecc71",       # Emerald Green
        "Dyna_PPO_Statistical_MLP": "#e74c3c",  # Red
        "Random_Baseline": "#7f8c8d",          # Gray
    }

    # Plot 1: 2D World Trajectory (X vs Y) in SNES Level Space
    for name, r in benchmark_results.items():
        c = colors.get(name, "#3498db")
        ax1.plot(r["trajectory_x"], r["trajectory_y"], label=name, color=c, linewidth=2.5, alpha=0.9)
    ax1.invert_yaxis()
    ax1.set_title("Dyna-PPO Zero-Shot Model-to-Real Trajectory Execution in Super Mario World", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Level Horizontal Coordinate X (Pixels)")
    ax1.set_ylabel("Level Vertical Coordinate Y (Pixels - Inverted)")
    ax1.legend(loc="best", fontsize=10)

    # Plot 2: Forward Progress Over Time (Frames)
    for name, r in benchmark_results.items():
        c = colors.get(name, "#3498db")
        frames = np.arange(len(r["trajectory_x"]))
        prog = np.array(r["trajectory_x"]) - r["trajectory_x"][0]
        ax2.plot(frames, prog, label=name, color=c, linewidth=2.5)
    ax2.set_title("Cumulative Forward Progress on Physical SNES Hardware", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Simulation Frame (60 Hz)")
    ax2.set_ylabel("Forward Displacement ΔX (Pixels)")
    ax2.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "dyna_ppo_snes_trajectories.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Trajectory comparison figure saved to: {fig_path}")

    # Summary table
    print("\n====================================================================")
    print("  DYNA-PPO MODEL-TO-REAL TRANSFER BENCHMARK SUMMARY")
    print("====================================================================")
    print(f"{'Policy Controller':32s} | {'Progress (px)':16s} | {'Mean vx':12s} | {'Frames Alive':12s}")
    print("-" * 80)
    for name, s in summary_metrics.items():
        prog_str = f"{s['total_progress_pixels']:+7.1f} px"
        print(f"{name:32s} | {prog_str:16s} | {s['mean_vx']:+5.1f} subpix | {s['survived_frames']:4d}/{max_frames}")
    print("====================================================================")

    return summary_metrics


if __name__ == "__main__":
    run_zero_shot_model_to_real_benchmark()
