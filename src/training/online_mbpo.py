"""
online_mbpo.py
Full Closed-Loop Model-Based Policy Optimization (MBPO / Dyna-Style Active Learning).

Implements the complete active reinforcement learning cycle:
1. Active Real Interaction: Gathers genuine transitions from the real SNES console (at >2,700 FPS) into D_env.
2. Physics World Model Adaptation: Periodically fine-tunes the Hard Residual PINN on D_env.
3. Branched Model Rollouts: Samples states s ~ D_env and generates k-step imaginary rollouts (k=5)
   inside the in-GPU vectorized PINN simulator, eliminating compounding error.
4. Policy Optimization: Updates the Actor-Critic agent on the generated rollouts via PPO.
5. Continual Iteration: Closes the model-environment loop.
"""

import json
import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from src.environment.pinn_sim_env import PINNVectorEnv
from src.environment.snes_emulator import SnesLibretroEmulator
from src.models import DeepPINNEnsemble, HardResidualPINNDynamics
from src.planning.mpc_planner import ACTION_MATRIX
from src.training.dyna_ppo import ActorCritic, DynaPPOTrainer
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    DATASET_GAMEPLAY,
    RESULTS_DIR,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
)

logger = get_logger(__name__)


class RealReplayBuffer:
    """Circular replay buffer storing authentic console transitions."""

    def __init__(self, capacity: int = 50000, state_dim: int = 8, action_dim: int = 6):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.dones[self.ptr] = float(d)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def sample_states(self, batch_size: int) -> np.ndarray:
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return self.states[idx]

    def __len__(self) -> int:
        return self.size


def train_online_mbpo(
    num_iterations: int = 3,
    real_steps_per_iter: int = 1000,
    model_rollout_steps: int = 40000,
    branch_horizon_k: int = 10,
    output_dir: str = RESULTS_DIR,
    use_safe_ensemble: bool = False,
) -> Dict:
    logger.info("====================================================================")
    if use_safe_ensemble:
        logger.info("  SAFE CLOSED-LOOP MBPO (DEEP ENSEMBLE + EPISTEMIC TRUNCATION)      ")
    else:
        logger.info("  CLOSED-LOOP MODEL-BASED POLICY OPTIMIZATION (ONLINE MBPO)          ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"MBPO Compute Device: {device} | Safe Mode: {use_safe_ensemble}")

    core_path = CORE_PATH
    rom_path = ROM_PATH
    state_path = STATE_YOSHI_ISLAND_1

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # 1. Initialize PINN World Model & Policy
    if use_safe_ensemble:
        world_model = DeepPINNEnsemble(num_models=5, state_dim=8, action_dim=6).to(device)
        ens_dir = os.path.join(output_dir, "checkpoints_ensemble")
        if os.path.exists(ens_dir):
            for i, member in enumerate(world_model.members):
                m_path = os.path.join(ens_dir, f"ensemble_pinn_seed_{42 + i * 17}.pt")
                if os.path.exists(m_path):
                    member.load_state_dict(
                        torch.load(m_path, map_location=device, weights_only=True)
                    )
            logger.info("Preloaded Deep Ensemble member weights.")
        model_optimizers = [
            torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            for m in world_model.members
        ]
    else:
        world_model = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
        pinn_best_path = os.path.join(output_dir, "checkpoints", "pinn_hard_best.pt")
        if os.path.exists(pinn_best_path):
            world_model.load_state_dict(
                torch.load(pinn_best_path, map_location=device, weights_only=True)
            )
            logger.info("Preloaded base Hard Residual PINN dynamics checkpoint.")
        model_optimizers = [torch.optim.AdamW(world_model.parameters(), lr=1e-3, weight_decay=1e-4)]

    policy_agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=128).to(device)

    # 2. Replay Buffer
    replay_buffer = RealReplayBuffer(capacity=50000)

    # Pre-populate buffer with initial offline dataset
    raw_dataset = np.load(DATASET_GAMEPLAY)
    init_s, init_a, init_ns = (
        raw_dataset["states"][:2000],
        raw_dataset["actions"][:2000],
        raw_dataset["next_states"][:2000],
    )
    for i in range(len(init_s)):
        replay_buffer.add(init_s[i], init_a[i], 0.0, init_ns[i], False)
    logger.info(f"Replay buffer seeded with {len(replay_buffer)} authentic transitions.")

    # 3. Setup Emulator for real interactions
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    iter_progress_history = []
    iter_survival_history = []
    t0 = time.time()

    for it in range(1, num_iterations + 1):
        logger.info(f"\n--- MBPO Iteration {it}/{num_iterations} ---")

        # Step A: Collect real transitions with current policy in authentic SNES emulator
        emu.load_state(initial_savestate)
        emu.enable_gameplay_mode()
        for _ in range(5):
            emu.step_frame()

        s_dict = emu.get_smw_state()
        curr_s = np.array(
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
        x_start = curr_s[0]
        max_x = curr_s[0]
        survived_frames = 0

        for step in range(real_steps_per_iter):
            curr_s_t = torch.tensor(curr_s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_idx = int(
                    torch.argmax(policy_agent.actor(policy_agent.in_norm(curr_s_t)), dim=-1).item()
                )

            action_vec = ACTION_MATRIX[action_idx]
            from src.evaluation.evaluate_policy_snes import action_vector_to_dict

            emu.set_input(action_vector_to_dict(action_vec))
            emu.step_frame()

            ns_dict = emu.get_smw_state()
            next_s = np.array(
                [
                    ns_dict["x"],
                    ns_dict["y"],
                    ns_dict["vx"],
                    ns_dict["vy"],
                    ns_dict["c_ground"],
                    ns_dict["c_ceiling"],
                    ns_dict["c_left"],
                    ns_dict["c_right"],
                ],
                dtype=np.float32,
            )

            fell_in_pit = next_s[1] > 450.0 or next_s[1] < 0.0 or ns_dict["air_state"] == 9
            delta_x = next_s[0] - curr_s[0]
            reward = float(
                2.0 * np.clip(delta_x, -5.0, 10.0) + 0.2 * (max(0.0, next_s[2]) / 16.0) + 0.05
            )
            if fell_in_pit:
                reward -= 100.0

            replay_buffer.add(curr_s, action_vec, reward, next_s, fell_in_pit)
            max_x = max(max_x, next_s[0])
            survived_frames += 1

            if fell_in_pit:
                emu.load_state(initial_savestate)
                emu.enable_gameplay_mode()
                for _ in range(5):
                    emu.step_frame()
                ns_dict = emu.get_smw_state()
                next_s = np.array(
                    [
                        ns_dict["x"],
                        ns_dict["y"],
                        ns_dict["vx"],
                        ns_dict["vy"],
                        ns_dict["c_ground"],
                        ns_dict["c_ceiling"],
                        ns_dict["c_left"],
                        ns_dict["c_right"],
                    ],
                    dtype=np.float32,
                )

            curr_s = next_s

        iter_progress = float(max_x - x_start)
        iter_progress_history.append(iter_progress)
        iter_survival_history.append(survived_frames)
        logger.info(
            f"  Real interaction completed: {real_steps_per_iter} frames | Buffer Size: {len(replay_buffer)}"
        )
        logger.info(f"  Max Real Console Progress: {iter_progress:+6.1f} px")

        # Step B: Fine-tune World Model on newly collected real buffer data
        world_model.train()
        criterion = nn.MSELoss()
        for _ in range(30):  # mini-batches
            b_s, b_a, _, b_ns, _ = replay_buffer.sample(batch_size=128)
            b_s_t = torch.tensor(b_s, dtype=torch.float32, device=device)
            b_a_t = torch.tensor(b_a, dtype=torch.float32, device=device)
            b_ns_t = torch.tensor(b_ns, dtype=torch.float32, device=device)

            if use_safe_ensemble:
                for member, opt in zip(world_model.members, model_optimizers):
                    opt.zero_grad()
                    pred = member(b_s_t, b_a_t)
                    loss = criterion(pred, b_ns_t)
                    loss.backward()
                    opt.step()
            else:
                model_optimizers[0].zero_grad()
                pred = world_model(b_s_t, b_a_t)
                loss = criterion(pred, b_ns_t)
                loss.backward()
                model_optimizers[0].step()

        # Step C: Branched Rollout Simulation & Policy Optimization
        # Sample starting states from buffer to eliminate compounding error
        sampled_initials = replay_buffer.sample_states(batch_size=512)
        initial_pool = torch.tensor(sampled_initials, dtype=torch.float32, device=device)

        sim_env = PINNVectorEnv(
            world_model=world_model,
            num_envs=512,
            max_episode_steps=branch_horizon_k,
            initial_state_pool=initial_pool,
            device=device,
            pessimism_beta=0.5 if use_safe_ensemble else 0.0,
            uncertainty_truncation_threshold=1.2 if use_safe_ensemble else 999.0,
        )

        trainer = DynaPPOTrainer(env=sim_env, actor_critic=policy_agent, device=device)
        trainer.train(total_timesteps=model_rollout_steps)
        logger.info(
            f"  Imagined Policy Optimization completed ({model_rollout_steps} transitions via branched PINN rollouts)."
        )

    emu.close()

    # Save final MBPO policy
    suffix = "_safe" if use_safe_ensemble else ""
    mbpo_policy_path = os.path.join(output_dir, "checkpoints", f"online_mbpo{suffix}_policy.pt")
    torch.save(policy_agent.state_dict(), mbpo_policy_path)
    logger.info(f"\nFinal MBPO Policy saved to: {mbpo_policy_path}")

    # Metrics
    metrics = {
        "num_iterations": num_iterations,
        "total_real_frames": num_iterations * real_steps_per_iter,
        "progress_per_iteration": iter_progress_history,
        "training_time_seconds": time.time() - t0,
        "is_safe_ensemble": use_safe_ensemble,
    }

    metrics_filename = f"online_mbpo{suffix}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"MBPO metrics saved to: {metrics_path}")

    # Generate MBPO convergence plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    iters = np.arange(1, num_iterations + 1)
    plot_color = "#3B82F6" if use_safe_ensemble else "#2ecc71"
    plot_label = (
        "Safe MBPO (Deep Ensemble E=5)" if use_safe_ensemble else "Standard MBPO (Hard PINN)"
    )
    ax.plot(
        iters, iter_progress_history, marker="o", color=plot_color, linewidth=2.5, label=plot_label
    )
    ax.set_title(
        f"Online MBPO Progress Across Iterations (SNES Console - {plot_label})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("MBPO Iteration (Real Interaction + Branched PINN Rollout)")
    ax.set_ylabel("Real Console Max Progress (Pixels)")
    ax.set_xticks(iters)
    ax.legend(loc="best")
    plt.tight_layout()

    fig_filename = f"online_mbpo{suffix}_convergence.png"
    fig_path = os.path.join(output_dir, "figures", fig_filename)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Convergence plot saved to: {fig_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Online MBPO Training")
    parser.add_argument(
        "--safe", action="store_true", help="Enable Safe MBRL with Deep PINN Ensemble"
    )
    parser.add_argument("--iterations", type=int, default=3, help="Number of MBPO iterations")
    parser.add_argument("--real_steps", type=int, default=1000, help="Real steps per iteration")
    parser.add_argument("--rollout_steps", type=int, default=40000, help="Imagined rollout steps")
    args = parser.parse_args()

    train_online_mbpo(
        num_iterations=args.iterations,
        real_steps_per_iter=args.real_steps,
        model_rollout_steps=args.rollout_steps,
        use_safe_ensemble=args.safe,
    )
