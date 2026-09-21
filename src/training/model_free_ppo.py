"""
model_free_ppo.py
Canonical Model-Free Reinforcement Learning Baseline:
Trains a standard Actor-Critic Proximal Policy Optimization (PPO) agent directly
on the authentic Super Mario World emulator (Snes9x Libretro via ctypes).

Provides the essential baseline for quantifying the exact Sample Efficiency Multiplier
of Model-Based Reinforcement Learning (Dyna-PPO / Hard PINN) versus pure Model-Free RL.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath("."))
from src.environment.snes_emulator import SnesLibretroEmulator
from src.planning.mpc_planner import ACTION_MATRIX
from src.training.dyna_ppo import ActorCritic


def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {
        "B": bool(vec[0] > 0.5),
        "Y": bool(vec[1] > 0.5),
        "UP": bool(vec[2] > 0.5),
        "DOWN": bool(vec[3] > 0.5),
        "LEFT": bool(vec[4] > 0.5),
        "RIGHT": bool(vec[5] > 0.5),
    }


class SnesSingleEnv:
    """Synchronous single-environment wrapper for fast ctypes SnesLibretroEmulator."""

    def __init__(
        self,
        rom_path: str = "data/raw/smw_usa.sfc",
        core_path: str = "src/environment/bin/snes9x_libretro.dll",
        state_path: str = "data/raw/smw_yoshi_island_1.state",
        max_episode_steps: int = 400,
        weight_progress: float = 2.0,
        weight_velocity: float = 0.2,
        alive_bonus: float = 0.05,
        pit_penalty: float = 100.0,
    ):
        self.emu = SnesLibretroEmulator(core_path)
        self.emu.load_rom(rom_path)
        with open(state_path, "rb") as f:
            self.initial_savestate = f.read()

        self.max_episode_steps = max_episode_steps
        self.weight_progress = weight_progress
        self.weight_velocity = weight_velocity
        self.alive_bonus = alive_bonus
        self.pit_penalty = pit_penalty

        self.step_count = 0
        self.prev_x = 0.0

    def reset(self) -> np.ndarray:
        self.emu.load_state(self.initial_savestate)
        self.emu.wram_buffer[0x0100] = 0x14
        for _ in range(5):
            self.emu.step_frame()

        s = self.emu.get_smw_state()
        self.prev_x = s["x"]
        self.step_count = 0
        return self._extract_obs(s)

    def _extract_obs(self, s: dict) -> np.ndarray:
        return np.array(
            [s["x"], s["y"], s["vx"], s["vy"], s["c_ground"], s["c_ceiling"], s["c_left"], s["c_right"]],
            dtype=np.float32,
        )

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        action_vec = ACTION_MATRIX[action_idx]
        self.emu.set_input(action_vector_to_dict(action_vec))
        self.emu.step_frame()

        s = self.emu.get_smw_state()
        curr_x = s["x"]
        curr_y = s["y"]
        curr_vx = s["vx"]

        delta_x = curr_x - self.prev_x
        self.prev_x = curr_x
        self.step_count += 1

        r_prog = self.weight_progress * np.clip(delta_x, -5.0, 10.0)
        r_vel = self.weight_velocity * (max(0.0, curr_vx) / 16.0)
        reward = float(r_prog + r_vel + self.alive_bonus)

        fell_in_pit = curr_y > 450.0 or curr_y < 0.0 or s["air_state"] == 9
        time_limit = self.step_count >= self.max_episode_steps
        done = bool(fell_in_pit or time_limit)

        if fell_in_pit:
            reward -= self.pit_penalty

        info = {"delta_x": delta_x, "x": curr_x, "fell_in_pit": fell_in_pit}
        return self._extract_obs(s), reward, done, info

    def close(self):
        self.emu.close()


def train_model_free_ppo(
    total_timesteps: int = 50000,
    rollout_length: int = 128,
    num_epochs: int = 4,
    minibatch_size: int = 32,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    output_dir: str = "results",
) -> Dict:
    print("====================================================================")
    print("  TRAINING CANONICAL MODEL-FREE PPO DIRECTLY ON SNES CONSOLE CORE    ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PPO Training Device: {device}")

    env = SnesSingleEnv()
    agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, eps=1e-5)

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    obs = env.reset()
    global_step = 0
    t0 = time.time()

    # Metrics tracking
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    step_history: List[int] = []
    return_history: List[float] = []

    curr_ep_return = 0.0
    curr_ep_length = 0

    num_iterations = total_timesteps // rollout_length

    for iteration in range(1, num_iterations + 1):
        obs_buf = np.zeros((rollout_length, 8), dtype=np.float32)
        act_buf = np.zeros(rollout_length, dtype=np.int64)
        rew_buf = np.zeros(rollout_length, dtype=np.float32)
        done_buf = np.zeros(rollout_length, dtype=np.float32)
        val_buf = np.zeros(rollout_length, dtype=np.float32)
        logp_buf = np.zeros(rollout_length, dtype=np.float32)

        for t in range(rollout_length):
            global_step += 1
            obs_buf[t] = obs

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, _, val = agent.get_action_and_value(obs_t)

            act_buf[t] = action.item()
            val_buf[t] = val.item()
            logp_buf[t] = logp.item()

            next_obs, rew, done, info = env.step(int(action.item()))
            rew_buf[t] = rew
            done_buf[t] = float(done)

            curr_ep_return += rew
            curr_ep_length += 1

            if done:
                episode_returns.append(curr_ep_return)
                episode_lengths.append(curr_ep_length)
                step_history.append(global_step)
                return_history.append(curr_ep_return)
                curr_ep_return = 0.0
                curr_ep_length = 0
                obs = env.reset()
            else:
                obs = next_obs

        # Bootstrap value with GAE
        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_val = agent.get_value(next_obs_t).item()

        advantages = np.zeros(rollout_length, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                next_non_terminal = 1.0 - done_buf[t]
                next_val_t = next_val
            else:
                next_non_terminal = 1.0 - done_buf[t]
                next_val_t = val_buf[t + 1]

            delta = rew_buf[t] + gamma * next_val_t * next_non_terminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + val_buf

        # Convert to tensors
        t_obs = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        t_act = torch.tensor(act_buf, dtype=torch.int64, device=device)
        t_logp = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        t_adv = torch.tensor(advantages, dtype=torch.float32, device=device)
        t_ret = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize advantages
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        # Optimize policy & value network
        dataset_size = rollout_length
        indices = np.arange(dataset_size)
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                _, new_logp, entropy, new_val = agent.get_action_and_value(t_obs[mb_idx], t_act[mb_idx])
                log_ratio = new_logp - t_logp[mb_idx]
                ratio = torch.exp(log_ratio)

                mb_adv = t_adv[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_val - t_ret[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if iteration % 25 == 0 or iteration == num_iterations:
            elapsed = time.time() - t0
            fps = global_step / elapsed
            mean_ep_ret = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            print(
                f"Iter {iteration:3d}/{num_iterations} | Real Steps: {global_step:6d} | "
                f"Mean Ret (last 20): {mean_ep_ret:+6.1f} | FPS: {fps:5.0f} ({elapsed:4.1f}s)"
            )

    env.close()

    # Save policy checkpoint
    ckpt_path = os.path.join(output_dir, "checkpoints", "model_free_ppo_policy.pt")
    torch.save(agent.state_dict(), ckpt_path)
    print(f"\nModel-Free PPO policy saved to: {ckpt_path}")

    # Metrics dictionary
    metrics = {
        "total_real_steps": global_step,
        "training_time_seconds": time.time() - t0,
        "mean_final_return": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
        "total_episodes": len(episode_returns),
        "step_history": step_history,
        "return_history": return_history,
    }

    metrics_path = os.path.join(output_dir, "model_free_ppo_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # Plot Model-Free learning curve
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    if return_history:
        ax.plot(step_history, return_history, alpha=0.3, color="#3498db", label="Episode Return")
        # Moving average
        window = 10
        if len(return_history) >= window:
            rolling = np.convolve(return_history, np.ones(window) / window, mode="valid")
            ax.plot(step_history[window - 1:], rolling, color="#2980b9", linewidth=2.5, label=f"{window}-Episode Moving Avg")
    ax.set_title("Canonical Model-Free PPO Learning Curve on Authentic SNES Console", fontsize=12, fontweight="bold")
    ax.set_xlabel("Real Console Simulation Frames")
    ax.set_ylabel("Episodic Return")
    ax.legend(loc="best")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "figures", "model_free_ppo_learning_curve.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {fig_path}")

    return metrics


if __name__ == "__main__":
    train_model_free_ppo(total_timesteps=40000)
