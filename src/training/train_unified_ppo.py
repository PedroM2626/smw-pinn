"""
train_unified_ppo.py
Trains an amortized Proximal Policy Optimization (PPO) Actor-Critic agent
inside the GPU-vectorized PINN simulation environment for high-speed,
hazard-aware real-time control (sub-millisecond latency) in place of online MPC.
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
from src.environment.pinn_sim_env import PINNVectorEnv
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.planning.mpc_planner import ACTION_MATRIX, ACTION_PRIMITIVES


class UnifiedActorCritic(nn.Module):
    """Actor-Critic architecture for 12D state and discrete action primitives."""

    def __init__(self, state_dim: int = 12, num_actions: int = 10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return Categorical(logits=logits), value.squeeze(-1)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self(x)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


def generate_initial_state_pool(device: torch.device, n_samples: int = 256) -> torch.Tensor:
    """Generates varied initial conditions with ground contact and hazard distributions."""
    pool = []
    for _ in range(n_samples):
        x = 16.0 + np.random.uniform(0.0, 50.0)
        y = 336.0
        vx = float(np.random.choice([0.0, 8.0, 16.0, 24.0]))
        vy = 0.0
        c_ground = 1.0
        c_ceiling = 0.0
        c_left = 0.0
        c_right = 0.0

        active = float(np.random.choice([0.0, 1.0, 1.0]))
        if active > 0.5:
            delta_x = np.random.uniform(35.0, 180.0)
            delta_y = 0.0
            vx_h = -16.0
        else:
            delta_x = 999.0
            delta_y = 0.0
            vx_h = 0.0

        pool.append([x, y, vx, vy, c_ground, c_ceiling, c_left, c_right, delta_x, delta_y, vx_h, active])

    return torch.tensor(pool, dtype=torch.float32, device=device)


def train_unified_ppo(
    total_timesteps: int = 200000,
    num_envs: int = 128,
    num_steps: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    checkpoint_path: str = "results/checkpoints/unified_ppo_policy_best.pt",
    output_metrics: str = "results/unified_ppo_metrics.json",
    output_figure: str = "results/figures/unified_ppo_learning_curve.png",
) -> Dict:
    print("====================================================================")
    print("  TRAINING UNIFIED DYNA-PPO AMORTIZED CONTROLLER (12D PINN)          ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Parallel Envs: {num_envs} | Steps/Env: {num_steps}")

    # Load World Model for Simulation
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)
    if os.path.exists("results/checkpoints/pinn_multi_entity_best.pt"):
        world_model.load_state_dict(torch.load("results/checkpoints/pinn_multi_entity_best.pt", map_location=device, weights_only=True))
    world_model.eval()

    # Create Vectorized Environment
    init_pool = generate_initial_state_pool(device, n_samples=512)
    env = PINNVectorEnv(
        world_model=world_model,
        num_envs=num_envs,
        max_episode_steps=120,
        device=device,
        initial_state_pool=init_pool,
    )

    agent = UnifiedActorCritic(state_dim=12, num_actions=env.num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    num_updates = total_timesteps // (num_envs * num_steps)
    batch_size = num_envs * num_steps
    minibatch_size = 256

    history_rewards = []
    history_progress = []
    history_losses = []

    t_start = time.time()
    best_return = -1e9

    obs = env.reset()

    for update in range(1, num_updates + 1):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        # Rollout in PINN simulation
        for step in range(num_steps):
            with torch.no_grad():
                action, logp, val = agent.get_action(obs)

            next_obs, reward, done, info = env.step(action)

            # Custom reward shaping for Mario navigation:
            # Reward forward movement and velocity
            delta_x = next_obs[:, 0] - obs[:, 0]
            vx = next_obs[:, 2]
            shaped_rew = delta_x * 2.5 + vx * 0.1

            # Hazard avoidance reward
            delta_xh = next_obs[:, 8].abs()
            delta_yh = next_obs[:, 9].abs()
            is_active = next_obs[:, 11] > 0.5
            hazard_collision = is_active & (delta_xh < 14.0) & (delta_yh < 16.0)
            shaped_rew = shaped_rew - hazard_collision.float() * 150.0

            # Evasive leap bonus (Mario jumping while hazard is close)
            in_air = next_obs[:, 4] < 0.5
            approaching = is_active & (next_obs[:, 8] > 0.0) & (next_obs[:, 8] < 45.0)
            shaped_rew = shaped_rew + (approaching & in_air).float() * 50.0

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(shaped_rew)
            val_buf.append(val)
            done_buf.append(done)

            obs = next_obs

        # GAE Calculation
        with torch.no_grad():
            _, _, next_val = agent.get_action(obs)

        obs_t = torch.stack(obs_buf)        # [num_steps, num_envs, 12]
        act_t = torch.stack(act_buf)        # [num_steps, num_envs]
        logp_t = torch.stack(logp_buf)      # [num_steps, num_envs]
        rew_t = torch.stack(rew_buf)        # [num_steps, num_envs]
        val_t = torch.stack(val_buf)        # [num_steps, num_envs]
        done_t = torch.stack(done_buf)      # [num_steps, num_envs]

        advantages = torch.zeros_like(rew_t)
        last_gae = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - done_t[t].float()
                next_values = next_val
            else:
                next_non_terminal = 1.0 - done_t[t+1].float()
                next_values = val_t[t+1]
            delta = rew_t[t] + gamma * next_values * next_non_terminal - val_t[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        returns = advantages + val_t

        # Flatten batch
        b_obs = obs_t.view(-1, 12)
        b_act = act_t.view(-1)
        b_logp = logp_t.view(-1)
        b_adv = advantages.view(-1)
        b_ret = returns.view(-1)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO Update
        epoch_losses = []
        for epoch in range(4):
            perm = torch.randperm(batch_size)
            for s in range(0, batch_size, minibatch_size):
                idx = perm[s : s + minibatch_size]
                dist, value = agent(b_obs[idx])
                new_logp = dist.log_prob(b_act[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv[idx]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()
                total_loss = pi_loss + 0.5 * v_loss - 0.01 * entropy

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                epoch_losses.append(total_loss.item())

        mean_ret = float(rew_t.sum(dim=0).mean().item())
        mean_prog = float((obs_t[-1, :, 0] - obs_t[0, :, 0]).mean().item())
        history_rewards.append(mean_ret)
        history_progress.append(mean_prog)
        history_losses.append(float(np.mean(epoch_losses)))

        if mean_ret > best_return:
            best_return = mean_ret
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(agent.state_dict(), checkpoint_path)

        if update % 5 == 0 or update == num_updates:
            print(
                f"Update {update:2d}/{num_updates} | "
                f"Episode Return: {mean_ret:6.1f} | "
                f"Mean Progress: {mean_prog:5.1f} px | "
                f"Loss: {np.mean(epoch_losses):6.2f} | "
                f"Throughput: {int(batch_size * update / (time.time() - t_start)):6d} FPS"
            )

    elapsed = time.time() - t_start
    print(f"\nPPO Training Complete in {elapsed:.1f} seconds. Checkpoint saved to: {checkpoint_path}")

    metrics = {
        "total_timesteps": total_timesteps,
        "elapsed_seconds": elapsed,
        "final_mean_return": float(history_rewards[-1]),
        "best_mean_return": float(best_return),
        "final_mean_progress": float(history_progress[-1]),
        "mean_training_throughput_fps": float(total_timesteps / elapsed),
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot Learning Curves
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(range(1, num_updates + 1), history_rewards, color="#1f77b4", linewidth=2.0, label="PPO Episode Return")
    ax1.set_ylabel("Mean Return", fontweight="bold")
    ax1.set_title("Unified Dyna-PPO Training inside Hard PINN Vectorized Simulation", fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right")

    ax2.plot(range(1, num_updates + 1), history_progress, color="#2ca02c", linewidth=2.0, label="Simulated Progress (px)")
    ax2.set_ylabel("Progress (px)", fontweight="bold")
    ax2.set_xlabel("PPO Iteration Updates", fontweight="bold")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    print(f"PPO Learning curve saved to: {output_figure}")

    return metrics


if __name__ == "__main__":
    train_unified_ppo()
