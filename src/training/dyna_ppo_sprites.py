"""
dyna_ppo_sprites.py
End-to-End Multi-Entity Amortized Policy Optimization (Dyna-PPO 12D).
Trains an Actor-Critic policy entirely inside the GPU-vectorized PINN simulation
with Mario kinematics and dynamic hazards (Rex).
Learns autonomous hazard evasion and leap timing end-to-end without hand-crafted rules.
"""

import json
import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.environment.pinn_sim_env import PINNVectorEnv
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.training.dyna_ppo import ActorCritic, DynaPPOTrainer
from src.utils.logging import get_logger
from src.utils.paths import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    results_file,
)

logger = get_logger(__name__)


def generate_multi_entity_initial_pool(
    device: torch.device,
    num_samples: int = 100,
) -> torch.Tensor:
    """
    Generates diverse initial states including Mario at ground level and
    approaching dynamic hazards at varying spatial offsets.
    """
    pool = []
    for _ in range(num_samples):
        # Mario base kinematics: [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
        mario_x = 16.0 + np.random.uniform(0.0, 30.0)
        mario_y = 336.0
        mario_vx = float(np.random.choice([0.0, 8.0, 16.0]))
        mario_vy = 0.0
        c_ground = 1.0
        c_ceiling = 0.0
        c_left = 0.0
        c_right = 0.0

        # Dynamic Hazard: delta_x, delta_y, vx_hazard, active
        active = float(np.random.choice([1.0, 1.0, 1.0, 0.0]))  # 75% active hazard
        if active > 0.5:
            delta_x = np.random.uniform(40.0, 150.0)
            delta_y = 0.0
            vx_hazard = -16.0  # Rex walking left
        else:
            delta_x = 300.0
            delta_y = 0.0
            vx_hazard = 0.0

        vec = [
            mario_x,
            mario_y,
            mario_vx,
            mario_vy,
            c_ground,
            c_ceiling,
            c_left,
            c_right,
            delta_x,
            delta_y,
            vx_hazard,
            active,
        ]
        pool.append(vec)

    return torch.tensor(pool, dtype=torch.float32, device=device)


def train_multi_entity_dyna_ppo(
    total_timesteps: int = 300000,
    num_envs: int = 256,
    num_steps: int = 64,
    checkpoint_dir: str = CHECKPOINTS_DIR,
    figures_dir: str = FIGURES_DIR,
) -> Dict:
    logger.info("====================================================================")
    logger.info("  TRAINING END-TO-END MULTI-ENTITY DYNA-PPO (12D WORLD MODEL)       ")
    logger.info("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} | Parallel Envs: {num_envs}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Instantiate Multi-Entity World Model
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    base_ckpt = os.path.join(checkpoint_dir, "pinn_hard_best.pt")
    if os.path.exists(base_ckpt):
        base_pinn.load_state_dict(torch.load(base_ckpt, map_location=device, weights_only=True))
        logger.info(f"Preloaded base Hard PINN weights from {base_ckpt}.")

    multi_world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)
    multi_world_model.eval()

    # 2. Setup Vectorized Simulation Environment
    init_pool = generate_multi_entity_initial_pool(device=device)
    sim_env = PINNVectorEnv(
        world_model=multi_world_model,
        num_envs=num_envs,
        max_episode_steps=300,
        device=device,
        weight_progress=2.5,
        weight_velocity=0.3,
        alive_bonus=0.05,
        pit_penalty=100.0,
        initial_state_pool=init_pool,
    )

    # 3. Setup 12D Actor-Critic Agent
    agent = ActorCritic(state_dim=12, num_actions=8, hidden_dim=128).to(device)
    trainer = DynaPPOTrainer(
        env=sim_env,
        actor_critic=agent,
        device=device,
        lr=3e-4,
    )

    # 4. Training Loop
    num_updates = total_timesteps // (num_envs * num_steps)
    batch_size = num_envs * num_steps
    logger.info(f"Total Updates: {num_updates} | Transitions per update: {batch_size}")

    obs = sim_env.reset()
    history_returns = []
    history_fps = []
    history_hazard_hits = []

    t_start = time.time()
    best_return = -float("inf")

    for update in range(1, num_updates + 1):
        t0 = time.time()

        obs_buf = torch.zeros((num_steps, num_envs, 12), dtype=torch.float32, device=device)
        act_buf = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device)
        logp_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        rew_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        done_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        val_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        hazard_hits_total = 0

        for step in range(num_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)

            next_obs, rewards, dones, info = sim_env.step(action)

            act_buf[step] = action
            logp_buf[step] = logprob
            rew_buf[step] = rewards
            done_buf[step] = dones.float()
            val_buf[step] = value

            hazard_hits_total += info.get("hazard_hits", 0)
            obs = next_obs

        with torch.no_grad():
            next_value = agent.get_value(next_obs)

        # Compute GAE advantages
        advantages = torch.zeros_like(rew_buf)
        lastgaelam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - done_buf[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - done_buf[t + 1]
                nextvalues = val_buf[t + 1]
            delta = rew_buf[t] + 0.99 * nextvalues * nextnonterminal - val_buf[t]
            advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
        returns = advantages + val_buf

        # Flatten rollout
        b_obs = obs_buf.reshape(-1, 12)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO Update
        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(4):
            np.random.shuffle(b_inds)
            minibatch_size = 256
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_act[mb_inds]
                )
                logratio = newlogprob - b_logp[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]

                mb_adv = b_adv[mb_inds]

                # Policy Loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue - b_ret[mb_inds]) ** 2).mean()

                # Entropy Loss
                ent_loss = entropy.mean()

                loss = pg_loss - 0.01 * ent_loss + 0.5 * v_loss

                trainer.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                trainer.optimizer.step()

        elapsed = time.time() - t0
        fps = int(batch_size / elapsed)
        mean_ret = float(rew_buf.sum(dim=0).mean().item())

        history_returns.append(mean_ret)
        history_fps.append(fps)
        history_hazard_hits.append(hazard_hits_total)

        if mean_ret > best_return:
            best_return = mean_ret
            best_ckpt = os.path.join(checkpoint_dir, "dyna_ppo_multi_entity_best.pt")
            torch.save(agent.state_dict(), best_ckpt)

        if update % 5 == 0 or update == num_updates:
            logger.info(
                f"Update {update:02d}/{num_updates} | "
                f"Mean Return: {mean_ret:+.2f} | "
                f"Hazard Hits: {hazard_hits_total:3d} | "
                f"Throughput: {fps:,} FPS | "
                f"Elapsed: {time.time() - t_start:.1f}s"
            )

    # Save metrics
    metrics = {
        "final_mean_return": float(np.mean(history_returns[-5:])),
        "best_return": float(best_return),
        "mean_fps": float(np.mean(history_fps)),
        "total_training_time_s": float(time.time() - t_start),
        "history_returns": history_returns,
        "history_hazard_hits": history_hazard_hits,
    }
    with open(results_file("dyna_ppo_multi_entity_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot learning curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_returns, color="#10B981", lw=2, label="12D Multi-Entity Policy")
    plt.xlabel("PPO Update")
    plt.ylabel("Mean Episode Return")
    plt.title("Autonomous Hazard Evasion Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_hazard_hits, color="#EF4444", lw=2, label="Hazard Hits per Update")
    plt.xlabel("PPO Update")
    plt.ylabel("Total Collisions")
    plt.title("Hazard Collision Reduction")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dyna_ppo_multi_entity_curve.png"), dpi=300)
    plt.close()

    logger.info("Multi-Entity Dyna-PPO training completed successfully.")
    return metrics


if __name__ == "__main__":
    train_multi_entity_dyna_ppo()
