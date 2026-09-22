"""
dyna_ppo.py
Amortized Policy Optimization (Dyna-PPO / Model-Based Policy Optimization).
Trains an Actor-Critic agent entirely within a learned World Model simulation environment.
Achieves ultra-fast policy learning by leveraging in-GPU vectorized rollouts.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.environment.pinn_sim_env import PINNVectorEnv
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ActorCritic(nn.Module):
    """
    Separate actor and critic networks with LayerNorm input conditioning
    to ensure scale invariance across coordinate and velocity dimensions.
    """

    def __init__(self, state_dim: int = 8, num_actions: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.in_norm = nn.LayerNorm(state_dim)

        # Policy network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

        # State-value function (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(state)
        return self.critic(x).squeeze(-1)

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.in_norm(state)
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x).squeeze(-1)

        return action, log_prob, entropy, value


class DynaPPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer operating over vectorized world model rollouts.
    """

    def __init__(
        self,
        env: PINNVectorEnv,
        actor_critic: ActorCritic,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.env = env
        self.agent = actor_critic.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def collect_rollouts(
        self, rollout_length: int = 128
    ) -> Tuple[torch.Tensor, ...]:
        """
        Collects T-step rollouts across all parallel environments in the world model.
        """
        num_envs = self.env.num_envs
        obs_buf = torch.zeros((rollout_length, num_envs, self.env.states.shape[1]), device=self.device)
        actions_buf = torch.zeros((rollout_length, num_envs), dtype=torch.int64, device=self.device)
        logprobs_buf = torch.zeros((rollout_length, num_envs), device=self.device)
        rewards_buf = torch.zeros((rollout_length, num_envs), device=self.device)
        dones_buf = torch.zeros((rollout_length, num_envs), device=self.device)
        values_buf = torch.zeros((rollout_length, num_envs), device=self.device)

        curr_obs = self.env.states

        for step in range(rollout_length):
            obs_buf[step] = curr_obs
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(curr_obs)

            next_obs, reward, done, _ = self.env.step(action)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            rewards_buf[step] = reward
            dones_buf[step] = done.float()
            values_buf[step] = value

            curr_obs = next_obs

        with torch.no_grad():
            next_value = self.agent.get_value(curr_obs)

        # Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards_buf)
        last_gae = 0.0
        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                next_non_terminal = 1.0 - dones_buf[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones_buf[t + 1]
                next_val = values_buf[t + 1]

            delta = rewards_buf[t] + self.gamma * next_val * next_non_terminal - values_buf[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_buf
        return obs_buf, actions_buf, logprobs_buf, returns, advantages, values_buf

    def update(
        self,
        obs_buf: torch.Tensor,
        actions_buf: torch.Tensor,
        logprobs_buf: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        update_epochs: int = 4,
        mini_batch_size: int = 2048,
    ) -> Dict[str, float]:
        """
        Executes PPO update passes over the flattened buffer.
        """
        b_obs = obs_buf.reshape(-1, self.env.states.shape[1])
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)

        b_size = b_obs.shape[0]
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(update_epochs):
            indices = torch.randperm(b_size, device=self.device)
            for start in range(0, b_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = self.agent.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()

                # Policy Clipped Surrogate Loss
                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Function Loss
                vf_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Entropy regularizer
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += ent_loss.item()
                total_approx_kl += approx_kl.item()
                n_updates += 1

        return {
            "policy_loss": total_pg_loss / max(1, n_updates),
            "value_loss": total_vf_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": total_approx_kl / max(1, n_updates),
        }

    def train(
        self,
        total_timesteps: int = 1_000_000,
        rollout_length: int = 128,
        update_epochs: int = 4,
        mini_batch_size: int = 2048,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Main PPO loop for high-throughput in-GPU policy learning.
        """
        batch_size = rollout_length * self.env.num_envs
        num_iterations = max(1, total_timesteps // batch_size)

        history = []
        t0 = time.time()

        if verbose:
            logger.info(f"Starting Dyna-PPO Training: {num_iterations} iterations ({batch_size} samples/iter)...")

        for iteration in range(1, num_iterations + 1):
            t_iter = time.time()
            obs, actions, logprobs, returns, advs, vals = self.collect_rollouts(rollout_length)
            metrics = self.update(obs, actions, logprobs, returns, advs, update_epochs, mini_batch_size)

            elapsed = time.time() - t_iter
            mean_return = float(returns.mean().item())
            fps = int(batch_size / max(1e-4, elapsed))

            metrics["iteration"] = iteration
            metrics["mean_return"] = mean_return
            metrics["fps"] = fps
            history.append(metrics)

            if verbose and (iteration % 5 == 0 or iteration == 1 or iteration == num_iterations):
                logger.info(
                    f"Iter {iteration:3d}/{num_iterations:3d} | "
                    f"Return: {mean_return:+6.2f} | "
                    f"PolLoss: {metrics['policy_loss']:+.4f} | "
                    f"ValLoss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.3f} | "
                    f"Throughput: {fps:6d} FPS"
                )

        total_time = time.time() - t0
        if verbose:
            logger.info(f"Dyna-PPO finished in {total_time:.2f}s ({int(total_timesteps / total_time)} overall FPS).")

        return history


def train_dyna_ppo_agents(
    checkpoints_dir: str = "results/checkpoints",
    output_dir: str = "results/checkpoints",
    total_timesteps: int = 800_000,
    num_envs: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, ActorCritic]:
    """
    Trains comparative policies inside the Hard PINN World Model vs. Statistical MLP World Model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dyna-PPO Optimization Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Initial state pool from genuine transitions
    initial_states = None
    dataset_path = "data/raw/smw_gameplay_dataset.npz"
    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
        states = data["states"]
        # Use grounded states from initial level positions
        grounded_mask = (states[:, 4] > 0.5) & (states[:, 1] >= 200) & (states[:, 1] <= 400)
        if grounded_mask.any():
            initial_states = torch.tensor(states[grounded_mask][:200], dtype=torch.float32)

    trained_policies = {}

    world_model_configs = [
        ("pinn_hard", HardResidualPINNDynamics(state_dim=8, action_dim=6), "pinn_hard_best.pt"),
        ("mlp", StatisticalMLPDynamics(state_dim=8, action_dim=6), "mlp_best.pt"),
    ]

    for model_name, model, ckpt_name in world_model_configs:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            logger.info(f"Warning: World model checkpoint {ckpt_path} not found. Skipping.")
            continue

        logger.info("\n====================================================================")
        logger.info(f"  TRAINING DYNA-PPO POLICY IN WORLD MODEL: {model_name.upper()}")
        logger.info("====================================================================")

        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        env = PINNVectorEnv(
            world_model=model,
            num_envs=num_envs,
            max_episode_steps=400,
            device=device,
            initial_state_pool=initial_states,
        )

        agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=128).to(device)
        trainer = DynaPPOTrainer(env=env, actor_critic=agent, device=device, lr=3e-4)

        trainer.train(total_timesteps=total_timesteps, rollout_length=128, verbose=True)

        policy_path = os.path.join(output_dir, f"dyna_ppo_{model_name}_policy.pt")
        torch.save(agent.state_dict(), policy_path)
        logger.info(f"Trained policy saved to: {policy_path}")
        trained_policies[model_name] = agent

    return trained_policies


if __name__ == "__main__":
    train_dyna_ppo_agents()
