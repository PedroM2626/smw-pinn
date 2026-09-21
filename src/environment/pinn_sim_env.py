"""
pinn_sim_env.py
High-throughput GPU-vectorized simulation environment powered by a learned World Model.
Simulates hundreds or thousands of parallel environments entirely in PyTorch tensor memory,
providing fast transitions for Dyna-style Model-Based Reinforcement Learning (Dyna-PPO / MBPO).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.planning.mpc_planner import ACTION_MATRIX


class PINNVectorEnv:
    """
    Vectorized environment running in-GPU forward dynamics via f_theta(s, a).
    Eliminates CPU-GPU memory transfer overhead during policy rollouts.
    """

    def __init__(
        self,
        world_model: nn.Module,
        num_envs: int = 512,
        max_episode_steps: int = 400,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weight_progress: float = 2.0,
        weight_velocity: float = 0.2,
        alive_bonus: float = 0.05,
        pit_penalty: float = 100.0,
        death_y: float = 450.0,
        initial_state_pool: Optional[torch.Tensor] = None,
    ):
        self.world_model = world_model.to(device)
        self.world_model.eval()
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.device = device

        self.weight_progress = weight_progress
        self.weight_velocity = weight_velocity
        self.alive_bonus = alive_bonus
        self.pit_penalty = pit_penalty
        self.death_y = death_y

        # Action tensor lookup: [num_actions=8, action_dim=6]
        self.action_tensor = torch.tensor(ACTION_MATRIX, dtype=torch.float32, device=device)
        self.num_actions = self.action_tensor.shape[0]

        # Initial state setup
        if initial_state_pool is not None:
            self.initial_state_pool = initial_state_pool.to(device)
        else:
            # Default level start: Mario at ground level in Yoshi's Island 1
            default_init = torch.tensor(
                [32.0, 350.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
                device=device,
            )
            self.initial_state_pool = default_init.unsqueeze(0)

        # Buffers
        self.states = torch.zeros((num_envs, 8), dtype=torch.float32, device=device)
        self.step_counts = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.total_episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=device)

        self.reset()

    def _sample_initial_states(self, n: int) -> torch.Tensor:
        num_pool = len(self.initial_state_pool)
        indices = torch.randint(0, num_pool, (n,), device=self.device)
        sampled = self.initial_state_pool[indices].clone()
        # Add small initial velocity exploration jitter
        sampled[:, 2] += torch.randn(n, device=self.device) * 0.5
        return sampled

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Resets all environments or a masked subset.
        """
        if env_mask is None:
            self.states = self._sample_initial_states(self.num_envs)
            self.step_counts.zero_()
            self.total_episode_rewards.zero_()
        else:
            n_reset = int(env_mask.sum().item())
            if n_reset > 0:
                self.states[env_mask] = self._sample_initial_states(n_reset)
                self.step_counts[env_mask] = 0
                self.total_episode_rewards[env_mask] = 0.0

        return self.states.clone()

    @torch.no_grad()
    def step(
        self, action_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Executes one vectorized step for all parallel environments.

        Args:
            action_indices: [num_envs] integer tensor with action primitive IDs (0 to 7)

        Returns:
            next_obs: [num_envs, 8]
            rewards: [num_envs]
            dones: [num_envs]
            info: dict with diagnostic metrics
        """
        # 1. Map discrete action indices to 6D Joypad vectors
        actions = self.action_tensor[action_indices]  # [num_envs, 6]

        prev_x = self.states[:, 0]

        # 2. Forward simulation through learned World Model
        next_states = self.world_model(self.states, actions)

        # 3. Compute rewards
        curr_x = next_states[:, 0]
        curr_y = next_states[:, 1]
        curr_vx = next_states[:, 2]

        delta_x = curr_x - prev_x
        # Scaled reward components: horizontal displacement and positive horizontal speed
        r_prog = self.weight_progress * torch.clamp(delta_x, -5.0, 10.0)
        r_vel = self.weight_velocity * (torch.clamp(curr_vx, min=0.0) / 16.0)
        rewards = r_prog + r_vel + self.alive_bonus

        # 4. Check terminal conditions
        fell_in_pit = (curr_y > self.death_y) | (curr_y < 0.0)
        time_limit = self.step_counts >= self.max_episode_steps
        dones = fell_in_pit | time_limit

        # Apply pit death penalty
        rewards = torch.where(fell_in_pit, rewards - self.pit_penalty, rewards)

        # Track stats
        self.step_counts += 1
        self.total_episode_rewards += rewards

        info = {
            "mean_reward": float(rewards.mean().item()),
            "mean_delta_x": float(delta_x.mean().item()),
            "pit_deaths": int(fell_in_pit.sum().item()),
            "time_outs": int(time_limit.sum().item()),
        }

        # 5. Auto-reset terminated environments
        if dones.any():
            next_states = next_states.clone()
            next_states[dones] = self._sample_initial_states(int(dones.sum().item()))
            self.step_counts[dones] = 0
            self.total_episode_rewards[dones] = 0.0

        self.states = next_states
        return self.states.clone(), rewards, dones, info
