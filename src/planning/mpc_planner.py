"""
mpc_planner.py
Model Predictive Control (MPC) and Trajectory Optimization for Model-Based RL (MBRL).
Evaluates action sequences using a learned World Model dynamics function f(s, a).
Supports GPU-vectorized Random Shooting and Cross-Entropy Method (CEM).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Discrete action primitives for Super Mario World
# Action vector format: [B (jump), Y (run), UP, DOWN, LEFT, RIGHT]
ACTION_PRIMITIVES = {
    "NOOP": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "WALK_RIGHT": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "RUN_RIGHT": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "WALK_JUMP_RIGHT": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "RUN_JUMP_RIGHT": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "JUMP_UP": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "WALK_LEFT": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "RUN_LEFT": np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
}

ACTION_MATRIX = np.stack(list(ACTION_PRIMITIVES.values()), axis=0)  # [num_actions, 6]


class TrajectoryObjective:
    """
    Standard platformer navigation objective:
    - Maximize forward horizontal displacement (progress along level).
    - Reward positive forward horizontal velocity.
    - Heavily penalize falling into pits (Y > death_y) or getting stuck against walls.
    """

    def __init__(
        self,
        weight_progress: float = 2.0,
        weight_velocity: float = 0.5,
        pit_penalty: float = 1000.0,
        death_y: float = 450.0,
        hazard_penalty: float = 600.0,
        leap_bonus: float = 200.0,
    ):
        self.weight_progress = weight_progress
        self.weight_velocity = weight_velocity
        self.pit_penalty = pit_penalty
        self.death_y = death_y
        self.hazard_penalty = hazard_penalty
        self.leap_bonus = leap_bonus

    def compute_trajectory_rewards(
        self,
        initial_states: torch.Tensor,     # [B, state_dim]
        predicted_trajectories: torch.Tensor,  # [B, H, state_dim]
    ) -> torch.Tensor:
        """
        Args:
            initial_states: [B, state_dim]
            predicted_trajectories: [B, H, state_dim]

        Returns:
            rewards: [B] tensor of scalar cumulative trajectory scores
        """
        init_x = initial_states[:, 0]  # [B]
        final_x = predicted_trajectories[:, -1, 0]  # [B]
        all_y = predicted_trajectories[:, :, 1]     # [B, H]
        all_vx = predicted_trajectories[:, :, 2]    # [B, H]

        # 1. Forward horizontal progress
        progress = final_x - init_x

        # 2. Mean forward velocity
        mean_vx = torch.mean(torch.clamp(all_vx, min=0.0), dim=1)

        # 3. Pit fall death penalty: any Y exceeding death_y
        fell_in_pit = (all_y > self.death_y).any(dim=1).float()

        rewards = (
            self.weight_progress * progress
            + self.weight_velocity * mean_vx
            - self.pit_penalty * fell_in_pit
        )

        # 4. Multi-Entity Hazard Collision Avoidance & Leap Optimization (if state_dim >= 12)
        if initial_states.shape[-1] >= 12:
            dx_hazard = predicted_trajectories[:, :, 8]       # [B, H]
            dy_hazard = predicted_trajectories[:, :, 9]       # [B, H]
            active_h = predicted_trajectories[:, :, 11]       # [B, H]

            # Detect fatal collision with active hazard hitbox in any future horizon step
            collided_hazard = (
                (active_h > 0.5)
                & (dx_hazard.abs() < 14.0)
                & (dy_hazard > -10.0)
                & (dy_hazard < 16.0)
            ).any(dim=1).float()

            # Detect clean evasive leap: hazard was in front, is passed horizontally,
            # while Mario leaped above the hazard's vertical collision zone
            passed_hazard = (
                (active_h[:, -1] > 0.5)
                & (initial_states[:, 8] > 0.0)
                & (predicted_trajectories[:, -1, 8] <= 0.0)
                & ((dy_hazard.max(dim=1).values > 16.0) | ((initial_states[:, 1] - all_y.min(dim=1).values) > 16.0))
            ).float()

            rewards = rewards - self.hazard_penalty * collided_hazard + self.leap_bonus * passed_hazard

        return rewards


class ModelPredictiveController:
    """
    GPU-vectorized Model Predictive Control (MPC) planner.
    Performs forward rollouts through the learned dynamics network f(s, a)
    to select the action sequence that maximizes objective reward.
    """

    def __init__(
        self,
        world_model: nn.Module,
        device: torch.device,
        horizon: int = 15,
        num_candidates: int = 256,
        cem_iterations: int = 3,
        elite_ratio: float = 0.1,
        objective: Optional[TrajectoryObjective] = None,
    ):
        self.world_model = world_model.to(device)
        self.world_model.eval()
        self.device = device
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.num_elites = max(2, int(num_candidates * elite_ratio))
        self.objective = objective or TrajectoryObjective()

        # Tensor representation of action primitives [num_actions, 6]
        self.action_tensor = torch.tensor(ACTION_MATRIX, dtype=torch.float32, device=device)
        self.num_actions = len(ACTION_PRIMITIVES)

    @torch.no_grad()
    def plan(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Plans optimal action for the current state using Cross-Entropy Method (CEM).

        Args:
            current_state: 1D numpy array of shape [state_dim]

        Returns:
            best_action: 1D numpy array of shape [action_dim=6]
            info: dictionary with planning diagnostic metrics
        """
        curr_s_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Uniform initial distribution over action primitives for each horizon step: [H, num_actions]
        logits = torch.zeros((self.horizon, self.num_actions), dtype=torch.float32, device=self.device)

        best_reward = -float("inf")
        best_sequence = None
        best_imagined_traj = None

        for it in range(self.cem_iterations):
            probs = torch.softmax(logits, dim=-1)  # [H, num_actions]

            # Sample candidate action sequences: [num_candidates, H]
            dist = torch.distributions.Categorical(probs=probs)
            sampled_indices = dist.sample((self.num_candidates,))  # [num_candidates, H]

            # Convert discrete indices to actual continuous action vectors: [num_candidates, H, action_dim]
            action_candidates = self.action_tensor[sampled_indices]

            # Simulate batch rollouts: [num_candidates, H, state_dim]
            init_batch = curr_s_tensor.expand(self.num_candidates, -1)  # [num_candidates, state_dim]
            sim_traj = self._simulate_batch(init_batch, action_candidates)

            # Evaluate objective rewards
            rewards = self.objective.compute_trajectory_rewards(init_batch, sim_traj)  # [num_candidates]

            # Select top elites
            top_vals, top_indices = torch.topk(rewards, k=self.num_elites)
            elite_action_indices = sampled_indices[top_indices]  # [num_elites, H]

            current_best_idx = top_indices[0]
            if top_vals[0].item() > best_reward:
                best_reward = top_vals[0].item()
                best_sequence = action_candidates[current_best_idx].cpu().numpy()
                best_imagined_traj = sim_traj[current_best_idx].cpu().numpy()

            # Update distribution toward elite actions
            # One-hot representation of elite choices: [num_elites, H, num_actions]
            elite_one_hot = torch.zeros(
                (self.num_elites, self.horizon, self.num_actions),
                device=self.device,
            )
            elite_one_hot.scatter_(2, elite_action_indices.unsqueeze(-1), 1.0)
            elite_mean_probs = elite_one_hot.mean(dim=0)  # [H, num_actions]

            # Smooth probability update (momentum)
            logits = torch.log(torch.clamp(elite_mean_probs, min=1e-6))

        # First action of optimal plan
        best_first_action = best_sequence[0]

        info = {
            "best_reward": best_reward,
            "best_sequence": best_sequence,
            "imagined_trajectory": best_imagined_traj,
            "imagined_final_x": float(best_imagined_traj[-1, 0]),
            "imagined_final_y": float(best_imagined_traj[-1, 1]),
        }
        return best_first_action, info

    def _simulate_batch(
        self,
        initial_states: torch.Tensor,      # [B, state_dim]
        action_sequences: torch.Tensor,    # [B, H, action_dim]
    ) -> torch.Tensor:
        """
        Vectorized forward autoregressive simulation through the neural dynamics model.
        """
        B, H, A = action_sequences.shape
        D = initial_states.shape[1]

        trajectory = torch.zeros((B, H, D), dtype=torch.float32, device=self.device)
        curr_state = initial_states

        for t in range(H):
            curr_action = action_sequences[:, t, :]  # [B, action_dim]
            next_state = self.world_model(curr_state, curr_action)
            trajectory[:, t, :] = next_state
            curr_state = next_state

        return trajectory
