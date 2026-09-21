"""
test_dyna_ppo.py
Rigorous unit tests for GPU-vectorized PINN simulation environment and Dyna-PPO agent.
"""

import numpy as np
import pytest
import torch
from src.environment.pinn_sim_env import PINNVectorEnv
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics
from src.training.dyna_ppo import ActorCritic, DynaPPOTrainer


def test_pinn_vector_env_shapes_and_step():
    device = torch.device("cpu")
    model = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    num_envs = 16

    env = PINNVectorEnv(
        world_model=model,
        num_envs=num_envs,
        max_episode_steps=50,
        device=device,
    )

    init_obs = env.reset()
    assert init_obs.shape == (num_envs, 8)
    assert not torch.isnan(init_obs).any()

    # Random discrete action indices in [0, 7]
    actions = torch.randint(0, 8, (num_envs,), device=device)
    next_obs, rewards, dones, info = env.step(actions)

    assert next_obs.shape == (num_envs, 8)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    assert dones.dtype == torch.bool
    assert not torch.isnan(next_obs).any()
    assert not torch.isnan(rewards).any()
    assert "mean_reward" in info


def test_actor_critic_forward_and_sampling():
    device = torch.device("cpu")
    agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=64).to(device)

    batch_size = 32
    state = torch.randn(batch_size, 8, device=device)

    action, log_prob, entropy, value = agent.get_action_and_value(state)

    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size,)

    assert (action >= 0).all() and (action < 8).all()
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(value).any()

    # Backpropagation check
    loss = -log_prob.mean() + value.mean()
    loss.backward()

    for p in agent.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_dyna_ppo_training_step():
    device = torch.device("cpu")
    model = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    num_envs = 8
    rollout_len = 8

    env = PINNVectorEnv(world_model=model, num_envs=num_envs, max_episode_steps=20, device=device)
    agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=32).to(device)
    trainer = DynaPPOTrainer(env=env, actor_critic=agent, device=device)

    obs, actions, logprobs, returns, advs, vals = trainer.collect_rollouts(rollout_length=rollout_len)
    assert obs.shape == (rollout_len, num_envs, 8)
    assert actions.shape == (rollout_len, num_envs)
    assert returns.shape == (rollout_len, num_envs)

    metrics = trainer.update(
        obs, actions, logprobs, returns, advs, update_epochs=2, mini_batch_size=32
    )

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert not np.isnan(metrics["policy_loss"])
    assert not np.isnan(metrics["value_loss"])
