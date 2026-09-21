"""
test_model_free_ppo.py
Unit tests for SnesSingleEnv environment wrapper and Model-Free PPO iteration.
"""

import os
import pytest
import numpy as np
import torch
from src.training.model_free_ppo import SnesSingleEnv
from src.training.dyna_ppo import ActorCritic


def test_snes_single_env_step():
    core_path = "src/environment/bin/snes9x_libretro.dll"
    rom_path = "data/raw/smw_usa.sfc"
    state_path = "data/raw/smw_yoshi_island_1.state"

    if not os.path.exists(core_path) or not os.path.exists(rom_path) or not os.path.exists(state_path):
        pytest.skip("Emulator binary, ROM, or savestate not found.")

    env = SnesSingleEnv(core_path=core_path, rom_path=rom_path, state_path=state_path, max_episode_steps=50)
    obs = env.reset()

    assert obs.shape == (8,)
    assert isinstance(obs, np.ndarray)

    next_obs, rew, done, info = env.step(action_idx=2)  # RUN_RIGHT

    assert next_obs.shape == (8,)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert "delta_x" in info

    env.close()


def test_actor_critic_model_free_shapes():
    agent = ActorCritic(state_dim=8, num_actions=8, hidden_dim=64)
    obs = torch.randn(4, 8)
    action, logp, entropy, val = agent.get_action_and_value(obs)

    assert action.shape == (4,)
    assert logp.shape == (4,)
    assert entropy.shape == (4,)
    assert val.shape == (4,)
