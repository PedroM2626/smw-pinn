"""
test_multi_entity.py
Unit tests verifying Multi-Entity 12D PINN dynamics, analytical relative kinematic
consistency, and vectorized simulation collision handling.
"""

import torch

from src.environment.pinn_sim_env import PINNVectorEnv
from src.models.pinn_multi_entity import MultiEntityPINNDynamics


def test_multi_entity_forward_shapes():
    model = MultiEntityPINNDynamics(state_dim=12, action_dim=6)
    states = torch.zeros(4, 12)
    actions = torch.zeros(4, 6)
    next_states = model(states, actions)
    assert next_states.shape == (4, 12)


def test_multi_entity_exact_relative_kinematics():
    """
    Verifies that MultiEntityPINNDynamics maintains exact analytical kinematic
    consistency for both Mario and the relative hazard position:
    Delta X_h(t+1) - Delta X_h(t) == (vx_h - vx_m) / 16.0
    """
    model = MultiEntityPINNDynamics(state_dim=12, action_dim=6)
    # Mario at (100, 300) with vx=16, vy=0
    # Hazard at offset dx=64, dy=0, vx_h=-16, active=1.0
    state = torch.tensor(
        [[100.0, 300.0, 16.0, 0.0, 1.0, 0.0, 0.0, 0.0, 64.0, 0.0, -16.0, 1.0]],
        dtype=torch.float32,
    )
    action = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    next_state = model(state, action)

    # Mario displacement
    m_dx = next_state[0, 0] - state[0, 0]
    m_expected_dx = next_state[0, 2] / 16.0
    assert torch.isclose(m_dx, m_expected_dx, atol=1e-5)

    # Relative hazard displacement
    h_ddx = next_state[0, 8] - state[0, 8]
    expected_relative_vx = next_state[0, 10] - next_state[0, 2]
    h_expected_ddx = expected_relative_vx / 16.0
    assert torch.isclose(h_ddx, h_expected_ddx, atol=1e-5)


def test_pinn_vector_env_12d_step():
    model = MultiEntityPINNDynamics(state_dim=12, action_dim=6)
    init_pool = torch.tensor(
        [[32.0, 350.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0, 0.0, -16.0, 1.0]],
        dtype=torch.float32,
    )
    env = PINNVectorEnv(
        world_model=model,
        num_envs=16,
        max_episode_steps=50,
        initial_state_pool=init_pool,
        device=torch.device("cpu"),
    )
    obs = env.reset()
    assert obs.shape == (16, 12)

    actions = torch.randint(0, 8, (16,))
    next_obs, rewards, dones, info = env.step(actions)
    assert next_obs.shape == (16, 12)
    assert rewards.shape == (16,)
    assert dones.shape == (16,)
    assert "hazard_hits" in info
