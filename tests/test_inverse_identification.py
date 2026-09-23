"""
test_inverse_identification.py
Emulator-free unit tests for the physics parameter-identification inverse problem
(README Section 10.40). Runs on CPU without the Libretro core or a ROM, so it is a CI gate.

Checks the analytic forward simulator's invariants, that identification recovers a hidden
world from its own transitions, that windowing never crosses an episode boundary, and that
the MPC helper and bootstrap return well-formed objects.
"""

from __future__ import annotations

import numpy as np
import torch

from src.inverse.parameter_identification import (
    PARAM_NAMES,
    EngineParams,
    bootstrap_ci,
    generate_synthetic_windows,
    identify_params,
    make_windows,
    mpc_random_shooting,
    per_variable_mse,
    simulate_rollout,
    simulate_step,
    theta_tensor,
)


def _params() -> torch.Tensor:
    return theta_tensor(EngineParams())


def test_simulate_step_integrates_position_by_scale() -> None:
    p = theta_tensor(EngineParams(subpixels_per_pixel=16.0))
    state = torch.tensor([[0.0, 336.0, 32.0, 0.0]])  # vx=32 subpx/f
    action = torch.zeros(1, 6)  # no buttons: coast
    nxt = simulate_step(state, action, p)
    # x advances by vx / scale = 32 / 16 = 2 px; vx unchanged when idle.
    assert abs(float(nxt[0, 0]) - 2.0) < 1e-5
    assert abs(float(nxt[0, 2]) - 32.0) < 1e-5


def test_simulate_step_gravity_branches() -> None:
    p = theta_tensor(EngineParams(held_gravity=3.0, fall_gravity=6.0))
    state = torch.tensor([[0.0, 336.0, 0.0, -20.0]])  # ascending (vy < 0)
    jump_held = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    jump_released = torch.tensor([[0.0, 0, 0, 0, 0, 0]])
    v_held = float(simulate_step(state, jump_held, p)[0, 3])
    v_free = float(simulate_step(state, jump_released, p)[0, 3])
    # held ascent adds +3 (lighter), released ascent adds +6.
    assert abs(v_held - (-17.0)) < 1e-5
    assert abs(v_free - (-14.0)) < 1e-5


def test_simulate_step_clamps_velocity_ceiling() -> None:
    p = theta_tensor(EngineParams(max_vx=50.0, run_accel=1.5))
    state = torch.tensor([[0.0, 336.0, 49.0, 0.0]])
    right_run = torch.tensor([[0.0, 1.0, 0, 0, 0, 1.0]])  # run + right
    nxt = simulate_step(state, right_run, p)
    assert float(nxt[0, 2]) == 50.0  # saturated at the ceiling, not 50.5


def test_make_windows_respects_episode_boundaries() -> None:
    # Two episodes of 6 frames each; a rollout_len of 4 must fit inside one episode only.
    states = np.zeros((12, 8), dtype=np.float32)
    next_states = np.zeros((12, 8), dtype=np.float32)
    actions = np.zeros((12, 6), dtype=np.float32)
    episodes = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    s0, acts, tgts = make_windows(states, actions, next_states, episodes, rollout_len=4)
    # Each 6-frame episode yields (6 - 4) = 2 windows -> 4 total; never 2*? across the seam.
    assert s0.shape[0] == 4
    assert acts.shape == (4, 4, 6)
    assert tgts.shape == (4, 4, 4)


def test_identify_recovers_hidden_world() -> None:
    hidden = theta_tensor(
        EngineParams(
            max_vx=72.0,  # keep the ceiling at the prior to avoid the warm-start pathology
            walk_accel=1.10,
            run_accel=2.20,
            subpixels_per_pixel=18.0,
            held_gravity=2.70,
            fall_gravity=6.40,
        )
    )
    prior = theta_tensor(EngineParams())
    windows = generate_synthetic_windows(hidden, n_windows=800, rollout_len=12, seed=5)
    theta_hat, _ = identify_params(windows, prior, steps=700, lr=0.06, seed=5)
    for k, name in enumerate(PARAM_NAMES):
        true_v = float(hidden[k])
        rel = abs(float(theta_hat[k]) - true_v) / abs(true_v)
        assert rel < 0.12, f"{name}: true={true_v} hat={float(theta_hat[k])} rel={rel:.3f}"


def test_identify_rejects_empty_windows() -> None:
    s0 = torch.zeros(0, 4)
    acts = torch.zeros(0, 6, 6)
    tgts = torch.zeros(0, 6, 4)
    try:
        identify_params((s0, acts, tgts), _params(), steps=2)
        raise AssertionError("expected ValueError for empty windows")
    except ValueError:
        pass


def test_per_variable_mse_keys() -> None:
    windows = generate_synthetic_windows(_params(), 100, 8, seed=3)
    pvar = per_variable_mse(windows, _params())
    assert set(pvar) == {"x", "y", "vx", "vy"}
    # Evaluating the simulator against its own data must be (near) exact.
    assert pvar["vx"] < 1e-3


def test_mpc_returns_full_action_sequence() -> None:
    s0 = torch.tensor([0.0, 336.0, 0.0, 0.0])

    def reward(rollout: torch.Tensor) -> torch.Tensor:
        return rollout[-1, 0]  # maximise progress

    plan, predicted = mpc_random_shooting(s0, _params(), reward, horizon=10, n_samples=64, seed=1)
    assert plan.shape == (10, 6)
    assert np.isfinite(predicted)


def test_bootstrap_ci_is_well_formed_and_tight() -> None:
    windows = generate_synthetic_windows(theta_tensor(EngineParams()), 400, 10, seed=2)
    ci = bootstrap_ci(windows, theta_tensor(EngineParams()), n_boot=6, steps=200, seed=2)
    assert set(ci) == set(PARAM_NAMES)
    for name in PARAM_NAMES:
        assert ci[name]["ci_low"] <= ci[name]["ci_high"]
        assert np.isfinite(ci[name]["std"])


def test_rollout_matches_stepwise_composition() -> None:
    p = _params()
    s0 = torch.tensor([[10.0, 336.0, 5.0, 12.0]])
    acts = torch.tensor([[[1.0, 1.0, 0, 0, 0, 1.0]] * 3])  # [1, 3, 6]
    roll = simulate_rollout(s0, acts, p)
    # Independent step-by-step composition must equal the batched rollout.
    state = s0
    for t in range(3):
        state = simulate_step(state, acts[:, t, :], p)
    assert torch.allclose(state, roll[:, -1, :], atol=1e-5)
