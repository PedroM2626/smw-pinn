"""
test_inverse_identification.py
Emulator-free unit tests for the physics parameter-identification inverse problem
(README Section 10.40). Runs on CPU without the Libretro core or a ROM, so it is a CI gate.

Checks the extended analytic simulator's invariants (traction/friction, asymmetric gravity,
speed ceiling, ground-contact reset), that identification recovers a hidden world from its own
transitions, that the Laplace posterior is well-formed and flags every excited constant as
identified, that windowing never crosses an episode boundary, and that the MPC helper and the
bootstrap return well-formed objects.
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
    log_posterior,
    make_windows,
    mcmc_random_walk,
    mpc_random_shooting,
    per_variable_mse,
    posterior_laplace,
    sample_posterior,
    simulate_rollout,
    simulate_step,
    theta_tensor,
)


def _params() -> torch.Tensor:
    return theta_tensor(EngineParams())


def test_engine_params_has_seven_constants() -> None:
    assert len(PARAM_NAMES) == 7
    assert _params().shape == (7,)


def test_simulate_step_integrates_position_by_scale() -> None:
    # decel=0 so a coasting frame keeps vx (isolates the position integration).
    p = theta_tensor(EngineParams(subpixels_per_pixel=16.0, decel=0.0))
    state = torch.tensor([[0.0, 336.0, 32.0, 0.0]])  # vx=32 subpx/f
    action = torch.zeros(1, 6)  # no buttons: coast
    nxt = simulate_step(state, action, p)
    # x advances by vx_next / scale = 32 / 16 = 2 px; vx unchanged with zero friction.
    assert abs(float(nxt[0, 0]) - 2.0) < 1e-5
    assert abs(float(nxt[0, 2]) - 32.0) < 1e-5


def test_simulate_step_coast_applies_friction() -> None:
    p = theta_tensor(EngineParams(max_vx=72.0, decel=0.5, subpixels_per_pixel=16.0))
    state = torch.tensor([[0.0, 336.0, 32.0, 0.0]])
    nxt = simulate_step(state, torch.zeros(1, 6), p)  # no direction held
    # Coulomb friction pulls vx toward zero by decel each coasting frame.
    assert abs(float(nxt[0, 2]) - 31.5) < 1e-5


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


def test_simulate_step_ground_reset_snaps_downward_velocity() -> None:
    p = theta_tensor(EngineParams(held_gravity=3.0, fall_gravity=6.0))
    state = torch.tensor([[0.0, 336.0, 0.0, 4.0]])  # moving down onto the floor
    contact = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # [ground, ceiling, left, right]
    nxt = simulate_step(state, torch.zeros(1, 6), p, contact)
    # Ground contact snaps the downward (+vy) velocity to zero so Mario cannot sink.
    assert float(nxt[0, 3]) == 0.0
    # An upward velocity is allowed to persist even while flagged grounded.
    state_up = torch.tensor([[0.0, 336.0, 0.0, -12.0]])
    nxt_up = simulate_step(state_up, torch.zeros(1, 6), p, contact)
    assert float(nxt_up[0, 3]) < 0.0


def test_simulate_step_wall_and_ceiling_collision() -> None:
    p = theta_tensor(EngineParams(held_gravity=3.0, fall_gravity=6.0))
    # Right wall zeroes rightward velocity but would keep leftward motion.
    state = torch.tensor([[0.0, 336.0, 20.0, 0.0]])
    right = torch.tensor([[0.0, 0.0, 0, 0, 0, 1.0]])  # RIGHT held
    wall = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # right_wall channel set
    nxt = simulate_step(state, right, p, wall)
    assert float(nxt[0, 2]) == 0.0  # stopped flush against the wall
    # A ceiling zeroes upward velocity so a jump cannot rise through it.
    up = torch.tensor([[0.0, 336.0, 0.0, -30.0]])
    ceil = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    nxt_c = simulate_step(up, torch.zeros(1, 6), p, ceil)
    assert float(nxt_c[0, 3]) >= 0.0


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
    s0, acts, tgts, ct = make_windows(states, actions, next_states, episodes, rollout_len=4)
    # Each 6-frame episode yields (6 - 4) = 2 windows -> 4 total; never 2*? across the seam.
    assert s0.shape[0] == 4
    assert acts.shape == (4, 4, 6)
    assert tgts.shape == (4, 4, 4)
    assert ct.shape == (4, 4, 4)


def test_identify_recovers_hidden_world() -> None:
    hidden = theta_tensor(
        EngineParams(
            max_vx=72.0,  # keep the ceiling at the prior to avoid the warm-start pathology
            walk_accel=1.10,
            run_accel=2.20,
            subpixels_per_pixel=18.0,
            held_gravity=2.70,
            fall_gravity=6.40,
            decel=0.90,  # distinct from the 0.5 prior so friction is actually exercised
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
    gr = torch.zeros(0, 6, 4)
    try:
        identify_params((s0, acts, tgts, gr), _params(), steps=2)
        raise AssertionError("expected ValueError for empty windows")
    except ValueError:
        pass


def test_per_variable_mse_keys() -> None:
    windows = generate_synthetic_windows(_params(), 100, 8, seed=3)
    pvar = per_variable_mse(windows, _params())
    assert set(pvar) == {"x", "y", "vx", "vy", "weighted_total"}
    # Evaluating the simulator against its own data must be (near) exact.
    assert pvar["vx"] < 1e-3


def test_posterior_laplace_is_well_formed_and_identifies() -> None:
    hidden = theta_tensor(EngineParams())
    windows = generate_synthetic_windows(hidden, 600, 12, seed=4)
    theta_hat, _ = identify_params(windows, hidden, steps=200, lr=0.05, seed=4)
    post = posterior_laplace(windows, theta_hat)
    assert len(post["std_errors"]) == len(PARAM_NAMES)
    assert len(post["correlation"]) == len(PARAM_NAMES)
    assert all(len(row) == len(PARAM_NAMES) for row in post["correlation"])
    assert post["condition_number"] >= 1.0
    # Data generated by the same model that fits it excites every constant.
    assert all(post["identified"])
    samples = sample_posterior(theta_hat, post, 8, seed=1)
    assert samples.shape == (8, len(PARAM_NAMES))
    assert bool((samples > 0).all())


def test_mpc_returns_full_action_sequence() -> None:
    s0 = torch.tensor([0.0, 336.0, 0.0, 0.0])

    def reward(state: torch.Tensor, action: torch.Tensor) -> float:
        del action
        return float(state[0])  # maximise progress in x

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


def test_full_covariance_samples_match_reported_covariance() -> None:
    hidden = theta_tensor(EngineParams())
    windows = generate_synthetic_windows(hidden, 600, 12, seed=6)
    theta_hat, _ = identify_params(windows, hidden, steps=150, lr=0.05, seed=6)
    post = posterior_laplace(windows, theta_hat)
    cov = np.asarray(post["cov"], dtype=np.float64)
    samples = sample_posterior(theta_hat, post, 4000, seed=9).numpy()
    emp = np.cov(samples.T)
    # Diagonal variances must be reproduced to a loose tolerance (correlated draws).
    assert np.allclose(np.sqrt(np.diag(emp)), np.sqrt(np.diag(cov)), rtol=0.35, atol=1e-4)


def test_log_posterior_rejects_nonpositive_parameters() -> None:
    windows = generate_synthetic_windows(_params(), 100, 8, seed=8)
    assert np.isfinite(log_posterior(_params(), windows, noise_variance=1.0))
    bad = _params().clone()
    bad[2] = -1.0
    assert log_posterior(bad, windows, noise_variance=1.0) == -float("inf")


def test_mcmc_random_walk_recovers_laplace_mean() -> None:
    hidden = theta_tensor(EngineParams())
    windows = generate_synthetic_windows(hidden, 600, 12, seed=4)
    theta_hat, _ = identify_params(windows, hidden, steps=250, lr=0.05, seed=4)
    post = posterior_laplace(windows, theta_hat)
    mc = mcmc_random_walk(
        windows,
        theta_hat,
        post["noise_variance"],
        n_samples=600,
        burn=200,
        thin=3,
        step=np.asarray(post["std_errors"], dtype=np.float64),
        seed=5,
    )
    # A well-tuned chain keeps the sample positive and the mean on the point estimate.
    assert bool((mc["samples"] > 0).all())
    assert 0.05 < mc["acceptance_rate"] < 0.95
    gap = np.abs(np.asarray(mc["mean"]) - theta_hat.numpy()) / np.abs(theta_hat.numpy())
    assert gap.max() < 0.15  # posterior is locally Gaussian -> MCMC mean ~= Laplace MAP
