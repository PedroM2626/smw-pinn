"""Tests for multi-start rollouts and per-variable metrics."""

import numpy as np
import torch

from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics


def _fake_test_set(n=300):
    rng = np.random.default_rng(3)
    states = rng.uniform(0, 500, size=(n, 8)).astype(np.float32)
    states[:, 4:] = rng.integers(0, 2, size=(n, 4)).astype(np.float32)
    actions = rng.integers(0, 2, size=(n, 6)).astype(np.float32)
    next_states = states + rng.normal(0, 2, size=(n, 8)).astype(np.float32)
    return states, actions, next_states


def test_multistart_shapes_and_stats():
    device = torch.device("cpu")
    model = StatisticalMLPDynamics(hidden_dims=[16])
    states, actions, next_states = _fake_test_set()
    ev = RolloutEvaluator(device=device)
    res = ev.evaluate_rollout_multistart(
        model, "mlp", states, actions, next_states, horizon=20, num_starts=4
    )
    assert res["num_starts"] == 4
    assert len(res["mean_drifts"]) == 4
    assert res["mean_drift_mean"] >= 0
    assert res["mean_drift_std"] >= 0
    assert 0.0 <= res["kinematic_violation_rate_mean"] <= 1.0


def test_multistart_single_start_matches_single_rollout():
    device = torch.device("cpu")
    model = StatisticalMLPDynamics(hidden_dims=[16])
    states, actions, next_states = _fake_test_set()
    ev = RolloutEvaluator(device=device)
    single = ev.evaluate_rollout(model, "mlp", states[0], actions[:10], next_states[:10])
    multi = ev.evaluate_rollout_multistart(
        model, "mlp", states, actions, next_states, horizon=10, num_starts=1
    )
    assert multi["mean_drifts"][0] == single["mean_drift"]


def test_hard_pinn_zero_violation_multistart():
    # Structural guarantee: predicted position always equals input position plus
    # *predicted* velocity / 16, by graph construction, on every rollout step
    # from every start (independent of weights or training).
    model = HardResidualPINNDynamics(hidden_dims=[16, 16])
    model.eval()
    states, actions, _ = _fake_test_set()
    starts = [0, 60, 120, 180, 240]
    with torch.no_grad():
        for s in starts:
            curr = torch.tensor(states[s], dtype=torch.float32).unsqueeze(0)
            for t in range(s, s + 30):
                act = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0)
                nxt = model(curr, act)
                assert torch.allclose(nxt[:, 0], curr[:, 0] + nxt[:, 2] / 16.0, atol=1e-6)
                assert torch.allclose(nxt[:, 1], curr[:, 1] + nxt[:, 3] / 16.0, atol=1e-6)
                curr = nxt


def test_per_variable_perfect_prediction():
    tgt = torch.tensor([[10.0, 20.0, 4.0, -8.0, 1.0, 0.0, 1.0, 0.0]])
    m = compute_per_variable_metrics(tgt.clone(), tgt.clone())
    assert m["x"]["mse"] == 0.0
    assert m["x"]["mae"] == 0.0
    assert m["c_ground"]["accuracy"] == 1.0
    assert m["c_ground"]["f1"] == 1.0


def test_per_variable_splits_scales():
    # Aggregate MSE would be dominated by x; per-variable must expose each channel.
    pred = torch.zeros(50, 8)
    tgt = torch.zeros(50, 8)
    tgt[:, 0] = 1000.0  # huge x error
    tgt[:, 4] = 1.0  # contact error too
    m = compute_per_variable_metrics(pred, tgt)
    assert m["x"]["mse"] > 1e5
    assert m["y"]["mse"] == 0.0
    assert m["c_ground"]["accuracy"] == 0.0


def test_per_variable_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        compute_per_variable_metrics(torch.zeros(10, 4), torch.zeros(10, 4))
