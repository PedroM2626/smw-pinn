"""Regression guard over committed benchmark JSONs.

Fails loudly if a code change silently degrades the published numbers.
Legacy keys are required; newer keys (multistart, per-variable, effect sizes)
are validated only when present so old artifacts still pass.
"""

import json
import os

import pytest

RESULTS = "results"


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present (skip regression in smoke envs)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_benchmark_hard_pinn_regression():
    m = _load("benchmark_metrics.json")
    single = m["single_step_results"]
    roll = m["rollout_metrics"]
    hard = single["Hard_Residual_PINN"]
    mlp = single["Statistical_MLP"]
    # Absolute guardrails from the published benchmark.
    assert hard["test_loss_data"] < 2.0
    assert hard["test_kinematic_error"] < 0.01
    assert hard["test_loss_data"] < mlp["test_loss_data"] / 10.0
    assert roll["Hard_Residual_PINN"]["kinematic_violations"] == 0
    assert roll["Statistical_MLP"]["kinematic_violations"] > 0
    # New keys, when produced by the current code, must be consistent.
    if "rollout_multistart" in m:
        ms = m["rollout_multistart"]["Hard_Residual_PINN"]
        assert ms["kinematic_violation_rate_mean"] == 0.0
        assert ms["mean_drift_mean"] >= 0
    if "per_variable_metrics" in m:
        pv = m["per_variable_metrics"]["Hard_Residual_PINN"]
        assert set(pv) == {"x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right"}
        assert pv["x"]["mse"] < mlp["test_loss_data"]


def test_multiseed_hard_pinn_regression():
    m = _load("multiseed_benchmark_metrics.json")
    stats = m["descriptive_statistics"]
    hard_mse = stats["Hard_Residual_PINN"]["test_mse"]
    assert hard_mse["mean"] < 2.0
    assert hard_mse["std"] < 1.0
    hyp = m["hypothesis_testing"]["Hard_PINN_vs_Statistical_MLP"]["test_mse"]
    assert hyp["p_value_ttest"] < 0.05
    if "cohen_dz" in hyp:
        assert hyp["cohen_dz"] < -1.0  # large effect in favour of Hard PINN


def test_sample_efficiency_monotonic_advantage():
    m = _load("sample_efficiency_metrics.json")
    res = m["results"]
    # Hard PINN at N=200 must beat the MLP at N=5000 (paper claim >25x).
    n_list = m["sample_sizes"]
    i200 = n_list.index(200)
    i5000 = n_list.index(5000)
    assert res["Hard_Residual_PINN"]["test_mse"][i200] < res["Statistical_MLP"]["test_mse"][i5000]
