"""
test_symbolic_regression.py
Emulator-free unit tests for the symbolic-regression inverse method (README Section 10.43).

Covers the transition-bank plumbing, the per-law feature blocks, the analytic wrapper that
lets the 10.40 parametric model be composed and probed through the *same* interface as a
discovered law, the probe stage's exact recovery of a known world, and a tiny-budget genetic
programming fit. Everything runs on CPU with a GP budget of a few seconds, so it is a CI gate.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from src.inverse.parameter_identification import (
    EngineParams,
    generate_synthetic_windows,
    simulate_rollout,
    simulate_step,
    theta_tensor,
)
from src.inverse.symbolic_regression import (
    LAW_FEATURES,
    LAW_NAMES,
    TransitionBank,
    bank_from_transitions,
    bank_from_windows,
    evaluate_law,
    excitation_weights,
    fit_law,
    fit_law_bank,
    law_matrices,
    model_rollout,
    parametric_laws,
    probe_constants,
    relative_error,
    structural_tags,
    summarize_constants,
    symbolic_step,
    truth_vector,
)

# A GP budget of seconds: population x generations x samples dominates the runtime.
TINY_GP: Dict[str, Any] = {
    "population_size": 60,
    "generations": 4,
    "max_train": 300,
    "parsimony_coefficient": 1e-3,
}
HIDDEN = EngineParams(
    max_vx=48.0,
    walk_accel=1.0,
    run_accel=1.8,
    subpixels_per_pixel=20.0,
    held_gravity=2.4,
    fall_gravity=5.2,
    decel=0.6,
)


def _windows(n_windows: int = 120, rollout_len: int = 10, seed: int = 5) -> Any:
    return generate_synthetic_windows(
        theta_tensor(HIDDEN), n_windows=n_windows, rollout_len=rollout_len, seed=seed
    )


def test_bank_from_windows_shapes_and_stride() -> None:
    windows = _windows(n_windows=20, rollout_len=6)
    bank = bank_from_windows(windows)
    assert bank.n == 20 * 6
    assert bank.states.shape == (120, 4)
    assert bank.contact.shape == (120, 4)
    assert set(bank.contact.ravel().tolist()) <= {0.0, 1.0}
    thinned = bank_from_windows(_thinned(windows, 2))
    assert thinned.n == 60
    # Chaining frames through the observed trajectory: within a window, the state of
    # frame t+1 is the next state of frame t, so no fabricated state enters the bank.
    stacked_states = bank.states.reshape(20, 6, 4)
    stacked_next = bank.next_states.reshape(20, 6, 4)
    assert np.allclose(stacked_states[:, 1:, :2], stacked_next[:, :-1, :2])


def _thinned(windows: Any, stride: int) -> Any:
    return tuple(w[::stride] for w in windows)


def test_bank_validates_shapes() -> None:
    with pytest.raises(ValueError):
        TransitionBank(np.zeros((4, 3)), np.zeros((4, 6)), np.zeros((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        TransitionBank(np.zeros((4, 4)), np.zeros((5, 6)), np.zeros((4, 4)), np.zeros((4, 4)))


def test_bank_from_transitions_splits_contact_channels() -> None:
    raw_states = np.zeros((6, 8))
    raw_states[:, 2] = np.arange(6)
    raw_next = raw_states.copy()
    raw_next[:, 4] = 1.0  # grounded
    raw_next[:, 7] = 1.0  # right wall
    bank = bank_from_transitions(raw_states, np.zeros((6, 6)), raw_next)
    assert bank.states.shape == (6, 4)
    assert bank.next_states.shape == (6, 4)
    assert np.allclose(bank.contact[:, 0], 1.0)
    assert np.allclose(bank.contact[:, 3], 1.0)
    assert np.allclose(bank.contact[:, 1], 0.0)


def test_law_matrices_follow_the_feature_preset() -> None:
    bank = bank_from_windows(_windows(n_windows=10, rollout_len=5))
    for preset in ("synthetic", "real", "full"):
        mats = law_matrices(bank, preset)
        assert set(mats) == set(LAW_NAMES)
        for name in LAW_NAMES:
            X, y = mats[name]
            assert X.shape == (bank.n, len(LAW_FEATURES[preset][name]))
        # Velocity laws fit increments, position laws fit displacements.
        assert np.allclose(mats["dvx"][1], bank.next_states[:, 2] - bank.states[:, 2])
        assert np.allclose(mats["dx"][1], bank.next_states[:, 0] - bank.states[:, 0])
    with pytest.raises(ValueError):
        law_matrices(bank, "not-a-preset")


def test_analytic_wrapper_composes_into_the_real_simulator() -> None:
    """The law interface must not change the physics: parity with ``simulate_step``."""
    windows = _windows(n_windows=8, rollout_len=5)
    s0, acts, _, contact = (np.asarray(t, dtype=np.float64) for t in windows)
    wrapped = parametric_laws(truth_vector(HIDDEN), "synthetic")
    state = s0.copy()
    for t in range(acts.shape[1]):
        reference = (
            simulate_step(
                torch.as_tensor(state, dtype=torch.float32),
                torch.as_tensor(acts[:, t, :], dtype=torch.float32),
                theta_tensor(HIDDEN),
                torch.as_tensor(contact[:, t, :], dtype=torch.float32),
            )
            .numpy()
            .astype(np.float64)
        )
        state = symbolic_step(state, acts[:, t, :], contact[:, t, :], wrapped)
        assert np.allclose(state, reference, atol=1e-4)


def test_model_rollout_matches_stepwise_composition() -> None:
    windows = _windows(n_windows=6, rollout_len=4)
    s0, acts, _, contact = (np.asarray(t, dtype=np.float64) for t in windows)
    wrapped = parametric_laws(truth_vector(HIDDEN), "synthetic")
    roll = model_rollout(wrapped, s0, acts, contact)
    assert roll.shape == (6, 4, 4)
    state = s0.copy()
    for t in range(4):
        state = symbolic_step(state, acts[:, t, :], contact[:, t, :], wrapped)
        assert np.allclose(roll[:, t, :], state, atol=1e-9)
    # The same rollout through the library integrator must agree (float32 tolerance).
    ref = simulate_rollout(windows[0], windows[1], theta_tensor(HIDDEN), windows[3]).numpy()
    assert np.abs(roll - ref).max() < 1e-3


def test_probe_constants_recovers_a_known_world_exactly() -> None:
    bank = bank_from_windows(_windows(n_windows=150, rollout_len=12))
    truth = truth_vector(HIDDEN)
    table = relative_error(probe_constants(parametric_laws(truth, "synthetic"), bank), truth)
    for name in (
        "max_vx",
        "walk_accel",
        "run_accel",
        "subpixels_per_pixel",
        "held_gravity",
        "fall_gravity",
        "decel",
    ):
        assert np.isfinite(table[name]["value"]), name
        assert table[name]["rel_error_pct"] < 1.0, f"{name}: {table[name]}"
    assert table["max_vx"]["fixed_point_found"] == 1.0
    assert table["subpixels_per_pixel"]["identity_residual"] < 1e-6


def test_probe_reports_a_missing_rigid_bound_as_missing() -> None:
    """A model with no ceiling must be reported as having no fixed point, not as the
    data maximum - the whole point of probing instead of copying the support."""
    bank = bank_from_windows(_windows(n_windows=40, rollout_len=6))
    no_bound = parametric_laws(truth_vector(HIDDEN), "synthetic")

    class _Linear:
        name = "dvx"
        feature_names = LAW_FEATURES["synthetic"]["dvx"]

        def increment(self, X: np.ndarray) -> np.ndarray:
            X = np.atleast_2d(X)
            return X[:, 1] * 1.0  # a constant drive, never saturating

    no_bound["dvx"] = _Linear()  # type: ignore[assignment]
    table = relative_error(probe_constants(no_bound, bank), truth_vector(HIDDEN))
    assert table["max_vx"]["fixed_point_found"] == 0.0
    assert np.isnan(table["max_vx"]["value"])
    assert table["max_vx"]["overshoot_px_per_frame"] > 0.0


def test_probe_constants_accepts_a_partial_model() -> None:
    """Single-law probing must work: the budget control reads one law at a time."""
    bank = bank_from_windows(_windows(n_windows=40, rollout_len=6))
    truth = truth_vector(HIDDEN)
    laws = parametric_laws(truth, "synthetic")
    horizontal = relative_error(probe_constants({"dvx": laws["dvx"]}, bank), truth)
    assert set(horizontal) >= {"max_vx", "walk_accel", "run_accel", "decel"}
    assert np.isnan(horizontal["subpixels_per_pixel"]["rel_error_pct"])
    assert horizontal["run_accel"]["rel_error_pct"] < 1.0
    vertical = relative_error(probe_constants({"dvy": laws["dvy"]}, bank), truth)
    assert np.isfinite(vertical["held_gravity"]["value"])
    assert "fixed_point_found" not in vertical["max_vx"]
    assert vertical["held_gravity"]["rel_error_pct"] < 1.0


def test_probe_rejects_a_degenerate_zero_drive_as_a_discovered_bound() -> None:
    """A law that never accelerates satisfies v_next <= v everywhere: that is not a ceiling."""
    bank = bank_from_windows(_windows(n_windows=40, rollout_len=6))

    class _Zero:
        name = "dvx"
        feature_names = LAW_FEATURES["synthetic"]["dvx"]

        def increment(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(np.atleast_2d(X).shape[0])

    table = relative_error(probe_constants({"dvx": _Zero()}, bank), truth_vector(HIDDEN))
    assert table["max_vx"]["fixed_point_found"] == 0.0
    assert np.isnan(table["max_vx"]["value"])
    assert table["max_vx"]["drive_response_px_per_frame"] == 0.0


def test_fit_law_reports_the_null_predictor_reference() -> None:
    """hold_r2 is meaningless without the mean-predictor baseline it must beat."""
    bank = bank_from_windows(_windows(n_windows=100, rollout_len=8))
    X, y = law_matrices(bank, "synthetic")["dx"]
    law, _ = fit_law("dx", X, y, LAW_FEATURES["synthetic"]["dx"], seed=0, **TINY_GP)
    assert np.isfinite(law.fit_mean)
    # The discovered identity beats predicting the fit-row mean by orders of magnitude.
    assert law.hold_r2 > law.hold_null_r2 + 0.5


def test_fit_law_recovers_the_kinematic_identity() -> None:
    """The position law is exactly linear in the post-step velocity: GP must find it."""
    bank = bank_from_windows(_windows(n_windows=150, rollout_len=10))
    X, y = law_matrices(bank, "synthetic")["dx"]
    law, hold_idx = fit_law("dx", X, y, LAW_FEATURES["synthetic"]["dx"], seed=0, **TINY_GP)
    assert hold_idx.size > 0
    assert law.hold_r2 > 0.95
    assert law.n_nodes <= 5
    assert structural_tags(law)["is_identity"] or law.n_nodes <= 5
    metrics = evaluate_law(law, X, y)
    assert metrics["mae"] < 0.05 * float(np.abs(y).mean())


def test_fit_law_rejects_an_empty_bank() -> None:
    with pytest.raises(ValueError):
        fit_law("dvx", np.zeros((0, 3)), np.zeros(0), ["vx", "dir", "run"], seed=0, **TINY_GP)


def test_fit_law_bank_and_bagging_average_the_seeds() -> None:
    bank = bank_from_windows(_windows(n_windows=60, rollout_len=8))
    laws = fit_law_bank(bank, "synthetic", seeds=(0, 1), **TINY_GP)
    assert set(laws.laws) == set(LAW_NAMES)
    assert all(len(v) == 2 for v in laws.per_seed.values())
    assert laws.mean_nodes() > 0
    assert laws.total_seconds() > 0
    bagged = laws.bagged()
    X, y = law_matrices(bank, "synthetic")["dvx"]
    mean_of_members = np.mean([law.increment(X[:20]) for law in laws.per_seed["dvx"]], axis=0)
    assert np.allclose(bagged["dvx"].increment(X[:20]), mean_of_members)
    assert evaluate_law(bagged["dvx"], X, y)["mae"] >= 0.0


def test_excitation_weights_target_the_under_excited_branches() -> None:
    bank = bank_from_windows(_windows(n_windows=30, rollout_len=6))
    assert excitation_weights(bank, "dvx", 0.0) is None
    w = excitation_weights(bank, "dvx", 4.0)
    assert w is not None and w.shape == (bank.n,)
    coast = bank.direction == 0
    assert np.all(w[coast] > 1.0) and np.all(w[~coast] == 1.0)
    wv = excitation_weights(bank, "dvy", 2.0)
    assert wv is not None and wv.max() > 1.0
    assert excitation_weights(bank, "dx", 2.0) is None


def test_summarize_constants_reports_recovery_rates() -> None:
    bank = bank_from_windows(_windows(n_windows=50, rollout_len=8))
    truth = truth_vector(HIDDEN)
    tables: List[Dict[str, Dict[str, float]]] = []
    for rep in range(2):
        tables.append(
            relative_error(probe_constants(parametric_laws(truth, "synthetic"), bank), truth)
        )
    summary = summarize_constants(tables)
    for name in ("max_vx", "decel", "held_gravity"):
        assert summary[name]["recovery_rate"] == 1.0
        assert summary[name]["n_recovered"] == 2.0
        assert summary[name]["median_rel_error_pct"] < 1.0


def test_normalisation_uses_only_the_fit_rows() -> None:
    """The excitation scale is data the estimator has; evaluation rows are not its to touch."""
    bank = bank_from_windows(_windows(n_windows=80, rollout_len=8))
    X, y = law_matrices(bank, "synthetic")["dx"]
    law, hold_idx = fit_law("dx", X, y, LAW_FEATURES["synthetic"]["dx"], seed=3, **TINY_GP)
    fit_mask = np.ones(len(y), dtype=bool)
    fit_mask[hold_idx] = False
    assert np.isclose(law.y_scale, float(np.abs(y[fit_mask]).max()))
    assert np.allclose(law.x_scale, np.abs(X[fit_mask]).max(axis=0))
