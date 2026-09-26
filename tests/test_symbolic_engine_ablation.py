"""
test_symbolic_engine_ablation.py
Unit tests for the three-engine discovery ablation (README Section 10.43.9).

The study's whole point is a *structural* read-out - was the rigid bound discovered, was
the gravity gate - so these tests pin the read-out itself down: it must accept a planted
clamp, reject an unbounded map, and refuse a degenerate map that never accelerates (which
would otherwise satisfy v_next <= v everywhere and be misread as a bound at zero). The
optional PySR leg is also covered: an unavailable engine must be recorded as unavailable,
never silently dropped from the comparison.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

import src.evaluation.symbolic_engine_ablation_benchmark as study


def test_bound_fixed_point_finds_a_planted_clamp() -> None:
    cap = 48.0
    probe = study._bound_fixed_point(lambda v: np.minimum(v + 1.8, cap), cap)
    assert probe["fixed_point_found"] == 1.0
    assert probe["bound_value"] == pytest.approx(cap, abs=0.6)
    assert probe["overshoot"] == 0.0


def test_bound_fixed_point_rejects_an_unbounded_map() -> None:
    cap = 48.0
    probe = study._bound_fixed_point(lambda v: v + 1.8, cap)
    assert probe["fixed_point_found"] == 0.0
    assert np.isnan(probe["bound_value"])
    assert probe["overshoot"] > 0.0


def test_bound_fixed_point_rejects_a_map_that_never_accelerates() -> None:
    """A zero-drive law satisfies v_next <= v everywhere; that is not a discovered bound."""
    cap = 48.0
    probe = study._bound_fixed_point(lambda v: v.copy(), cap)
    assert probe["fixed_point_found"] == 0.0
    assert np.isnan(probe["bound_value"])


def test_nanmean_ignores_non_finite_and_reports_nan_when_empty() -> None:
    assert study._nanmean([1.0, 2.0, float("nan")]) == 1.5
    assert np.isnan(study._nanmean([float("nan"), float("inf")]))


def test_rel_handles_missing_estimates() -> None:
    """A template that never posited the quantity must not report an error for it."""
    assert np.isnan(study._rel(None, 48.0))
    assert np.isnan(study._rel(float("nan"), 48.0))
    assert study._rel(24.0, 48.0) == 50.0


def test_probe_designs_follow_the_laws_own_feature_names() -> None:
    """The structural probes must address a law by feature name, not by column position."""
    from types import SimpleNamespace

    v = np.array([1.0, 2.0])
    narrow = SimpleNamespace(feature_names=["vx", "dir", "run"])
    X = study._sprint_rows(narrow, v)
    assert X.shape == (2, 3)
    assert np.allclose(X[:, 0], v) and np.allclose(X[:, 1], 1.0) and np.allclose(X[:, 2], 1.0)

    wide = SimpleNamespace(feature_names=["vx", "ground", "right", "dir", "run"])
    X = study._sprint_rows(wide, v)
    assert X.shape == (2, 5)
    assert np.allclose(X[:, 1], 0.0) and np.allclose(X[:, 2], 0.0)

    vy = np.array([-3.0, 4.0])
    vertical = SimpleNamespace(feature_names=["vy", "jump", "asc"])
    Y = study._jump_rows(vertical, vy, held=1.0)
    assert np.allclose(Y[:, 0], vy) and np.allclose(Y[:, 1], 1.0)
    assert np.allclose(Y[:, 2], [1.0, 0.0])  # only the rising frame is ascending


def test_pysr_leg_records_unavailability_rather_than_skipping(monkeypatch) -> None:
    monkeypatch.setattr(study, "pysr_available", lambda: False)
    out = study.run_pysr(
        _tiny_bank(),
        _tiny_bank(),
        np.zeros(7),
        niterations=1,
        max_train=10,
        seed=0,
    )
    assert out["available"] is False
    assert "reason" in out


def _tiny_bank():
    return study.bank_from_transitions(
        np.zeros((3, 4)), np.zeros((3, 6)), np.zeros((3, 4)), np.zeros((3, 4))
    )


_BINDING = {"hidden_world": 0.271, "real_telemetry": 0.0}


def _engine_block(
    r2: float, bound_rate: float, gate_rate: float, bound: float, overshoot: float, tier: float
) -> Dict[str, Any]:
    return {
        "available": True,
        "dvx_heldout_r2_mean": r2,
        "rigid_bound_discovered_rate": bound_rate,
        "bound_selected_by_bic_rate": bound_rate,
        "bound_value_mean": bound,
        "bound_overshoot_px_per_frame_mean": overshoot,
        "gravity_gate_discovered_rate": gate_rate,
        "tier_separation_mean": tier,
    }


def test_bound_reading_only_claims_an_accuracy_advantage_that_was_measured() -> None:
    """The wording has to follow the numbers, in both directions."""
    truth = np.array([48.0, 1.0, 1.8, 20.0, 2.4, 5.2, 0.6])
    templates = _engine_block(1.0, 1.0, 1.0, 48.0, 0.0, -2.8)
    better = {
        "gplearn": _engine_block(0.8, 0.0, 0.0, float("nan"), 1.8, -0.3),
        "pysr": _engine_block(0.99, 0.0, 0.0, float("nan"), 0.03, 0.0),
        "templates": templates,
    }
    reading = study._bound_reading(better, truth, _BINDING)
    assert "while missing the same structure" in reading
    assert "0.990 vs 0.800" in reading
    # The closing clause quotes the measured binding-frame fractions, not a assumed one.
    assert "binds on 27.1% of held-out synthetic frames" in reading
    assert "0.0% of the telemetry test split" in reading

    worse = {
        "gplearn": _engine_block(0.95, 0.0, 0.0, float("nan"), 1.8, -0.3),
        "pysr": _engine_block(0.4, 0.0, 0.0, float("nan"), 0.03, 0.0),
        "templates": templates,
    }
    reading = study._bound_reading(worse, truth, _BINDING)
    assert "cannot separate" in reading
    assert "while missing the same structure" not in reading


def test_bound_reading_falls_back_when_the_structure_is_not_in_the_repertoire() -> None:
    truth = np.array([48.0, 1.0, 1.8, 20.0, 2.4, 5.2, 0.6])
    matrix = {
        "gplearn": _engine_block(0.8, 0.0, 0.0, float("nan"), 1.8, -0.3),
        "pysr": {"available": False, "reason": "not installed"},
        "templates": _engine_block(0.9, 0.0, 0.0, float("nan"), 0.0, -1.0),
    }
    assert "not recovered even as an explicit candidate" in study._bound_reading(
        matrix, truth, _BINDING
    )


def test_real_reading_reports_each_engine_and_the_criterion_split() -> None:
    prior = np.array([72.0, 1.0, 1.8, 20.0, 2.4, 5.2, 0.6])
    real = {
        "gplearn": {
            "dvx_eval_r2": 0.6,
            "structure": {"rigid_bound_discovered": False, "bound_value": float("nan")},
        },
        "pysr": {"available": False},
        "templates": {
            "dvx_eval_r2": 0.99,
            "structure": {"rigid_bound_discovered": True, "bound_value": 47.775},
            "horizontal": {"bic_selected": "H6_contact", "tail_rmse_selected": "H5_coast"},
        },
    }
    text = study._real_reading(real, prior)
    assert "pysr" not in text  # an engine that did not run is not reported as if it had
    assert "47.775" in text and "no bound" in text
    assert "different structures" in text


def test_pysr_availability_is_a_plain_bool() -> None:
    assert isinstance(study.pysr_available(), bool)
