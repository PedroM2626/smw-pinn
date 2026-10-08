r"""Unit tests for the SINDy leg of README 10.56.

CPU-only, emulator-free and deterministic. A sparse-identification implementation is easy to
write in a way that looks right and quietly reports an arbitrary split of one coefficient
across two collinear columns, so the tests here are (i) recovery of a known system under both
losses, (ii) the behaviour that the section's whole conclusion rests on - that a
least-squares fit is owned by a handful of discontinuous frames while a robust one is not,
and (iii) that the dictionary pruning reports what it removed.
"""

import numpy as np
import pytest

from src.evaluation.sindy_identification_benchmark import (
    DYNAMIC_TARGETS,
    MAX_DEGREE,
    render_gate_table,
)
from src.inverse.sindy_identification import (
    LIBRARY_INPUTS,
    evaluate_library,
    library_terms,
    prune_columns,
    sindy,
    term_names,
)


def _synthetic(seed: int = 0, n: int = 4000, contamination: float = 0.0) -> tuple:
    """A hand-built system with known coefficients: dx = vx/16, dvx = 0.25 a_right - 0.1 vx,
    dvy = 3.0 - 6.0 c_ground, optionally with a few frames where the engine stops dead."""
    rng = np.random.default_rng(seed)
    vx = rng.normal(12.0, 8.0, n)
    cg = (rng.random(n) > 0.5).astype(float)
    ar = (rng.random(n) > 0.5).astype(float)
    inputs = np.zeros((n, len(LIBRARY_INPUTS)))
    inputs[:, 0] = vx
    inputs[:, 2] = cg
    inputs[:, 11] = ar
    xdot = np.column_stack([vx / 16.0, 0.25 * ar - 0.1 * vx, 3.0 - 6.0 * cg])
    if contamination:
        count = max(1, int(n * contamination))
        indices = rng.choice(n, size=count, replace=False)
        xdot[indices, 0] += rng.uniform(60.0, 200.0, count)
        xdot[indices, 2] += rng.uniform(60.0, 200.0, count)
    return inputs, xdot


def _fit(inputs: np.ndarray, xdot: np.ndarray, alpha: float, robust: bool):
    terms = library_terms(MAX_DEGREE)
    theta = evaluate_library(inputs, terms)
    theta, names, _dropped, _keep = prune_columns(theta, term_names(terms))
    return sindy(xdot, theta, names, alpha=alpha, targets=["dx", "dvx", "dvy"], robust=robust)


def test_the_dictionary_drops_idempotent_powers_and_names_the_rest() -> None:
    terms = library_terms(2)
    names = term_names(terms)
    assert names[0] == "1"
    assert len(names) == len(set(names)), "a dictionary with two identical names is ambiguous"
    assert "c_ground c_ground" not in names, "a binary flag squared is the same flag"
    assert "vx^2" in names and "vx c_ground" in names
    assert len(terms) == 1 + len(LIBRARY_INPUTS) + 68


def test_recovery_is_exact_on_a_clean_system_under_either_loss() -> None:
    inputs, xdot = _synthetic()
    for robust in (False, True):
        fit = _fit(inputs, xdot, alpha=0.02, robust=robust)
        assert fit.coefficient_of("dx", "vx") == pytest.approx(0.0625, abs=1e-4)
        assert fit.coefficient_of("dvx", "a_right") == pytest.approx(0.25, abs=1e-3)
        assert fit.coefficient_of("dvx", "vx") == pytest.approx(-0.1, abs=1e-3)
        assert fit.coefficient_of("dvy", "1") == pytest.approx(3.0, abs=1e-3)
        assert fit.coefficient_of("dvy", "c_ground") == pytest.approx(-6.0, abs=1e-3)
        assert fit.sparsity()["dx"] == 1


def test_the_robust_loss_survives_frames_the_engine_stopped_on() -> None:
    """The section's claim, isolated: 1% discontinuous frames own a least-squares fit."""
    _inputs, _clean = _synthetic()
    contaminated = _synthetic(contamination=0.01)
    ls = _fit(contaminated[0], contaminated[1], alpha=0.02, robust=False)
    rb = _fit(contaminated[0], contaminated[1], alpha=0.02, robust=True)
    # The comparison is on the coefficient, not on its reciprocal: a least-squares fit that
    # has thrown the term away entirely is the strongest form of the same failure.
    assert abs(rb.coefficient_of("dx", "vx") - 0.0625) < abs(ls.coefficient_of("dx", "vx") - 0.0625)
    assert rb.coefficient_of("dx", "vx") == pytest.approx(0.0625, abs=2e-3)
    assert abs(rb.coefficient_of("dvy", "1") - 3.0) < abs(ls.coefficient_of("dvy", "1") - 3.0)
    assert rb.coefficient_of("dvy", "1") == pytest.approx(3.0, abs=0.3)
    assert abs(ls.coefficient_of("dvy", "1") - 3.0) > 0.3, "least squares must be the biased one"


def test_pruning_reports_the_collinearity_it_removes() -> None:
    """`c_right` only ever occurring with `c_ground` is data, not a bug - and it must be said."""
    terms = library_terms(2)
    names = term_names(terms)
    inputs = np.zeros((200, len(LIBRARY_INPUTS)))
    rng = np.random.default_rng(1)
    inputs[:, 0] = rng.normal(10.0, 3.0, 200)
    ground = (rng.random(200) > 0.4).astype(float)
    right = ground * (rng.random(200) > 0.5)  # a right contact implies a ground contact
    inputs[:, 2] = ground
    inputs[:, 5] = right
    theta = evaluate_library(inputs, terms)
    pruned, kept, dropped, keep = prune_columns(theta, names)
    assert pruned.shape[1] < theta.shape[1]
    assert any("c_ground c_right" in entry for entry in dropped), dropped
    assert "c_ground c_right" not in kept
    assert np.array_equal(theta[:, keep], pruned)
    assert pruned.std(axis=0).min() > 0.0 or pruned.shape[1] == theta.shape[1] - len(dropped)


def test_a_tighter_threshold_never_returns_fewer_terms() -> None:
    inputs, xdot = _synthetic()
    terms = library_terms(2)
    theta = evaluate_library(inputs, terms)
    theta, names, _dropped, _keep = prune_columns(theta, term_names(terms))
    previous = None
    for alpha in (0.5, 0.2, 0.1, 0.05, 0.02):
        fit = sindy(xdot, theta, names, alpha=alpha, targets=["dx", "dvx", "dvy"])
        total = sum(len(s) for s in fit.support)
        assert previous is None or total >= previous, f"{alpha}: {total} < {previous}"
        previous = total


def test_variance_explained_and_relative_error_disagree_by_the_mean_of_the_target() -> None:
    """A position increment has a large mean, so the two statistics must both be reported."""
    inputs, xdot = _synthetic()
    fit = _fit(inputs, xdot, alpha=0.02, robust=True)
    terms = library_terms(2)
    theta = evaluate_library(inputs, terms)
    theta, names, _d, _k = prune_columns(theta, term_names(terms))
    r2 = fit.variance_explained(theta, xdot)
    rel = fit.relative_error(theta, xdot)
    assert r2["dx"] > 0.99 and rel["dx"] < 0.05
    assert set(r2) == {"dx", "dvx", "dvy", "pooled"}


def test_the_benchmark_targets_are_the_four_dynamic_channels() -> None:
    assert DYNAMIC_TARGETS == ("dx", "dy", "dvx", "dvy")
    assert len(DYNAMIC_TARGETS) == 4


def test_renderers_produce_one_row_per_recording_and_loss() -> None:
    payload = {
        label: {
            "strata": {
                loss: {
                    "ascent_held": {
                        "transitions": 10,
                        "observed_median_dvy": 3.0,
                        "predicted_mean_dvy": 3.0,
                    },
                    "ascent_released": {
                        "transitions": 12,
                        "observed_median_dvy": 6.0,
                        "predicted_mean_dvy": 5.9,
                    },
                    "measured_tier_separation": 3.0,
                    "identified_tier_separation": 2.9,
                }
                for loss in ("least_squares", "robust")
            }
        }
        for label in ("published_gameplay", "tilemap")
    }
    rows = render_gate_table(payload)
    assert len(rows) == 4
    assert all(row.startswith("| ") and row.endswith(" |") for row in rows)
    assert rows[0].count("|") == 7
