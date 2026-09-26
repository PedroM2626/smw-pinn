"""
test_symbolic_tilemap_residual.py
Unit tests for the tilemap-conditioned residual control (README Section 10.43.8).

Emulator-free and CPU-only. Covers the terrain descriptors and their orientation, the
five conditioning designs (including the shuffled-geometry placebo), the estimator-free
correlation diagnostic, the least-squares control, the split/patch alignment against the
canonical loader, and - most importantly - the verdict rule, which must refuse to call a
"less catastrophic" negative R^2 an explanation and must refuse to credit a gain the
placebo also achieves.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.evaluation.symbolic_tilemap_residual_benchmark import (
    CHANNELS,
    GEOM_FEATURES,
    STATE_FEATURES,
    _ols_r2,
    build_verdict,
    condition_designs,
    geometry_descriptors,
    interaction_block,
    load_tilemap_splits,
    signal_diagnostics,
    state_block,
)
from src.inverse.symbolic_regression import TransitionBank
from src.utils.paths import DATASET_TILEMAP

N = 24


def _bank() -> TransitionBank:
    rng = np.random.RandomState(0)
    states = rng.uniform(-10, 10, (N, 4))
    actions = np.zeros((N, 6))
    actions[np.arange(N) % 3 == 0, 5] = 1.0  # RIGHT
    actions[np.arange(N) % 5 == 0, 4] = 1.0  # LEFT
    actions[np.arange(N) % 4 == 0, 1] = 1.0  # RUN
    actions[np.arange(N) % 7 == 0, 0] = 1.0  # JUMP
    next_states = states + rng.uniform(-1, 1, (N, 4))
    contact = (rng.uniform(0, 1, (N, 4)) > 0.7).astype(np.float64)
    return TransitionBank(states, actions, next_states, contact)


def _patches() -> np.ndarray:
    """A 7x7 stack whose solid cells sit at known compass positions and vary across frames.

    Variation matters: a constant terrain column has no correlation to compute, so a
    degenerate stack would let a broken descriptor pass the diagnostic tests.
    """
    p = np.zeros((N, 7, 7), dtype=np.int64)
    idx = np.arange(N)
    p[idx % 2 == 0, 5:7, 2:5] = 1  # ground below Mario, every other frame
    p[idx % 3 == 0, 2:5, 6] = 1  # wall to his right
    p[idx % 4 == 0, 0:2, 2:5] = 2  # ceiling above
    p[idx % 5 == 0, 3, 4] = 3  # a slope tile beside him
    p[idx % 7 == 0, 6, 6] = 1  # an isolated corner block
    return p


def test_geometry_descriptors_read_the_right_cells() -> None:
    geom = geometry_descriptors(_patches())
    assert geom.shape == (N, len(GEOM_FEATURES))
    by_name = {name: geom[:, j] for j, name in enumerate(GEOM_FEATURES)}
    even = np.arange(N) % 2 == 0
    assert np.all(by_name["solid_below"][even] == 1.0)
    assert np.all(by_name["solid_below"][~even] == 0.0)
    assert np.all(by_name["solid_ahead_right"][np.arange(N) % 3 == 0] > 0.0)
    assert np.all(by_name["solid_ahead_left"] == 0.0)  # no wall was placed to his left
    assert np.all(by_name["ceiling_above"][np.arange(N) % 4 == 0] == 1.0)
    assert np.all(by_name["center_solid"] == 0.0)  # the center cell stays air
    assert np.all(by_name["slope_fraction"][np.arange(N) % 5 == 0] > 0.0)
    assert 0.0 < by_name["patch_fill"].mean() < 1.0


def test_slope_class_is_distinguished_from_solid() -> None:
    """`solid` counts class 3 as terrain, `slope_fraction` isolates it."""
    geom = geometry_descriptors(_patches())
    slope = geom[:, GEOM_FEATURES.index("slope_fraction")]
    assert slope.min() == 0.0 and slope.max() > 0.0
    # Every frame with a slope tile also registers as solid somewhere in the patch.
    assert np.all(geom[:, GEOM_FEATURES.index("patch_fill")][slope > 0] > 0.0)


def test_state_block_column_order() -> None:
    bank = _bank()
    X = state_block(bank)
    assert X.shape == (N, len(STATE_FEATURES))
    assert np.allclose(X[:, STATE_FEATURES.index("vx")], bank.states[:, 2])
    assert np.allclose(X[:, STATE_FEATURES.index("dir")], bank.direction)
    assert np.allclose(X[:, STATE_FEATURES.index("ground")], bank.contact[:, 0])


def test_interaction_block_picks_the_facing_side() -> None:
    bank = _bank()
    geom = geometry_descriptors(_patches())
    inter = interaction_block(bank, geom)
    assert inter.shape == (N, 4)
    facing_right = bank.direction > 0
    if facing_right.any():
        assert np.allclose(inter[facing_right, 0], (bank.direction * geom[:, 1])[facing_right])
    # No wall was placed to the left, so the left-facing product must vanish.
    assert np.allclose(inter[:, 1], 0.0)
    # The facing-side product follows the direction sign, not a fixed column.
    assert np.allclose(
        inter[:, 2], (bank.states[:, 2] * np.where(bank.direction >= 0, geom[:, 1], geom[:, 2]))
    )


def test_condition_designs_shapes_and_placebo() -> None:
    bank, patches = _bank(), _patches()
    designs = condition_designs(bank, patches, np.random.RandomState(0))
    assert set(designs) == {
        "state",
        "state+geom",
        "state+geom+inter",
        "state+full",
        "state+geom-shuffled",
    }
    assert designs["state"][0].shape == (N, 11)
    assert designs["state+geom"][0].shape == (N, 18)
    assert designs["state+geom+inter"][0].shape == (N, 22)
    assert designs["state+full"][0].shape == (N, 11 + 49)
    assert len(designs["state+full"][1]) == 60
    assert designs["state+geom-shuffled"][0].shape == designs["state+geom+inter"][0].shape
    # The placebo keeps the observable block identical and permutes only the terrain block.
    assert np.allclose(
        designs["state+geom-shuffled"][0][:, :11],
        designs["state+geom+inter"][0][:, :11],
    )
    assert not np.allclose(
        designs["state+geom-shuffled"][0][:, 11:18],
        designs["state+geom+inter"][0][:, 11:18],
    )
    assert designs["state+geom-shuffled"][1][11].endswith("_shuffled")


def test_signal_diagnostics_finds_a_planted_dependency() -> None:
    bank = _bank()
    patches = _patches()
    geom = geometry_descriptors(patches)
    residual = np.zeros((N, 4))
    residual[:, 3] = 5.0 * geom[:, 0] + 0.01 * bank.states[:, 2]
    out = signal_diagnostics(bank, residual, patches)
    assert out["solid_below"]["vy"] == pytest.approx(1.0, abs=1e-2)
    assert out["ceiling_above"]["vx"] == pytest.approx(0.0, abs=1e-9)


def test_ols_control_recovers_a_linear_target() -> None:
    rng = np.random.RandomState(1)
    X = rng.uniform(-1, 1, (200, 3))
    y = 2.0 * X[:, 0] - X[:, 2] + 0.5
    out = _ols_r2(X[:150], y[:150], X[150:], y[150:])
    assert out["test_r2"] > 0.999
    assert out["train_r2"] > 0.999


def _row(median: float, ols: float = 0.0) -> Dict[str, Any]:
    return {
        "gp_test_r2_median": median,
        "gp_test_r2_mean": median,
        "gp_test_r2_best": median,
        "gp_test_r2_worst": median,
        "ols": {"test_r2": ols},
    }


def _results(state: float, geom: float, full: float, placebo: float) -> Dict[str, Any]:
    conditions = {
        "state": state,
        "state+geom": geom,
        "state+geom+inter": geom,
        "state+full": full,
        "state+geom-shuffled": placebo,
    }
    return {
        c: {ch: _row(v) for ch, v in zip(CHANNELS, vals)}
        for c, vals in ((k, [v, v, v, v]) for k, v in conditions.items())
    }


def _diag() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        "test": {g: {ch: 0.1 for ch in CHANNELS} for g in GEOM_FEATURES},
        "train": {g: {ch: 0.1 for ch in CHANNELS} for g in GEOM_FEATURES},
    }


def test_verdict_rejects_less_catastrophic_as_explanation() -> None:
    """-8 -> -2 is a gain of +6 and still explains nothing."""
    results = _results(state=-8.0, geom=-2.0, full=-2.0, placebo=-8.0)
    verdict = build_verdict(results, _diag())
    assert verdict["terrain_explains_channels"] == []
    assert verdict["observation_gap_claim_supported_for_vx"] is False


def test_verdict_rejects_a_gain_the_placebo_also_achieves() -> None:
    results = _results(state=0.10, geom=0.40, full=0.40, placebo=0.42)
    verdict = build_verdict(results, _diag())
    assert verdict["terrain_explains_channels"] == []


def _mixed_results(
    state: float, geom: float, placebo: float, vertical_only: bool
) -> Dict[str, Any]:
    """Per-condition rows, optionally with the improvement confined to the vertical channels."""
    values = {
        "state": state,
        "state+geom": geom,
        "state+geom+inter": geom,
        "state+full": geom,
        "state+geom-shuffled": placebo,
    }
    out: Dict[str, Any] = {}
    for condition, value in values.items():
        row: Dict[str, Any] = {}
        for channel in CHANNELS:
            vertical = channel in ("y", "vy")
            moved = (vertical and vertical_only) or not vertical_only
            r2 = value if moved else state
            row[channel] = _row(r2)
        out[condition] = row
    return out


def test_verdict_accepts_a_real_gain() -> None:
    verdict = build_verdict(_mixed_results(0.10, 0.40, 0.05, vertical_only=False), _diag())
    assert set(verdict["terrain_explains_channels"]) == set(CHANNELS)
    assert verdict["observation_gap_claim_supported_for_vx"] is True


def test_verdict_does_not_let_a_vertical_gain_certify_the_horizontal_claim() -> None:
    verdict = build_verdict(_mixed_results(0.10, 0.40, 0.05, vertical_only=True), _diag())
    assert verdict["terrain_explains_channels"] == ["y", "vy"]
    assert verdict["terrain_does_not_explain_channels"] == ["x", "vx"]
    assert verdict["observation_gap_claim_supported_for_vx"] is False


def test_split_reconstruction_matches_the_canonical_loader() -> None:
    """The patch alignment rests on assertions, so this checks they actually hold."""
    data = load_tilemap_splits(DATASET_TILEMAP)
    n_train = data["train_states"].shape[0]
    assert data["train_patches"].shape == (n_train, 7, 7)
    assert data["test_patches"].shape[0] == data["test_states"].shape[0]
    assert data["train_states"].shape[1] == 8
    # Patches are tile-class codes, and the recording is binary in this stage.
    assert set(np.unique(data["train_patches"]).tolist()) <= {0, 1, 2, 3}
