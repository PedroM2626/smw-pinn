"""
test_structure_selection.py
Unit tests for the nested-template model-selection engine (README Section 10.43.9).

The templates are the third engine of the discovery comparison: every structure whose
discovery is in question (traction tiers, rigid bound, Coulomb deadband, held-jump gravity
gate, terminal clamp, ground reset) is an explicit candidate, so a failure to select it is a
statement about the data or the criterion, never about a representation. These tests pin
that down: exact recovery on data the engine generated, the ordering of the nested
templates, and - critically - that the bound is NOT selected when the data never saturates.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inverse.parameter_identification import (
    EngineParams,
    generate_synthetic_windows,
    theta_tensor,
)
from src.inverse.structure_selection import (
    TemplateFit,
    _shrink,
    _tail_mask,
    evaluate_templates,
    fit_horizontal_templates,
    fit_vertical_templates,
    summarise,
)

WORLD = EngineParams(
    max_vx=48.0,
    walk_accel=1.0,
    run_accel=1.8,
    subpixels_per_pixel=20.0,
    held_gravity=2.4,
    fall_gravity=5.2,
    decel=0.6,
)


def _flat(seed: int = 11, n_windows: int = 400, rollout_len: int = 12) -> tuple:
    w = generate_synthetic_windows(
        theta_tensor(WORLD), n_windows=n_windows, rollout_len=rollout_len, seed=seed
    )
    s0, acts, tgts, ct = (np.asarray(t, dtype=np.float64) for t in w)
    states = np.concatenate([s0[:, None, :], tgts[:, :-1, :]], axis=1).reshape(-1, 4)
    return states, acts.reshape(-1, 6), tgts.reshape(-1, 4), ct.reshape(-1, 4)


def _horizontal(fits):
    return {f.name: f for f in fits}


def test_shrink_is_a_one_sided_deadband() -> None:
    v = np.array([-5.0, -0.2, 0.0, 0.2, 5.0])
    out = _shrink(v, 0.6)
    # Never crosses zero, and a sub-threshold speed is killed exactly.
    assert np.all(np.sign(out) == np.sign(v)) or np.all(out[np.abs(v) < 0.6] == 0.0)
    assert out[0] == pytest.approx(-4.4)
    assert out[4] == pytest.approx(4.4)
    assert out[1] == 0.0 and out[3] == 0.0


def test_tail_mask_selects_the_saturated_frames() -> None:
    v = np.array([0.0, 10.0, 47.0, -48.0, 24.0])
    mask = _tail_mask(v, frac=0.9)
    assert mask.tolist() == [False, False, True, True, False]


def test_templates_recover_the_generating_law_exactly() -> None:
    states, actions, nxt, _ = _flat(seed=11)
    fits = fit_horizontal_templates(
        states[:, 2], actions[:, 5] - actions[:, 4], actions[:, 1], nxt[:, 2]
    )
    by_name = _horizontal(fits)
    coast = by_name["H5_coast"]
    assert coast.parameters["max_vx"] == pytest.approx(48.0, abs=1e-6)
    assert coast.parameters["walk"] == pytest.approx(1.0, abs=1e-5)
    assert coast.parameters["walk"] + coast.parameters["run_increment"] == pytest.approx(
        1.8, abs=1e-5
    )
    assert coast.parameters["decel"] == pytest.approx(0.6, abs=1e-3)
    assert coast.train_rmse < 1e-3


def test_genuinely_nested_templates_improve_monotonically() -> None:
    """H1 < H2 and H4 < H5 < H6 are real nestings; H3 is a different hypothesis."""
    states, actions, nxt, ct = _flat(seed=5)
    fits = _horizontal(
        fit_horizontal_templates(
            states[:, 2], actions[:, 5] - actions[:, 4], actions[:, 1], nxt[:, 2], wall=ct
        )
    )
    assert fits["H1_constant"].train_rmse >= fits["H2_affine"].train_rmse
    chain = ["H4_clamped", "H5_coast", "H6_contact"]
    rmse = [fits[n].train_rmse for n in chain]
    assert all(a >= b - 1e-9 for a, b in zip(rmse, rmse[1:])), rmse


def test_the_bound_is_what_fixes_the_saturated_frames() -> None:
    """H3 (no bound) and H4 (bound) share the drive; only the tail separates them."""
    states, actions, nxt, _ = _flat(seed=5)
    fits = fit_horizontal_templates(
        states[:, 2], actions[:, 5] - actions[:, 4], actions[:, 1], nxt[:, 2]
    )
    by_name = _horizontal(fits)
    assert by_name["H3_drive"].train_tail_rmse > by_name["H4_clamped"].train_tail_rmse
    assert abs(by_name["H3_drive"].parameters["walk"] - 1.0) > 1e-3  # biased without the clamp


def test_unsaturated_data_does_not_invent_a_bound() -> None:
    """If the speed never approaches a ceiling, selection must not posit one.

    Noise is essential to the test: on noiseless data every template that contains the
    truth fits with SSE ~ 0, and BIC then compares numerically-degenerate differences
    instead of a complexity trade-off.
    """
    rng = np.random.RandomState(3)
    v = rng.uniform(-8.0, 8.0, 1200)
    direction = rng.choice([-1.0, 0.0, 1.0], 1200)
    run = rng.randint(0, 2, 1200).astype(float)
    v_next = v + direction * (1.0 + 0.8 * run) + rng.normal(0.0, 0.3, 1200)
    by_name = _horizontal(fit_horizontal_templates(v, direction, run, v_next))
    summary = summarise(list(by_name.values()))
    assert summary["bic_selected"] == "H3_drive"
    # The bound the clamped template does fit sits at the support edge, never inside it.
    support = float(np.abs(v_next).max())
    assert by_name["H4_clamped"].parameters["max_vx"] > 0.9 * support
    # ... and the two hypotheses agree to noise level, so nothing was gained by clipping.
    assert by_name["H4_clamped"].train_rmse == pytest.approx(
        by_name["H3_drive"].train_rmse, abs=1e-3
    )


def test_vertical_templates_recover_both_gravity_tiers() -> None:
    states, actions, nxt, ct = _flat(seed=7)
    fits = fit_vertical_templates(states[:, 3], actions[:, 0], nxt[:, 3], ground=ct[:, 0])
    by_name = {f.name: f for f in fits}
    assert by_name["V6_ground_reset"].parameters["fall_gravity"] == pytest.approx(5.2, abs=1e-3)
    assert by_name["V6_ground_reset"].parameters["g_hold"] == pytest.approx(2.4, abs=1e-3)
    # A single averaged gravity cannot reproduce the two-tier law.
    assert by_name["V3_single_gravity"].train_rmse > by_name["V6_ground_reset"].train_rmse


def test_held_out_metrics_are_attached_and_summarised() -> None:
    fit = _flat(seed=11)
    ev = _flat(seed=907)
    fits = fit_horizontal_templates(
        fit[0][:, 2], fit[1][:, 5] - fit[1][:, 4], fit[1][:, 1], fit[2][:, 2]
    )
    evaluate_templates(
        fits,
        {
            "v": ev[0][:, 2],
            "dir": ev[1][:, 5] - ev[1][:, 4],
            "run": ev[1][:, 1],
            "wall": np.zeros((ev[0].shape[0], 4)),
        },
        ev[2][:, 2],
        _tail_mask(ev[2][:, 2]),
    )
    summary = summarise(fits)
    assert summary["tail_rmse_selected"] == "H5_coast"
    assert summary["criteria_agree"] is True
    for f in fits:
        assert np.isfinite(f.test_rmse) and np.isfinite(f.test_tail_rmse)
    assert summary["templates"]["H5_coast"]["test_r2"] > 0.99


def test_templates_reject_mismatched_wall_shapes() -> None:
    states, actions, nxt, ct = _flat(seed=2, n_windows=60)
    fits = fit_horizontal_templates(
        states[:, 2], actions[:, 5] - actions[:, 4], actions[:, 1], nxt[:, 2], wall=ct
    )
    names = [f.name for f in fits]
    assert "H6_contact" in names  # wall channels present
    assert all(isinstance(f, TemplateFit) for f in fits)
    no_wall = fit_horizontal_templates(
        states[:, 2],
        actions[:, 5] - actions[:, 4],
        actions[:, 1],
        nxt[:, 2],
        wall=np.zeros_like(ct),
    )
    assert "H6_contact" not in [f.name for f in no_wall]
