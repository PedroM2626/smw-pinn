"""
test_sprint_excitation.py
Unit tests for the excitation-targeted ceiling study (README Section 10.45).

The study exists because one measured number - zero held-out frames near the speed support -
turned a claim about the estimator into a claim about the data collection policy. These tests
pin the two things that could go wrong with that argument: the profile must count the frames
that actually exist, and the verdict must report the direction the clamp estimate moved rather
than the direction that would be convenient.
"""

from __future__ import annotations

import os

import numpy as np

import src.evaluation.sprint_excitation_benchmark as study


def _block(binding_fraction: float, clamp: float, support: float) -> dict:
    return {
        "profile": {"max_abs_vx": support},
        "heldout_profile": {"fraction_at_90pct_of_support": binding_fraction},
        "templates": {"structure": {"bound_value": clamp}},
    }


def test_excitation_profile_counts_frames_against_the_training_support() -> None:
    states = np.zeros((5, 8))
    nxt = np.array([[0, 0, v, 0, 0, 0, 0, 0] for v in (40.0, 36.0, 20.0, -35.0, -10.0)])
    out = study.excitation_profile(states, nxt, support=40.0)
    assert out["max_abs_vx"] == 40.0
    assert out["fraction_at_90pct_of_support"] == 0.4  # 40 and 36 only
    assert out["fraction_at_95pct_of_support"] == 0.2  # 40 only
    assert out["train_transitions"] == 5


def test_excitation_profile_handles_a_silent_recording() -> None:
    """A recording with no motion has no support; the study must not divide by it silently."""
    nxt = np.zeros((4, 8))
    out = study.excitation_profile(np.zeros((4, 8)), nxt, support=0.0)
    assert out["max_abs_vx"] == 0.0
    assert out["fraction_at_90pct_of_support"] == 1.0  # 0 >= 0 for every frame


def test_verdict_reports_the_direction_the_estimate_actually_moved() -> None:
    results = {
        "published_gameplay": _block(0.0, 47.775, 49.0),
        "sprint_targeted": _block(0.41, 37.0, 37.0),
    }
    out = study.build_verdict(results, reference=72.0)
    assert out["ceiling_became_measurable"] is True
    assert out["best_excited_dataset"] == "sprint_targeted"
    # 47.775 is closer to 72 than 37 is, so the honest reading is the negative one.
    assert "does not move the estimate toward the WRAM reference" in out["reading"]
    assert "0.00%" in out["reading"] and "41.00%" in out["reading"]


def test_verdict_gives_credit_when_more_excitation_closes_the_gap() -> None:
    results = {
        "published_gameplay": _block(0.0, 40.0, 41.0),
        "sprint_targeted": _block(0.5, 68.0, 70.0),
    }
    out = study.build_verdict(results, reference=72.0)
    assert "does move the estimate" in out["reading"]
    assert out["template_clamp_estimate"]["sprint_targeted"] == 68.0


def test_verdict_says_nothing_changed_when_binding_stays_zero() -> None:
    results = {
        "published_gameplay": _block(0.0, 47.0, 49.0),
        "sprint_targeted": _block(0.0, 46.0, 48.0),
    }
    assert study.build_verdict(results, reference=72.0)["ceiling_became_measurable"] is False


def test_the_study_refuses_to_compare_against_a_missing_recording(monkeypatch) -> None:
    real = os.path.isfile
    monkeypatch.setattr(study.os.path, "isfile", lambda p: False if "nope" in str(p) else real(p))
    assert study.main(["--sprint-dataset", "nope.npz"]) == 1
