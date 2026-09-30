"""
test_learned_structure_probe.py
Unit tests for the structural probes applied to learned dynamics (README Section 10.46).

The probe is the whole study, so the tests are about what a crossing is allowed to mean.
The 10.43 acceptance rule - "the map must accelerate somewhere below the crossing" - was
written for a genetic program that might collapse to a constant, and it is not enough for a
network extrapolating past its recording: an MLP in this repository accelerates by twelve
pixels a frame and then falls back through the diagonal at a *negative* velocity, which the
old rule would have reported as a velocity ceiling. These tests pin the stricter
classification down, because a study whose headline is "every learned model contains the
constraint" would be wrong if the classifier were loose.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.evaluation.learned_structure_probe_benchmark as study


class _MapModel(nn.Module):
    """A dynamics model whose vx channel is an arbitrary function of the current vx."""

    def __init__(self, fn: callable) -> None:  # noqa: ANN401
        super().__init__()
        self.fn = fn
        self.anchor = nn.Parameter(torch.zeros(1))  # so device_of() has a parameter to read

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = state.clone()
        vx = state[:, 2].detach().cpu().numpy().astype(np.float64)
        out[:, 2] = torch.as_tensor(self.fn(vx), dtype=state.dtype, device=state.device)
        return out


def _probe(fn: callable, cap: float = 49.0, rows: int = 193) -> Dict[str, float]:  # noqa: ANN401
    return study.probe_ceiling(_MapModel(fn).eval(), cap, torch.device("cpu"), rows=rows)


def test_a_planted_stable_ceiling_is_recognised() -> None:
    out = _probe(lambda v: np.minimum(v + 1.8, 48.0))
    assert out["has_ceiling"] == 1.0
    assert out["ceiling_like_fixed_point"] == pytest.approx(48.0, abs=0.6)
    assert out["local_slope_at_ceiling"] < 0.0  # the map pulls back toward the level


def test_a_negative_crossing_is_not_reported_as_a_ceiling() -> None:
    """Exactly the artifact the smoke run exposed: strong acceleration, then a collapse below zero.

    A model that adds twelve pixels a frame and then throws the velocity negative has no
    ceiling, but the 10.43 rule - which only asks that the map accelerate somewhere - reports
    its first crossing as one, at a negative level.
    """

    def fn(v: np.ndarray) -> np.ndarray:
        return np.where(v < 20.0, v + 12.0, -5.0)

    out = _probe(fn)
    assert out["fixed_point_found_10_43_rule"] == 1.0  # the loose rule would accept this
    assert out["bound_value_10_43_rule"] == pytest.approx(-5.0)
    assert out["has_ceiling"] == 0.0
    assert not np.isfinite(out["ceiling_like_fixed_point"])


def test_an_unstable_crossing_is_rejected() -> None:
    """The map touches the diagonal at v = 20 and accelerates again above it: not a ceiling."""
    # rows=50 over [0, 49] puts v = 20 exactly on the probe grid, so the tangency is seen.
    out = _probe(lambda v: v + 0.01 * (v - 20.0) ** 2, rows=50)
    assert out["accelerating"] == 1.0
    assert out["crossings_in_range"] >= 1
    assert out["has_ceiling"] == 0.0


def test_a_map_that_never_accelerates_has_no_ceiling() -> None:
    """The identity map is the flat-model case, and it also exposes a float32 subtlety.

    These checkpoints are evaluated in single precision, so v_next - v is noise of order
    1e-5 rather than exact zero. The 10.43 rule compares against 1e-6 and therefore reports
    a fixed point on the noise; the classification uses a physical tolerance (1e-3 px/frame,
    far below the engine's smallest real increment) and correctly reports no ceiling.
    """
    out = _probe(lambda v: v.copy())
    assert out["has_ceiling"] == 0.0
    assert out["accelerating"] == 0.0
    assert out["fixed_point_found_10_43_rule"] == 1.0  # accepted on round-off alone


def test_a_clamp_above_the_observed_support_cannot_explain_a_lower_crossing() -> None:
    """The attribution the imposed/learned distinction actually needs."""
    probes = {
        "shell": {
            "available": True,
            "has_ceiling": True,
            "ceiling": {
                "has_ceiling": 1.0,
                "ceiling_like_fixed_point": 23.0,
            },
            "bound_imposed_by_construction": True,
            "shell_clamp_constant": 72.0,
            "shell_clamp_binds_within_support": False,
            "gravity_gate": {"tier_separation": 0.0},
            "gravity_gate_found": False,
            "acceleration_gain_px_per_frame": 1.7,
            "ceiling_is_plausible": True,
        }
    }
    surrogate = {
        "identification_on_real_transitions": {},
        "real_max_relative_error_pct": 80.0,
        "per_surrogate": {"shell": {"available": True, "max_relative_error_pct": 95.0}},
    }
    verdict = study.build_verdict(probes, surrogate)
    assert verdict["fixed_point_imposed_by_architecture"] == []
    assert verdict["fixed_point_learned"] == ["shell"]


def test_a_fold_back_after_huge_acceleration_is_not_counted_as_a_bound() -> None:
    """The plateau must be reached with plausible traction, or it is an extrapolation artifact."""
    probes = {
        "runaway": {
            "available": True,
            "ceiling": {"has_ceiling": 1.0, "ceiling_like_fixed_point": 30.0},
            "bound_imposed_by_construction": False,
            "shell_clamp_constant": float("nan"),
            "shell_clamp_binds_within_support": False,
            "acceleration_gain_px_per_frame": 12.3,
            "ceiling_is_plausible": False,
            "gravity_gate": {"tier_separation": 0.0},
            "gravity_gate_found": False,
        }
    }
    surrogate = {
        "identification_on_real_transitions": {},
        "real_max_relative_error_pct": 84.1,
        "per_surrogate": {"runaway": {"available": True, "max_relative_error_pct": 148.0}},
    }
    verdict = study.build_verdict(probes, surrogate)
    assert verdict["crossing_without_plausible_traction"] == ["runaway"]
    assert verdict["fixed_point_learned"] == []
    assert "extrapolation artifact" in verdict["reading"]


def test_load_model_records_a_missing_checkpoint_as_unavailable(tmp_path) -> None:
    assert (
        study.load_model("definitely_absent.pt", lambda: nn.Linear(2, 2), torch.device("cpu"))
        is None
    )


def test_device_of_reads_the_loaded_device() -> None:
    model = _MapModel(lambda v: v)
    assert study.device_of(model).type in {"cpu", "cuda"}
