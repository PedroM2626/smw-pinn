"""
rollout_diagnostics.py
Separating what the published rollout metric actually measures (README 10.48).

`RolloutEvaluator.evaluate_rollout` reports a *kinematic violation* whenever

    | (x_{t+1} - x_t) - v_t / 16 | > 0.2 pixels

with `v_t` the velocity the model predicted on the **previous** frame. Two
different properties move that number: whether the model integrates position from
velocity at all, and how far its velocity estimate changes from one frame to the
next. A model that satisfies the discrete kinematics exactly is still flagged
whenever two consecutive predicted velocities differ by more than 3.2
sub-pixels/frame - so the figure published since Section 8 as the signature of a
hard kinematic shell is partly a smoothness measurement.

This module computes the three quantities separately from a predicted trajectory,
so a violation rate can be attributed instead of asserted:

* ``integration_error_*`` - the residual of the model's own integration identity,
  measured against the velocity it actually used;
* ``jump_rate`` - the fraction of frames where consecutive predicted velocities
  differ by more than the tolerance's equivalent of 3.2 sub-pixels/frame;
* ``bound_exceedance_rate`` - the fraction of predicted states outside the
  engine's velocity bounds, which is what a clamp is for.

It also re-scores the published predicate against real telemetry, which is the
reference that says how much of a violation rate is a property of Mario's physics
rather than of the model.
"""

import math
from typing import Any, Dict

import numpy as np

SUBPIXELS_PER_PIXEL = 16.0
MAX_VX = 72.0
TERMINAL_VY = 64.0
MIN_VY = -80.0

# The published tolerance, restated in velocity units: |dx - v/16| > 0.2 px is
# triggered by a velocity change of more than 0.2 * 16 = 3.2 sub-pixels/frame.
PUBLISHED_TOLERANCE_PX = 0.2
TOLERANCE_AS_VELOCITY_JUMP = PUBLISHED_TOLERANCE_PX * SUBPIXELS_PER_PIXEL


def published_predicate_rate(pred_traj: np.ndarray) -> float:
    """Reproduce the 10.27 violation predicate: position vs the *previous* velocity."""
    if pred_traj.shape[0] < 2:
        return 0.0
    dx = np.diff(pred_traj[:, 0])
    v_prev = pred_traj[:-1, 2]
    return float(np.mean(np.abs(dx - v_prev / SUBPIXELS_PER_PIXEL) > PUBLISHED_TOLERANCE_PX))


def integration_residual(pred_traj: np.ndarray) -> Dict[str, float]:
    """Residual of the model's own identity: dx against the velocity it advanced by."""
    if pred_traj.shape[0] < 2:
        return {
            "integration_error_mean_px": 0.0,
            "integration_error_median_px": 0.0,
            "integration_error_p95_px": 0.0,
        }
    dx = np.diff(pred_traj[:, 0])
    v_next = pred_traj[1:, 2]
    err = np.abs(dx - v_next / SUBPIXELS_PER_PIXEL)
    return {
        "integration_error_mean_px": float(err.mean()),
        # A screen wrap or a wall contact moves x by hundreds of pixels in one frame,
        # so the mean is a tail statistic here; the median is the honest central value.
        "integration_error_median_px": float(np.median(err)),
        "integration_error_p95_px": float(np.quantile(err, 0.95)),
    }


def velocity_jump_rate(pred_traj: np.ndarray) -> Dict[str, float]:
    """How often the model's velocity estimate moves more than the tolerance allows."""
    if pred_traj.shape[0] < 2:
        return {"jump_rate": 0.0, "jump_median": 0.0, "jump_p95": 0.0}
    jump = np.abs(np.diff(pred_traj[:, 2]))
    return {
        "jump_rate": float(np.mean(jump > TOLERANCE_AS_VELOCITY_JUMP)),
        "jump_median": float(np.median(jump)),
        "jump_p95": float(np.quantile(jump, 0.95)),
    }


def bound_exceedance(pred_traj: np.ndarray) -> Dict[str, float]:
    """Fraction of predicted states the engine's own velocity bounds would clip."""
    vx, vy = pred_traj[:, 2], pred_traj[:, 3]
    out = np.where(np.abs(vx) > MAX_VX, 1.0, 0.0) + np.where(vy > TERMINAL_VY, 1.0, 0.0)
    out = out + np.where(vy < MIN_VY, 1.0, 0.0)
    return {
        "bound_exceedance_rate": float(np.mean(out > 0)),
        "max_abs_vx": float(np.max(np.abs(vx))) if vx.size else 0.0,
    }


def decompose(pred_traj: np.ndarray) -> Dict[str, float]:
    """All three quantities, plus the published predicate, for one trajectory."""
    out: Dict[str, float] = {"published_violation_rate": published_predicate_rate(pred_traj)}
    out.update(integration_residual(pred_traj))
    out.update(velocity_jump_rate(pred_traj))
    out.update(bound_exceedance(pred_traj))
    return out


def aggregate(runs: np.ndarray) -> Dict[str, float]:
    """Average the per-start decompositions of a rollout ensemble.

    Args:
        runs: array of shape [num_starts, horizon, state_dim].
    """
    parts = [decompose(runs[i]) for i in range(runs.shape[0])]
    keys = parts[0].keys()
    return {k: float(np.mean([p[k] for p in parts])) for k in keys}


def traction_sensitivity(rows: Dict[str, Any], thresholds: tuple) -> Dict[str, Any]:
    """Re-classify the probed ceilings for every traction threshold, from stored fields.

    The acceptance rule that 10.46 added is a chosen constant, so the classification
    has to be readable without re-running anything: every probe record already keeps
    the raw 10.43 rule, the stable-crossing value and the measured traction gain.
    """
    out: Dict[str, Any] = {}
    for threshold in thresholds:
        accepted, artifacts = [], {}
        for artifact_name, probes in rows.items():
            for label, block in probes.items():
                if not block.get("available"):
                    continue
                ceiling = block["ceiling"]
                gain = block["acceleration_gain_px_per_frame"]
                value = ceiling["ceiling_like_fixed_point"]
                plausible = (
                    ceiling["has_ceiling"] > 0.5
                    and value is not None
                    and math.isfinite(value)
                    and value > 0.0
                    and gain <= threshold
                )
                if plausible:
                    accepted.append(label)
                    artifacts.setdefault(artifact_name, []).append(label)
        out[f"{threshold}"] = {
            "accepted": len(accepted),
            "by_artifact": {name: len(labels) for name, labels in artifacts.items()},
        }
    return out
