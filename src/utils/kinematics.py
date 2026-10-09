r"""
kinematics.py
The two ways a discrete model can read section 4.1, named in one place.

README §4.1 states the position update with the velocity the frame *starts* with
($X_{t+1} = X_t + v_{x,t}/16$), which is what the recorded console does - the median
residual against that velocity is 0.0000 px and it holds exactly on most frames
(`results/physics_claim_audit_metrics.json`, §10.54). The published shells instead advance
position with the velocity they predict for the next frame, which costs one sub-pixel of lag
on every accelerating frame and is what §10.53 measures as a whole prediction target.

Because both forms are legitimate objects of study and several classes can be built either
way, the mode names live here: a typo raises at construction rather than silently training
the other model. ``NEXT`` is the default every published artifact was recorded with.
"""

from typing import Tuple

CARRIED = "carried"
NEXT = "next"
POSITION_VELOCITY_MODES: Tuple[str, ...] = (NEXT, CARRIED)

# Vertical ground rule names for `GroundContactConsistencyLoss`.
CONTACT_ZERO_VELOCITY = "zero_velocity"
CONTACT_ZERO_INCREMENT = "zero_increment"
CONTACT_RULES: Tuple[str, ...] = (CONTACT_ZERO_VELOCITY, CONTACT_ZERO_INCREMENT)


def check_position_velocity(mode: str) -> str:
    """Return `mode` when it names one of the two position-integration conventions."""
    if mode not in POSITION_VELOCITY_MODES:
        raise ValueError(
            f"position_velocity={mode!r} is not one of {POSITION_VELOCITY_MODES}; "
            "carried advances with v_t (section 4.1 as written), next with the predicted "
            "v_(t+1) (the published shells)"
        )
    return mode


def check_contact_rule(rule: str) -> str:
    """Return `rule` when it names one of the two grounded-frame objectives."""
    if rule not in CONTACT_RULES:
        raise ValueError(
            f"contact rule {rule!r} is not one of {CONTACT_RULES}; zero_velocity is the "
            "published penalty, zero_increment is the rule README 4.3.5 now states"
        )
    return rule


def set_position_velocity(model: object, mode: str) -> str:
    """Switch a trained model's integration convention and return the mode it had.

    The shells decide the convention in their forward pass, so the same weights can be
    scored under both - the cleanest control 10.57 has: one line of graph differs, and no
    training noise enters the contrast. Raises when the model has no such switch, rather
    than silently scoring a model that never had the mode.
    """
    checked = check_position_velocity(mode)
    previous = getattr(model, "position_velocity", None)
    if previous is None:
        raise TypeError(
            f"{type(model).__name__} has no position_velocity switch; it integrates position "
            "with one hardcoded velocity, which is exactly the disagreement 10.54 records"
        )
    model.position_velocity = checked  # type: ignore[attr-defined]
    return str(previous)
