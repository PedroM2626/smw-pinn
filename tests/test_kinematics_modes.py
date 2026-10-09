r"""
test_kinematics_modes.py
The corrected forms of README section 4 are reachable, and the defaults are not moved.

Section 10.54 established that four physics claims are stated one way in the prose and
implemented another. The corrected behaviours are now constructible - position advancing with
the frame's own velocity, the ground flag suppressing the gravity *step* rather than the
vertical velocity, and the rollout predicate scored against the speed class the telemetry can
actually reach - but every published artifact was recorded with the old defaults, so what this
file proves is the pair of statements that matters:

1. asking for the corrected mode changes exactly one channel and nothing else, and changes it
   in the direction section 4 states;
2. asking for nothing keeps the published arithmetic bit-for-bit, so no artifact's provenance
   is invalidated by the existence of the flag.

CPU-only, deterministic.

Run:  pytest tests/test_kinematics_modes.py -q
"""

import numpy as np
import pytest
import torch

from src.evaluation.rollout_evaluator import RolloutEvaluator
from src.losses.physics_losses import (
    CompositePINNLoss,
    GroundContactConsistencyLoss,
)
from src.models.analytical_kinematics import AnalyticalKinematicsDynamics
from src.models.output_projection import ProjectedDynamics
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.residual_dynamics import ResidualDynamics
from src.utils.kinematics import (
    CARRIED,
    CONTACT_ZERO_INCREMENT,
    CONTACT_ZERO_VELOCITY,
    NEXT,
    check_contact_rule,
    check_position_velocity,
)

SUBPIXELS = 16.0


class _ConstantAux(torch.nn.Module):
    """A base net whose increments are fixed, so the graph is the only variable."""

    def __init__(self, dvx: float = 4.0, dvy: float = -3.0, aux: int = 4):
        super().__init__()
        self.dvx, self.dvy, self.aux = dvx, dvy, aux

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        n = state.shape[0]
        return torch.cat(
            [
                torch.full((n, 1), self.dvx),
                torch.full((n, 1), self.dvy),
                torch.full((n, self.aux), 0.25),
            ],
            dim=-1,
        )


def _states() -> torch.Tensor:
    return torch.tensor(
        [[100.0, 40.0, 24.0, -6.0, 0.0, 0.0, 0.0, 1.0], [5.0, 9.0, -8.0, 30.0, 1.0, 0.0, 0.0, 0.0]]
    )


def _actions(jump: float = 0.0) -> torch.Tensor:
    a = torch.zeros(2, 6)
    a[:, 0] = jump
    a[:, 5] = 1.0  # RIGHT
    return a


def test_mode_names_are_validated_at_construction() -> None:
    assert check_position_velocity(CARRIED) == "carried"
    assert check_contact_rule(CONTACT_ZERO_VELOCITY) == "zero_velocity"
    with pytest.raises(ValueError, match="position_velocity"):
        check_position_velocity("carried_or_something_else")
    with pytest.raises(ValueError, match="contact rule"):
        check_contact_rule("no_rule")
    for factory in (
        lambda mode: ResidualDynamics(_ConstantAux(), position_velocity=mode),
        lambda mode: ProjectedDynamics(_ConstantAux(), position_velocity=mode),
        lambda mode: HardResidualPINNDynamics(position_velocity=mode),
    ):
        with pytest.raises(ValueError):
            factory("previous")  # type: ignore[arg-type]


def test_carried_mode_changes_position_only() -> None:
    """Velocity output identical, position advanced with the frame's own velocity."""
    state, action = _states(), _actions()
    kwargs = dict(base=_ConstantAux(), state_dim=8, hard=True)
    nxt = ResidualDynamics(position_velocity=NEXT, **kwargs)(state, action)  # type: ignore[arg-type]
    car = ResidualDynamics(position_velocity=CARRIED, **kwargs)(  # type: ignore[arg-type]
        state, action
    )
    assert torch.equal(nxt[:, 2:], car[:, 2:])  # velocities and contacts untouched
    assert torch.allclose(car[:, 0], state[:, 0] + state[:, 2] / SUBPIXELS)
    assert torch.allclose(car[:, 1], state[:, 1] + state[:, 3] / SUBPIXELS)
    # ...and the published form is the one that uses the value it just computed.
    assert torch.allclose(nxt[:, 0], state[:, 0] + nxt[:, 2] / SUBPIXELS)
    lag = (nxt[:, 0] - car[:, 0]).abs()
    assert torch.all(lag > 0) and torch.allclose(lag, (nxt[:, 2] - state[:, 2]).abs() / SUBPIXELS)


def test_default_construction_is_the_published_arithmetic() -> None:
    """No flag passed: the three shells must match the next-frame formula exactly."""
    state, action = _states(), _actions()
    residual = ResidualDynamics(base=_ConstantAux())(state, action)
    assert torch.allclose(residual[:, 0], state[:, 0] + residual[:, 2] / SUBPIXELS)
    projected = ProjectedDynamics(base=_StateNet())(state, action)
    assert torch.allclose(projected[:, 0], state[:, 0] + projected[:, 2] / SUBPIXELS)
    torch.manual_seed(42)
    flagship = HardResidualPINNDynamics()
    named = HardResidualPINNDynamics(position_velocity="next")
    named.load_state_dict(flagship.state_dict())
    with torch.no_grad():
        assert torch.equal(flagship(state, action), named(state, action))


class _StateNet(torch.nn.Module):
    """A state-output base that predicts an out-of-window velocity, to test the projection."""

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = state.clone()
        out[:, 2] = 100.0  # above the 72.0 window
        out[:, 3] = 90.0
        return out


def test_projection_carried_leaves_position_to_the_input() -> None:
    state, action = _states(), _actions()
    carried = ProjectedDynamics(base=_StateNet(), position_velocity=CARRIED)(state, action)
    assert torch.allclose(carried[:, 0], state[:, 0] + state[:, 2] / SUBPIXELS)
    assert torch.allclose(carried[:, 2], torch.full((2,), 72.0))  # velocity still projected


def test_ground_rule_zero_increment_keeps_the_velocity_it_had() -> None:
    """The §4.3.5 rule as measured: on the ground the gravity *step* is skipped."""
    state = torch.tensor([[0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 0.0]])  # grounded, falling slowly
    action = torch.zeros(1, 6)
    published = AnalyticalKinematicsDynamics()(state, action)
    corrected = AnalyticalKinematicsDynamics(ground_rule=CONTACT_ZERO_INCREMENT)(state, action)
    assert published[0, 3].item() == pytest.approx(0.0)  # the retracted rest state
    assert corrected[0, 3].item() == pytest.approx(6.0)  # the step was not applied
    assert torch.allclose(published[:, 2], corrected[:, 2])  # horizontal rule is shared


def test_contact_penalty_default_is_the_published_square_and_the_other_is_the_step() -> None:
    state, action = _states(), _actions()
    state[:, 4] = 1.0
    predicted = state.clone()
    predicted[:, 3] = 6.0
    published = GroundContactConsistencyLoss()(state, action, predicted)
    increment = GroundContactConsistencyLoss(rule=CONTACT_ZERO_INCREMENT)(state, action, predicted)
    assert torch.allclose(
        published, torch.tensor(36.0)
    )  # mean of hat_vy**2 over the two grounded frames
    expected_step = ((predicted[:, 3] - state[:, 3]) ** 2).mean()
    assert torch.allclose(increment, expected_step)
    # A model that only ever keeps its velocity is exactly zero under the corrected rule.
    still = state.clone()
    assert GroundContactConsistencyLoss(rule=CONTACT_ZERO_INCREMENT)(state, action, still) == 0.0
    with pytest.raises(ValueError):
        CompositePINNLoss(contact_rule="whatever")  # type: ignore[arg-type]


def test_predicate_bounds_are_parameters_and_the_tighter_class_flags_more_frames() -> None:
    """Scoring the same rollout at 48.0 must flag at least what 72.0 flags."""

    class _FastModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            self.step += 1
            out = state.clone()
            out[:, 0] = out[:, 0] + out[:, 2] / SUBPIXELS
            out[:, 2] = 60.0 if self.step % 2 else 50.0  # above the run class, below the P-meter
            return out

    model = _FastModel()
    states = np.tile(np.array([[0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype="float32"), (8, 1))
    actions = np.zeros((8, 6), dtype="float32")
    device = torch.device("cpu")
    published = RolloutEvaluator(device=device).evaluate_rollout(
        model, "test", states[0], actions, states
    )
    run_class = RolloutEvaluator(device=device, max_vx=48.0).evaluate_rollout(
        model, "test", states[0], actions, states
    )
    assert published["velocity_violations"] == 0
    assert run_class["velocity_violations"] == 8
    assert np.allclose(published["predicted_trajectory"], run_class["predicted_trajectory"])


def test_tolerance_is_a_parameter_of_the_same_predicate() -> None:
    class _LaggingModel(torch.nn.Module):
        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            out = state.clone()
            out[:, 0] = out[:, 0] + (out[:, 2] + 1.6) / SUBPIXELS  # 0.1 px of lag per frame
            return out

    states = np.tile(np.array([[0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype="float32"), (4, 1))
    actions = np.zeros((4, 6), dtype="float32")
    evaluator = RolloutEvaluator(torch.device("cpu"))
    published = evaluator.evaluate_rollout(_LaggingModel(), "test", states[0], actions, states)
    tight = RolloutEvaluator(torch.device("cpu"), tolerance_px=0.002).evaluate_rollout(
        _LaggingModel(), "test", states[0], actions, states
    )
    assert published["kinematic_violations"] == 0  # inside the published 0.2 px
    assert tight["kinematic_violations"] == 4  # the same trajectory, a tighter ruler
