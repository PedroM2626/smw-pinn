"""Tests for the closed-form engine-rules baseline (src/models/analytical_kinematics.py).

These lock in the *guarantees* the model claims (exact discrete integration,
documented gravity asymmetry, saturation tiers) and the honesty properties of the
identification routine (deterministic, improves the loss, stays in range).
"""

import numpy as np
import pytest
import torch

from src.models.analytical_kinematics import (
    G_FALL,
    G_HOLD,
    IDENTIFIABLE,
    INITIAL_GUESS,
    TERMINAL_VY,
    VX_RUN,
    VX_WALK,
    AnalyticalKinematicsDynamics,
    EngineRuleParameters,
    velocity_prediction_mse,
)


def _state(x=100.0, y=300.0, vx=0.0, vy=0.0, ground=1.0, ceil=0.0, left=0.0, right=0.0):
    return torch.tensor([[x, y, vx, vy, ground, ceil, left, right]], dtype=torch.float32)


# Action layout: [B (jump), Y (run), UP, DOWN, LEFT, RIGHT]
def _action(jump=0.0, run=0.0, left=0.0, right=0.0):
    return torch.tensor([[jump, run, 0.0, 0.0, left, right]], dtype=torch.float32)


def test_model_has_no_learnable_weights():
    model = AnalyticalKinematicsDynamics()
    assert model.num_parameters == 0
    assert list(model.parameters()) == []


def test_position_update_is_the_exact_kinematic_identity():
    """X_{t+1} = X_t + vx_{t+1}/16 to float32 precision (section 4.1)."""
    model = AnalyticalKinematicsDynamics()
    state = _state(x=512.0, y=200.0, vx=40.0, vy=-12.0)
    action = _action(run=1.0, right=1.0)
    nxt = model(state, action)
    assert float(nxt[0, 0] - (512.0 + float(nxt[0, 2]) / 16.0)) == pytest.approx(0.0, abs=1e-4)
    assert float(nxt[0, 1] - (200.0 + float(nxt[0, 3]) / 16.0)) == pytest.approx(0.0, abs=1e-4)


def test_held_ascent_uses_g_held_and_released_descent_uses_g_fall():
    model = AnalyticalKinematicsDynamics()
    rising = _state(vy=-30.0, ground=0.0)
    held = model(rising, _action(jump=1.0))[0, 3].item()
    released = model(rising, _action())[0, 3].item()
    assert held == pytest.approx(-30.0 + G_HOLD)
    assert released == pytest.approx(-30.0 + G_FALL)


def test_vertical_saturation_is_clamped():
    model = AnalyticalKinematicsDynamics()
    falling = _state(vy=70.0, ground=0.0)
    assert model(falling, _action())[0, 3].item() == pytest.approx(TERMINAL_VY)
    rocket = _state(vy=-90.0, ground=0.0)
    assert model(rocket, _action(jump=1.0))[0, 3].item() == pytest.approx(-80.0)


def test_ground_rest_rule_zeroes_downward_velocity():
    """Section 4.3.5: solid floor, no jump command -> vy_{t+1} = 0."""
    model = AnalyticalKinematicsDynamics()
    resting = _state(vy=25.0, ground=1.0)
    assert model(resting, _action())[0, 3].item() == pytest.approx(0.0)


def test_traction_tiers_and_friction():
    model = AnalyticalKinematicsDynamics(
        params=EngineRuleParameters(a_traction_walk=2.0, a_traction_run=6.0, a_friction=1.0)
    )
    walking = _state(vx=0.0, ground=1.0)
    assert model(walking, _action(right=1.0))[0, 2].item() == pytest.approx(2.0)
    running = model(walking, _action(run=1.0, right=1.0))[0, 2].item()
    assert running == pytest.approx(6.0)
    # Friction pulls an uncommanded velocity back toward zero by a_friction.
    coasting = _state(vx=10.0, ground=1.0)
    assert model(coasting, _action())[0, 2].item() == pytest.approx(9.0)
    # Above-tier momentum is neither shaved off nor extended while pushing the same way
    # (the tiers cap commanded acceleration, they do not clamp an existing speed).
    fast = _state(vx=60.0, ground=1.0)
    assert float(model(fast, _action(run=1.0, right=1.0))[0, 2]) == pytest.approx(60.0)
    assert VX_WALK < VX_RUN


def test_contact_channels_are_propagated_by_persistence():
    model = AnalyticalKinematicsDynamics()
    state = _state(ground=1.0, left=1.0)
    nxt = model(state, _action(jump=1.0, right=1.0))
    assert torch.allclose(nxt[0, 4:], state[0, 4:])


def test_state_dim_guard_is_explicit():
    with pytest.raises(ValueError, match="8D player state"):
        AnalyticalKinematicsDynamics(state_dim=12)


def _synthetic_transitions(n: int = 600, seed: int = 0):
    """Synthetic trajectories generated with known scalars, so the fit has a target.

    The input mix deliberately exercises every branch of the rule set (traction,
    friction to rest, skidding by reversing while moving, repeated takeoffs),
    otherwise unexercised parameters are legitimately unidentifiable.
    """
    gen = np.random.default_rng(seed)
    params = EngineRuleParameters(
        a_traction_walk=1.5,
        a_traction_run=4.0,
        a_friction=1.5,
        a_skid=5.0,
        jump_impulse=-70.0,
        jump_run_gain=0.25,
    )
    model = AnalyticalKinematicsDynamics(params=params)
    states, actions, next_states = [], [], []
    s = _state(x=100.0, vx=0.0, vy=0.0, ground=1.0)
    for _ in range(n):
        jump, run, right, left = [float(gen.integers(0, 2)) for _ in range(4)]
        a = _action(jump=jump, run=run, right=right, left=left)
        nxt = model(s, a)
        states.append(s.numpy())
        actions.append(a.numpy())
        next_states.append(nxt.numpy())
        s = nxt
    return (
        torch.tensor(np.concatenate(states), dtype=torch.float32),
        torch.tensor(np.concatenate(actions), dtype=torch.float32),
        torch.tensor(np.concatenate(next_states), dtype=torch.float32),
    )


def test_identification_recovers_the_generating_scalars():
    """System identification inside the graph, not beside it: it must converge back."""
    states, actions, next_states = _synthetic_transitions()
    model = AnalyticalKinematicsDynamics()  # starts from INITIAL_GUESS
    before = velocity_prediction_mse(model, states, actions, next_states, torch.device("cpu"))
    trace = model.fit_engine_rules(states, actions, next_states, device=torch.device("cpu"))
    after = velocity_prediction_mse(model, states, actions, next_states, torch.device("cpu"))

    assert after < before
    assert after < 1e-4, f"fit left MSE {after}"
    assert model.params.a_traction_walk == pytest.approx(1.5, abs=0.25)
    assert model.params.a_traction_run == pytest.approx(4.0, abs=0.5)
    assert model.params.a_friction == pytest.approx(1.5, abs=0.25)
    assert model.params.jump_impulse == pytest.approx(-70.0, abs=1.0)
    assert model.params.jump_run_gain == pytest.approx(0.25, abs=0.05)
    # The trace is auditable: one row per parameter per sweep, always improving.
    assert len(trace) >= len(IDENTIFIABLE)
    assert trace[-1]["velocity_mse"] <= trace[0]["velocity_mse"]


def test_unexercised_parameters_are_not_fabricated():
    """If the data never exercises a branch, the scalar must stay at its prior guess.

    Silence here is the honest outcome: a grid search would otherwise "fit" a
    parameter that the observations say nothing about.
    """
    gen = np.random.default_rng(11)
    params = EngineRuleParameters(a_skid=6.0)
    model = AnalyticalKinematicsDynamics(params=params)
    states, actions, next_states = [], [], []
    s = _state(x=100.0, ground=1.0)
    for _ in range(300):  # always pushing right, always on the ground: no skid frames
        a = _action(jump=float(gen.integers(0, 2)), run=1.0, right=1.0)
        nxt = model(s, a)
        states.append(s.numpy())
        actions.append(a.numpy())
        next_states.append(nxt.numpy())
        s = nxt
    fitted = AnalyticalKinematicsDynamics()
    fitted.fit_engine_rules(
        torch.tensor(np.concatenate(states)),
        torch.tensor(np.concatenate(actions)),
        torch.tensor(np.concatenate(next_states)),
        device=torch.device("cpu"),
    )
    assert fitted.params.a_skid == INITIAL_GUESS["a_skid"]


def test_identification_stays_inside_published_ranges():
    states, actions, next_states = _synthetic_transitions(seed=7)
    model = AnalyticalKinematicsDynamics()
    model.fit_engine_rules(states, actions, next_states, sweeps=2, device=torch.device("cpu"))
    for name, (lo, hi, _) in IDENTIFIABLE.items():
        value = getattr(model.params, name)
        assert lo - 1e-9 <= value <= hi + 1e-9, f"{name}={value} left [{lo}, {hi}]"


def test_identification_is_deterministic():
    states, actions, next_states = _synthetic_transitions(seed=3)
    runs = []
    for _ in range(2):
        model = AnalyticalKinematicsDynamics()
        model.fit_engine_rules(states, actions, next_states, sweeps=2, device=torch.device("cpu"))
        runs.append(model.params.as_dict())
    assert runs[0] == runs[1]


def test_every_declared_scalar_has_a_guess_and_a_range():
    assert set(IDENTIFIABLE) == set(INITIAL_GUESS) == set(EngineRuleParameters().__dict__)
    for name, (lo, hi, step) in IDENTIFIABLE.items():
        assert lo < hi and step > 0
        assert lo <= INITIAL_GUESS[name] <= hi, f"initial guess for {name} outside its range"
