"""Unit tests for the effective-velocity target of README 10.53.

CPU-only and emulator-free. The claims worth testing here are the ones the study's
prose rests on: that mode ``next`` *is* the published shell rather than a re-derivation,
that the carried convention makes position independent of the weights, that the offset
mode nests it, and that a carried graph satisfies the repository's kinematic predicate at
any tolerance with no physics in the model at all.
"""

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.effective_velocity_benchmark import (
    FAMILIES,
    MECHANISMS,
    MODES,
    PUBLISHED_MECHANISM,
    TIGHTEST_TOLERANCE,
    arm_label,
    aux_dim_for,
    build_arm,
    checkpoint_name,
    checkpoint_registry,
    grid_cells,
    next_velocity_violation_key,
    published_violation_key,
    telemetry_reference,
    training_plan,
    violation_keys,
)
from src.evaluation.rollout_diagnostics import TOLERANCE_LADDER, violation_rate_at_tolerance
from src.models import EffectiveVelocityDynamics, ResidualDynamics, StatisticalMLPDynamics

STATE_DIM = 8
ACTION_DIM = 6
MAX_VX = 72.0


class _StubBase(nn.Module):
    """A base operator that emits a fixed auxiliary vector for every input."""

    def __init__(self, aux: torch.Tensor):
        super().__init__()
        self.aux = aux

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.aux.expand(state.shape[0], -1)


def _states(batch: int = 5, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    state = torch.rand(batch, STATE_DIM, generator=gen) * 20.0
    state[:, 2] = torch.linspace(-90.0, 90.0, batch)  # vx beyond the engine bound
    state[:, 3] = torch.linspace(-100.0, 100.0, batch)
    state[:, 4:] = (state[:, 4:] > 0.5).float()
    return state


def _actions(batch: int = 5, seed: int = 1) -> torch.Tensor:
    return torch.rand(batch, ACTION_DIM, generator=torch.Generator().manual_seed(seed))


def _wrapped(base: nn.Module, mode: str, hard: bool = False) -> EffectiveVelocityDynamics:
    return EffectiveVelocityDynamics(base=base, state_dim=STATE_DIM, mode=mode, hard=hard)


def test_next_mode_is_bit_identical_to_the_published_residual_shell() -> None:
    """The convention axis has to be the only thing the study moves."""
    state, action = _states(), _actions()
    for hard in (False, True):
        base = _StubBase(torch.tensor([[12.0, -30.0, 1.0, 0.0, 0.0, 0.0]]))
        shell = ResidualDynamics(base=base, state_dim=STATE_DIM, hard=hard)
        rebuilt = _wrapped(base, mode="next", hard=hard)
        assert torch.equal(shell(state, action), rebuilt(state, action))


def test_carried_mode_advances_position_with_the_velocity_the_frame_starts_with() -> None:
    state, action = _states(), _actions()
    aux = torch.tensor([[5.0, 9.0, 1.0, 0.0, 0.0, 0.0]])
    model = _wrapped(_StubBase(aux), mode="carried")
    out = model(state, action)

    assert torch.equal(out[:, 0], state[:, 0] + state[:, 2] / 16.0)
    assert torch.equal(out[:, 1], state[:, 1] + state[:, 3] / 16.0)
    assert torch.equal(out[:, 2], state[:, 2] + 5.0), "the reported velocity is still v_{t+1}"


def test_carried_position_does_not_depend_on_the_network_output() -> None:
    """Two different weightings of the same convention predict the same position."""
    state, action = _states(), _actions()
    positions = []
    for scale in (0.0, 1.0, 40.0):
        aux = torch.full((1, STATE_DIM - 2), float(scale))
        positions.append(_wrapped(_StubBase(aux), mode="carried")(state, action)[:, 0])
    assert torch.equal(positions[0], positions[1])
    assert torch.equal(positions[1], positions[2])


def test_offset_mode_nests_the_carried_convention_at_zero_offset() -> None:
    state, action = _states(), _actions()
    carried = _wrapped(_StubBase(torch.tensor([[3.0, 4.0, 1.0, 0.0, 0.0, 0.0]])), mode="carried")(
        state, action
    )
    zero_offset = _wrapped(
        _StubBase(torch.tensor([[3.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])), mode="offset"
    )(state, action)
    assert torch.equal(carried, zero_offset)


def test_offset_mode_can_move_position_and_the_hard_flag_leaves_it_alone() -> None:
    state, action = _states(), _actions()
    aux = torch.tensor([[3.0, 4.0, 16.0, -16.0, 1.0, 0.0, 0.0, 0.0]])
    model = _wrapped(_StubBase(aux), mode="offset", hard=True)
    out = model(state, action)

    assert torch.equal(out[:, 0], state[:, 0] + (state[:, 2] + 16.0) / 16.0)
    assert torch.equal(out[:, 2], (state[:, 2] + 3.0).clamp(-MAX_VX, MAX_VX))
    assert model.bounds_imposed_by_construction is True


def test_hard_flag_bounds_the_reported_velocity_but_never_the_effective_one() -> None:
    """A wall stop displaces Mario by less than his velocity while |vx| can exceed 72."""
    state = torch.tensor([[10.0, 20.0, 140.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    action = torch.zeros(1, ACTION_DIM)
    aux = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    carried = _wrapped(_StubBase(aux), mode="carried", hard=True)(state, action)
    assert float(carried[0, 2]) == MAX_VX, "the reported velocity is saturated"
    assert float(carried[0, 0]) == 10.0 + 140.0 / 16.0, "position keeps the engine's displacement"


def test_carried_graph_satisfies_the_published_predicate_at_any_tolerance() -> None:
    """No penalty, no clamp: the predicate is met because the graph integrates that way."""
    model = _wrapped(
        StatisticalMLPDynamics(
            state_dim=STATE_DIM - 2, action_dim=ACTION_DIM, sensor_dim=STATE_DIM + ACTION_DIM
        ),
        mode="carried",
    )
    model.eval()
    state = _states(batch=1)
    rows = []
    with torch.no_grad():
        for _ in range(30):
            state = model(state, _actions(batch=1))
            rows.append(state.squeeze(0).numpy().copy())
    trajectory = np.stack(rows)
    for tolerance in TOLERANCE_LADDER:
        assert violation_rate_at_tolerance(trajectory, tolerance) == 0.0


def test_auxiliary_width_and_checkpoint_names() -> None:
    assert aux_dim_for("next") == 6
    assert aux_dim_for("carried") == 6
    assert aux_dim_for("offset") == 8, "two extra channels for the effective-velocity head"
    assert published_violation_key(0.2) == "published_violation_at_0.2px"
    assert next_velocity_violation_key(0.002) == "next_velocity_violation_at_0.002px"
    assert len(violation_keys()) == 2 * 4
    assert TIGHTEST_TOLERANCE == min(0.2, 0.05, 0.01, 0.002)


def test_plan_is_the_full_cross_and_only_the_unconstrained_cells_publish() -> None:
    cells = grid_cells()
    assert len(cells) == len(FAMILIES) * len(MODES) * len(MECHANISMS) == 27
    plan = training_plan()
    assert [label for label, *_ in plan[:1]] == [arm_label("MLP", "next", "none")]
    published = [label for label, *_ in plan if label.endswith(f"/{PUBLISHED_MECHANISM}")]
    assert len(published) == 9
    registry = checkpoint_registry()
    assert set(registry) == {
        f"{family.lower()}_{mode}_{PUBLISHED_MECHANISM}" for family in FAMILIES for mode in MODES
    }
    assert set(registry.values()) == {checkpoint_name(label) for label in published}


def test_every_arm_builds_and_returns_a_full_state() -> None:
    state, action = _states(batch=2), _actions(batch=2)
    for family in FAMILIES:
        for mode in MODES:
            for mechanism in MECHANISMS:
                model = build_arm(family, mode, mechanism, STATE_DIM, ACTION_DIM)
                out = model(state, action)
                assert out.shape == (2, STATE_DIM)
                assert model.bounds_imposed_by_construction == (mechanism == "hard")
                assert model.position_is_predicted == (mode != "carried")


def test_telemetry_reference_separates_the_two_conventions() -> None:
    """Synthesise transitions that obey the carried rule and check what is measured."""
    states = np.zeros((4, STATE_DIM), dtype=np.float32)
    states[:, 2] = [0.0, 16.0, 32.0, 48.0]
    states[:, 3] = [0.0, -16.0, 16.0, 0.0]
    next_states = states.copy()
    next_states[:, 0] = states[:, 0] + states[:, 2] / 16.0
    next_states[:, 1] = states[:, 1] + states[:, 3] / 16.0
    next_states[:, 2] = states[:, 2] + 8.0  # every frame accelerates
    next_states[:, 3] = states[:, 3]

    out = telemetry_reference(states, next_states)
    assert out["x_carried_residual_mean_px"] == 0.0
    assert out["x_carried_residual_median_px"] == 0.0
    assert out["x_effective_exact_rate"] == 1.0
    assert out["x_next_residual_mean_px"] == 0.5, "8 sub-pixels of lag per frame over 16"
    assert out["x_next_residual_median_px"] == 0.5
    assert out["y_carried_residual_mean_px"] == 0.0
    assert out["n_transitions"] == 4.0


def test_closed_loop_registry_matches_the_files_the_study_publishes() -> None:
    """A renamed cell would silently drop a row from the console study."""
    from src.evaluation.physics_injection_mpc_benchmark import EFFECTIVE_ARMS, STUDIES

    assert set(EFFECTIVE_ARMS) == set(checkpoint_registry())
    for key, (checkpoint, factory) in EFFECTIVE_ARMS.items():
        assert checkpoint == checkpoint_registry()[key]
        model = factory()
        assert isinstance(model, EffectiveVelocityDynamics)
        assert model(torch.zeros(2, STATE_DIM), torch.zeros(2, ACTION_DIM)).shape == (2, 8)
        assert model.hard is False
    assert STUDIES["effective"][1] == "effective_velocity_mpc_metrics.json"
