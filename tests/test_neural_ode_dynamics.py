r"""Unit tests for the NeuralODE integrator arm of README 10.55.

CPU-only and emulator-free. The section's claims are all of one shape - "this solver *is*
that arm, numerically" - so each one is tested against the object it is supposed to equal
rather than against a hand-computed expectation, and the order of the integration is checked
where the algebra is not obvious (midpoint and rk4 on a constant-acceleration field).
"""

import pytest
import torch
import torch.nn as nn

from src.evaluation.neural_ode_integrator_benchmark import (
    EXPECTED_C,
    FAMILIES,
    MECHANISMS,
    aux_dim_for,
    build_arm,
    checkpoint_name,
    checkpoint_registry,
    grid_cells,
    publishes,
    second_order_coefficient,
    training_plan,
)
from src.models import (
    EffectiveVelocityDynamics,
    NeuralODEDynamics,
    ResidualDynamics,
    StatisticalMLPDynamics,
)
from src.models.neural_ode_dynamics import FIELD_EVALUATIONS, SOLVERS

STATE_DIM = 8
ACTION_DIM = 6
AUX_DIM = STATE_DIM - 2


class _ConstantField(nn.Module):
    """A field with a fixed acceleration and fixed contact logits."""

    def __init__(self, accel: torch.Tensor):
        super().__init__()
        self.aux = torch.cat([accel.reshape(-1), torch.tensor([1.0, 0.0, 0.0, 0.0])])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.aux.unsqueeze(0).expand(state.shape[0], -1)


def _states(batch: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    state = torch.rand(batch, STATE_DIM, generator=gen) * 20.0
    state[:, 2] = torch.linspace(-30.0, 90.0, batch)
    state[:, 3] = torch.linspace(-100.0, 100.0, batch)
    state[:, 4:] = (state[:, 4:] > 0.5).float()
    return state


def _action(batch: int = 4) -> torch.Tensor:
    return torch.zeros(batch, ACTION_DIM)


def test_euler_is_the_carried_convention_of_10_53_bit_for_bit() -> None:
    state, action = _states(), _action()
    accel = torch.tensor([3.0, -4.0])
    euler = NeuralODEDynamics(base=_ConstantField(accel), solver="euler")
    carried = EffectiveVelocityDynamics(base=_ConstantField(accel), mode="carried")
    assert torch.equal(euler(state, action), carried(state, action))


def test_symplectic_is_the_published_shell_bit_for_bit() -> None:
    """The velocity-first split is what every shell since 10.27 has done."""
    state, action = _states(seed=3), _action()
    accel = torch.tensor([3.0, -4.0])
    symplectic = NeuralODEDynamics(base=_ConstantField(accel), solver="symplectic")
    shell = ResidualDynamics(base=_ConstantField(accel), hard=False)
    assert torch.equal(symplectic(state, action), shell(state, action))

    big = torch.tensor([900.0, 900.0])
    bounded = NeuralODEDynamics(base=_ConstantField(big), solver="symplectic", bounded=True)
    hard_shell = ResidualDynamics(base=_ConstantField(big), hard=True)
    assert torch.equal(bounded(state, action), hard_shell(state, action))


def test_second_order_coefficients_follow_the_method_not_the_order() -> None:
    r"""On a constant field the position step is $v_t/16 + c\,a/16$ with a known $c$."""
    state, action = _states(seed=5, batch=1), _action(1)
    accel = torch.tensor([[8.0, -12.0]])
    moved = []
    for solver in SOLVERS:
        model = NeuralODEDynamics(base=_ConstantField(accel), solver=solver)
        out = model(state, action)
        moved.append((solver, out[:, :2] - state[:, :2]))
    for solver, delta in moved:
        fitted = second_order_coefficient(
            delta.numpy(), state[:, 2:4].numpy(), accel.expand(1, 2).numpy()
        )
        assert fitted["c_x"] == pytest.approx(EXPECTED_C[solver], abs=1e-5), solver
        assert fitted["c_y"] == pytest.approx(EXPECTED_C[solver], abs=1e-5), solver


def test_midpoint_and_rk4_agree_on_a_constant_field_and_differ_on_a_real_one() -> None:
    state, action = _states(seed=7, batch=1), _action(1)
    midpoint = NeuralODEDynamics(base=_ConstantField(torch.tensor([[5.0, 5.0]])), solver="midpoint")
    rk4 = NeuralODEDynamics(base=_ConstantField(torch.tensor([[5.0, 5.0]])), solver="rk4")
    assert torch.allclose(midpoint(state, action)[:, :4], rk4(state, action)[:, :4], atol=1e-5)

    net = StatisticalMLPDynamics(state_dim=AUX_DIM, action_dim=ACTION_DIM, sensor_dim=14)
    net.eval()
    with torch.no_grad():
        # The same weights under both methods: a field that varies with the state separates
        # the second-order methods, which is why they are separate arms and not one.
        mp = NeuralODEDynamics(base=net, solver="midpoint")(state, action)
        rk = NeuralODEDynamics(base=net, solver="rk4")(state, action)
    assert not torch.allclose(mp[:, :4], rk[:, :4], atol=1e-6)


def test_bounding_projects_inside_the_step_and_reports_itself() -> None:
    state = torch.tensor([[10.0, 20.0, 200.0, 200.0, 1.0, 0.0, 0.0, 0.0]])
    action = _action(1)
    model = NeuralODEDynamics(
        base=_ConstantField(torch.tensor([[0.0, 0.0]])), solver="euler", bounded=True
    )
    out = model(state, action)
    assert float(out[0, 2]) == 72.0 and float(out[0, 3]) == 64.0
    assert model.bounds_imposed_by_construction is True
    # Position is still advanced with the velocity the step started with, which the engine
    # had already applied: the projection constrains the state, not the arithmetic.
    assert float(out[0, 0]) == pytest.approx(10.0 + 200.0 / 16.0)


def test_contact_channels_are_read_once_and_passed_through() -> None:
    state, action = _states(seed=9), _action()
    model = NeuralODEDynamics(base=_ConstantField(torch.tensor([[1.0, 1.0]])), solver="rk4")
    out = model(state, action)
    assert out.shape == (state.shape[0], STATE_DIM)
    assert torch.equal(out[:, 4:8], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, 4))


def test_field_evaluation_cost_is_what_makes_rk4_the_expensive_arm() -> None:
    assert FIELD_EVALUATIONS == {"euler": 1, "symplectic": 1, "midpoint": 2, "rk4": 4}
    for solver in SOLVERS:
        model = NeuralODEDynamics(base=_ConstantField(torch.tensor([[1.0, 1.0]])), solver=solver)
        assert model.field_evaluations == FIELD_EVALUATIONS[solver]


def test_plan_is_the_full_cross_and_only_the_free_cells_publish() -> None:
    assert grid_cells()[0] == ("MLP", "euler", "free")
    assert len(grid_cells()) == len(FAMILIES) * len(SOLVERS) * len(MECHANISMS) == 24
    assert len(training_plan()) == 24
    published = [label for label, *_ in training_plan() if publishes(label)]
    assert len(published) == 12
    assert all(label.endswith("/free") for label in published)
    registry = checkpoint_registry()
    assert set(registry) == {f"{f.lower()}_{s}_free" for f in FAMILIES for s in SOLVERS}
    assert set(registry.values()) == {checkpoint_name(label) for label in published}


def test_every_arm_builds_and_emits_a_full_state() -> None:
    state, action = _states(batch=2), _action(2)
    assert aux_dim_for() == 6
    for family in FAMILIES:
        for solver in SOLVERS:
            for mechanism in MECHANISMS:
                model = build_arm(family, solver, mechanism, STATE_DIM, ACTION_DIM)
                out = model(state, action)
                assert out.shape == (2, STATE_DIM)
                assert model.bounds_imposed_by_construction == (mechanism == "bounded")


def test_closed_loop_registry_matches_the_checkpoints_the_grid_publishes() -> None:
    from src.evaluation.physics_injection_mpc_benchmark import STUDIES

    assert set(STUDIES) == {"grid", "projection", "effective", "ode", "corrected"}
    assert STUDIES["ode"][1] == "neural_ode_integrator_mpc_metrics.json"


def test_the_grid_reproduces_its_published_weights_byte_for_byte() -> None:
    """The control behind 10.55's parity claim: re-running the command changed nothing.

    `results/checkpoints/ode_published.sha256` was written from the *first* run of
    ``--seeds 42,43,44,45,46`` and the twelve files in the repository come from the second.
    Their hashes matching is what licenses "the same convention, second code path" gaps are
    the parameterisation rather than the machine - and it is why the clamped cells, which
    publish no weights and did move between those two runs, carry no claim in the section.
    """
    import hashlib
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    listing = repo / "results" / "checkpoints" / "ode_published.sha256"
    assert listing.is_file(), "the published-checkpoint hash listing is part of 10.55's evidence"
    lines = [line for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 12, "twelve unconstrained arms publish weights"
    for line in lines:
        digest, _, recorded = line.partition(" ")
        target = repo / recorded.strip().lstrip("*")
        assert target.is_file(), f"{target} is gone; the section's parity control has no evidence"
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, (
            f"{target.name} was regenerated after 10.55's closed loop flew it"
        )
