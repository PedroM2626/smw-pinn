"""Unit tests for the physics-injection grid of README 10.47.

CPU-only and emulator-free: the shell, the parameterisation and the grid plan are
checked directly, and the integration path is proven equal to the published
Physics-Constrained DeepONet's own forward pass rather than asserted.
"""

import torch
import torch.nn as nn

from src.evaluation.operator_physics_injection_benchmark import (
    MAX_VX,
    MIN_VY,
    REFERENCE_ARMS,
    SOFT_LAMBDAS,
    TERMINAL_VY,
    ResidualDynamics,
    _velocity_compliance,
    arm_label,
    build_arm,
    build_loss,
    grid_cells,
    training_plan,
)
from src.losses.physics_losses import CompositePINNLoss
from src.models import (
    DeepONetDynamics,
    FNODynamics,
    HardResidualPINNDynamics,
    PhysicsConstrainedDeepONetDynamics,
    StatisticalMLPDynamics,
)

STATE_DIM = 8
ACTION_DIM = 6


class _StubBase(nn.Module):
    """A base operator that returns a fixed increment/auxiliary vector."""

    def __init__(self, residual: torch.Tensor):
        super().__init__()
        self.residual = residual
        self.calls = 0

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.residual.expand(state.shape[0], -1)


def _states(batch: int = 6, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    state = torch.rand(batch, STATE_DIM, generator=gen) * 20.0
    state[:, 2] = torch.linspace(-90.0, 90.0, batch)  # vx beyond the engine bound
    state[:, 3] = torch.linspace(-100.0, 100.0, batch)
    state[:, 4:] = (state[:, 4:] > 0.5).float()
    return state


def test_hard_shell_saturates_velocity_and_integrates_position_exactly() -> None:
    state = _states()
    action = torch.zeros(state.shape[0], ACTION_DIM)
    residual = torch.tensor([[200.0, 300.0, 1.0, 0.0, 0.0, 0.0]])
    model = ResidualDynamics(_StubBase(residual), state_dim=STATE_DIM, hard=True)

    out = model(state, action)
    assert torch.all(out[:, 2] == MAX_VX), out[:, 2]
    assert torch.all(out[:, 3] == TERMINAL_VY), out[:, 3]
    assert torch.allclose(out[:, 0], state[:, 0] + MAX_VX / 16.0)
    assert torch.allclose(out[:, 1], state[:, 1] + TERMINAL_VY / 16.0)


def test_unbounded_mode_keeps_position_consistent_but_not_velocity() -> None:
    state = _states()
    action = torch.zeros(state.shape[0], ACTION_DIM)
    residual = torch.tensor([[200.0, -300.0, 0.0, 0.0, 0.0, 0.0]])
    model = ResidualDynamics(_StubBase(residual), state_dim=STATE_DIM, hard=False)

    out = model(state, action)
    assert torch.all(out[:, 2] == state[:, 2] + 200.0), "no saturation without the shell"
    assert torch.all(out[:, 3] == state[:, 3] - 300.0)
    assert torch.allclose(out[:, 0], state[:, 0] + out[:, 2] / 16.0)
    assert torch.allclose(out[:, 1], state[:, 1] + out[:, 3] / 16.0)


def test_wrapper_integration_path_equals_the_published_physics_constrained_deeponet() -> None:
    """The shared shell must be the same object, not a re-derivation, so numeric
    equality against the published class is checkable."""
    state = _states(seed=3)
    action = torch.rand(state.shape[0], ACTION_DIM, generator=torch.Generator().manual_seed(3))
    pc = PhysicsConstrainedDeepONetDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    pc.eval()

    with torch.no_grad():
        sensors = torch.cat([state, action], dim=-1)
        coefficients = pc.branch(sensors)
        basis = pc.trunk(pc.residual_query_coords)
        aux = torch.einsum("bp,qp->bq", coefficients, basis) + pc.output_bias
        wrapped = ResidualDynamics(_StubBase(aux), state_dim=STATE_DIM, hard=True)
        expected = pc(state, action)
        got = wrapped(state, action)

    assert torch.equal(got, expected)


def test_hard_shell_matches_the_published_hard_pinn_invariant() -> None:
    """Both hard families must leave the kinematics residual identically zero."""
    state = _states(seed=5)
    action = torch.zeros(state.shape[0], ACTION_DIM)
    aux = torch.zeros(1, STATE_DIM - 2)
    hard_wrapper = ResidualDynamics(_StubBase(aux), state_dim=STATE_DIM, hard=True)
    published = HardResidualPINNDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    for model in (hard_wrapper, published):
        out = model(state, action)
        vx, vy = out[:, 2], out[:, 3]
        assert bool((vx.abs() <= MAX_VX).all())
        assert bool((vy <= TERMINAL_VY).all() and (vy >= MIN_VY).all())
        assert torch.allclose(out[:, 0], state[:, 0] + vx / 16.0)
        assert torch.allclose(out[:, 1], state[:, 1] + vy / 16.0)


def test_grid_plan_covers_every_cell_except_state_hard() -> None:
    cells = grid_cells()
    assert len(cells) == 15
    assert all(not (target == "state" and mechanism == "hard") for _, target, mechanism in cells)
    families = {family for family, _, _ in cells}
    assert families == {"MLP", "DeepONet", "FNO"}
    plan = training_plan()
    assert len(plan) == 15 + len(REFERENCE_ARMS)
    assert all(
        label == arm_label(family, target, mechanism) or label.startswith("Published_")
        for label, _, family, target, mechanism in plan[: len(cells)]
    )


def test_sensor_dim_defaults_reproduce_the_published_parameter_counts() -> None:
    """10.42 published these counts; the sensor_dim parameter may not move them."""
    counts = {
        "DeepONet": sum(p.numel() for p in DeepONetDynamics().parameters()),
        "PC_DeepONet": sum(p.numel() for p in PhysicsConstrainedDeepONetDynamics().parameters()),
        "FNO": sum(p.numel() for p in FNODynamics(width=32, modes=6, n_layers=2).parameters()),
        "MLP": sum(p.numel() for p in StatisticalMLPDynamics().parameters()),
    }
    assert counts["DeepONet"] == 52744
    assert counts["PC_DeepONet"] == 52742
    assert counts["FNO"] == 14537

    residual_deeponet = DeepONetDynamics(state_dim=6, action_dim=ACTION_DIM, sensor_dim=14)
    assert residual_deeponet.branch[0].in_features == 14
    assert residual_deeponet(torch.zeros(2, 8), torch.zeros(2, 6)).shape == (2, 6)


def test_build_arm_shapes_and_shell_flags() -> None:
    for family in ("MLP", "DeepONet", "FNO"):
        state_model = build_arm(family, "state", "none", STATE_DIM, ACTION_DIM, 64, 32, 6, 2)
        assert state_model(_states(2), torch.zeros(2, ACTION_DIM)).shape == (2, STATE_DIM)
        for mechanism in ("none", "soft", "hard"):
            residual_model = build_arm(
                family, "residual", mechanism, STATE_DIM, ACTION_DIM, 64, 32, 6, 2
            )
            out = residual_model(_states(2), torch.zeros(2, ACTION_DIM))
            assert out.shape == (2, STATE_DIM)
            assert residual_model.bounds_imposed_by_construction == (mechanism == "hard")


def test_soft_mechanism_uses_the_published_composite_penalty() -> None:
    soft = build_loss("soft")
    assert isinstance(soft, CompositePINNLoss)
    assert (soft.lambda_kin, soft.lambda_bound, soft.lambda_contact) == (
        SOFT_LAMBDAS["lambda_kin"],
        SOFT_LAMBDAS["lambda_bound"],
        SOFT_LAMBDAS["lambda_contact"],
    )
    assert isinstance(build_loss("none"), nn.SmoothL1Loss)
    assert isinstance(build_loss("hard"), nn.SmoothL1Loss)


def test_velocity_compliance_sees_an_unbounded_output_as_violating() -> None:
    state = _states(seed=7)
    action = torch.zeros(state.shape[0], ACTION_DIM)
    device = torch.device("cpu")
    aux = torch.tensor([[MAX_VX * 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    unbounded = ResidualDynamics(_StubBase(aux), state_dim=STATE_DIM, hard=False)
    bounded = ResidualDynamics(_StubBase(aux), state_dim=STATE_DIM, hard=True)
    assert _velocity_compliance(unbounded, state, action, device)["out_of_bounds_rate"] == 1.0
    assert _velocity_compliance(bounded, state, action, device)["out_of_bounds_rate"] == 0.0
    assert _velocity_compliance(bounded, state, action, device)["max_abs_vx_predicted"] == MAX_VX


def test_probe_registry_wraps_every_arm_with_the_right_attribution() -> None:
    """The 10.46 probes must know which grid cell's ceiling is the constructor's."""
    from src.evaluation.operator_physics_injection_benchmark import probe_registry

    rows = probe_registry()
    assert len(rows) == 15
    state = _states(2)
    action = torch.zeros(2, ACTION_DIM)
    for label, checkpoint, factory, imposed, shell_constant in rows:
        assert checkpoint == f"physinj_{label.replace('/', '_').lower()}_best.pt"
        model = factory()
        assert model(state, action).shape == (2, STATE_DIM)
        assert imposed == (label.endswith("/hard"))
        if imposed:
            assert shell_constant == MAX_VX


def test_closed_loop_registry_names_the_checkpoints_the_grid_publishes() -> None:
    """A renamed grid cell would otherwise silently drop rows from the closed loop."""
    from src.evaluation.operator_physics_injection_benchmark import arm_label, checkpoint_name
    from src.evaluation.physics_injection_mpc_benchmark import GRID_CHECKPOINTS

    expected = {
        "fno_residual_hard": arm_label("FNO", "residual", "hard"),
        "fno_residual_none": arm_label("FNO", "residual", "none"),
        "fno_state_soft": arm_label("FNO", "state", "soft"),
        "deeponet_residual_none": arm_label("DeepONet", "residual", "none"),
        "deeponet_state_soft": arm_label("DeepONet", "state", "soft"),
        "mlp_residual_none": arm_label("MLP", "residual", "none"),
    }
    assert set(GRID_CHECKPOINTS) == set(expected)
    for key, label in expected.items():
        assert GRID_CHECKPOINTS[key][0] == checkpoint_name(label)
        assert arm_label(*label.split("/")) == label


def test_projection_registry_matches_the_files_10_51_publishes() -> None:
    """A renamed projected arm would silently drop a row from the closed loop."""
    from src.evaluation.physics_injection_mpc_benchmark import PROJECTION_ARMS, STUDIES
    from src.evaluation.projection_cell_benchmark import (
        FAMILIES,
        build_projected,
        projection_checkpoint_name,
    )

    assert set(PROJECTION_ARMS) == {f"{f.lower()}_state_projected" for f in FAMILIES}
    for family in FAMILIES:
        checkpoint, factory = PROJECTION_ARMS[f"{family.lower()}_state_projected"]
        assert checkpoint == projection_checkpoint_name(family)
        model = factory()
        assert model(torch.zeros(2, 8), torch.zeros(2, 6)).shape == (2, 8)
        assert model.bounds_imposed_by_construction is True
        assert isinstance(model, type(build_projected(family)))
    assert set(STUDIES) == {"grid", "projection", "effective", "ode"}
    assert STUDIES["projection"][1] == "projection_cell_mpc_metrics.json"
    assert STUDIES["effective"][1] == "effective_velocity_mpc_metrics.json"
