"""
test_inverse_world_models.py
Unit tests for the MPC-facing wrappers of the two inverse answers (README Section 10.44).

Sections 10.40 and 10.43 produce a parameter vector and a bag of genetic-program trees;
the planner needs an ``nn.Module`` with ``(state, action) -> next_state`` over the 8D
state. These tests pin the conversion down - above all the parity between the wrapper
and the library integrator it delegates to, because the closed-loop comparison is only
meaningful if the model the planner rolls out is the model the study measured.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import src.evaluation.inverse_model_mpc_benchmark as mpc
from src.inverse.parameter_identification import (
    EngineParams,
    simulate_step,
    theta_tensor,
)
from src.inverse.symbolic_regression import parametric_laws
from src.models.inverse_world_models import (
    IdentifiedKinematicsDynamics,
    SymbolicKinematicsDynamics,
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


def _batch(n: int = 5, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    state = torch.zeros(n, 8)
    state[:, 0] = torch.rand(n, generator=g) * 500.0
    state[:, 1] = 336.0
    state[:, 2] = torch.rand(n, generator=g) * 96.0 - 48.0
    state[:, 3] = torch.rand(n, generator=g) * 60.0 - 30.0
    state[:, 4:] = (torch.rand(n, 4, generator=g) > 0.7).float()
    action = torch.zeros(n, 6)
    action[torch.arange(n) % 2, 5] = 1.0
    action[torch.arange(n) % 3 == 0, 1] = 1.0
    action[torch.arange(n) % 4 == 0, 0] = 1.0
    return state, action


def test_identified_wrapper_matches_the_library_integrator() -> None:
    state, action = _batch()
    model = IdentifiedKinematicsDynamics(theta_tensor(WORLD))
    out = model(state, action)
    assert out.shape == (state.shape[0], 8)
    expected = simulate_step(state[:, :4], action, theta_tensor(WORLD), state[:, 4:8])
    assert torch.allclose(out[:, :4], expected, atol=1e-4)
    # The collision byte is not derivable from kinematics: it is propagated unchanged.
    assert torch.equal(out[:, 4:], state[:, 4:])


def test_identified_wrapper_accepts_numpy_and_reports_seven_constants() -> None:
    model = IdentifiedKinematicsDynamics(WORLD.as_vector())
    assert model.num_parameters == 7
    assert "theta" in dict(model.named_buffers())
    restored = IdentifiedKinematicsDynamics(WORLD.as_vector())
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored.theta, model.theta)


def test_identified_wrapper_validates_its_inputs() -> None:
    with pytest.raises(ValueError):
        IdentifiedKinematicsDynamics(np.zeros(5))
    with pytest.raises(ValueError):
        IdentifiedKinematicsDynamics(WORLD.as_vector(), state_dim=12)
    model = IdentifiedKinematicsDynamics(WORLD.as_vector())
    with pytest.raises(ValueError):
        model(torch.zeros(2, 12), torch.zeros(2, 6))


def test_symbolic_wrapper_reproduces_the_parametric_map_through_the_same_interface() -> None:
    """The 10.40 map driven as *laws* must equal it driven as a parameter vector.

    This is the parity that makes the closed-loop comparison honest: the planner sees one
    composition convention, not two.
    """
    state, action = _batch(seed=3)
    laws = parametric_laws(theta_tensor(WORLD).numpy().astype(np.float64), "real")
    symbolic = SymbolicKinematicsDynamics(laws)
    identified = IdentifiedKinematicsDynamics(theta_tensor(WORLD))
    out_sym = symbolic(state, action)
    out_par = identified(state, action)
    assert out_sym.shape == out_par.shape == (state.shape[0], 8)
    assert torch.allclose(out_sym[:, :4], out_par[:, :4], atol=1e-4)
    assert torch.equal(out_sym[:, 4:], state[:, 4:])


def test_symbolic_wrapper_with_a_hand_made_law() -> None:
    class _Law:
        def __init__(self, name: str, features, fn) -> None:
            self.name = name
            self.feature_names = list(features)
            self.n_nodes = 1
            self.expression = f"{name}-hand"
            self._fn = fn

        def increment(self, X: np.ndarray) -> np.ndarray:
            return self._fn(np.atleast_2d(X))

    laws = {
        "dvx": _Law(
            "dvx",
            ["vx", "dir", "run", "ground", "ceiling", "left", "right"],
            lambda X: X[:, 1] * 2.0,
        ),
        "dvy": _Law(
            "dvy",
            ["vy", "jump", "ground", "ceiling", "left", "right"],
            lambda X: np.full(X.shape[0], 5.0),
        ),
        "dx": _Law("dx", ["vx_next"], lambda X: X[:, 0] / 10.0),
        "dy": _Law("dy", ["vy_next"], lambda X: X[:, 0] / 10.0),
    }
    model = SymbolicKinematicsDynamics(laws)
    assert model.num_parameters == 4  # one node per hand-made law
    assert set(model.expressions()) == set(laws)
    state, action = _batch(seed=4)
    out = model(state, action)
    direction = action[:, 5] - action[:, 4]
    vx_next = state[:, 2] + direction * 2.0
    assert torch.allclose(out[:, 2], vx_next, atol=1e-5)
    assert torch.allclose(out[:, 0], state[:, 0] + vx_next / 10.0, atol=1e-5)
    assert torch.allclose(out[:, 3], state[:, 3] + 5.0, atol=1e-5)


def test_symbolic_wrapper_requires_every_law_and_the_right_state_width() -> None:
    with pytest.raises(ValueError):
        SymbolicKinematicsDynamics({"dvx": object(), "dvy": object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        SymbolicKinematicsDynamics(
            {"dvx": object(), "dvy": object(), "dx": object(), "dy": object()},  # type: ignore[dict-item]
            state_dim=12,
        )


def test_wrappers_preserve_device_dtype_and_batch_shape() -> None:
    state, action = _batch(seed=6)
    model = IdentifiedKinematicsDynamics(theta_tensor(WORLD))
    out = model(state.double(), action.double())
    assert out.dtype == torch.float64
    assert out.device == state.device
    big = model(state.repeat(3, 1), action.repeat(3, 1))
    assert big.shape == (state.shape[0] * 3, 8)


def test_action_agreement_counts_shared_frames() -> None:
    """The reference controller agrees with itself; a shorter sequence scores per shared frame."""
    outcomes = {
        "reference": {"action_sequence": "RRRRr"},
        "same": {"action_sequence": "RRRRr"},
        "half": {"action_sequence": "RRrr"},
    }
    out = mpc._agreement(outcomes, "reference")
    assert out["same"]["sequence_agreement"] == 1.0
    assert out["same"]["first_action_agreement"] == 1.0
    assert out["half"]["sequence_agreement"] == pytest.approx(0.5)
    assert out["half"]["first_action_agreement"] == 1.0


def test_closed_loop_figure_is_written(tmp_path) -> None:
    outcomes = {
        "established_wram_engine_rules": {"progress_px": 605.0, "termination": "timeout"},
        "symbolically_discovered_10_43": {"progress_px": 60.0, "termination": "pit/death"},
    }
    agreement = {
        "established_wram_engine_rules": {"sequence_agreement": 1.0},
        "symbolically_discovered_10_43": {"sequence_agreement": 0.12},
    }
    path = str(tmp_path / "figures" / "closed_loop.png")
    assert mpc._render_figure(outcomes, agreement, path) == path
    assert os.path.getsize(path) > 1000


def test_the_study_refuses_to_run_without_the_emulator(monkeypatch) -> None:
    """A missing core must produce a diagnostic and a non-zero exit, never a fabricated row."""
    monkeypatch.setattr(mpc, "hardware_present", lambda: False)
    assert mpc.main([]) == 1
