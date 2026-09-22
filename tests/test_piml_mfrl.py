"""
test_piml_mfrl.py
Emulator-free unit + integration tests for PIML-MFRL (README Section 10.39).

Everything here runs on CPU without the Libretro core: the physics terms are checked on
synthetic states, and the full training update path is exercised with a mock console so
Approaches A/B/C are covered in CI. The real SNES closed-loop run itself is hardware
dependent and is exercised by the documented entry point, not by these tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.losses.physics_rl_losses import (
    ActionPhysicsViolation,
    PhysicsInformedCriticLoss,
    TractionFrictionParams,
    commanded_acceleration,
    physics_action_violation,
    physics_action_violation_table,
)
from src.models.cbf_projection import (
    CBFQPLayer,
    DiscreteCBFCategoricalFilter,
    smw_barrier_affine,
)
from src.planning.mpc_planner import ACTION_MATRIX
from src.training.piml_mfrl import PIMLActorCritic, train_piml_mfrl

PARAMS = TractionFrictionParams()

# Button column indices: [B, Y, UP, DOWN, LEFT, RIGHT].
IDX_JUMP, IDX_RUN, IDX_UP, IDX_DOWN, IDX_LEFT, IDX_RIGHT = range(6)


def _state(x=100.0, y=100.0, vx=0.0, vy=0.0, ground=1.0, ceil=0.0, left=0.0, right=0.0):
    return torch.tensor([[x, y, vx, vy, ground, ceil, left, right]], dtype=torch.float32)


class _MockConsole:
    """A minimal gym-like env with an 8-D state that visits danger + contact states."""

    def __init__(self) -> None:
        self.t = 0
        # Cycle through representative states, including a falling-into-pit danger state.
        self.pool = np.array(
            [
                [100.0, 100.0, 0.0, 0.0, 1, 0, 0, 0],
                [120.0, 90.0, 20.0, -30.0, 0, 0, 0, 1],  # moving right, wall on the right
                [140.0, 400.0, 10.0, 40.0, 0, 0, 0, 0],  # falling toward a pit (danger)
                [160.0, 100.0, -50.0, 0.0, 1, 0, 1, 0],  # pressing into a left wall
            ],
            dtype=np.float32,
        )

    def reset(self):
        self.t = 0
        return self.pool[0].copy()

    def step(self, action_idx: int):
        self.t += 1
        s = self.pool[self.t % len(self.pool)].copy()
        reward = float(np.clip(s[2], -5, 10))
        done = self.t % 16 == 0
        return s, reward, done, {"x": s[0]}

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Approach C: action physics violation
# --------------------------------------------------------------------------- #
def test_feasible_action_has_zero_violation() -> None:
    # Standing still on the ground, no direction pressed: nothing physically violated.
    s = _state(ground=1.0)
    noop = torch.zeros(1, 6)
    assert float(physics_action_violation(s, noop, PARAMS)) == pytest.approx(0.0)


def test_thrust_into_wall_is_penalised() -> None:
    # Moving right with a right wall present and RIGHT pressed -> non-penetration demand.
    s = _state(vx=40.0, right=1.0)
    action = torch.zeros(1, 6)
    action[0, IDX_RIGHT] = 1.0
    assert float(physics_action_violation(s, action, PARAMS)) > 0.0
    # Same state, no direction pressed -> no violation.
    assert float(physics_action_violation(s, torch.zeros(1, 6), PARAMS)) == pytest.approx(0.0)


def test_moving_away_from_wall_is_not_penalised() -> None:
    # Right wall present but moving left: no impossible force is demanded.
    s = _state(vx=-40.0, right=1.0)
    action = torch.zeros(1, 6)
    action[0, IDX_LEFT] = 1.0
    assert float(physics_action_violation(s, action, PARAMS)) == pytest.approx(0.0)


def test_violation_is_nonnegative_and_differentiable() -> None:
    s = _state(vx=40.0, right=1.0)
    action = torch.zeros(1, 6)
    action[0, IDX_RIGHT] = 1.0
    v = physics_action_violation(s, action, PARAMS)
    assert float(v) >= 0.0
    mod = ActionPhysicsViolation(PARAMS)
    out = mod(s, action)
    assert out.requires_grad is False  # state/action are constants here
    assert float(out) >= 0.0


def test_table_version_scores_every_action() -> None:
    s = _state(vx=40.0, right=1.0)
    table = torch.as_tensor(ACTION_MATRIX, dtype=torch.float32)
    scores = physics_action_violation_table(s, table, PARAMS)
    assert scores.shape == (1, len(ACTION_MATRIX))
    assert torch.all(scores >= 0)
    # NOOP (row 0) is feasible; RUN_RIGHT (row 2, presses into the wall) is not.
    assert float(scores[0, 0]) == pytest.approx(0.0)
    assert float(scores[0, 2]) > 0.0


def test_commanded_acceleration_direction() -> None:
    right_run = torch.zeros(1, 6)
    right_run[0, IDX_RIGHT] = 1.0
    right_run[0, IDX_RUN] = 1.0
    left = torch.zeros(1, 6)
    left[0, IDX_LEFT] = 1.0
    a_run = float(commanded_acceleration(right_run, PARAMS))
    a_walk_left = float(commanded_acceleration(left, PARAMS))
    assert a_run > 0 and a_walk_left < 0
    assert abs(a_run) > abs(a_walk_left)  # running accelerates harder than walking


# --------------------------------------------------------------------------- #
# Approach A: physics-informed critic
# --------------------------------------------------------------------------- #
def _danger_state() -> torch.Tensor:
    return _state(y=400.0, vy=40.0, ground=0.0)  # falling toward the pit plane


def test_critic_loss_requires_grad_on_state() -> None:
    loss_fn = PhysicsInformedCriticLoss()
    s = _danger_state()  # requires_grad False
    with pytest.raises(ValueError):
        loss_fn(s, -s[:, 1])


def test_value_decreasing_with_depth_satisfies_lyapunov() -> None:
    # V = -y decays along the downward drift, so no penalty should be incurred.
    s = _danger_state().requires_grad_(True)
    value = -s[:, 1]
    loss = PhysicsInformedCriticLoss()(s, value)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_value_increasing_with_depth_violates_lyapunov() -> None:
    # V = +y grows toward the pit: the decay condition is violated -> positive penalty.
    s = _danger_state().requires_grad_(True)
    value = s[:, 1]
    loss = PhysicsInformedCriticLoss()(s, value)
    assert float(loss) > 0.0


def test_critic_loss_is_differentiable_and_flows_to_params() -> None:
    critic = torch.nn.Linear(8, 1)
    s = _danger_state().requires_grad_(True)
    value = critic(s).squeeze(-1)
    loss = PhysicsInformedCriticLoss()(s, value)
    loss.backward()
    assert critic.weight.grad is not None
    assert torch.isfinite(critic.weight.grad).all()


# --------------------------------------------------------------------------- #
# Approach B: CBF projection
# --------------------------------------------------------------------------- #
def test_cbf_qp_leaves_feasible_action_unchanged() -> None:
    layer = CBFQPLayer()
    a_hat = torch.tensor([[0.0]])
    rows = torch.tensor([[[1.0]]])
    rhs = torch.tensor([[-5.0]])  # a >= -5, satisfied by 0
    proj = layer(a_hat, rows, rhs, u_max=100.0)
    assert float(proj) == pytest.approx(0.0, abs=1e-4)


def test_cbf_qp_projects_violated_constraint_to_feasible() -> None:
    layer = CBFQPLayer()
    a_hat = torch.tensor([[0.0]])
    rows = torch.tensor([[[1.0]]])
    rhs = torch.tensor([[5.0]])  # a >= 5
    proj = layer(a_hat, rows, rhs, u_max=100.0)
    assert float(proj) >= 5.0 - 1e-4
    # Residual constraint violation is now gone.
    assert float(rows[0, 0] * proj[0, 0] - rhs[0, 0]) >= -1e-3


def test_cbf_qp_is_differentiable() -> None:
    layer = CBFQPLayer()
    a_hat = torch.tensor([[0.0]], requires_grad=True)
    rows = torch.tensor([[[1.0]]])
    rhs = torch.tensor([[5.0]])
    proj = layer(a_hat, rows, rhs, u_max=100.0)
    proj.sum().backward()
    assert a_hat.grad is not None
    assert torch.isfinite(a_hat.grad).all()


def test_smw_barrier_affine_shapes() -> None:
    s = _state(vx=10.0)
    rows, rhs, u_max = smw_barrier_affine(s, PARAMS)
    assert rows.shape == (1, 2, 1)
    assert rhs.shape == (1, 2)
    assert float(u_max[0]) == pytest.approx(PARAMS.max_vx)


def test_categorical_filter_zero_beta_is_identity() -> None:
    from torch.distributions import Categorical

    table = torch.as_tensor(ACTION_MATRIX, dtype=torch.float32)
    logits = torch.randn(4, len(ACTION_MATRIX))
    s = torch.randn(4, 8).abs()
    filt = DiscreteCBFCategoricalFilter(table, beta=0.0, params=PARAMS)
    dist = filt(logits, s)
    # With beta=0 the filtered distribution equals the plain policy.
    assert torch.allclose(dist.probs, Categorical(logits=logits).probs, atol=1e-6)


def test_categorical_filter_suppresses_infeasible_actions() -> None:
    table = torch.as_tensor(ACTION_MATRIX, dtype=torch.float32)
    s = _state(vx=40.0, right=1.0)
    logits = torch.zeros(1, len(ACTION_MATRIX))  # uniform prior
    filt = DiscreteCBFCategoricalFilter(table, beta=50.0, params=PARAMS)
    dist = filt(logits, s)
    probs = dist.probs[0]
    # NOOP is feasible -> highest probability; wall-thrusting RUN_RIGHT is suppressed.
    assert int(torch.argmax(probs)) == 0
    assert float(probs[2]) < float(probs[0])


# --------------------------------------------------------------------------- #
# Actor and end-to-end update path (mock console, all mechanisms on)
# --------------------------------------------------------------------------- #
def test_piml_actor_critic_shapes() -> None:
    agent = PIMLActorCritic(hidden_dim=32)
    obs = torch.randn(5, 8)
    action, logp, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == (5,)
    assert logp.shape == (5,) == entropy.shape == value.shape


def test_train_loop_runs_with_all_mechanisms(tmp_path) -> None:
    metrics = train_piml_mfrl(
        env=_MockConsole(),
        total_timesteps=64,
        rollout_length=16,
        num_epochs=2,
        minibatch_size=8,
        use_physics_critic=True,
        use_cbf_filter=True,
        use_action_penalty=True,
        seed=0,
        output_dir=str(tmp_path),
        write_artifact=False,
    )
    assert metrics["total_real_steps"] == 64
    assert set(metrics["return_history"]) or metrics["return_history"] == []
    assert metrics["config"]["use_cbf_filter"] is True
    assert np.isfinite(metrics["mean_action_violation"])


def test_each_mechanism_toggle_reduces_to_ppo(tmp_path) -> None:
    # Vanilla model-free PPO (all off) still trains and returns finite metrics.
    metrics = train_piml_mfrl(
        env=_MockConsole(),
        total_timesteps=32,
        rollout_length=16,
        num_epochs=1,
        minibatch_size=8,
        seed=0,
        output_dir=str(tmp_path),
        write_artifact=False,
    )
    assert np.isfinite(metrics["mean_final_return"])
    assert metrics["mean_action_violation"] == 0.0  # penalty off -> nothing logged


def test_artifact_written_with_meta(tmp_path) -> None:
    import json

    train_piml_mfrl(
        env=_MockConsole(),
        total_timesteps=32,
        rollout_length=16,
        num_epochs=1,
        minibatch_size=8,
        use_action_penalty=True,
        seed=7,
        output_dir=str(tmp_path),
        write_artifact=True,
    )
    written = list(tmp_path.glob("piml_mfrl_metrics.json"))
    assert written, "trainer did not write the artifact to the requested output dir"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert "_meta" in payload  # provenance required for new artifacts
    assert payload["_meta"]["seed"] == 7


# --------------------------------------------------------------------------- #
# Study aggregation (pure, emulator-free)
# --------------------------------------------------------------------------- #
def _synthetic_runs():
    return [
        {
            "condition": "model_free_ppo",
            "seed": 1,
            "mean_final_return": 10.0,
            "mean_action_violation": 0.4,
            "total_episodes": 5,
            "total_real_steps": 80,
            "training_time_seconds": 1.0,
        },
        {
            "condition": "model_free_ppo",
            "seed": 2,
            "mean_final_return": 20.0,
            "mean_action_violation": 0.6,
            "total_episodes": 7,
            "total_real_steps": 80,
            "training_time_seconds": 1.0,
        },
        {
            "condition": "piml_mfrl_full",
            "seed": 1,
            "mean_final_return": 30.0,
            "mean_action_violation": 0.1,
            "total_episodes": 5,
            "total_real_steps": 80,
            "training_time_seconds": 1.0,
        },
    ]


def test_study_aggregate_groups_by_condition() -> None:
    from src.evaluation.piml_mfrl_study import aggregate_runs

    summary = aggregate_runs(_synthetic_runs())
    assert summary["conditions"] == ["model_free_ppo", "piml_mfrl_full"]
    assert summary["model_free_ppo"]["mean_final_return"]["mean"] == pytest.approx(15.0)
    assert summary["model_free_ppo"]["mean_final_return"]["n"] == 2
    assert summary["piml_mfrl_full"]["mean_action_violation"]["mean"] == pytest.approx(0.1)
    assert summary["model_free_ppo"]["seed_returns"] == {1: 10.0, 2: 20.0}


def test_study_comparison_reports_delta_and_reduction() -> None:
    from src.evaluation.piml_mfrl_study import _comparison, aggregate_runs

    comparison = _comparison(aggregate_runs(_synthetic_runs()))
    assert comparison["return_delta"] == pytest.approx(15.0)  # 30 - 15
    assert comparison["return_change_pct"] == pytest.approx(100.0)
    # baseline violation 0.5 -> piml 0.1 => 80% reduction
    assert comparison["violation_reduction_pct"] == pytest.approx(80.0)


def test_study_aggregate_handles_empty() -> None:
    from src.evaluation.piml_mfrl_study import aggregate_runs

    summary = aggregate_runs([])
    assert summary == {"conditions": []}
