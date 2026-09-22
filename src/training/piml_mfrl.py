"""
piml_mfrl.py
Physics-Informed Model-Free Reinforcement Learning (PIML-MFRL), README Section 10.39.

Trains a genuine model-free PPO agent *directly on the authentic Super Mario World
console* (the ctypes Snes9x Libretro core through ``SnesSingleEnv``), but couples the
known discrete engine physics into the update in three independently toggleable ways:

* **Approach A - Physics-Informed Critic** (``--use-physics-critic``): a Control-
  Lyapunov / HJB decay penalty ``grad_s V . s_dot <= -alpha V`` on danger states, added
  to the critic objective (``src/losses/physics_rl_losses.PhysicsInformedCriticLoss``).
* **Approach B - Physics-Constrained Actor / CBF filter** (``--use-cbf-filter``): a
  differentiable projection of the policy onto the safe-action set before execution
  (``src/models/cbf_projection.DiscreteCBFCategoricalFilter``).
* **Approach C - Physical regularisation of the PPO surrogate** (``--use-action-penalty``):
  a penalty on actions that demand impossible contact / over-saturation forces, added
  to the clipped surrogate (``src/losses/physics_rl_losses.ActionPhysicsViolation``).

None of the three rolls the state forward, so the critic still only estimates the
return of *real* observed states: the agent stays model-free while being physics-aware.

The entry point runs on real hardware (needs the Libretro core + ROM); the trainer
accepts an injected environment so the exact same update path is exercised in CI by
``tests/test_piml_mfrl.py`` with an emulator-free mock console.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.losses.physics_rl_losses import (
    PhysicsInformedCriticLoss,
    TractionFrictionParams,
    physics_action_violation,
)
from src.models.cbf_projection import DiscreteCBFCategoricalFilter
from src.planning.mpc_planner import ACTION_MATRIX
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import RESULTS_DIR
from src.utils.provenance import write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

STATE_DIM = 8
NUM_ACTIONS = 8


class PIMLActorCritic(nn.Module):
    """Actor-critic with an optional discrete CBF safety filter on the action head.

    Mirrors the architecture of ``src.training.dyna_ppo.ActorCritic`` (LayerNorm input,
    two hidden Tanh layers) but exposes the raw logits so the Categorical policy can be
    projected through :class:`DiscreteCBFCategoricalFilter` when ``cbf_filter`` is set.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        cbf_filter: Optional[DiscreteCBFCategoricalFilter] = None,
    ) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(state_dim)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.cbf_filter = cbf_filter

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(self.in_norm(state)).squeeze(-1)

    def _dist(self, state: torch.Tensor, logits: torch.Tensor) -> Categorical:
        if self.cbf_filter is not None:
            return self.cbf_filter(logits, state)
        return Categorical(logits=logits)

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.in_norm(state)
        logits = self.actor(x)
        dist = self._dist(state, logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(state)


def _make_real_env() -> object:
    """Build the real SNES environment lazily (keeps the module importable without hw)."""
    from src.training.model_free_ppo import SnesSingleEnv

    return SnesSingleEnv()


def train_piml_mfrl(
    env: Optional[object] = None,
    total_timesteps: int = 40000,
    rollout_length: int = 128,
    num_epochs: int = 4,
    minibatch_size: int = 32,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    use_physics_critic: bool = False,
    use_cbf_filter: bool = False,
    use_action_penalty: bool = False,
    lambda_critic: float = 0.1,
    lambda_action: float = 0.01,
    cbf_beta: float = 1.0,
    lyapunov_alpha: float = 1.0,
    seed: int = 42,
    output_dir: str = RESULTS_DIR,
    write_artifact: bool = True,
) -> Dict:
    """Run PIML-MFRL and (optionally) publish ``results/piml_mfrl_metrics.json``.

    Args:
        env: an environment exposing ``reset()/step(int)/close()`` with an 8-D state and
            an integer action in ``[0, 8)``. When ``None`` the real SNES console is used.
        use_physics_critic / use_cbf_filter / use_action_penalty: enable Approaches A/B/C.
        lambda_critic / lambda_action: weights of Approaches A and C in the objective.
        cbf_beta: projection strength of Approach B.
        write_artifact: when False, run without writing any file (used by tests).

    Returns:
        The metrics dictionary (also written to disk when ``write_artifact`` is True).
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_table = torch.as_tensor(ACTION_MATRIX, dtype=torch.float32, device=device)

    params = TractionFrictionParams()
    cbf_filter = (
        DiscreteCBFCategoricalFilter(action_table.cpu(), beta=cbf_beta, params=params).to(device)
        if use_cbf_filter
        else None
    )
    agent = PIMLActorCritic(cbf_filter=cbf_filter).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, eps=1e-5)

    critic_loss_fn = PhysicsInformedCriticLoss(alpha=lyapunov_alpha, params=params)
    action_table_np = np.asarray(ACTION_MATRIX, dtype=np.float32)

    own_env = env is None
    environment = _make_real_env() if own_env else env
    assert environment is not None  # for mypy; _make_real_env always returns an env

    logger.info("=== PIML-MFRL: model-free PPO + physics (A/B/C) on real SNES ===")
    logger.info(
        "critic(A)=%s cbf(B)=%s action_penalty(C)=%s | device=%s",
        use_physics_critic,
        use_cbf_filter,
        use_action_penalty,
        device,
    )

    obs = np.asarray(environment.reset(), dtype=np.float32)  # type: ignore[attr-defined]
    global_step = 0
    t0 = time.time()
    episode_returns: List[float] = []
    step_history: List[int] = []
    return_history: List[float] = []
    violation_history: List[float] = []
    curr_return = 0.0

    num_iterations = max(1, total_timesteps // rollout_length)

    for iteration in range(1, num_iterations + 1):
        obs_buf = np.zeros((rollout_length, STATE_DIM), dtype=np.float32)
        act_buf = np.zeros(rollout_length, dtype=np.int64)
        rew_buf = np.zeros(rollout_length, dtype=np.float32)
        done_buf = np.zeros(rollout_length, dtype=np.float32)
        val_buf = np.zeros(rollout_length, dtype=np.float32)
        logp_buf = np.zeros(rollout_length, dtype=np.float32)

        for t in range(rollout_length):
            global_step += 1
            obs_buf[t] = obs
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, _, val = agent.get_action_and_value(obs_t)
            act_buf[t] = action.item()
            val_buf[t] = val.item()
            logp_buf[t] = logp.item()

            next_obs, rew, done, _info = environment.step(int(action.item()))  # type: ignore[attr-defined]
            rew_buf[t] = rew
            done_buf[t] = float(done)
            curr_return += rew
            if done:
                episode_returns.append(curr_return)
                step_history.append(global_step)
                return_history.append(curr_return)
                curr_return = 0.0
                obs = np.asarray(environment.reset(), dtype=np.float32)  # type: ignore[attr-defined]
            else:
                obs = np.asarray(next_obs, dtype=np.float32)

        with torch.no_grad():
            bootstrap = agent.get_value(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ).item()

        advantages = np.zeros(rollout_length, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(rollout_length)):
            non_terminal = 1.0 - done_buf[t]
            next_val = bootstrap if t == rollout_length - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * non_terminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        t_obs = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        t_act = torch.as_tensor(act_buf, dtype=torch.int64, device=device)
        t_logp = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        t_adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        t_ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        indices = np.arange(rollout_length)
        iter_violation = 0.0
        iter_critic_phys = 0.0
        n_mb = 0
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_length, minibatch_size):
                mb = indices[start : start + minibatch_size]
                mb_obs = t_obs[mb]
                _, new_logp, entropy, new_val = agent.get_action_and_value(mb_obs, t_act[mb])
                ratio = torch.exp(new_logp - t_logp[mb])

                mb_adv = t_adv[mb]
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((new_val - t_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss

                # Approach C: penalise the executed action's physics violation.
                if use_action_penalty:
                    buttons = torch.as_tensor(
                        action_table_np[t_act[mb].cpu().numpy()], dtype=torch.float32, device=device
                    )
                    viol = physics_action_violation(mb_obs, buttons, params).mean()
                    loss = loss + lambda_action * viol
                    iter_violation += float(viol.item())

                # Approach A: Lyapunov/HJB decay penalty on the critic.
                if use_physics_critic:
                    state_g = mb_obs.detach().clone().requires_grad_(True)
                    value_g = agent.get_value(state_g)
                    phys = critic_loss_fn(state_g, value_g)
                    loss = loss + lambda_critic * phys
                    iter_critic_phys += float(phys.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                n_mb += 1

        if n_mb:
            violation_history.append(iter_violation / n_mb)
        if iteration % 25 == 0 or iteration == num_iterations:
            elapsed = time.time() - t0
            mean_ret = float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
            logger.info(
                "Iter %3d/%d | steps=%6d | mean_ret(last20)=%+7.1f | fps=%5.0f",
                iteration,
                num_iterations,
                global_step,
                mean_ret,
                global_step / max(1e-6, elapsed),
            )

    if own_env:
        environment.close()  # type: ignore[attr-defined]

    metrics: Dict[str, object] = {
        "config": {
            "use_physics_critic": use_physics_critic,
            "use_cbf_filter": use_cbf_filter,
            "use_action_penalty": use_action_penalty,
            "lambda_critic": lambda_critic,
            "lambda_action": lambda_action,
            "cbf_beta": cbf_beta,
            "lyapunov_alpha": lyapunov_alpha,
            "total_timesteps": total_timesteps,
        },
        "total_real_steps": global_step,
        "training_time_seconds": time.time() - t0,
        "mean_final_return": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
        "total_episodes": len(episode_returns),
        "mean_action_violation": float(np.mean(violation_history)) if violation_history else 0.0,
        "step_history": step_history,
        "return_history": return_history,
    }

    if write_artifact:
        path = _write_artifact(metrics, output_dir, seed)
        logger.info("PIML-MFRL metrics written to: %s", path)
    return metrics


def _write_artifact(metrics: Dict[str, object], output_dir: str, seed: int) -> str:
    from src.utils.paths import results_file

    target = results_file("piml_mfrl_metrics.json")
    if output_dir and os.path.abspath(output_dir) != os.path.abspath(RESULTS_DIR):
        os.makedirs(output_dir, exist_ok=True)
        target = os.path.join(output_dir, "piml_mfrl_metrics.json")
    return write_metrics(
        target,
        metrics,
        seed=seed,
        command="python -m src.training.piml_mfrl",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML file; CLI flags override it")
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=40000)
    parser.add_argument("--rollout-length", dest="rollout_length", type=int, default=128)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", dest="minibatch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", dest="clip_coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", dest="ent_coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", dest="vf_coef", type=float, default=0.5)
    parser.add_argument("--lambda-critic", dest="lambda_critic", type=float, default=0.1)
    parser.add_argument("--lambda-action", dest="lambda_action", type=float, default=0.01)
    parser.add_argument("--cbf-beta", dest="cbf_beta", type=float, default=1.0)
    parser.add_argument("--lyapunov-alpha", dest="lyapunov_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    parser.add_argument("--use-physics-critic", dest="use_physics_critic", action="store_true")
    parser.add_argument("--use-cbf-filter", dest="use_cbf_filter", action="store_true")
    parser.add_argument("--use-action-penalty", dest="use_action_penalty", action="store_true")
    parser.add_argument(
        "--use-all",
        dest="use_all",
        action="store_true",
        help="enable Approaches A, B and C together (PIML-MFRL full)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    flags = vars(args)
    if flags.pop("use_all", False):
        flags["use_physics_critic"] = True
        flags["use_cbf_filter"] = True
        flags["use_action_penalty"] = True
    flags.pop("config", None)
    train_piml_mfrl(**flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
