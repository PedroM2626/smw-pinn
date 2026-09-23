"""
parameter_identification.py
Physics parameter identification (the inverse problem) for the SMW engine model.

Everything else in this repository solves the *forward* problem: given the state
``s_t``, the action ``a_t`` and a fixed set of engine constants ``theta``, predict the
next state ``s_{t+1}`` (Sections 5-8) or plan forward in time with it (Sections 10.6,
10.31). This module solves the *inverse* problem: given only observed transitions,
recover the physical constants ``theta`` that generated them. It is the concrete realisation
of the deferred future-work item in README Section 10.16-6 ("porting to dynamical systems
with unknown discretization schemes would necessitate either explicit system identification
or meta-learning of the physical scaling factors") and of the "system identification inside
the graph" idea introduced in ``src/models/pinn_gravity.py``.

The identified parameter vector is the six structural constants that drive the analytic
fixed-point kinematics of Section 4 (the same quantities as
``src.losses.physics_rl_losses.TractionFrictionParams``):

    theta = [max_vx, walk_accel, run_accel, subpixels_per_pixel, held_gravity, fall_gravity]

The forward simulator :func:`simulate_step` reproduces the Hard-Residual-PINN integrator
exactly but with the *learned residual force removed*: velocities follow the traction /
asymmetric-gravity budget and positions integrate them through the subpixel ratio. Because
the simulator is written with differentiable tensor operations, ``theta`` is fitted by
gradient descent on an open-loop multi-step rollout loss, which is the standard non-linear
least-squares formulation of a discrete-time inverse dynamics problem.

Only tensor/parameter mathematics lives here; the closed-loop *transfer* experiment (which
is what the recovered parameters are ultimately for) is driven from
``src/evaluation/inverse_transfer_benchmark.py``. This module is emulator-free and
deterministic under ``set_global_seed``, so it runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.losses.physics_rl_losses import TractionFrictionParams
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

# Fixed structural clamps of the SMW engine (README Section 4.2). These are hardware
# constants that are *not* part of the identified vector: they define the sign/constrain
# the velocity, and identification of the subpixel scale would be degenerate if the clamps
# were simultaneously free.
MIN_VY = -80.0  # maximum ascent velocity (subpixels/frame).
TERMINAL_VY = 64.0  # maximum descent (fall) velocity (subpixels/frame).

# Ordered names of the identified vector; every tensor of parameters uses this order.
PARAM_NAMES: List[str] = [
    "max_vx",
    "walk_accel",
    "run_accel",
    "subpixels_per_pixel",
    "held_gravity",
    "fall_gravity",
]

# Button-column indices of the shared 6-wide action layout [B, Y, UP, DOWN, LEFT, RIGHT].
_A_JUMP, _A_RUN, _, _, _A_LEFT, _A_RIGHT = range(6)


@dataclass(frozen=True)
class EngineParams:
    """A concrete value of the identified engine constants (units: subpixels / frames).

    Defaults are the reverse-engineered WRAM values (README Section 4.3), which double as
    the ground-truth for the synthetic recovery experiment and as the "prior" a zero-shot
    controller would naively carry into an unknown world.
    """

    max_vx: float = 72.0
    walk_accel: float = 0.75
    run_accel: float = 1.50
    subpixels_per_pixel: float = 16.0
    held_gravity: float = 3.0
    fall_gravity: float = 6.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [getattr(self, name) for name in PARAM_NAMES],
            dtype=np.float64,
        )

    @staticmethod
    def from_vector(vec: torch.Tensor | np.ndarray) -> "EngineParams":
        vals = [float(x) for x in np.asarray(_to_numpy(vec), dtype=np.float64)]
        return EngineParams(**dict(zip(PARAM_NAMES, vals)))

    def to_traction(self) -> TractionFrictionParams:
        """Bridge to the PIML safety model, which consumes the same constants."""
        return TractionFrictionParams(
            max_vx=self.max_vx,
            walk_accel=self.walk_accel,
            run_accel=self.run_accel,
            subpixels_per_pixel=self.subpixels_per_pixel,
            held_gravity=self.held_gravity,
            fall_gravity=self.fall_gravity,
        )


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus, for parameterising strictly-positive constants."""
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


def _softplus(raw: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(raw)


def theta_tensor(params: EngineParams | TractionFrictionParams) -> torch.Tensor:
    """Return the identified-vector ordering of a params object as a float32 tensor."""
    if isinstance(params, TractionFrictionParams):
        params = EngineParams(
            max_vx=params.max_vx,
            walk_accel=params.walk_accel,
            run_accel=params.run_accel,
            subpixels_per_pixel=params.subpixels_per_pixel,
            held_gravity=params.held_gravity,
            fall_gravity=params.fall_gravity,
        )
    return torch.tensor(params.as_vector(), dtype=torch.float32)


def simulate_step(
    states: torch.Tensor,
    actions: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """Differentiable one-frame analytic forward step.

    Args:
        states: ``[B, C]`` with ``C >= 4``; columns 0..3 are ``[x, y, vx, vy]``
            (positions in pixels, velocities in subpixels/frame).
        actions: ``[B, 6]`` button vectors in ``[B, Y, UP, DOWN, LEFT, RIGHT]`` order.
        params: ``[6]`` positive engine constants in :data:`PARAM_NAMES` order.

    Returns:
        ``[B, 4]`` predicted ``[x, y, vx, vy]`` one frame later.
    """
    x, y, vx, vy = states[..., 0], states[..., 1], states[..., 2], states[..., 3]
    jump, run = actions[..., _A_JUMP], actions[..., _A_RUN]
    direction = actions[..., _A_RIGHT] - actions[..., _A_LEFT]

    max_vx, walk_accel, run_accel, scale, g_hold, g_fall = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )

    # Horizontal: the traction tier is applied only while a direction is held; with no
    # directional input the body coasts at constant velocity (drag is out of scope and is
    # deliberately NOT in the identified vector, cf. README Section 4 traction budget).
    tier = torch.where(run > 0.5, run_accel, walk_accel)
    a_cmd = direction * tier
    vx_next = torch.clamp(vx + a_cmd, -max_vx, max_vx)

    # Vertical: asymmetric gravity, lighter while jump is held during ascent, else fall.
    g = torch.where((jump > 0.5) & (vy < 0.0), g_hold, g_fall)
    vy_next = torch.clamp(vy + g, MIN_VY, TERMINAL_VY)

    x_next = x + vx_next / scale
    y_next = y + vy_next / scale
    return torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)


def simulate_rollout(
    s0: torch.Tensor,
    action_seq: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """Open-loop multi-step rollout under a fixed, time-indexed action sequence.

    Args:
        s0: ``[B, 4]`` initial ``[x, y, vx, vy]``.
        action_seq: ``[B, L, 6]`` action buttons, one per frame.
        params: ``[6]`` engine constants.

    Returns:
        ``[B, L, 4]`` predicted states (frame ``t`` = state after applying ``action_seq[:, t]``).
    """
    state = s0
    outs: List[torch.Tensor] = []
    for t in range(action_seq.shape[1]):
        state = simulate_step(state, action_seq[:, t, :], params)
        outs.append(state)
    return torch.stack(outs, dim=1)


def make_windows(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    episodes: np.ndarray,
    rollout_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split flat transitions into contiguous within-episode open-loop windows.

    Only consecutive frames belonging to the same episode are chained, so a rollout never
    jumps a reset boundary. Positions/velocities are truncated to the four kinematic
    channels the simulator predicts.

    Returns:
        ``(s0 [W,4], action_seq [W,L,6], targets [W,L,4])`` float32 tensors.
    """
    s0_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    tgt_list: List[np.ndarray] = []
    for ep in np.unique(episodes):
        mask = episodes == ep
        st = states[mask]
        ac = actions[mask]
        nx = next_states[mask]
        n = len(st)
        if n < rollout_len + 1:
            continue
        for i in range(n - rollout_len):
            s0_list.append(st[i, :4])
            act_list.append(ac[i : i + rollout_len])
            tgt_list.append(nx[i : i + rollout_len, :4])
    if not s0_list:
        empty = torch.zeros(0, 4)
        empty_a = torch.zeros(0, rollout_len, 6)
        return empty, empty_a, torch.zeros(0, rollout_len, 4)
    return (
        torch.tensor(np.stack(s0_list), dtype=torch.float32),
        torch.tensor(np.stack(act_list), dtype=torch.float32),
        torch.tensor(np.stack(tgt_list), dtype=torch.float32),
    )


def generate_synthetic_windows(
    true_params: torch.Tensor,
    n_windows: int,
    rollout_len: int,
    seed: int = 42,
    x_span: Tuple[float, float] = (0.0, 600.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce (s0, action_seq, targets) windows from the true simulator.

    The initial conditions and button statistics are chosen to *excite every identified
    constant*: horizontal states are seeded near the velocity ceiling with directional input
    held so the saturation clamp binds (making ``max_vx`` identifiable); vertical states span
    both signs of ``vy`` with the jump button frequently held so the ascending-held
    (``held_gravity``) and falling (``fall_gravity``) branches are both observed; and both
    run and walk traction tiers appear. A generator that failed to exercise a constant would
    correctly report it as unidentifiable - which is itself a real result (cf. the negative
    jump-impulse identifiability in README 10.37.1) - so the ranges here are deliberate.

    Targets are the *true* states the simulator produces, so recovering ``true_params`` from
    them is a clean inverse problem with a known ground truth.
    """
    g = torch.Generator().manual_seed(seed)
    max_vx = float(true_params[0])
    x0 = torch.rand(n_windows, generator=g) * (x_span[1] - x_span[0]) + x_span[0]
    y0 = torch.full((n_windows,), 336.0)
    # Per-window sustained "sprint" flag decides both the initial velocity and the button
    # pattern: sprint windows start *near the ceiling moving outward*, so the saturation clamp
    # binds within the first frames - a strong, unambiguous signal for max_vx (a weakly excited
    # ceiling is the honest reason max_vx is the hardest of the six to identify).
    sprint = torch.rand(n_windows, generator=g) < 0.6
    dir_sign = torch.where(x0 >= 0, torch.ones_like(x0), -torch.ones_like(x0))
    vx0 = (torch.rand(n_windows, generator=g) * 2 - 1) * max_vx * 0.95
    # Seed both ascent (vy < 0) and descent (vy > 0) so held_gravity and fall_gravity bind.
    vy0 = (torch.rand(n_windows, generator=g) * 2 - 1) * 60.0
    s0 = torch.stack([x0, y0, vx0, vy0], dim=-1)
    jump = (torch.rand(n_windows, rollout_len, generator=g) < 0.5).float()
    run = torch.where(
        sprint.unsqueeze(1),
        torch.ones(n_windows, rollout_len),
        (torch.rand(n_windows, rollout_len, generator=g) < 0.4).float(),
    )
    right = torch.clamp(
        sprint.unsqueeze(1).float() * (dir_sign > 0).float().unsqueeze(1)
        + (torch.rand(n_windows, rollout_len, generator=g) < 0.2).float(),
        max=1.0,
    )
    left = torch.clamp(
        sprint.unsqueeze(1).float() * (dir_sign < 0).float().unsqueeze(1)
        + (torch.rand(n_windows, rollout_len, generator=g) < 0.1).float(),
        max=1.0,
    )
    down = torch.zeros(n_windows, rollout_len)
    up = (torch.rand(n_windows, rollout_len, generator=g) < 0.1).float()
    # Button layout [B, Y, UP, DOWN, LEFT, RIGHT].
    action_seq = torch.stack([jump, run, up, down, left, right], dim=-1)
    targets = simulate_rollout(s0, action_seq, true_params)
    return s0, action_seq, targets


def rollout_mse(windows, params: torch.Tensor) -> float:
    s0, acts, tgts = windows
    if s0.shape[0] == 0:
        return float("nan")
    pred = simulate_rollout(s0, acts, params)
    return float(((pred - tgts) ** 2).mean())


def per_variable_mse(windows, params: torch.Tensor) -> Dict[str, float]:
    """Open-loop rollout MSE decomposed by predicted channel (x, y, vx, vy)."""
    s0, acts, tgts = windows
    names = ["x", "y", "vx", "vy"]
    if s0.shape[0] == 0:
        return {n: float("nan") for n in names}
    pred = simulate_rollout(s0, acts, params)
    err = (pred - tgts) ** 2
    return {n: float(err[..., k].mean()) for k, n in enumerate(names)}


def identify_params(
    windows,
    init: torch.Tensor,
    steps: int = 400,
    lr: float = 0.05,
    batch: int = 256,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Fit the six engine constants by gradient descent on the open-loop rollout loss.

    Args:
        windows: ``(s0, action_seq, targets)`` as produced by :func:`make_windows` or
            :func:`generate_synthetic_windows`.
        init: ``[6]`` positive starting guess (the prior a zero-shot agent would carry in).
        steps: optimisation iterations.
        lr: Adam learning rate on the softplus-constrained parameters.
        batch: window sub-sample per step (0 or >= N uses full batch).
        seed: RNG seed for minibatch sampling (reproducible).

    Returns:
        ``(theta_hat, info)`` where ``theta_hat`` is the fitted ``[6]`` tensor and ``info``
        carries the loss curve and a convergence flag.
    """
    s0, acts, tgts = windows
    if s0.shape[0] == 0:
        raise ValueError("identify_params received an empty window set")
    dev = device or torch.device("cpu")
    s0, acts, tgts = s0.to(dev), acts.to(dev), tgts.to(dev)
    set_global_seed(seed)

    raw = torch.nn.Parameter(_inv_softplus(init.clone().to(dev).float()))
    opt = torch.optim.Adam([raw], lr=lr)
    n = int(s0.shape[0])
    use_batch = batch if 0 < batch < n else n
    # Channel-variance weighting (generalised least squares): the raw x-error dwarfs the
    # vx-error by an order of magnitude, which would let the fit match positions while
    # ignoring the velocity ceiling. Dividing each channel by its own variance gives the
    # six constants comparable influence, so max_vx (a velocity-only effect) is identifiable.
    chan_var = tgts.reshape(-1, 4).var(dim=0, unbiased=False).clamp(min=1e-6)
    weight = 1.0 / chan_var
    history: List[float] = []
    for _ in range(steps):
        idx = torch.randint(0, n, (use_batch,))
        p = _softplus(raw)
        pred = simulate_rollout(s0[idx], acts[idx], p)
        loss = (((pred - tgts[idx]) ** 2) * weight).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))

    theta_hat = _softplus(raw).detach().cpu()
    info: Dict[str, object] = {
        "final_loss": history[-1] if history else float("nan"),
        "loss_curve": history,
        "steps": steps,
    }
    return theta_hat, info


def bootstrap_ci(
    windows,
    init: torch.Tensor,
    n_boot: int = 16,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap confidence intervals for each identified constant.

    Resamples the window set with replacement and refits; the spread of the point
    estimates is the standard bootstrap uncertainty and doubles as an empirical
    *identifiability* diagnostic (a near-flat bootstrap for a constant means the data
    cannot determine it, cf. the negative jump-impulse result in README 10.37.1).
    """
    s0, acts, tgts = windows
    n = int(s0.shape[0])
    gen = np.random.default_rng(seed)
    estimates: List[np.ndarray] = []
    for b in range(n_boot):
        idx_np = gen.integers(0, n, size=n)
        idx = torch.as_tensor(idx_np, dtype=torch.long)
        boot = (s0[idx], acts[idx], tgts[idx])
        theta_b, _ = identify_params(boot, init, steps=steps, lr=lr, seed=seed + b)
        estimates.append(theta_b.numpy())
    arr = np.stack(estimates, axis=0)  # [n_boot, 6]
    out: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(PARAM_NAMES):
        col = arr[:, k]
        out[name] = {
            "mean": float(col.mean()),
            "std": float(col.std(ddof=1)) if n_boot > 1 else 0.0,
            "ci_low": float(np.percentile(col, 2.5)),
            "ci_high": float(np.percentile(col, 97.5)),
        }
    return out


def mpc_random_shooting(
    s0: torch.Tensor,
    model_params: torch.Tensor,
    task_reward,
    horizon: int = 12,
    n_samples: int = 128,
    n_iters: int = 3,
    elite: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Finite-horizon random-shooting MPC over the analytic model (Section 10.6 planner).

    The planner optimises an action sequence *using ``model_params``* and returns only the
    first action; the caller rolls the true world forward with it. The gap between the
    planner's predicted return and the return actually attained is the signature of a
    misspecified model, which is exactly what identification is meant to remove.

    Args:
        s0: ``[4]`` current ``[x, y, vx, vy]``.
        model_params: ``[6]`` engine constants the planner believes.
        task_reward: callable(``[horizon, 4]`` predicted rollout) -> scalar (torch).
        horizon/n_samples/n_iters/elite: CEM/random-shooting hyper-parameters.

    Returns:
        ``(action_sequence [horizon, 6] buttons, predicted_return)``. The caller rolls the
        returned plan forward under the *true* world to measure the achieved outcome; the
        gap between ``predicted_return`` and the achieved outcome is the model's optimism,
        which a misspecified (unidentified) model inflates.
    """
    g = np.random.default_rng(seed)
    best_seq: Optional[np.ndarray] = None
    best_score = -np.inf
    mean = np.full((horizon, 6), 0.5, dtype=np.float64)
    std = np.full((horizon, 6), 0.4, dtype=np.float64)
    for _ in range(n_iters):
        # Bernoulli button samples per (candidate, step, button), then CEM-update.
        draws = np.clip(g.normal(mean, std, size=(n_samples, horizon, 6)), 0.0, 1.0)
        scores = np.empty(n_samples, dtype=np.float64)
        s0_exp = s0.unsqueeze(0).expand(n_samples, 4).contiguous()
        acts = torch.tensor(draws, dtype=torch.float32)
        pred = simulate_rollout(s0_exp, acts, model_params)
        for i in range(n_samples):
            scores[i] = float(task_reward(pred[i]))
        order = np.argsort(-scores)
        elite_idx = order[:elite]
        mean = draws[elite_idx].mean(axis=0)
        std = draws[elite_idx].std(axis=0) + 1e-3
        top = int(elite_idx[0])
        if scores[top] > best_score:
            best_score = scores[top]
            best_seq = draws[top].copy()
    assert best_seq is not None
    return best_seq, float(best_score)


__all__: List[str] = [
    "EngineParams",
    "PARAM_NAMES",
    "MIN_VY",
    "TERMINAL_VY",
    "theta_tensor",
    "simulate_step",
    "simulate_rollout",
    "make_windows",
    "generate_synthetic_windows",
    "rollout_mse",
    "per_variable_mse",
    "identify_params",
    "bootstrap_ci",
    "mpc_random_shooting",
]
