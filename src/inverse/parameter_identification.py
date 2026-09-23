"""Inverse-problem core: identify the engine's physics constants from
transition data and use the recovered parameters for zero-shot control
transfer (README section 10.40).

The forward problem the rest of this repository solves is *"given a world
model, predict / plan"*.  This module inverts it: *"given observed
transitions, recover the physical constants that generated them"*.  The
generator is the exact one-step kinematics encoded by the analytic
traction/friction parameter set
(``src.losses.physics_rl_losses.TractionFrictionParams``), re-derived here as
a closed-form, fully differentiable integrator so the constants can be fit by
gradient descent and -- crucially -- evaluated analytically in the MPC planner
of section 10.36, unlike the learned networks.

The analytic world model is the extended hybrid kinematic map:
  * horizontal: directional input accelerates ``vx`` toward ``+/-max_vx``
    (the run tier applies when the run button is held); releasing every
    direction applies Coulomb-style friction ``decel`` toward zero;
  * vertical: held-jump ascent integrates ``g_hold``, everything else
    ``g_fall``, clamped to the engine's jump-impulse and terminal-velocity
    bounds (the asymmetry the residual PINN is built to learn);
  * ground contact: when the grounded flag is set, downward velocity is
    reset to zero (Mario cannot sink through the floor).

Why this is a clean inverse problem.  Six of the seven constants are
*genuinely free* in the engine (they are not derivable from tile geometry);
the jump impulse and terminal velocity are fixed structural bounds.  Fitting
the free constants against rollouts, with a per-channel variance weighting so
the sub-pixel velocity channels are not swamped by the wide horizontal
position channel, recovers them to sub-percent accuracy on synthetic data
and improves real-data prediction -- and a Laplace/Gauss-Newton posterior
turns the point estimate into a full uncertainty statement, exposing which
directions of parameter space the data can and cannot resolve.

No emulator and no network: the whole pipeline runs on CPU from a recorded
dataset (or a synthetic ground truth), so it is a hardware-free, deterministic
gate.  ``src/evaluation/inverse_transfer_benchmark.py`` composes these into
the published benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.losses.physics_rl_losses import TractionFrictionParams

# Fixed structural bounds of the one-step map: the upward jump-impulse cap
# and the downward terminal velocity, in sub-pixels per frame.  These are not
# identified -- they are the clamp levels the engine hard-codes.
MIN_VY = -80.0
TERMINAL_VY = 64.0

# The seven identified constants, in the fixed order of the theta vector.
PARAM_NAMES = [
    "max_vx",
    "walk_accel",
    "run_accel",
    "subpixels_per_pixel",
    "held_gravity",
    "fall_gravity",
    "decel",
]

# Action bit indices, matching ACTION_NAMES in src/training/physics_rl_losses.py.
_A_JUMP, _A_RUN, _A_LEFT, _A_RIGHT = 0, 1, 4, 5


@dataclass(frozen=True)
class EngineParams:
    """The physical constants of the extended one-step kinematic map.

    Defaults are the WRAM-measured Super Mario World values used throughout
    this repository (README section 2); they double as the informative prior
    for the identification experiments.  ``decel`` is the coast-down
    friction applied when no direction is held, and ``ground_vy_reset``
    encodes the (structural, not free) floor contact handled in
    :func:`simulate_step`.
    """

    max_vx: float = 72.0
    walk_accel: float = 0.75
    run_accel: float = 1.5
    subpixels_per_pixel: float = 16.0
    held_gravity: float = 3.0
    fall_gravity: float = 6.0
    decel: float = 0.5

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.max_vx,
                self.walk_accel,
                self.run_accel,
                self.subpixels_per_pixel,
                self.held_gravity,
                self.fall_gravity,
                self.decel,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray | Tensor) -> "EngineParams":
        arr = np.asarray(
            vec.detach().cpu().numpy() if isinstance(vec, Tensor) else vec, dtype=np.float64
        )
        return cls(*arr)

    def to_traction(self) -> TractionFrictionParams:
        """Bridge to the analytic MPC planner (``src/planning/mpc_planner``).

        The planner's ``decel`` is fixed (0.0), so this bridge carries the six
        constants it models; the identified coast-down friction is used only
        by :func:`simulate_step` here.
        """
        return TractionFrictionParams(
            max_vx=self.max_vx,
            walk_accel=self.walk_accel,
            run_accel=self.run_accel,
            subpixels_per_pixel=self.subpixels_per_pixel,
            held_gravity=self.held_gravity,
            fall_gravity=self.fall_gravity,
        )


def theta_tensor(params: EngineParams | Tensor) -> Tensor:
    """Return ``params`` (an :class:`EngineParams` or a length-7 tensor) as a
    float32 tensor of the seven constants."""
    if isinstance(params, Tensor):
        return params.detach().to(torch.float32).clone()
    return torch.tensor(params.as_vector(), dtype=torch.float32)


def _softplus_inv(y: Tensor) -> Tensor:
    """Inverse of ``F.softplus`` on the positive orthant; guarantees the
    unconstrained optimiser stays in the physically positive domain even at
    the zero boundary (clamped to a small floor)."""
    y = y.clamp(min=1e-6)
    return y + torch.log(-torch.expm1(-y))


def simulate_step(
    states: Tensor, actions: Tensor, params: Tensor, contact: Optional[Tensor] = None
) -> Tensor:
    """One differentiable frame of the extended kinematic map.

    ``states`` is ``[..., 4]`` as ``[x, y, vx, vy]`` (position in pixel units,
    velocity in sub-pixels per frame); ``actions`` is ``[..., 6]`` one-hot
    button channels.  ``params`` is the length-7 theta tensor.  ``contact`` is
    an optional ``[..., 4]`` 0/1 flag vector ordered ``[ground, ceiling,
    left_wall, right_wall]`` (engine collision byte ``$7E:0077`` bits, matching
    state channels 4-7); when present it applies the rigid-body *collision
    response* -- ground kills downward velocity, ceiling kills upward velocity,
    a right wall kills rightward velocity and a left wall kills leftward
    velocity -- so the state cannot penetrate terrain.  Returns ``[..., 4]``.
    """
    x = states[..., 0]
    y = states[..., 1]
    vx = states[..., 2]
    vy = states[..., 3]
    jump = actions[..., _A_JUMP]
    run = actions[..., _A_RUN]
    direction = actions[..., _A_RIGHT] - actions[..., _A_LEFT]

    max_vx = params[0]
    walk = params[1]
    run_a = params[2]
    scale = params[3]
    g_hold = params[4]
    g_fall = params[5]
    decel = params[6]

    # Horizontal: directional input accelerates toward the +/-max_vx speed
    # cap; releasing every direction applies Coulomb friction toward zero.
    tier = torch.where(run > 0.5, run_a, walk)
    a_cmd = direction * tier
    vx_drive = torch.clamp(vx + a_cmd, -max_vx, max_vx)
    moving = direction != 0
    vx_fric = torch.where(
        vx > 0, torch.clamp(vx - decel, min=0.0), torch.clamp(vx + decel, max=0.0)
    )
    vx_next = torch.where(moving, vx_drive, vx_fric)

    # Vertical: held-jump ascent integrates g_hold, everything else g_fall;
    # clamped to the engine's structural bounds.
    g = torch.where((jump > 0.5) & (vy < 0), g_hold, g_fall)
    vy_next = torch.clamp(vy + g, MIN_VY, TERMINAL_VY)

    # Rigid collision response on the four terrain-contact channels: each one
    # zeroes only the velocity component that would push into the surface, so
    # Mario can still slide along / away from it (the classic SMW wall-slide).
    if contact is not None:
        c_ground = contact[..., 0]
        c_ceiling = contact[..., 1]
        c_left = contact[..., 2]
        c_right = contact[..., 3]
        vy_next = torch.where(c_ground > 0.5, torch.clamp(vy_next, max=0.0), vy_next)
        vy_next = torch.where(c_ceiling > 0.5, torch.clamp(vy_next, min=0.0), vy_next)
        vx_next = torch.where(c_right > 0.5, torch.clamp(vx_next, max=0.0), vx_next)
        vx_next = torch.where(c_left > 0.5, torch.clamp(vx_next, min=0.0), vx_next)

    x_next = x + vx_next / scale
    y_next = y + vy_next / scale
    return torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)


def simulate_rollout(
    s0: Tensor, action_seq: Tensor, params: Tensor, contact: Optional[Tensor] = None
) -> Tensor:
    """Unroll :func:`simulate_step``L`` times.

    ``s0`` is ``[B, 4]``, ``action_seq`` is ``[B, L, 6]``, ``contact`` is an
    optional ``[B, L, 4]`` terrain-contact sequence.  Returns predicted
    next-states ``[B, L, 4]`` (state at ``t = 1 .. L``), all differentiable in
    ``params``.
    """
    state = s0
    outs = []
    for t in range(action_seq.shape[1]):
        ct = None if contact is None else contact[:, t, :]
        state = simulate_step(state, action_seq[:, t, :], params, ct)
        outs.append(state)
    return torch.stack(outs, dim=1)


def make_windows(
    states: Tensor, actions: Tensor, next_states: Tensor, episodes: Tensor, rollout_len: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cut per-episode rollouts from a dataset for multi-step fitting.

    ``states`` / ``next_states`` are ``[N, >=8]`` (the first four channels are
    the fitted state; channels 4-7 of ``next_states`` are the terrain-contact
    flags ``[ground, ceiling, left_wall, right_wall]``).  ``episodes`` tags each
    transition.  Only windows fully inside one episode are kept, so no rollout
    crosses a reset boundary.  Returns ``(s0, actions, targets, contact)`` with
    shapes ``[W, 4] / [W, L, 6] / [W, L, 4] / [W, L, 4]``.
    """
    states = torch.as_tensor(states)
    actions = torch.as_tensor(actions)
    next_states = torch.as_tensor(next_states)
    episodes = torch.as_tensor(episodes)
    n = states.shape[0]
    s0s, acts, tgts, cts = [], [], [], []
    i = 0
    while i < n - rollout_len:
        if episodes[i] != episodes[i + rollout_len]:
            i += 1
            continue
        s0s.append(states[i, :4])
        acts.append(actions[i : i + rollout_len])
        tgts.append(next_states[i : i + rollout_len, :4])
        cts.append((next_states[i : i + rollout_len, 4:8] > 0.5).float())
        i += 1
    if not s0s:
        empty = torch.zeros((0, 4))
        return (
            empty,
            torch.zeros((0, rollout_len, 6)),
            torch.zeros((0, rollout_len, 4)),
            torch.zeros((0, rollout_len, 4)),
        )
    return torch.stack(s0s), torch.stack(acts), torch.stack(tgts), torch.stack(cts)


def generate_synthetic_windows(
    true_params: EngineParams,
    n_windows: int,
    rollout_len: int,
    seed: int,
    x_span: Tuple[float, float] = (0.0, 600.0),
    floor_y: float = 336.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build ground-truth rollouts under ``true_params`` with a floor at
    ``floor_y``.  Deliberately spans the regimes that make each constant
    observable: sustained sprints that saturate ``max_vx``, release frames
    that coast under ``decel``, and held-jump ascents versus falls (the
    gravity asymmetry).  The ground-contact channel is generated coherently
    with the floor, so the collision response is exercised too.
    """
    g = torch.Generator().manual_seed(seed)
    p = theta_tensor(true_params)
    max_vx_true = float(p[0])
    lo, hi = x_span
    sign = torch.where(
        torch.rand((n_windows,), generator=g) < 0.5, torch.ones(n_windows), -torch.ones(n_windows)
    )
    x0 = sign * (lo + (hi - lo) * torch.rand((n_windows,), generator=g))
    y0 = torch.full((n_windows,), floor_y)
    # Start near the speed cap so the ceiling clamp is active from step one.
    vx0 = sign * max_vx_true * 0.95 * (0.4 + 0.6 * torch.rand((n_windows,), generator=g))
    # Half start grounded (vy 0), half mid-jump (vy<0) to exercise both g_tiers.
    vy0 = torch.where(
        torch.rand((n_windows,), generator=g) < 0.5,
        torch.zeros(n_windows),
        -60.0 * torch.rand((n_windows,), generator=g),
    )
    s0 = torch.stack([x0, y0, vx0, vy0], dim=-1)

    actions = torch.zeros((n_windows, rollout_len, 6))
    run = torch.rand((n_windows, rollout_len), generator=g)
    jump = torch.rand((n_windows, rollout_len), generator=g)
    actions[:, :, _A_RUN] = (run < 0.7).float()
    right = (
        (sign > 0).unsqueeze(-1) & (torch.rand((n_windows, rollout_len), generator=g) < 0.6)
    ).float()
    left = (
        (sign < 0).unsqueeze(-1) & (torch.rand((n_windows, rollout_len), generator=g) < 0.6)
    ).float()
    actions[:, :, _A_RIGHT] = right
    actions[:, :, _A_LEFT] = left
    # Press Y for the first third of held-jump windows to exercise g_hold.
    jump_len = max(1, rollout_len // 3)
    actions[:, :jump_len, _A_JUMP] = (jump[:, :jump_len] < 0.5).float()

    # Roll with a coherent floor-contact flag (only the ground channel is
    # active in this synthetic world; walls / ceiling are exercised on real
    # data via make_windows).
    state = s0.clone()
    outs, contact_flags = [], []
    zeros = torch.zeros(n_windows)
    for t in range(rollout_len):
        cg = (state[:, 1] >= floor_y - 1e-3).float()
        ct = torch.stack([cg, zeros, zeros, zeros], dim=-1)  # [B, 4]
        state = simulate_step(state, actions[:, t, :], p, ct)
        # Keep Mario from sinking: clamp y back to the floor when grounded.
        state = torch.stack(
            [state[:, 0], torch.clamp(state[:, 1], max=floor_y), state[:, 2], state[:, 3]], dim=-1
        )
        outs.append(state)
        contact_flags.append(ct)
    targets = torch.stack(outs, dim=1)
    contact = torch.stack(contact_flags, dim=1)  # [B, L, 4]
    return s0, actions, targets, contact


def _channel_weights(targets: Tensor) -> Tensor:
    """Per-channel inverse-variance weights, shape ``[4]``."""
    var = targets.reshape(-1, 4).var(dim=0, unbiased=False).clamp(min=1e-6)
    return 1.0 / var


def identify_params(
    windows: Tuple[Tensor, Tensor, Tensor, Tensor],
    init: EngineParams | Tensor,
    steps: int = 400,
    lr: float = 0.05,
    batch: int = 256,
    seed: int = 42,
) -> Tuple[Tensor, dict]:
    """Recover the seven physical constants by weighted multi-step
    least-squares regression of the simulator onto observed rollouts.

    The channel-weighted objective (inverse per-target variance) keeps the
    wide ``x`` channel from swamping the sub-pixel velocity channels.  Only
    genuine free constants are fit; the jump impulse / terminal velocity are
    structural.  Returns ``(theta_hat, info)`` where ``theta_hat`` is the
    length-7 estimate and ``info`` carries the final loss and per-variable
    training MSE.
    """
    s0, acts, tgts, gr = windows
    if s0.shape[0] == 0:
        raise ValueError("identify_params needs at least one transition window")
    weight = _channel_weights(tgts)

    if isinstance(init, EngineParams):
        init_vec = theta_tensor(init)
    else:
        init_vec = init
    raw = _softplus_inv(init_vec).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    loss_curve = []
    for _ in range(steps):
        opt.zero_grad()
        idx = torch.randint(
            0,
            s0.shape[0],
            (min(batch, s0.shape[0]),),
            generator=torch.Generator().manual_seed(seed),
        )
        params = torch.nn.functional.softplus(raw)
        pred = simulate_rollout(s0[idx], acts[idx], params, gr[idx])
        loss = (((pred - tgts[idx]) ** 2) * weight).mean()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss))
    theta_hat = torch.nn.functional.softplus(raw).detach()
    with torch.no_grad():
        train_mse = _per_variable_mse(windows, theta_hat)
    info = {"final_loss": loss_curve[-1], "loss_curve": loss_curve, "train_mse": train_mse}
    return theta_hat, info


def _per_variable_mse(windows: Tuple[Tensor, Tensor, Tensor, Tensor], params: Tensor) -> dict:
    s0, acts, tgts, gr = windows
    weight = _channel_weights(tgts)
    pred = simulate_rollout(s0, acts, params, gr)
    sq = ((pred - tgts) ** 2).reshape(-1, 4).mean(dim=0)
    weighted = float((((pred - tgts) ** 2).reshape(-1, 4) * weight).mean())
    return {
        "x": float(sq[0]),
        "y": float(sq[1]),
        "vx": float(sq[2]),
        "vy": float(sq[3]),
        "weighted_total": weighted,
    }


def per_variable_mse(
    windows: Tuple[Tensor, Tensor, Tensor, Tensor], params: EngineParams | Tensor
) -> dict:
    """Per-variable multi-step prediction MSE of a parameter set on windows."""
    p = theta_tensor(params) if isinstance(params, EngineParams) else params
    return _per_variable_mse(windows, p)


def rollout_mse(
    windows: Tuple[Tensor, Tensor, Tensor, Tensor], params: EngineParams | Tensor
) -> float:
    """Scalar weighted multi-step prediction MSE of a parameter set."""
    return float(per_variable_mse(windows, params)["weighted_total"])


def bootstrap_ci(
    windows: Tuple[Tensor, Tensor, Tensor, Tensor],
    init: EngineParams | Tensor,
    n_boot: int = 16,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> dict:
    """Frequentist non-parametric bootstrap over transition windows.

    Refits :func:`identify_params` on resampled windows to obtain a percentile
    interval per constant.  A parameter whose interval collapses onto a single
    point (near-zero width) is *locally degenerate* in this data -- the
    identifiability limit is a property of the excitation, not the estimator.
    Complements the Laplace posterior (which is parametric/Gaussian).
    """
    s0, acts, tgts, gr = windows
    n = s0.shape[0]
    boots = []
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        it = torch.as_tensor(idx)
        sub: Tuple[Tensor, Tensor, Tensor, Tensor] = (s0[it], acts[it], tgts[it], gr[it])
        theta, _ = identify_params(sub, init, steps=steps, lr=lr, seed=b)
        boots.append(theta.numpy())
    boots_arr = np.stack(boots)  # [n_boot, 7]
    return {
        name: {
            "mean": float(boots_arr[:, j].mean()),
            "std": float(boots_arr[:, j].std(ddof=1)) if n_boot > 1 else 0.0,
            "ci_low": float(np.percentile(boots_arr[:, j], 2.5)),
            "ci_high": float(np.percentile(boots_arr[:, j], 97.5)),
        }
        for j, name in enumerate(PARAM_NAMES)
    }


def posterior_laplace(
    windows: Tuple[Tensor, Tensor, Tensor, Tensor],
    theta_hat: Tensor,
    n_windows: int = 160,
    ridge: float = 1e-8,
) -> dict:
    """Laplace / Gauss-Newton posterior around the point estimate.

    With residual vector ``r(theta)`` (channel-weighted one-step errors over a
    subsample of windows), the Fisher information is ``H = J^T J`` with
    ``J = d r / d theta``; the posterior covariance is ``sigma^2 (H + ridge I)^-1``
    where ``sigma^2`` is the residual variance.  The eigen-spectrum of ``H`` is
    the identifiability statement: near-zero eigenvalues are the directions of
    parameter space the transition data cannot resolve (the structural limit
    that also caps any learned model -- the jump impulse never enters
    ``theta``, and a floor-capped constant is locally unidentifiable without
    warm-starting inside the active branch).  Returns per-parameter standard
    errors, the correlation matrix, the eigenvalues / condition number, and
    relative-error flags.
    """
    s0, acts, tgts, gr = windows
    m = min(n_windows, s0.shape[0])
    sub = (s0[:m], acts[:m], tgts[:m], gr[:m])
    weight = _channel_weights(tgts[:m])
    sw = weight.sqrt()

    def residual(theta: Tensor) -> Tensor:
        pred = simulate_rollout(sub[0], sub[1], theta, sub[3])
        return ((pred - sub[2]) * sw).reshape(-1)

    theta = theta_hat.detach().clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(residual, theta)  # [N, 7]
    r = residual(theta).detach()
    sigma2 = float((r**2).mean())
    h = jac.T @ jac
    cov = sigma2 * torch.linalg.inv(h + ridge * torch.eye(len(theta_hat), dtype=theta_hat.dtype))
    std = torch.sqrt(cov.clamp(min=0).diagonal())
    d = torch.diag(1.0 / std.clamp(min=1e-12))
    corr = d @ cov @ d
    eigvals = torch.linalg.eigvalsh(h).flip(0)  # descending
    cond = float(eigvals[0] / eigvals[-1].clamp(min=1e-30))
    rel_err = (std / theta_hat.abs().clamp(min=1e-6)).tolist()
    return {
        "theta_hat": theta_hat.detach().tolist(),
        "std_errors": std.detach().tolist(),
        "cov": cov.detach().tolist(),
        "correlation": corr.detach().tolist(),
        "eigenvalues": eigvals.detach().tolist(),
        "condition_number": cond,
        "noise_variance": sigma2,
        "relative_std": rel_err,
        "identified": [bool(e < 0.05) for e in rel_err],
    }


def sample_posterior(theta_hat: Tensor, post: dict, n_samples: int, seed: int) -> Tensor:
    """Draw samples from the full-covariance Laplace posterior.

    Uses a Cholesky factor of the reported covariance ``post["cov"]`` (falling
    back to the diagonal of squared standard errors) so parameter *correlations*
    -- the near-degenerate directions the Fisher spectrum flags -- are honoured,
    rather than assuming coordinate independence.  Samples are clamped to the
    positive orthant.  Used for a posterior-predictive credible interval.
    """
    mean = np.asarray(post["theta_hat"], dtype=np.float64)
    d = len(mean)
    if post.get("cov") is not None:
        cov = np.asarray(post["cov"], dtype=np.float64) + 1e-12 * np.eye(d)
    else:
        cov = np.diag(np.maximum(np.asarray(post["std_errors"], dtype=np.float64), 1e-6) ** 2)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.clip(np.diag(cov), 0.0, None)) + 1e-9)
    gen = np.random.default_rng(seed)
    z = gen.standard_normal((n_samples, d))
    draws = mean[None, :] + z @ chol.T
    return torch.tensor(np.clip(draws, 1e-3, None), dtype=torch.float32)


def _weighted_residual(windows: Tuple[Tensor, Tensor, Tensor, Tensor], theta: Tensor) -> Tensor:
    """Channel-weighted multi-step residual tensor for a parameter vector."""
    s0, acts, tgts, gr = windows
    sw = _channel_weights(tgts).sqrt()
    pred = simulate_rollout(s0, acts, theta, gr)
    return (pred - tgts) * sw


def log_posterior(
    theta: Tensor,
    windows: Tuple[Tensor, Tensor, Tensor, Tensor],
    noise_variance: float,
    n_windows: int = 160,
) -> float:
    """Unnormalised log-posterior: Gaussian likelihood on the weighted
    residuals with variance ``noise_variance`` and a hard positivity prior on
    every constant (the softplus parameter domain).  ``-inf`` outside the
    domain so a sampler never leaves the physically valid region."""
    if not bool((theta > 0).all()):
        return -float("inf")
    s0 = windows[0]
    m = min(n_windows, s0.shape[0])
    sub = (windows[0][:m], windows[1][:m], windows[2][:m], windows[3][:m])
    resid = _weighted_residual(sub, theta)
    ssr = float((resid**2).sum())
    return -0.5 * ssr / max(noise_variance, 1e-12)


def mcmc_random_walk(
    windows: Tuple[Tensor, Tensor, Tensor, Tensor],
    theta_start: Tensor,
    noise_variance: float,
    n_samples: int = 3000,
    burn: int = 1000,
    thin: int = 5,
    step: Optional[np.ndarray] = None,
    step_frac: float = 0.05,
    seed: int = 0,
) -> dict:
    """Random-walk Metropolis posterior sampler on the *true* likelihood.

    Unlike the Laplace approximation this makes no Gaussian/curvature assumption
    -- it targets the exact posterior implied by the simulator residuals -- so
    its agreement with (or departure from) the Laplace summary is a direct probe
    of posterior non-Gaussianity, the regime where the linearised uncertainty is
    wrong.  The Gaussian proposal scale defaults to ``step_frac`` of each
    coordinate but should be matched to the posterior width (pass the Laplace
    standard errors as ``step``) -- on near-noiseless data the posterior is a
    sharp ridge and an un-scaled step gives zero acceptance.  The chain runs
    fully deterministically from a fixed seed.  Returns the retained samples,
    the per-constant mean / standard error / 95% credible interval and the
    acceptance rate (a good RW-Metropolis tune lies roughly in [0.15, 0.5]).
    """
    gen = np.random.default_rng(seed)
    theta = theta_start.detach().double().clone()
    step_vec = (
        np.asarray(step, dtype=np.float64)
        if step is not None
        else step_frac * np.abs(theta.numpy()).clip(min=1e-3)
    )
    lp = log_posterior(theta, windows, noise_variance)
    keep = n_samples
    chain = np.zeros((keep, len(theta)))
    accepted = 0
    idx = 0
    post_burn_iters = 0
    for it in range(burn + n_samples * thin):
        prop = theta + torch.tensor(gen.normal(0.0, step_vec), dtype=torch.float64)
        lpp = log_posterior(prop, windows, noise_variance)
        if it >= burn:
            post_burn_iters += 1
        if float(np.log(gen.random() + 1e-300)) < (lpp - lp):
            theta, lp = prop, lpp
            if it >= burn:
                accepted += 1
        if it >= burn and (it - burn) % thin == 0 and idx < keep:
            chain[idx] = theta.numpy()
            idx += 1
    chain = chain[:idx]
    mean = chain.mean(axis=0)
    std = chain.std(axis=0, ddof=1) if idx > 1 else np.zeros(len(mean))
    lo = np.percentile(chain, 2.5, axis=0)
    hi = np.percentile(chain, 97.5, axis=0)
    return {
        "samples": torch.tensor(chain, dtype=torch.float32),
        "mean": mean.tolist(),
        "std_errors": std.tolist(),
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
        "acceptance_rate": accepted / max(post_burn_iters, 1),
        "n_samples": int(idx),
    }


def mpc_random_shooting(
    s0: Tensor,
    model_params: Tensor,
    task_reward,
    horizon: int = 12,
    n_samples: int = 128,
    n_iters: int = 3,
    elite: int = 16,
    seed: int = 0,
) -> Tuple[Tensor, float]:
    """Sample-MPC (random shooting + CEM) using the identified model.

    Actions are one-hot 6-channel frames.  ``task_reward(s_t, a_t)`` returns a
    per-step scalar reward for the predicted state / action.  Returns the best
    action sequence ``[horizon, 6]`` and its predicted return.  The optimiser
    is seed-fixed so the MPC is deterministic; the *model* is the only thing
    that varies across the comparison conditions, so a quality difference
    isolates the transfer penalty of a mis-identified parameter set.
    """
    gen = torch.Generator().manual_seed(seed)
    mean = torch.full((horizon, 6), 0.5)
    std = torch.full((horizon, 6), 0.4)
    best_return = -float("inf")
    best_seq = torch.zeros((horizon, 6))
    for _ in range(n_iters):
        noise = torch.randn((n_samples, horizon, 6), generator=gen)
        logits = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        # RIGHT always active (forward progress); among the {LEFT, JUMP, RUN}
        # competition channels take the argmax; run is a modifier of RIGHT.
        logits[:, :, _A_RIGHT] += 3.0
        comp = [_A_LEFT, _A_JUMP, _A_RUN]
        arg = logits[:, :, comp].argmax(dim=-1)  # [n_samples, horizon] index into comp
        samples = torch.zeros((n_samples, horizon, 6))
        samples[:, :, _A_RIGHT] = 1.0
        picked = torch.tensor(comp)[arg]  # [n_samples, horizon] actual channel ids
        samples.scatter_(2, picked.unsqueeze(-1), 1.0)
        returns = []
        for k in range(n_samples):
            state = s0.clone()
            total = 0.0
            for t in range(horizon):
                state = simulate_step(state, samples[k, t], model_params)
                total += float(task_reward(state, samples[k, t]))
            returns.append(total)
        returns_t = torch.tensor(returns)
        order = returns_t.argsort(descending=True)
        elites = samples[order[:elite]]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(min=0.1)
        if float(returns_t.max()) > best_return:
            best_return = float(returns_t.max())
            best_seq = samples[int(order[0])]
    return best_seq, best_return
