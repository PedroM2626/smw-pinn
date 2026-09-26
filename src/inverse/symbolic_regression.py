"""Symbolic regression for the inverse problem: *discovering* the engine's law
instead of *fitting constants inside* a law one already assumes (README section
10.43).

Section 10.40 solves the inverse problem parametrically: the analytic traction /
friction / asymmetric-gravity map is posited, its seven constants are unknown, and
Adam plus a Laplace posterior recovers them.  That answer is only as good as the
posited structure.  This module removes the structure from the assumptions: genetic
programming (``gplearn``) evolves the functional form of each one-step update law
directly from observed transitions, and a *probe* stage reads the physical constants
back out of whatever expression was discovered.  Three questions then become
answerable with evidence rather than opinion:

1. **Is the engine's law discoverable from data alone?**  Which terms survive a
   structure-free search (the discrete integration identity, the run tier), and which
   do not (the unilateral velocity bound, the held-jump gravity gate, the coast
   deadband)?
2. **What is the price of not knowing the structure?**  The discovered laws are
   compared with the parametric identification of section 10.40 on the same hidden
   world, constant by constant, and with the published learned models on real WRAM
   telemetry.
3. **What does a discovered law buy a controller?**  The same held-out control
   battery as section 10.40-E3 scores the symbolic world model's optimism.

Two representation issues drive every design choice below, and the study measures
both instead of asserting them:

*Terminal-scale limit.*  GP terminals are drawn uniformly from a bounded
``const_range``, but the engine's horizontal law mixes a per-frame increment of ~1
sub-pixel with a rigid ceiling of ~48 sub-pixels -- two constants an order of
magnitude apart, so no single normalisation lets the search express both.  Every law
is therefore fitted in *excitation-normalised* coordinates (each channel divided by
the maximum its own observed data reaches), and the ceiling is probed as the fixed
point of the discovered map, reported as *not found* when the map has none.

*Discontinuity.*  The gravity tier is gated by ``jump held AND ascending`` and the
collision response by a contact bit.  Tree GP must build those gates out of protected
division and ``min``/``max``, which is why the study reports per-seed recovery rates
and a bagged ensemble rather than one lucky draw.

Everything runs on CPU from the recorded dataset or a synthetic ground truth -- no
emulator, no GPU -- so ``src/evaluation/symbolic_inverse_benchmark.py`` is a CI-safe
study.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor

from src.inverse.parameter_identification import (
    PARAM_NAMES,
    simulate_step,
    theta_tensor,
)

# Piecewise-affine primitive set: ``min``/``max`` are what a rigid unilateral
# constraint looks like symbolically, and gplearn's protected ``div`` is the only
# route to a soft switch (a signed denominator turns a velocity into a gate).
GP_FUNCTION_SET: List[str] = ["add", "sub", "mul", "div", "neg", "abs", "min", "max"]

# The four one-step laws that compose the symbolic world model.  Velocity channels are
# fitted as increments (their physical magnitude *is* an acceleration, so the
# normalised target keeps the terminals O(1)); position channels are fitted as
# increments against the post-step velocity, which is exactly the discrete integration
# identity -- so their recovery tests whether GP rediscovers kinematics.
LAW_NAMES: Tuple[str, ...] = ("dvx", "dvy", "dx", "dy")

# Feature presets.  "synthetic" matches the excitation available in the hidden world of
# section 10.40 (a floor, no walls or ceiling); "real" adds every terrain-contact
# channel recorded in WRAM, so on genuine gameplay the collision response is part of
# what the search may or may not discover.
LAW_FEATURES: Dict[str, Dict[str, List[str]]] = {
    "synthetic": {
        "dvx": ["vx", "dir", "run"],
        "dvy": ["vy", "jump", "ground"],
        "dx": ["vx_next"],
        "dy": ["vy_next"],
    },
    "real": {
        "dvx": ["vx", "dir", "run", "ground", "ceiling", "left", "right"],
        "dvy": ["vy", "jump", "ground", "ceiling", "left", "right"],
        "dx": ["vx_next"],
        "dy": ["vy_next"],
    },
    # "full" hands the search every observable channel of the 8D telemetry.  It exists
    # for the grey-box residual control, where the question is precisely whether ANY
    # closed form of the observation explains what the analytic model missed.
    "full": {
        "dvx": ["x", "y", "vx", "vy", "dir", "run", "jump", "ground", "ceiling", "left", "right"],
        "dvy": ["x", "y", "vx", "vy", "dir", "run", "jump", "ground", "ceiling", "left", "right"],
        "dx": ["vx_next"],
        "dy": ["vy_next"],
    },
}


@dataclass
class TransitionBank:
    """Single-step transitions cut from one or more rollout windows.

    ``states`` / ``next_states`` are ``[N, 4]`` as ``[x, y, vx, vy]`` (pixels and
    sub-pixels per frame), ``actions`` is ``[N, 6]`` one-hot button channels and
    ``contact`` the ``[N, 4]`` terrain-contact vector ordered ``[ground, ceiling,
    left_wall, right_wall]`` -- the convention of
    :func:`src.inverse.parameter_identification.simulate_step`, which keeps the
    symbolic and parametric models directly comparable.
    """

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    contact: np.ndarray

    def __post_init__(self) -> None:
        n = self.states.shape[0] if self.states.ndim == 2 else -1
        if self.states.ndim != 2 or self.states.shape[1] != 4:
            raise ValueError(f"states must be [N, 4], got {self.states.shape}")
        if self.next_states.shape != (n, 4):
            raise ValueError(f"next_states must be [N, 4], got {self.next_states.shape}")
        if self.actions.shape != (n, 6):
            raise ValueError(f"actions must be [N, 6], got {self.actions.shape}")
        if self.contact.shape != (n, 4):
            raise ValueError(f"contact must be [N, 4], got {self.contact.shape}")

    @property
    def n(self) -> int:
        return int(self.states.shape[0])

    @property
    def direction(self) -> np.ndarray:
        """Signed horizontal input ``right - left`` in {-1, 0, +1}."""
        return self.actions[:, 5] - self.actions[:, 4]

    @property
    def jump(self) -> np.ndarray:
        return self.actions[:, 0]

    @property
    def run(self) -> np.ndarray:
        return self.actions[:, 1]

    @property
    def velocity_support(self) -> Tuple[float, float]:
        """Observed ``max |vx|`` and ``max |vy|``: the reachable-set support."""
        return (
            float(np.abs(self.states[:, 2]).max()) if self.n else 0.0,
            float(np.abs(self.states[:, 3]).max()) if self.n else 0.0,
        )

    @property
    def step_support(self) -> Tuple[float, float]:
        """Observed ``max |dx|`` and ``max |dy|`` of a single frame."""
        step = np.abs(self.next_states[:, :2] - self.states[:, :2])
        return (
            float(step[:, 0].max()) if self.n else 0.0,
            float(step[:, 1].max()) if self.n else 0.0,
        )


def bank_subset(bank: TransitionBank, idx: np.ndarray) -> TransitionBank:
    """Index every array of a bank with the same selection."""
    return TransitionBank(
        bank.states[idx], bank.actions[idx], bank.next_states[idx], bank.contact[idx]
    )


def bank_from_windows(windows: Sequence[Any], stride: int = 1) -> TransitionBank:
    """Flatten ``(s0, actions, targets, contact)`` windows into single steps.

    ``windows`` is the 4-tuple returned by
    :func:`src.inverse.parameter_identification.make_windows` or
    ``generate_synthetic_windows``: ``[W, 4] / [W, L, 6] / [W, L, 4] / [W, L, 4]``.
    Each window contributes its frames chained through the *observed* trajectory, so
    ``s_t`` is the previous row and no fabricated state enters the fit.  ``stride``
    thins the bank -- genetic-programming cost is linear in samples and neighbouring
    frames of one rollout are highly correlated, so thinning buys speed without
    removing regimes.
    """
    s0, actions, targets, contact = (np.asarray(w, dtype=np.float64) for w in windows)
    if s0.shape[0] == 0:
        return TransitionBank(
            np.zeros((0, 4)), np.zeros((0, 6)), np.zeros((0, 4)), np.zeros((0, 4))
        )
    states = np.concatenate([s0[:, None, :], targets[:, :-1, :]], axis=1)
    bank = TransitionBank(
        states.reshape(-1, 4),
        actions.reshape(-1, 6),
        targets.reshape(-1, 4),
        contact.reshape(-1, 4),
    )
    if stride > 1:
        return bank_subset(bank, np.arange(0, bank.n, stride))
    return bank


def bank_from_transitions(
    states: Any,
    actions: Any,
    next_states: Any,
    contact: Optional[Any] = None,
    stride: int = 1,
) -> TransitionBank:
    """Build a bank from already-flat transition arrays (the real dataset case).

    The canonical WRAM arrays are ``[N, 8]``: channels 0-3 are the fitted state and
    channels 4-7 the terrain-contact byte, so wider input is split automatically and
    ``contact`` may be omitted.  Thresholded at 0.5 exactly as ``make_windows`` does,
    which keeps the collision flags identical to the 10.40-E2 pipeline.
    """
    ns = np.asarray(next_states, dtype=np.float64)
    if contact is None:
        contact = ns[:, 4:8] > 0.5 if ns.shape[1] > 4 else np.zeros((ns.shape[0], 4))
    bank = TransitionBank(
        np.asarray(states, dtype=np.float64)[:, :4],
        np.asarray(actions, dtype=np.float64),
        ns[:, :4],
        np.asarray(contact, dtype=np.float64),
    )
    if stride > 1:
        return bank_subset(bank, np.arange(0, bank.n, stride))
    return bank


def law_matrices(
    bank: TransitionBank, preset: str = "synthetic"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return ``law -> (X, y)`` in *physical* units for one bank.

    Velocity laws regress the one-frame increment on the driving state; position laws
    regress the one-frame displacement on the post-step velocity, the form in which
    the identity ``x_{t+1} = x_t + v_{t+1}/scale`` becomes a univariate fit.
    """
    if preset not in LAW_FEATURES:
        raise ValueError(
            f"unknown feature preset {preset!r}, expected one of {sorted(LAW_FEATURES)}"
        )
    columns: Dict[str, np.ndarray] = {
        "x": bank.states[:, 0],
        "y": bank.states[:, 1],
        "vx": bank.states[:, 2],
        "vy": bank.states[:, 3],
        "dir": bank.direction,
        "run": bank.run,
        "jump": bank.jump,
        "ground": bank.contact[:, 0],
        "ceiling": bank.contact[:, 1],
        "left": bank.contact[:, 2],
        "right": bank.contact[:, 3],
        "vx_next": bank.next_states[:, 2],
        "vy_next": bank.next_states[:, 3],
    }
    targets: Dict[str, np.ndarray] = {
        "dvx": bank.next_states[:, 2] - bank.states[:, 2],
        "dvy": bank.next_states[:, 3] - bank.states[:, 3],
        "dx": bank.next_states[:, 0] - bank.states[:, 0],
        "dy": bank.next_states[:, 1] - bank.states[:, 1],
    }
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in LAW_NAMES:
        feats = LAW_FEATURES[preset][name]
        out[name] = (np.stack([columns[f] for f in feats], axis=1), targets[name])
    return out


@dataclass
class SymbolicLaw:
    """One fitted GP law plus everything needed to evaluate and probe it.

    ``x_scale`` / ``y_scale`` hold the excitation normalisation, so :meth:`increment`
    always speaks in physical units even though the search ran on ``[-1, 1]``
    terminals.  ``n_nodes`` and ``expression`` are the parsimony observables;
    ``raw_fitness`` is the *normalised* training fitness GP minimised, without the
    parsimony term.
    """

    name: str
    feature_names: List[str]
    estimator: Any
    x_scale: np.ndarray
    y_scale: float
    expression: str
    n_nodes: int
    depth: int
    raw_fitness: float
    fit_seconds: float
    seed: int
    fit_mean: float = 0.0
    train_mae: float = 0.0
    train_r2: float = 0.0
    hold_mae: float = 0.0
    hold_r2: float = 0.0
    hold_rmse: float = 0.0
    hold_null_r2: float = 0.0

    def increment(self, X_phys: np.ndarray) -> np.ndarray:
        """Predict the physical-unit increment for physical-unit features."""
        X = np.asarray(X_phys, dtype=np.float64)
        flat = X.ndim == 1
        if flat:
            X = X[None, :]
        out = np.asarray(self.estimator.predict(X / self.x_scale) * self.y_scale, dtype=np.float64)
        return out.reshape(-1)[0] if flat else out


@dataclass
class BaggedLaw:
    """Seed ensemble of :class:`SymbolicLaw`: the mean of their increments.

    Genetic programming is a randomised search, so one draw is an anecdote.  Averaging
    per-seed increments is the cheapest variance reduction available and is reported
    beside the single-run accuracy to separate "the representation cannot express the
    law" from "this draw was unlucky".
    """

    name: str
    feature_names: List[str]
    members: List[SymbolicLaw]

    def increment(self, X_phys: np.ndarray) -> np.ndarray:
        stack = np.stack([law.increment(X_phys) for law in self.members], axis=0)
        return stack.mean(axis=0)


@dataclass
class AnalyticLaw:
    """One channel of the section-10.40 parametric map, on the law interface.

    Implements ``increment`` by calling the real differentiable simulator, so the
    identified model can be probed, composed and rolled out through *exactly* the same
    code as a discovered law.  A unit test asserts the composition is identical to
    ``simulate_rollout``, which is what makes the two estimators comparable.
    """

    name: str
    feature_names: List[str]
    theta: np.ndarray

    def increment(self, X_phys: np.ndarray) -> np.ndarray:
        X = np.asarray(X_phys, dtype=np.float64)
        flat = X.ndim == 1
        if flat:
            X = X[None, :]
        cols = {feat: X[:, j] for j, feat in enumerate(self.feature_names)}
        n = X.shape[0]
        states = np.zeros((n, 4))
        actions = np.zeros((n, 6))
        contact = np.zeros((n, 4))
        if "vx" in cols:
            states[:, 2] = cols["vx"]
        if "vy" in cols:
            states[:, 3] = cols["vy"]
        if "dir" in cols:
            actions[:, 5] = np.maximum(cols["dir"], 0.0)
            actions[:, 4] = np.maximum(-cols["dir"], 0.0)
        if "run" in cols:
            actions[:, 1] = cols["run"]
        if "jump" in cols:
            actions[:, 0] = cols["jump"]
        for i, chan in enumerate(("ground", "ceiling", "left", "right")):
            if chan in cols:
                contact[:, i] = cols[chan]
        nxt = _simulate_step_numpy(states, actions, contact, self.theta)
        scale = float(self.theta[3])
        if self.name == "dvx":
            out = nxt[:, 2] - states[:, 2]
        elif self.name == "dvy":
            out = nxt[:, 3] - states[:, 3]
        elif self.name == "dx":
            # The position law's feature *is* the post-step velocity, so the analytic
            # counterpart is the integration identity itself.
            out = cols["vx_next"] / scale
        else:
            out = cols["vy_next"] / scale
        return out.reshape(-1)[0] if flat else out


def _simulate_step_numpy(
    states: np.ndarray, actions: np.ndarray, contact: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """One frame of the analytic map in float64, reusing the library integrator."""
    nxt = simulate_step(
        torch.as_tensor(states, dtype=torch.float64),
        torch.as_tensor(actions, dtype=torch.float64),
        torch.as_tensor(theta, dtype=torch.float64),
        torch.as_tensor(contact, dtype=torch.float64),
    )
    return nxt.numpy().astype(np.float64)


def parametric_laws(theta: Any, preset: str = "synthetic") -> Dict[str, Any]:
    """Wrap a parameter vector as law objects on the world-model interface."""
    vec = np.asarray(theta, dtype=np.float64).reshape(-1)
    return {name: AnalyticLaw(name, LAW_FEATURES[preset][name], vec) for name in LAW_NAMES}


LawLike = Any  # anything exposing ``increment(X) -> np.ndarray``
WorldModel = Dict[str, LawLike]


@dataclass
class LawBank:
    """A named collection of laws that composes into a world model."""

    laws: Dict[str, SymbolicLaw]
    preset: str
    seeds: List[int] = field(default_factory=list)
    per_seed: Dict[str, List[SymbolicLaw]] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.laws

    def expressions(self) -> Dict[str, str]:
        return {name: law.expression for name, law in self.laws.items()}

    def mean_nodes(self) -> float:
        if not self.laws:
            return 0.0
        return float(np.mean([law.n_nodes for law in self.laws.values()]))

    def total_seconds(self) -> float:
        return float(sum(law.fit_seconds for law in self.laws.values()))

    def bagged(self) -> WorldModel:
        """Seed-averaged world model over the same laws."""
        return {
            name: (BaggedLaw(name, laws[0].feature_names, laws) if len(laws) > 1 else laws[0])
            for name, laws in self.per_seed.items()
            if laws
        }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(np.abs(y_true - y_pred).mean())
    var = float(np.var(y_true))
    r2 = 1.0 - float(np.mean((y_true - y_pred) ** 2)) / var if var > 1e-18 else float("nan")
    return mae, r2


def fit_law(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int = 0,
    population_size: int = 500,
    generations: int = 25,
    parsimony_coefficient: float = 1e-3,
    const_range: Tuple[float, float] = (-4.0, 4.0),
    tournament_size: int = 20,
    max_train: int = 4000,
    hold_frac: float = 0.2,
    sample_weight: Optional[np.ndarray] = None,
    n_jobs: int = 1,
) -> Tuple[SymbolicLaw, np.ndarray]:
    """Evolve one symbolic law and return it with its held-out row indices.

    The normalisation is computed from the *fit* rows only: the observed excitation is
    data the experiment has, the evaluation rows are not theirs to touch.  ``max_train``
    caps how many rows the search may see, which is the dominant cost term.
    """
    n = int(X.shape[0])
    if n == 0:
        raise ValueError(f"fit_law({name}) needs at least one transition")
    rng = np.random.RandomState(1234 + seed)
    order = rng.permutation(n)
    n_hold = int(round(hold_frac * n))
    hold_idx = np.sort(order[:n_hold])
    fit_idx = order[n_hold:]
    if max_train < fit_idx.size:
        fit_idx = rng.choice(fit_idx, size=max_train, replace=False)

    x_scale = np.abs(X[fit_idx]).max(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    y_scale = float(np.abs(y[fit_idx]).max()) or 1.0

    est = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        const_range=const_range,
        parsimony_coefficient=parsimony_coefficient,
        function_set=list(GP_FUNCTION_SET),
        p_crossover=0.9,
        p_subtree_mutation=0.06,
        p_hoist_mutation=0.02,
        p_point_mutation=0.02,
        max_samples=1.0,
        feature_names=list(feature_names),
        n_jobs=n_jobs,
        verbose=0,
        random_state=seed,
    )
    t0 = time.time()
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)[fit_idx]
    est.fit(X[fit_idx] / x_scale, y[fit_idx] / y_scale, sample_weight=weight)
    fit_seconds = time.time() - t0

    # gplearn 0.4.2 exposes the winning program only through this attribute.
    program = est._program
    law = SymbolicLaw(
        name=name,
        feature_names=list(feature_names),
        estimator=est,
        x_scale=x_scale,
        y_scale=y_scale,
        expression=str(program),
        n_nodes=int(program.length_),
        depth=int(program.depth_),
        raw_fitness=float(program.raw_fitness_),
        fit_seconds=fit_seconds,
        seed=seed,
    )
    law.fit_mean = float(np.mean(y[fit_idx]))
    law.train_mae, law.train_r2 = _mae_r2(y[fit_idx], law.increment(X[fit_idx]))
    if hold_idx.size:
        pred = law.increment(X[hold_idx])
        law.hold_mae, law.hold_r2 = _mae_r2(y[hold_idx], pred)
        law.hold_rmse = float(np.sqrt(np.mean((y[hold_idx] - pred) ** 2)))
        law.hold_null_r2 = _mae_r2(y[hold_idx], np.full_like(y[hold_idx], law.fit_mean))[1]
    return law, hold_idx


def evaluate_law(law: LawLike, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE / MSE / R-squared of one law against a transition bank slice."""
    pred = law.increment(X)
    err = np.asarray(y, dtype=np.float64) - pred
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mse": float(np.mean(err**2)),
        "r2": _mae_r2(y, pred)[1],
    }


def fit_law_bank(
    bank: TransitionBank,
    preset: str = "synthetic",
    seeds: Sequence[int] = (0,),
    eval_bank: Optional[TransitionBank] = None,
    weights: Optional[Dict[str, np.ndarray]] = None,
    **gp_kwargs: Any,
) -> LawBank:
    """Fit every law of one preset, once per GP seed.

    When ``eval_bank`` is given, each law's ``hold_*`` metrics are computed on that
    *independent* transition bank rather than a random split of the fit rows -- for
    the synthetic study that is the difference between "the search memorised
    correlated frames of one rollout" and "the discovered law predicts unseen
    rollouts".
    """
    fit_mats = law_matrices(bank, preset)
    eval_mats = {} if eval_bank is None else law_matrices(eval_bank, preset)
    per_seed: Dict[str, List[SymbolicLaw]] = {name: [] for name in LAW_NAMES}
    for name in LAW_NAMES:
        X, y = fit_mats[name]
        features = LAW_FEATURES[preset][name]
        for seed in seeds:
            law, _ = fit_law(
                name,
                X,
                y,
                features,
                seed=seed,
                sample_weight=None if weights is None else weights.get(name),
                **gp_kwargs,
            )
            if eval_bank is not None:
                Xe, ye = eval_mats[name]
                metrics = evaluate_law(law, Xe, ye)
                law.hold_mae, law.hold_r2, law.hold_rmse = (
                    metrics["mae"],
                    metrics["r2"],
                    metrics["rmse"],
                )
                law.hold_null_r2 = _mae_r2(ye, np.full_like(ye, law.fit_mean))[1]
            per_seed[name].append(law)
    return LawBank(
        laws={name: per_seed[name][0] for name in LAW_NAMES},
        preset=preset,
        seeds=list(seeds),
        per_seed=per_seed,
    )


# ---------------------------------------------------------------------------
# Composition: discovered laws as an autoregressive world model
# ---------------------------------------------------------------------------


def _feature_matrix(
    feature_names: Sequence[str],
    states: np.ndarray,
    actions: np.ndarray,
    contact: np.ndarray,
    next_velocity: np.ndarray,
) -> np.ndarray:
    columns: Dict[str, np.ndarray] = {
        "vx": states[:, 2],
        "vy": states[:, 3],
        "dir": actions[:, 5] - actions[:, 4],
        "run": actions[:, 1],
        "jump": actions[:, 0],
        "ground": contact[:, 0],
        "ceiling": contact[:, 1],
        "left": contact[:, 2],
        "right": contact[:, 3],
        "vx_next": next_velocity[:, 0],
        "vy_next": next_velocity[:, 1],
    }
    return np.stack([columns[name] for name in feature_names], axis=1)


def symbolic_step(
    states: np.ndarray,
    actions: np.ndarray,
    contact: np.ndarray,
    model: WorldModel,
) -> np.ndarray:
    """One frame of a world model built from increment laws: ``[N, 4] -> [N, 4]``.

    ``model`` maps law names to anything exposing ``increment(X)`` -- a
    :class:`SymbolicLaw`, a :class:`BaggedLaw` or the :class:`AnalyticLaw` wrapper of
    the parametric map.  Velocity laws consume the driving state; position laws consume
    the *model's own* next velocity, so the integration identity is applied exactly as
    the analytic simulator applies it and both models roll out under identical
    bookkeeping.  No clamping is imposed: a law that never discovered the velocity
    ceiling should overshoot, and that overshoot is a measured outcome.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    contact = np.asarray(contact, dtype=np.float64)
    zero_v = np.zeros((states.shape[0], 2))
    vx_law, vy_law = model["dvx"], model["dvy"]
    dx_law, dy_law = model["dx"], model["dy"]
    dvx = vx_law.increment(_feature_matrix(vx_law.feature_names, states, actions, contact, zero_v))
    dvy = vy_law.increment(_feature_matrix(vy_law.feature_names, states, actions, contact, zero_v))
    next_velocity = np.stack([states[:, 2] + dvx, states[:, 3] + dvy], axis=1)
    dx = dx_law.increment(
        _feature_matrix(dx_law.feature_names, states, actions, contact, next_velocity)
    )
    dy = dy_law.increment(
        _feature_matrix(dy_law.feature_names, states, actions, contact, next_velocity)
    )
    return np.stack(
        [states[:, 0] + dx, states[:, 1] + dy, next_velocity[:, 0], next_velocity[:, 1]], axis=1
    )


def model_rollout(
    model: WorldModel,
    s0: np.ndarray,
    action_seq: np.ndarray,
    contact_seq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unroll :func:`symbolic_step`; mirrors ``simulate_rollout``'s contract.

    ``s0`` is ``[B, 4]``, ``action_seq`` ``[B, L, 6]`` and ``contact_seq`` an optional
    ``[B, L, 4]`` exogenous terrain-contact sequence (the contact byte is a
    measurement, not something the structure-free model is asked to infer).  Returns
    states at ``t = 1 .. L`` as ``[B, L, 4]``.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    action_seq = np.asarray(action_seq, dtype=np.float64)
    batch, horizon = int(action_seq.shape[0]), int(action_seq.shape[1])
    contact_seq = (
        np.zeros((batch, horizon, 4))
        if contact_seq is None
        else np.asarray(contact_seq, dtype=np.float64)
    )
    state = s0.copy()
    outs: List[np.ndarray] = []
    for t in range(horizon):
        state = symbolic_step(state, action_seq[:, t, :], contact_seq[:, t, :], model)
        outs.append(state)
    return np.stack(outs, axis=1)


# ---------------------------------------------------------------------------
# Probe stage: reading physical constants back out of a discovered law
# ---------------------------------------------------------------------------


def probe_constants(
    model: WorldModel,
    bank: TransitionBank,
    probe_rows: int = 120,
    velocity_probe_scale: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Read the seven section-10.40 constants out of a discovered world model.

    Each constant is defined by the *response* of the model at designed probes, never by
    the value of a GP terminal -- tree GP estimates constants poorly, and the plateau of
    the discovered map is the physically meaningful quantity anyway:

    * ``walk_accel`` / ``run_accel``: median driven increment over the lower half of the
      observed speed range, with and without the run button;
    * ``decel``: median coasting decrement over the same range with no direction held;
    * ``max_vx``: the *fixed point* of the driven map -- the speed at which pushing stops
      making Mario faster.  A law that never discovered the ceiling has no fixed point
      inside the explored range, reported explicitly rather than papered over with the
      data maximum;
    * ``subpixels_per_pixel``: reciprocal slope of the ``dx`` law plus the worst residual
      from that straight line, i.e. how non-kinematic the discovered integration is;
    * ``held_gravity`` / ``fall_gravity``: median increment on ascending probes with and
      without jump held, plus the descending consistency probe.

    ``velocity_probe_scale`` extends the probe beyond the observed support (a law may
    saturate only outside it; the overshoot is still reported as such).
    """
    vx_sup, vy_sup = bank.velocity_support
    step_x, step_y = bank.step_support

    def has(law_name: str) -> bool:
        return law_name in model

    rows = max(probe_rows, 8)
    drive = np.linspace(0.0, max(vx_sup * velocity_probe_scale, 1e-6), rows)
    mid = drive[drive <= 0.5 * max(vx_sup, 1e-6)]
    if mid.size == 0:
        mid = drive[: max(1, rows // 2)]

    def vx_increment(v: np.ndarray, direction: float, run: float) -> np.ndarray:
        law = model["dvx"]
        n = v.shape[0]
        cols = {
            "vx": v,
            "dir": np.full(n, direction),
            "run": np.full(n, run),
            "ground": np.zeros(n),
            "ceiling": np.zeros(n),
            "left": np.zeros(n),
            "right": np.zeros(n),
        }
        return law.increment(np.stack([cols[f] for f in law.feature_names], axis=1))

    def vy_increment(v: np.ndarray, jump: float) -> np.ndarray:
        law = model["dvy"]
        n = v.shape[0]
        cols = {
            "vy": v,
            "jump": np.full(n, jump),
            "ground": np.zeros(n),
            "ceiling": np.zeros(n),
            "left": np.zeros(n),
            "right": np.zeros(n),
        }
        return law.increment(np.stack([cols[f] for f in law.feature_names], axis=1))

    out: Dict[str, Dict[str, float]] = {}
    if has("dvx"):
        out["walk_accel"] = {"value": float(np.median(vx_increment(mid, 1.0, 0.0)))}
        out["run_accel"] = {"value": float(np.median(vx_increment(mid, 1.0, 1.0)))}
        out["decel"] = {"value": float(-np.median(vx_increment(mid, 0.0, 0.0)))}
        driven_next = drive + vx_increment(drive, 1.0, 1.0)
        drive_response = vx_increment(drive, 1.0, 1.0)
        # A fixed point is only a discovered *bound* if the map actually accelerates
        # below it: a degenerate zero-drive law satisfies v_next <= v everywhere and
        # would otherwise be read as a ceiling at the bottom of the probe range.
        accelerates = bool(np.any(drive_response > 1e-6))
        saturating = np.where(driven_next <= drive + 1e-9)[0]
        if saturating.size and accelerates:
            ceiling, found = float(driven_next[saturating[0]]), 1.0
        else:
            ceiling, found = float("nan"), 0.0
        out["max_vx"] = {
            "value": ceiling,
            "fixed_point_found": found,
            "reachable_support": float(vx_sup),
            "overshoot_px_per_frame": float(max(0.0, driven_next.max() - vx_sup)),
            # Median driven increment: zero here means the law found no force at all,
            # which is why a fixed point alone does not certify a discovered bound.
            "drive_response_px_per_frame": float(np.median(drive_response)),
        }

    if has("dvy"):
        ascending = np.linspace(-0.9 * max(vy_sup, 1e-6), -0.1 * max(vy_sup, 1e-6), rows)
        descending = np.linspace(0.1 * max(vy_sup, 1e-6), 0.9 * max(vy_sup, 1e-6), rows)
        out["held_gravity"] = {"value": float(np.median(vy_increment(ascending, 1.0)))}
        out["fall_gravity"] = {"value": float(np.median(vy_increment(ascending, 0.0)))}
        out["fall_gravity_descending"] = {"value": float(np.median(vy_increment(descending, 0.0)))}

    def position_scale(law_name: str, v: np.ndarray) -> Dict[str, float]:
        law = model[law_name]
        inc = law.increment(np.stack([v], axis=1))
        slope = float(np.sum(inc * v) / max(np.sum(v**2), 1e-18))
        return {
            "value": 1.0 / slope if abs(slope) > 1e-12 else float("nan"),
            "slope": slope,
            "identity_residual": float(np.abs(inc - slope * v).max()),
        }

    if has("dx"):
        out["subpixels_per_pixel"] = {
            **position_scale("dx", np.linspace(-vx_sup, vx_sup, rows)),
            "max_observed_step_px": step_x,
        }
    if has("dy"):
        out["subpixels_per_pixel_from_y"] = {
            **position_scale("dy", np.linspace(-vy_sup, vy_sup, rows)),
            "max_observed_step_px": step_y,
        }
    return out


def relative_error(
    probe: Dict[str, Dict[str, float]], truth: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Attach the ground-truth value and relative error to each probed constant."""
    named = {
        "max_vx": 0,
        "walk_accel": 1,
        "run_accel": 2,
        "subpixels_per_pixel": 3,
        "held_gravity": 4,
        "fall_gravity": 5,
        "decel": 6,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, idx in named.items():
        row = dict(probe.get(name, {}))
        t = float(np.asarray(truth, dtype=np.float64).reshape(-1)[idx])
        est = float(row.get("value", float("nan")))
        row["true"] = t
        row["rel_error_pct"] = (
            float(100.0 * abs(est - t) / abs(t)) if np.isfinite(est) and t != 0.0 else float("nan")
        )
        out[name] = row
    for extra in ("fall_gravity_descending", "subpixels_per_pixel_from_y"):
        out[extra] = dict(probe.get(extra, {}))
    return out


def summarize_constants(
    tables: Sequence[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-replicate constant tables into mean / std / recovery-rate rows."""
    keys = list(PARAM_NAMES) + ["subpixels_per_pixel_from_y", "fall_gravity_descending"]
    out: Dict[str, Dict[str, float]] = {}
    for name in keys:
        values, rels = [], []
        for table in tables:
            row = table.get(name, {})
            value = float(row.get("value", float("nan")))
            if np.isfinite(value):
                values.append(value)
            err = float(row.get("rel_error_pct", float("nan")))
            if np.isfinite(err):
                rels.append(err)
        out[name] = {
            "n_recovered": float(len(values)),
            "n_replicates": float(len(tables)),
            "mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median_rel_error_pct": float(np.median(rels)) if rels else float("nan"),
            "recovery_rate": float(len(rels) / max(len(tables), 1)),
        }
    return out


def excitation_weights(bank: TransitionBank, law: str, strength: float) -> Optional[np.ndarray]:
    """Sample weights that over-represent a poorly-excited regime (the ablation).

    ``dvx`` weights coasting frames (``dir == 0``), where the friction deadband lives;
    ``dvy`` frames that are simultaneously ascending and jump-held, where the
    held-gravity tier lives.  ``strength`` = 0 returns ``None``, i.e. the passive
    observation baseline exactly.
    """
    if strength <= 0.0:
        return None
    if law == "dvx":
        indicator = (bank.direction == 0).astype(np.float64)
    elif law == "dvy":
        indicator = ((bank.states[:, 3] < 0) & (bank.jump > 0.5)).astype(np.float64)
    else:
        return None
    return 1.0 + strength * indicator


def structural_tags(law: SymbolicLaw) -> Dict[str, bool]:
    """Which primitives the discovered expression actually used."""
    text = law.expression
    return {
        "uses_clamp": ("max(" in text) or ("min(" in text),
        "uses_abs": "abs(" in text,
        "uses_ratio": "div(" in text,
        "is_identity": law.n_nodes == 1,
        "depends_on_input": any(tok in text for tok in ("dir", "run", "jump")),
    }


def truth_vector(params: Any) -> np.ndarray:
    """The seven constants as a float64 vector, accepting dataclass or tensor."""
    return theta_tensor(params).numpy().astype(np.float64)
