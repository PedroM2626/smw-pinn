"""Nested-template model selection: the third point of the discovery comparison
(README section 10.43.9).

Section 10.43 asked whether tree genetic programming can *discover* the engine's
one-step laws and reported a structural negative: the rigid velocity ceiling is never
found, because a gplearn terminal is drawn uniformly from a bounded ``const_range`` and
a clamp at +/-48 sub-pixels is not in that representation. That section's own
limitations list conceded that this is a statement about *one* representation. This
module supplies the control that concedes nothing: it hands the *same data* a candidate
set in which every structure of interest is explicitly expressible, and asks only what
the data prefers.

Six nested horizontal templates, from "a constant" up to the full posited law, each with
its parameter count known and its non-linear level (the bound) estimated by a
deterministic grid search with an inner least-squares solve on the currently-unclamped
rows:

* ``H1_constant``  v' = c
* ``H2_affine``    v' = a0 + a1 v                        (velocity-dependent gain)
* ``H3_drive``     v' = v + d (g0 + g1 rho)              (traction tiers, no bound)
* ``H4_clamped``   v' = clip(H3, +/-M)                   (+ one rigid bound)
* ``H5_coast``     H4 with d = 0 shrunk toward zero      (+ Coulomb deadband)
* ``H6_contact``   H5 with a facing wall zeroing the     (+ collision response)
                   velocity driven into it

and the vertical analogue ``V1_constant, V2_affine, V3_single_gravity, V4_held_gate,
V5_clamped, V6_ground_reset``. Gravity tiers are estimated on free-flight rows only,
because a grounded frame carries no information about them - the same identifiability
logic as Section 10.40, applied to template fitting rather than to constants.

Three selection criteria are reported side by side and they can disagree: BIC on the fit
rows, held-out RMSE on an independent bank, and held-out RMSE restricted to the
near-bound frames. The third column is what actually decides whether a bound is
supported: a rigid clamp only earns its keep where the data saturates, and that is a
small fraction of passively-recorded transitions. Reading BIC alone is precisely how a
structure-free search comes to prefer the unbounded affine map.

numpy-only, deterministic, no new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Candidate bound levels are searched over the observed velocity support: a bound above
# every observed speed is no more identifiable here than it was for the genetic program.
_CLAMP_GRID = 41
_COAST_GRID = 41


@dataclass
class TemplateFit:
    """One fitted template: parameters, complexity, scores and a held-out predictor."""

    name: str
    n_parameters: int
    parameters: Dict[str, float]
    structure_present: bool
    structure_note: str
    predict: Callable[[Dict[str, np.ndarray]], np.ndarray] = field(
        repr=False, default=lambda b: np.zeros(np.asarray(b["v"]).shape)
    )
    train_rmse: float = float("nan")
    train_r2: float = float("nan")
    train_tail_rmse: float = float("nan")
    bic: float = float("nan")
    test_rmse: float = float("nan")
    test_r2: float = float("nan")
    test_tail_rmse: float = float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst if sst > 1e-18 else float("nan")


def _bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    n = y_true.shape[0]
    sse = max(float(np.sum((y_true - y_pred) ** 2)), 1e-30)
    return float(n * np.log(sse / n) + k * np.log(n))


def _lsq(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _shrink(v: np.ndarray, mu: float) -> np.ndarray:
    """Coulomb soft threshold toward zero: the engine's coast branch."""
    return np.sign(v) * np.maximum(np.abs(v) - mu, 0.0)


def _tail_mask(v_next: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Frames at or beyond ``frac`` of the largest speed the data ever reached."""
    cap = float(np.abs(v_next).max())
    return np.abs(v_next) >= frac * cap


def fit_horizontal_templates(
    v: np.ndarray,
    direction: np.ndarray,
    run: np.ndarray,
    v_next: np.ndarray,
    wall: Optional[np.ndarray] = None,
) -> List[TemplateFit]:
    """Fit the nested horizontal templates to ``(v, dir, run) -> v_next``."""
    v = np.asarray(v, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    run = np.asarray(run, dtype=np.float64)
    v_next = np.asarray(v_next, dtype=np.float64)
    n = v.shape[0]
    driven = direction != 0
    cap = float(np.abs(v_next).max())
    grid = (
        np.linspace(0.5 * cap, min(1.5 * cap, 4.0 * cap), _CLAMP_GRID)
        if cap > 0
        else np.array([1.0])
    )
    tail = _tail_mask(v_next)
    fits: List[TemplateFit] = []

    def register(
        name: str,
        k: int,
        params: Dict[str, float],
        predictor: Callable[[Dict[str, np.ndarray]], np.ndarray],
        present: bool,
        note: str,
    ) -> None:
        batch = {
            "v": v,
            "dir": direction,
            "run": run,
            "wall": np.zeros((n, 4)) if wall is None else np.asarray(wall, dtype=np.float64),
        }
        pred = predictor(batch)
        fits.append(
            TemplateFit(
                name=name,
                n_parameters=k,
                parameters=params,
                structure_present=present,
                structure_note=note,
                predict=predictor,
                train_rmse=float(np.sqrt(np.mean((v_next - pred) ** 2))),
                train_r2=_r2(v_next, pred),
                train_tail_rmse=float(np.sqrt(np.mean((v_next[tail] - pred[tail]) ** 2))),
                bic=_bic(v_next, pred, k),
            )
        )

    c = float(v_next.mean())
    register(
        "H1_constant",
        1,
        {"c": c},
        lambda b: np.full(np.asarray(b["v"]).shape, c),
        False,
        "no structure",
    )

    A2 = np.stack([np.ones(n), v], axis=1)
    a2 = _lsq(A2, v_next)
    register(
        "H2_affine",
        2,
        {"a0": float(a2[0]), "a1": float(a2[1])},
        lambda b: b["v"] * a2[1] + a2[0],
        False,
        "velocity-dependent gain, no input term",
    )

    A3 = np.stack([direction, direction * run], axis=1)
    a3 = _lsq(A3, v_next - v)
    register(
        "H3_drive",
        2,
        {"walk": float(a3[0]), "run_increment": float(a3[1])},
        lambda b: b["v"] + b["dir"] * (a3[0] + a3[1] * b["run"]),
        False,
        "two traction tiers, unbounded",
    )

    _, a4, m4 = _fit_clamped(v, A3, v_next, grid)
    register(
        "H4_clamped",
        3,
        {"walk": float(a4[0]), "run_increment": float(a4[1]), "max_vx": m4},
        lambda b: np.clip(b["v"] + b["dir"] * (a4[0] + a4[1] * b["run"]), -m4, m4),
        True,
        "rigid symmetric velocity bound",
    )

    _, a5, m5, mu5 = _fit_coast(v, A3, driven, v_next, m4, a4)
    register(
        "H5_coast",
        4,
        {"walk": float(a5[0]), "run_increment": float(a5[1]), "max_vx": m5, "decel": mu5},
        lambda b: np.clip(
            np.where(
                b["dir"] != 0,
                b["v"] + b["dir"] * (a5[0] + a5[1] * b["run"]),
                _shrink(b["v"], mu5),
            ),
            -m5,
            m5,
        ),
        True,
        "rigid bound plus Coulomb coast deadband",
    )

    if wall is not None and float(np.asarray(wall).sum()) > 0.0:

        def h6(b: Dict[str, np.ndarray]) -> np.ndarray:
            ww = b["wall"]
            base = np.clip(
                np.where(
                    b["dir"] != 0,
                    b["v"] + b["dir"] * (a5[0] + a5[1] * b["run"]),
                    _shrink(b["v"], mu5),
                ),
                -m5,
                m5,
            )
            into = ((b["dir"] > 0) & (ww[:, 1] > 0.5)) | ((b["dir"] < 0) & (ww[:, 2] > 0.5))
            return np.where(into, 0.0, base)

        register(
            "H6_contact",
            4,
            {"walk": float(a5[0]), "run_increment": float(a5[1]), "max_vx": m5, "decel": mu5},
            h6,
            True,
            "bound, deadband and facing-wall stop",
        )
    return fits


def _fit_clamped(
    v: np.ndarray, A: np.ndarray, v_next: np.ndarray, grid: np.ndarray, sweeps: int = 3
) -> Tuple[float, np.ndarray, float]:
    """Active-set fit of ``clip(v + A coef, +/-m)`` over a grid of candidate bounds."""
    coef = _lsq(A, v_next - v)
    best: Optional[Tuple[float, np.ndarray, float]] = None
    for m in grid:
        c = coef.copy()
        for _ in range(sweeps):
            inside = np.abs(v + A @ c) < m - 1e-12
            if inside.sum() <= A.shape[1]:
                break
            c = _lsq(A[inside], (v_next[inside] - v[inside]))
        pred = np.clip(v + A @ c, -m, m)
        sse = float(np.sum((v_next - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, c, float(m))
    assert best is not None
    return best


def _fit_coast(
    v: np.ndarray,
    A: np.ndarray,
    driven: np.ndarray,
    v_next: np.ndarray,
    m: float,
    coef: np.ndarray,
) -> Tuple[float, np.ndarray, float, float]:
    """Add the Coulomb coast branch: one extra parameter, estimated on a grid."""
    coast = ~driven
    # The deadband is a per-frame decrement, so its grid must be scaled by the observed
    # coast increments: a velocity-scaled grid is too coarse to resolve mu at all.
    increments = np.abs((v_next - v)[coast]) if coast.any() else np.abs(v) * 0.0
    scale = max(float(np.mean(increments)) if increments.size else 1.0, 1e-3)
    best: Optional[Tuple[float, np.ndarray, float, float]] = None
    for mu in np.linspace(0.0, 4.0 * scale, _COAST_GRID):
        # The tiers are re-estimated on the driven rows with the same active-set rule as
        # the bound-only template, so the coast branch cannot bias them through the frames
        # that were already pinned at the ceiling.
        c = _lsq(A[driven], (v_next[driven] - v[driven])) if driven.sum() > A.shape[1] else coef
        for _ in range(3):
            inside = driven & (np.abs(v + A @ c) < m - 1e-12)
            if inside.sum() <= A.shape[1]:
                break
            c = _lsq(A[inside], (v_next[inside] - v[inside]))
        pred = np.clip(np.where(driven, v + A @ c, _shrink(v, mu)), -m, m)
        sse = float(np.sum((v_next - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, c, m, float(mu))
    assert best is not None
    return best


def fit_vertical_templates(
    vy: np.ndarray,
    jump: np.ndarray,
    v_next_y: np.ndarray,
    ground: Optional[np.ndarray] = None,
) -> List[TemplateFit]:
    """Fit the nested vertical templates to ``(vy, jump) -> vy_next``.

    Gravity tiers are estimated on free-flight rows only: a grounded frame has already
    had its downward velocity reset, so it carries no information about ``g`` and would
    bias every tier estimate that included it.
    """
    vy = np.asarray(vy, dtype=np.float64)
    jump = np.asarray(jump, dtype=np.float64)
    v_next_y = np.asarray(v_next_y, dtype=np.float64)
    n = vy.shape[0]
    g = (
        np.zeros(n)
        if ground is None
        else (np.asarray(ground, dtype=np.float64) > 0.5).astype(np.float64)
    )
    free = g < 0.5
    rows = np.where(free)[0]
    asc = (vy < 0).astype(np.float64)
    gate = jump * asc
    cap = float(np.abs(v_next_y[free]).max()) if free.any() else float(np.abs(v_next_y).max())
    grid = (
        np.linspace(0.5 * cap, min(2.0 * cap, 4.0 * cap), _CLAMP_GRID)
        if cap > 0
        else np.array([1.0])
    )
    fits: List[TemplateFit] = []

    def register(
        name: str,
        k: int,
        params: Dict[str, float],
        predictor: Callable[[Dict[str, np.ndarray]], np.ndarray],
        present: bool,
        note: str,
    ) -> None:
        batch = {"vy": vy, "jump": jump, "asc": asc, "ground": g}
        pred = predictor(batch)
        fits.append(
            TemplateFit(
                name=name,
                n_parameters=k,
                parameters=params,
                structure_present=present,
                structure_note=note,
                predict=predictor,
                train_rmse=float(np.sqrt(np.mean((v_next_y - pred) ** 2))),
                train_r2=_r2(v_next_y, pred),
                train_tail_rmse=float(np.sqrt(np.mean((v_next_y[free] - pred[free]) ** 2))),
                bic=_bic(v_next_y, pred, k),
            )
        )

    c = float(v_next_y.mean())
    register(
        "V1_constant",
        1,
        {"c": c},
        lambda b: np.full(np.asarray(b["vy"]).shape, c),
        False,
        "no structure",
    )

    A2 = np.stack([np.ones(n), vy], axis=1)
    a2 = _lsq(A2, v_next_y)
    register(
        "V2_affine",
        2,
        {"a0": float(a2[0]), "a1": float(a2[1])},
        lambda b: b["vy"] * a2[1] + a2[0],
        False,
        "velocity-dependent gain",
    )

    g3 = float(np.mean(v_next_y[rows] - vy[rows])) if rows.size else 0.0
    register(
        "V3_single_gravity",
        1,
        {"gravity": g3},
        lambda b: b["vy"] + g3,
        False,
        "one gravity tier, free-flight fit",
    )

    A4 = np.stack([np.ones(rows.size), gate[rows]], axis=1)
    a4 = _lsq(A4, (v_next_y - vy)[rows]) if rows.size > 2 else np.array([g3, 0.0])
    register(
        "V4_held_gate",
        2,
        {
            "fall_gravity": float(a4[0]),
            "held_increment": float(a4[1]),
            "g_hold": float(a4[0] + a4[1]),
        },
        lambda b: b["vy"] + a4[0] + b["jump"] * b["asc"] * a4[1],
        True,
        "discontinuous held-jump gravity gate",
    )

    m5, a5 = _fit_vertical_clamp(vy, gate, v_next_y, rows, grid, a4)
    register(
        "V5_clamped",
        3,
        {
            "fall_gravity": float(a5[0]),
            "held_increment": float(a5[1]),
            "g_hold": float(a5[0] + a5[1]),
            "terminal_vy": m5,
        },
        lambda b: np.clip(b["vy"] + a5[0] + b["jump"] * b["asc"] * a5[1], -m5, m5),
        True,
        "held gate plus terminal-velocity clamp",
    )

    if float(g.sum()) > 0.0:

        def v6(b: Dict[str, np.ndarray]) -> np.ndarray:
            base = np.clip(b["vy"] + a5[0] + b["jump"] * b["asc"] * a5[1], -m5, m5)
            return np.where(b["ground"] > 0.5, np.minimum(base, 0.0), base)

        register(
            "V6_ground_reset",
            3,
            {
                "fall_gravity": float(a5[0]),
                "held_increment": float(a5[1]),
                "g_hold": float(a5[0] + a5[1]),
                "terminal_vy": m5,
            },
            v6,
            True,
            "gate, clamp and ground-contact downward reset",
        )
    return fits


def _fit_vertical_clamp(
    vy: np.ndarray,
    gate: np.ndarray,
    v_next_y: np.ndarray,
    rows: np.ndarray,
    grid: np.ndarray,
    coef: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Grid the terminal-velocity level, refitting tiers on the unclamped free-flight rows."""
    best: Optional[Tuple[float, float, np.ndarray]] = None
    for m in grid:
        c = coef.copy()
        for _ in range(3):
            raw = vy[rows] + c[0] + gate[rows] * c[1]
            inside = np.abs(raw) < m - 1e-12
            if inside.sum() <= 2:
                break
            A = np.stack([np.ones(int(inside.sum())), gate[rows][inside]], axis=1)
            c = _lsq(A, (v_next_y[rows] - vy[rows])[inside])
        pred = np.clip(vy[rows] + c[0] + gate[rows] * c[1], -m, m)
        sse = float(np.sum((v_next_y[rows] - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(m), c)
    assert best is not None
    return best[1], best[2]


def evaluate_templates(
    fits: Sequence[TemplateFit],
    batch: Dict[str, np.ndarray],
    target: np.ndarray,
    tail: Optional[np.ndarray] = None,
) -> None:
    """Attach held-out metrics to already-fitted templates (mutates in place)."""
    for f in fits:
        pred = f.predict(batch)
        f.test_rmse = float(np.sqrt(np.mean((target - pred) ** 2)))
        f.test_r2 = _r2(target, pred)
        if tail is not None and bool(tail.any()):
            f.test_tail_rmse = float(np.sqrt(np.mean((target[tail] - pred[tail]) ** 2)))


def summarise(fits: Sequence[TemplateFit]) -> Dict[str, object]:
    """Tabulate templates and report what each criterion selects."""
    table: Dict[str, object] = {}
    for f in fits:
        table[f.name] = {
            "n_parameters": f.n_parameters,
            "train_rmse": f.train_rmse,
            "train_r2": f.train_r2,
            "train_tail_rmse": f.train_tail_rmse,
            "test_rmse": f.test_rmse,
            "test_r2": f.test_r2,
            "test_tail_rmse": f.test_tail_rmse,
            "bic": f.bic,
            "structure_present": f.structure_present,
            "structure_note": f.structure_note,
            **{k: float(v) for k, v in f.parameters.items()},
        }
    bic_pick = min(fits, key=lambda f: f.bic)
    rmse_pick = min(fits, key=lambda f: f.test_rmse if np.isfinite(f.test_rmse) else np.inf)
    tailed = [f for f in fits if np.isfinite(f.test_tail_rmse)]
    tail_pick = min(tailed, key=lambda f: f.test_tail_rmse) if tailed else rmse_pick
    return {
        "templates": table,
        "bic_selected": bic_pick.name,
        "test_rmse_selected": rmse_pick.name,
        "tail_rmse_selected": tail_pick.name,
        "selected_structure_by_bic": bic_pick.structure_note,
        "selected_structure_by_tail": tail_pick.structure_note,
        "tail_selected_parameters": {k: float(v) for k, v in tail_pick.parameters.items()},
        "criteria_agree": bool(bic_pick.name == tail_pick.name == rmse_pick.name),
    }


__all__ = [
    "TemplateFit",
    "evaluate_templates",
    "fit_horizontal_templates",
    "fit_vertical_templates",
    "summarise",
]
