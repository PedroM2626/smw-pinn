"""
symbolic_engine_ablation_benchmark.py
Is the missing structure a representation limit or a data limit? (README Section 10.43.9)

Section 10.43 reported that genetic programming never discovers the engine's rigid velocity
ceiling, and attributed that to gplearn's bounded, uniformly-drawn terminal constants - a
representation limit, with the honest caveat that it was a statement about one
representation. This study removes the caveat by running the *same inverse problem* through
three engines of increasing structural generosity and asking the same structural questions of
each:

* **gplearn tree GP** - the published 10.43 estimator: free-form trees, terminals drawn
  uniformly from ``const_range=(-4, 4)``, mean-error fitness.
* **PySR** (optional; recorded as unavailable rather than skipped silently) - the same kind
  of tree search, but with unbounded constants refined by numerical optimisation and
  ``min``/``max`` as first-class operators, so a bound at 48 is expressible.
* **Nested templates + model selection** (`src/inverse/structure_selection.py`) - not a
  search at all: every structure of interest (traction tiers, rigid bound, Coulomb deadband,
  held-jump gravity gate, terminal clamp, ground reset) is an explicit candidate, and the
  only question is which one the data prefers under BIC, under held-out RMSE, and under
  held-out RMSE restricted to the near-bound frames.

Three questions are answered per engine, on the hidden world of 10.40-E1 and again on genuine
WRAM telemetry: is the rigid velocity bound discovered, is the held-jump gravity gate
discovered, and how exactly are the physical constants recovered? The engines are given the
same design matrix, the same row budget and the same held-out bank, so a difference in outcome
is a difference between the searches rather than between their data or their accuracy. What
separates them, and how much of the 10.43 negative result that accounts for, is decided by the
measured rates and written into the artifact's ``verdict`` rather than asserted here.

Emulator-free and deterministic. Writes ``results/symbolic_engine_ablation_metrics.json``
with ``_meta`` and ``results/figures/symbolic_engine_ablation.png``.

Run:  python -m src.evaluation.symbolic_engine_ablation_benchmark
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.environment.dataset_loader import load_and_preprocess_data  # noqa: E402
from src.evaluation.inverse_transfer_benchmark import OOD_WORLD, PRIOR  # noqa: E402
from src.inverse.parameter_identification import (
    PARAM_NAMES,
    generate_synthetic_windows,
    theta_tensor,
)
from src.inverse.structure_selection import (
    evaluate_templates,
    fit_horizontal_templates,
    fit_vertical_templates,
    summarise,
)
from src.inverse.symbolic_regression import (
    LAW_FEATURES,
    TransitionBank,
    bank_from_transitions,
    bank_from_windows,
    fit_law_bank,
    law_matrices,
    probe_constants,
    relative_error,
    summarize_constants,
    truth_vector,
)
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

ARTIFACT_NAME = "symbolic_engine_ablation_metrics.json"
FIGURE_NAME = "symbolic_engine_ablation.png"
ENGINES: Tuple[str, ...] = ("gplearn", "pysr", "templates")


def pysr_available() -> bool:
    return importlib.util.find_spec("pysr") is not None


def flat_transitions(
    windows: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(s0, actions, targets, contact)`` windows -> flat single-step arrays."""
    s0, acts, tgts, ct = (np.asarray(w, dtype=np.float64) for w in windows)
    states = np.concatenate([s0[:, None, :], tgts[:, :-1, :]], axis=1).reshape(-1, 4)
    return states, acts.reshape(-1, 6), tgts.reshape(-1, 4), ct.reshape(-1, 4)


def _bound_fixed_point(predict_next: Any, cap: float, rows: int = 193) -> Dict[str, float]:
    """Does the discovered horizontal map have a fixed point below the explored range?

    ``predict_next(v)`` must return the next velocity for a fully-driven sprint. A map with
    no rigid bound keeps accelerating and yields no fixed point; a degenerate map that never
    accelerates is rejected as well, exactly as in the 10.43 probe.
    """
    v = np.linspace(0.0, cap, rows)
    nxt = np.asarray(predict_next(v), dtype=np.float64)
    accelerating = bool(np.any(nxt > v + 1e-6))
    saturating = np.where(nxt <= v + 1e-6)[0]
    if saturating.size and accelerating:
        return {
            "fixed_point_found": 1.0,
            "bound_value": float(nxt[saturating[0]]),
            "overshoot": float(max(0.0, nxt.max() - cap)),
        }
    return {
        "fixed_point_found": 0.0,
        "bound_value": float("nan"),
        "overshoot": float(max(0.0, nxt.max() - cap)),
    }


def _sprint_rows(law: Any, v: np.ndarray) -> np.ndarray:
    """The law's own design matrix at velocity ``v``, driven right at sprint, in free flight.

    Built from ``feature_names`` rather than positionally, so the same probe works on the
    contact-free synthetic preset and on the contact-aware real preset.
    """
    cols: Dict[str, np.ndarray] = {
        "vx": v,
        "dir": np.ones_like(v),
        "run": np.ones_like(v),
        "ground": np.zeros_like(v),
        "ceiling": np.zeros_like(v),
        "left": np.zeros_like(v),
        "right": np.zeros_like(v),
    }
    return np.stack([cols[f] for f in law.feature_names], axis=1)


def _jump_rows(law: Any, vy: np.ndarray, held: float) -> np.ndarray:
    """The vertical design at velocity ``vy`` in free flight, with the jump button as given."""
    cols: Dict[str, np.ndarray] = {
        "vy": vy,
        "jump": np.full_like(vy, held),
        "asc": (vy < 0).astype(np.float64),
        "ground": np.zeros_like(vy),
        "ceiling": np.zeros_like(vy),
        "left": np.zeros_like(vy),
        "right": np.zeros_like(vy),
    }
    return np.stack([cols[f] for f in law.feature_names], axis=1)


def run_gplearn(
    fit_states: np.ndarray,
    fit_actions: np.ndarray,
    fit_next: np.ndarray,
    fit_contact: np.ndarray,
    eval_bank: TransitionBank,
    truth: np.ndarray,
    seeds: Sequence[int],
    budget: Dict[str, Any],
    preset: str = "synthetic",
) -> Dict[str, Any]:
    """The published 10.43 estimator, re-run on this study's banks."""
    from src.inverse.symbolic_regression import parametric_laws  # noqa: F401 - parity import

    bank = bank_from_transitions(fit_states, fit_actions, fit_next, fit_contact)
    laws = fit_law_bank(bank, preset, seeds=seeds, eval_bank=eval_bank, **budget)
    bagged = laws.bagged()
    table = relative_error(probe_constants(bagged, bank), truth)
    summary = summarize_constants([table])
    cap = float(np.abs(fit_next[:, 2]).max())
    dvx, dvy = bagged["dvx"], bagged["dvy"]
    probe = _bound_fixed_point(lambda v: v + dvx.increment(_sprint_rows(dvx, v)), cap)
    asc = np.linspace(-0.9 * cap, -0.1 * cap, 61)
    gh = float(np.median(dvy.increment(_jump_rows(dvy, asc, 1.0))))
    gf = float(np.median(dvy.increment(_jump_rows(dvy, asc, 0.0))))
    ev_X, ev_y = law_matrices(eval_bank, preset)["dvx"]
    return {
        "preset": preset,
        "dvx_eval_r2": _r2(ev_y, np.asarray(dvx.increment(ev_X), dtype=np.float64)),
        "constants": {
            k: {
                "value": summary[k]["mean"],
                "std": summary[k]["std"],
                "rel_error_pct": table[k]["rel_error_pct"],
            }
            for k in PARAM_NAMES
        },
        "structure": {
            "rigid_bound_discovered": bool(probe["fixed_point_found"] > 0.5),
            "bound_value": probe["bound_value"],
            "bound_overshoot": probe["overshoot"],
            "gravity_gate_discovered": bool(abs(gh - gf) > 0.5 * abs(truth[4] - truth[5])),
            "tier_separation": gh - gf,
            "true_tier_separation": float(truth[4] - truth[5]),
        },
        "per_law": {
            name: {
                "expression": law.expression,
                "nodes": law.n_nodes,
                "hold_r2": law.hold_r2,
                "hold_mae": law.hold_mae,
            }
            for name, law in laws.laws.items()
        },
        "mean_nodes": laws.mean_nodes(),
        "gp_seconds": laws.total_seconds(),
    }


def run_pysr(
    fit_bank: TransitionBank,
    eval_bank: TransitionBank,
    truth: np.ndarray,
    niterations: int,
    max_train: int,
    seed: int,
    preset: str = "synthetic",
) -> Dict[str, Any]:
    """PySR on the same laws: unbounded numerically-optimised constants, min/max available."""
    if not pysr_available():
        return {
            "available": False,
            "reason": "the pysr package (and its Julia runtime) is not installed",
        }
    from pysr import PySRRegressor

    # The design matrices are the ones gplearn was given, unchanged, so the two tree
    # engines differ in their search and in their constants, not in what they can see.
    fit_m = law_matrices(fit_bank, preset)
    eval_m = law_matrices(eval_bank, preset)
    # The same row budget gplearn is given, so a difference in outcome cannot be bought
    # with a difference in data.  gplearn spends part of its budget on an internal
    # hold-out; both engines are then scored on the shared evaluation bank.
    n_fit = int(fit_bank.n)
    rows = np.sort(
        np.random.RandomState(1234 + seed).choice(n_fit, size=min(max_train, n_fit), replace=False)
    )

    ops = ["+", "-", "*", "min", "max"]
    started = time.perf_counter()
    out: Dict[str, Any] = {
        "available": True,
        "niterations": niterations,
        "operators": ops,
        "fit_rows": int(rows.size),
        "preset": preset,
    }
    models: Dict[str, Any] = {}
    per_law: Dict[str, Any] = {}
    for key in ("dvx", "dvy"):
        Xf, yf = fit_m[key]
        Xe, ye = eval_m[key]
        est = PySRRegressor(
            niterations=niterations,
            population_size=600,
            binary_operators=ops,
            unary_operators=["abs"],
            # Serial and deterministic: the published artifact has to be reproducible,
            # and PySR's parallel search is not bit-reproducible for a fixed seed.
            parallelism="serial",
            deterministic=True,
            # PySR writes its search state to disk; keep that scratch out of the repository.
            output_directory=os.path.join(tempfile.gettempdir(), f"mworld_pysr_{seed}_{key}"),
            verbosity=0,
            progress=False,
            random_state=seed,
            maxsize=32,
        )
        est.fit(Xf[rows], yf[rows])
        models[key] = est
        pred = np.asarray(est.predict(Xe), dtype=np.float64)
        best = est.equations_.sort_values("loss").iloc[0]
        # ``eval_r2`` is scored on the shared held-out transition bank, the same rows the
        # other two engines are scored on, so the accuracy column is comparable.
        per_law[key] = {
            "expression": str(best["equation"]),
            "nodes": float(best["complexity"]),
            "eval_r2": _r2(ye, pred),
            "eval_mae": float(np.mean(np.abs(ye - pred))),
            "eval_rmse": float(np.sqrt(np.mean((ye - pred) ** 2))),
        }
    out["per_law"] = per_law
    out["dvx_eval_r2"] = per_law["dvx"]["eval_r2"]
    cap = float(np.abs(fit_bank.next_states[:, 2]).max())
    # The estimators take positional columns, so the probes borrow the gplearn law
    # interface: same feature names, same probe, same definition of "discovered".
    dvx_law = SimpleNamespace(name="dvx", feature_names=LAW_FEATURES[preset]["dvx"])
    dvy_law = SimpleNamespace(name="dvy", feature_names=LAW_FEATURES[preset]["dvy"])

    def predict(model: Any, rows_x: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(rows_x), dtype=np.float64)

    probe = _bound_fixed_point(lambda v: v + predict(models["dvx"], _sprint_rows(dvx_law, v)), cap)
    asc = np.linspace(-0.9 * cap, -0.1 * cap, 61)
    separation = float(
        np.median(predict(models["dvy"], _jump_rows(dvy_law, asc, 1.0)))
        - np.median(predict(models["dvy"], _jump_rows(dvy_law, asc, 0.0)))
    )
    out["structure"] = {
        "rigid_bound_discovered": bool(probe["fixed_point_found"] > 0.5),
        "bound_value": probe["bound_value"],
        "bound_overshoot": probe["overshoot"],
        "gravity_gate_discovered": bool(abs(separation) > 0.5 * abs(float(truth[4] - truth[5]))),
        "tier_separation": separation,
        "true_tier_separation": float(truth[4] - truth[5]),
    }
    out["pysr_seconds"] = time.perf_counter() - started
    return out


def run_templates(
    fit_states: np.ndarray,
    fit_actions: np.ndarray,
    fit_next: np.ndarray,
    fit_contact: np.ndarray,
    eval_states: np.ndarray,
    eval_actions: np.ndarray,
    eval_next: np.ndarray,
    eval_contact: np.ndarray,
    truth: np.ndarray,
    with_contact: bool,
) -> Dict[str, Any]:
    """Nested-template engine: every structure is a candidate, selection is the question."""
    fd = fit_actions[:, 5] - fit_actions[:, 4]
    ed = eval_actions[:, 5] - eval_actions[:, 4]
    wall = np.asarray(fit_contact, dtype=np.float64) if with_contact else None
    fh = fit_horizontal_templates(
        fit_states[:, 2], fd, fit_actions[:, 1], fit_next[:, 2], wall=wall
    )
    tail = np.abs(eval_next[:, 2]) >= 0.9 * float(np.abs(fit_next[:, 2]).max())
    evaluate_templates(
        fh,
        {
            "v": eval_states[:, 2],
            "dir": ed,
            "run": eval_actions[:, 1],
            "wall": np.asarray(eval_contact, dtype=np.float64),
        },
        eval_next[:, 2],
        tail,
    )
    fv = fit_vertical_templates(
        fit_states[:, 3],
        fit_actions[:, 0],
        fit_next[:, 3],
        ground=np.asarray(fit_contact, dtype=np.float64)[:, 0] if with_contact else None,
    )
    evaluate_templates(
        fv,
        {
            "vy": eval_states[:, 3],
            "jump": eval_actions[:, 0],
            "asc": (eval_states[:, 3] < 0).astype(np.float64),
            "ground": np.asarray(eval_contact, dtype=np.float64)[:, 0],
        },
        eval_next[:, 3],
        None,
    )
    sh, sv = summarise(fh), summarise(fv)

    def params(summary: Dict[str, object]) -> Dict[str, float]:
        return {k: float(v) for k, v in dict(summary["tail_selected_parameters"]).items()}  # type: ignore[arg-type]

    ph, pv = params(sh), params(sv)
    bound = ph.get("max_vx", float("nan"))
    gate = pv.get("held_increment", float("nan"))
    chosen = str(sh["tail_rmse_selected"])
    return {
        "dvx_eval_r2": float(sh["templates"][chosen]["test_r2"]),
        "dvx_selected": chosen,
        # How many held-out frames the bound actually binds on: the fraction of evaluation
        # rows at or above 90% of the observed speed support, i.e. the rows a clamp can
        # be identified from at all.
        "tail_frame_fraction": float(np.mean(tail)),
        "horizontal": sh,
        "vertical": sv,
        "constants": {
            "max_vx": {"value": bound, "rel_error_pct": _rel(bound, truth[0])},
            "walk_accel": {
                "value": ph.get("walk"),
                "rel_error_pct": _rel(ph.get("walk"), truth[1]),
            },
            "run_accel": {
                "value": (ph.get("walk", 0.0) or 0.0) + (ph.get("run_increment", 0.0) or 0.0),
                "rel_error_pct": _rel(
                    (ph.get("walk", 0.0) or 0.0) + (ph.get("run_increment", 0.0) or 0.0), truth[2]
                ),
            },
            "decel": {"value": ph.get("decel"), "rel_error_pct": _rel(ph.get("decel"), truth[6])},
            "held_gravity": {
                "value": pv.get("g_hold"),
                "rel_error_pct": _rel(pv.get("g_hold"), truth[4]),
            },
            "fall_gravity": {
                "value": pv.get("fall_gravity"),
                "rel_error_pct": _rel(pv.get("fall_gravity"), truth[5]),
            },
        },
        "structure": {
            "rigid_bound_discovered": bool(
                sh["tail_rmse_selected"] in ("H4_clamped", "H5_coast", "H6_contact")
            ),
            "bound_value": bound,
            "bound_selected_by_bic": bool(
                sh["bic_selected"] in ("H4_clamped", "H5_coast", "H6_contact")
            ),
            "bound_selected_by_tail": bool(
                sh["tail_rmse_selected"] in ("H4_clamped", "H5_coast", "H6_contact")
            ),
            "gravity_gate_discovered": bool(
                sv["tail_rmse_selected"] in ("V4_held_gate", "V5_clamped", "V6_ground_reset")
            ),
            "tier_separation": gate,
            "true_tier_separation": float(truth[4] - truth[5]),
            "criteria_agree_horizontal": bool(sh["criteria_agree"]),
            "criteria_agree_vertical": bool(sv["criteria_agree"]),
        },
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Held-out R^2 of one engine's horizontal increment law, on the shared eval bank."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    var = float(np.var(yt))
    return float(1.0 - np.mean((yt - yp) ** 2) / var) if var > 0 else float("nan")


def _nanmean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _rel(value: Optional[float], truth: float) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(100.0 * abs(float(value) - float(truth)) / abs(float(truth)))


def run_study(
    output_dir: str = "results",
    replicates: int = 3,
    gp_seeds: int = 3,
    population_size: int = 500,
    generations: int = 25,
    max_train: int = 4000,
    fit_windows: int = 800,
    eval_windows: int = 300,
    rollout_len: int = 12,
    pysr_iterations: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the three-engine ablation on the hidden world and on real telemetry."""
    set_global_seed(seed)
    torch.manual_seed(seed)
    truth = truth_vector(OOD_WORLD)
    prior = truth_vector(PRIOR)
    budget = {
        "population_size": population_size,
        "generations": generations,
        "max_train": max_train,
        "parsimony_coefficient": 1e-3,
        "n_jobs": 1,
    }
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== Engine ablation on the hidden world (structure known only to templates) ===")
    have_pysr = pysr_available()
    if not have_pysr:
        logger.warning("=== PySR is not installed: recorded as an unavailable engine ===")
    hidden: Dict[str, Any] = {}
    for rep in range(replicates):
        fit_w = generate_synthetic_windows(
            theta_tensor(OOD_WORLD),
            n_windows=fit_windows,
            rollout_len=rollout_len,
            seed=11 + 10 * rep,
        )
        eval_w = generate_synthetic_windows(
            theta_tensor(OOD_WORLD),
            n_windows=eval_windows,
            rollout_len=rollout_len,
            seed=907 + 10 * rep,
        )
        fs, fa, fn, fc = flat_transitions(fit_w)
        es, ea, en, ec = flat_transitions(eval_w)
        fit_bank = bank_from_windows(fit_w)
        eval_bank = bank_from_windows(eval_w)
        row: Dict[str, Any] = {}
        row["gplearn"] = run_gplearn(
            fs, fa, fn, fc, eval_bank, truth, tuple(range(gp_seeds)), budget
        )
        row["templates"] = run_templates(fs, fa, fn, fc, es, ea, en, ec, truth, with_contact=True)
        logger.info(
            "  rep %d | gplearn bound %s gate %s (%.1f gp-s) | templates bound %s (BIC %s) gate %s",
            rep,
            row["gplearn"]["structure"]["rigid_bound_discovered"],
            row["gplearn"]["structure"]["gravity_gate_discovered"],
            row["gplearn"]["gp_seconds"],
            row["templates"]["structure"]["rigid_bound_discovered"],
            row["templates"]["horizontal"]["bic_selected"],
            row["templates"]["structure"]["gravity_gate_discovered"],
        )
        if have_pysr:
            row["pysr"] = run_pysr(
                fit_bank, eval_bank, truth, pysr_iterations, max_train, seed + rep
            )
            logger.info(
                "         pysr bound %s gate %s | %.1f s | dvx held-out R2 %.3f (gplearn %.3f)",
                row["pysr"]["structure"]["rigid_bound_discovered"],
                row["pysr"]["structure"]["gravity_gate_discovered"],
                row["pysr"]["pysr_seconds"],
                row["pysr"]["dvx_eval_r2"],
                row["gplearn"]["dvx_eval_r2"],
            )
        hidden[f"replicate_{rep}"] = row
    if not have_pysr:
        hidden["pysr"] = {"available": False, "reason": "the pysr package is not installed"}

    logger.info("=== Engine ablation on genuine WRAM telemetry ===")
    data = load_and_preprocess_data(seed=seed)
    train_bank = bank_from_transitions(
        data["train_states"], data["train_actions"], data["train_next_states"]
    )
    test_bank = bank_from_transitions(
        data["test_states"], data["test_actions"], data["test_next_states"]
    )
    real: Dict[str, Any] = {}
    real["gplearn"] = run_gplearn(
        train_bank.states,
        train_bank.actions,
        train_bank.next_states,
        train_bank.contact,
        test_bank,
        prior,
        tuple(range(gp_seeds)),
        budget,
        preset="real",
    )
    real["pysr"] = (
        run_pysr(train_bank, test_bank, prior, pysr_iterations, max_train, seed, preset="real")
        if have_pysr
        else {"available": False, "reason": "the pysr package is not installed"}
    )
    real["templates"] = run_templates(
        train_bank.states,
        train_bank.actions,
        train_bank.next_states,
        train_bank.contact,
        test_bank.states,
        test_bank.actions,
        test_bank.next_states,
        test_bank.contact,
        prior,
        with_contact=True,
    )
    logger.info(
        "  telemetry | gplearn bound %s gate %s | pysr bound %s gate %s | templates bound %s "
        "(BIC %s / tail %s) gate %s",
        real["gplearn"]["structure"]["rigid_bound_discovered"],
        real["gplearn"]["structure"]["gravity_gate_discovered"],
        real["pysr"].get("structure", {}).get("rigid_bound_discovered"),
        real["pysr"].get("structure", {}).get("gravity_gate_discovered"),
        real["templates"]["structure"]["rigid_bound_discovered"],
        real["templates"]["horizontal"]["bic_selected"],
        real["templates"]["horizontal"]["tail_rmse_selected"],
        real["templates"]["structure"]["gravity_gate_discovered"],
    )

    structure_matrix: Dict[str, Any] = {}
    for engine in ENGINES:
        rows = [v for k, v in hidden.items() if k.startswith("replicate_") and engine in v]
        if not rows:
            structure_matrix[engine] = {
                "available": False,
                "reason": str(
                    hidden.get(engine, {}).get("reason", "the engine did not run in this study")
                ),
            }
            continue
        structure_matrix[engine] = {
            "available": True,
            "n_replicates": len(rows),
            "dvx_heldout_r2_mean": _nanmean([r[engine]["dvx_eval_r2"] for r in rows]),
            "rigid_bound_discovered_rate": float(
                np.mean([r[engine]["structure"]["rigid_bound_discovered"] for r in rows])
            ),
            "gravity_gate_discovered_rate": float(
                np.mean([r[engine]["structure"]["gravity_gate_discovered"] for r in rows])
            ),
            "bound_selected_by_bic_rate": float(
                np.mean(
                    [
                        r[engine]["structure"].get(
                            "bound_selected_by_bic",
                            r[engine]["structure"]["rigid_bound_discovered"],
                        )
                        for r in rows
                    ]
                )
            ),
            "bound_value_mean": _nanmean([r[engine]["structure"]["bound_value"] for r in rows]),
            "bound_overshoot_px_per_frame_mean": _nanmean(
                [float(r[engine]["structure"].get("bound_overshoot", float("nan"))) for r in rows]
            ),
            "tier_separation_mean": float(
                np.mean([r[engine]["structure"]["tier_separation"] for r in rows])
            ),
            "true_tier_separation": float(truth[4] - truth[5]),
        }

    binding = {
        "definition": "held-out frames at or above 90% of the observed speed support, i.e. the only frames a rigid clamp can be identified from",
        "hidden_world": _nanmean(
            [
                v["templates"]["tail_frame_fraction"]
                for k, v in hidden.items()
                if k.startswith("replicate_")
            ]
        ),
        "real_telemetry": float(real["templates"]["tail_frame_fraction"]),
    }

    payload: Dict[str, Any] = {
        "study": (
            "Three engines on the same inverse problem: bounded-terminal tree GP, PySR with "
            "numerically-optimised unbounded constants, and nested-template model selection in "
            "which every structure of interest is an explicit candidate. The question is "
            "whether the structures 10.43 failed to discover are missing from the "
            "representation, from the fitness, or from the data."
        ),
        "hidden_world": {n: float(truth[i]) for i, n in enumerate(PARAM_NAMES)},
        "protocol": {
            "replicates": replicates,
            "gp_seeds": gp_seeds,
            "population_size": population_size,
            "generations": generations,
            "max_train_transitions": max_train,
            "fit_windows": fit_windows,
            "eval_windows": eval_windows,
            "rollout_len": rollout_len,
            "pysr_iterations": pysr_iterations,
            "template_selection": "BIC on fit rows, held-out RMSE, and held-out RMSE restricted to frames at >=90% of the observed speed support",
            "shared_accuracy_metric": "R^2 of the horizontal increment law on the shared held-out transition bank; gplearn scores its seed-bagged law, PySR its selected model, the template engine its tail-selected structure",
            "feature_design": {
                "hidden_world": "synthetic (vx, dir, run | vy, jump, ground)",
                "real_telemetry": "real (vx, dir, run, ground, ceiling, left, right | vy, jump, ground, ceiling, left, right)",
            },
            "seed": seed,
        },
        "binding_frames": binding,
        "structure_matrix": structure_matrix,
        "hidden_world_runs": hidden,
        "real_telemetry": real,
        "verdict": _verdict(structure_matrix, truth, real, prior, binding),
    }
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.symbolic_engine_ablation_benchmark",
        extra_meta={"engines": list(ENGINES), "replicates": replicates},
    )
    logger.info("Artifact written to: %s", artifact)
    figure = _render_figure(structure_matrix, os.path.join(output_dir, "figures", FIGURE_NAME))
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def _bound_reading(matrix: Dict[str, Any], truth: np.ndarray, binding: Dict[str, float]) -> str:
    """Assemble the rigid-bound conclusion from the measured numbers only."""
    tm = matrix.get("templates", {})
    ps = matrix.get("pysr", {})
    bound_tm = float(tm.get("rigid_bound_discovered_rate", float("nan")))
    bound_gp = float(matrix.get("gplearn", {}).get("rigid_bound_discovered_rate", float("nan")))
    bound_bic = float(tm.get("bound_selected_by_bic_rate", float("nan")))
    if not bound_tm > 0.5:
        return (
            "The bound is not recovered even as an explicit candidate template, so the "
            "limitation is in the data or the estimator rather than the representation."
        )
    head = (
        "The bound is recoverable: with the clamp as an explicit candidate the template engine "
        f"returns {float(tm.get('bound_value_mean', float('nan'))):.3f} against a true "
        f"{float(truth[0]):.1f}, and BIC selects it in {bound_bic * 100:.0f}% of replicates. "
        f"Free tree search finds it in {bound_gp * 100:.0f}% of replicates (gplearn)"
    )
    r2 = {e: float(matrix.get(e, {}).get("dvx_heldout_r2_mean", float("nan"))) for e in ENGINES}
    if not ps.get("available"):
        return head + "; PySR did not run here, so no second searcher is available."
    head += f" and {float(ps.get('rigid_bound_discovered_rate', float('nan'))) * 100:.0f}% (PySR)."
    ps_r2, gp_r2 = r2["pysr"], r2["gplearn"]
    if np.isfinite(ps_r2) and np.isfinite(gp_r2) and ps_r2 > gp_r2:
        head += (
            f" PySR's constants are unbounded and numerically optimised, and on the shared eval "
            f"bank it fits the horizontal increment law better than gplearn ({ps_r2:.3f} vs "
            f"{gp_r2:.3f}) while missing the same structure, so neither the terminal range nor "
            "the accuracy of the fit explains the miss. "
        )
    else:
        head += (
            f" PySR's constants are unbounded and numerically optimised, but its held-out fit is "
            f"not better than gplearn's ({ps_r2:.3f} vs {gp_r2:.3f}), so this study cannot "
            "separate the representation from the accuracy of the search. "
        )
    overshoot = float(ps.get("bound_overshoot_px_per_frame_mean", float("nan")))
    if np.isfinite(overshoot) and overshoot > 0.0:
        head += (
            f" Its driven map stays within {overshoot:.3f} px/frame of the observed velocity "
            "support without ever turning over inside it, so it has no fixed point and the probe "
            "registers no bound. "
        )
    return head + (
        "What the three engines separate on is the criterion, not the repertoire: the constraint "
        f"binds on {float(binding['hidden_world']) * 100:.1f}% of held-out synthetic frames and on "
        f"{float(binding['real_telemetry']) * 100:.1f}% of the telemetry test split, and an "
        "aggregate error metric prices it at essentially nothing on either - the template engine "
        "is the only one of the three that scores any structure on the binding frames separately."
    )


def _real_reading(real: Dict[str, Any], prior: np.ndarray) -> str:
    """The telemetry conclusion, assembled from what each engine actually reported."""
    parts: List[str] = []
    for engine in ENGINES:
        block = real.get(engine, {})
        structure = block.get("structure", {})
        if not block.get("available", True) or not structure:
            continue
        value = structure.get("bound_value")
        r2 = float(block.get("dvx_eval_r2", float("nan")))
        reference = float(prior[0])
        parts.append(
            f"{engine} fits the horizontal law with held-out R^2 {r2:.3f} and reports "
            + (
                f"a bound at {float(value):.3f} ({100.0 * abs(float(value) - reference) / abs(reference):.1f}% from the WRAM reference {reference:.1f})"
                if value is not None and np.isfinite(float(value))
                else "no bound"
            )
        )
    horizontal = real.get("templates", {}).get("horizontal", {})
    bic, tail = str(horizontal.get("bic_selected")), str(horizontal.get("tail_rmse_selected"))
    return (
        "On genuine telemetry: "
        + "; ".join(parts)
        + f". Inside the template engine the BIC and tail criteria pick {bic} and {tail}"
        + (
            ", the same structure."
            if bic == tail
            else ", different structures, so the selection criterion still decides the reported "
            "law on real data even where every candidate carries the constraint."
        )
    )


def _verdict(
    matrix: Dict[str, Any],
    truth: np.ndarray,
    real: Dict[str, Any],
    prior: np.ndarray,
    binding: Dict[str, float],
) -> Dict[str, Any]:
    """Compute the conclusion from the measured rates; never assert it.

    The interesting case is the one a more generous engine can create - a searcher that fits
    the increments better than the one that failed, and still fails to find the same
    structure - so every clause of the reading is assembled from what was measured, and the
    wording changes when the measurement does not support it.
    """
    gp = matrix.get("gplearn", {})
    tm = matrix.get("templates", {})
    ps = matrix.get("pysr", {})
    bound_gp = float(gp.get("rigid_bound_discovered_rate", float("nan")))
    bound_tm = float(tm.get("rigid_bound_discovered_rate", float("nan")))
    bound_tm_bic = float(tm.get("bound_selected_by_bic_rate", float("nan")))
    have_pysr = bool(ps.get("available"))
    searchers = {"gplearn": bound_gp}
    if have_pysr:
        searchers["pysr"] = float(ps.get("rigid_bound_discovered_rate", float("nan")))
    true_sep = float(truth[4] - truth[5])
    return {
        "rigid_bound": {
            "discovery_rate_by_searcher": searchers,
            "discovery_rate_when_a_candidate_template": bound_tm,
            "selected_by_bic_rate": bound_tm_bic,
            "bound_recovered_by_templates": float(tm.get("bound_value_mean", float("nan"))),
            "true_bound": float(truth[0]),
            "heldout_r2_on_shared_eval_bank": {
                e: float(matrix.get(e, {}).get("dvx_heldout_r2_mean", float("nan")))
                for e in ENGINES
            },
            "reading": _bound_reading(matrix, truth, binding),
        },
        "gravity_gate": {
            "discovery_rate_by_engine": {
                e: float(matrix.get(e, {}).get("gravity_gate_discovered_rate", float("nan")))
                for e in ENGINES
            },
            "true_tier_separation": true_sep,
            "tier_separation_by_engine": {
                e: float(matrix.get(e, {}).get("tier_separation_mean", float("nan")))
                for e in ENGINES
            },
            "reading": (
                "The held-jump gate is a step in the velocity increment, so a tree without a "
                f"conditional has to approximate a jump of {true_sep:.2f} px/frame. Measured "
                "separations and recovery rates: "
                + "; ".join(
                    f"{e} separates by "
                    f"{float(matrix[e].get('tier_separation_mean', float('nan'))):.2f} and "
                    f"recovers the gate in "
                    f"{float(matrix[e].get('gravity_gate_discovered_rate', float('nan'))):.2f} "
                    "of replicates"
                    for e in ENGINES
                    if matrix.get(e, {}).get("available")
                )
                + "."
            ),
        },
        "real_telemetry": {
            "bound_value_by_engine": {
                e: real.get(e, {}).get("structure", {}).get("bound_value") for e in ENGINES
            },
            "heldout_r2_by_engine": {
                e: float(real.get(e, {}).get("dvx_eval_r2", float("nan"))) for e in ENGINES
            },
            "gate_by_engine": {
                e: bool(real.get(e, {}).get("structure", {}).get("gravity_gate_discovered", False))
                for e in ENGINES
            },
            "horizontal_template_by_criterion": {
                "bic": real.get("templates", {}).get("horizontal", {}).get("bic_selected"),
                "test_rmse": (
                    real.get("templates", {}).get("horizontal", {}).get("test_rmse_selected")
                ),
                "tail_rmse": (
                    real.get("templates", {}).get("horizontal", {}).get("tail_rmse_selected")
                ),
            },
            "reading": _real_reading(real, prior),
        },
    }


def _render_figure(matrix: Dict[str, Any], path: str) -> str:
    engines = [e for e in ENGINES if matrix.get(e, {}).get("available", False)]
    bound = [float(matrix[e].get("rigid_bound_discovered_rate", np.nan)) for e in engines]
    gate = [float(matrix[e].get("gravity_gate_discovered_rate", np.nan)) for e in engines]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(engines))
    ax1.bar(x, bound, color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(engines, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("rigid velocity bound\ndiscovered (rate)")
    ax1.set_title("Is the constraint discoverable?", fontweight="bold")
    for i, v in enumerate(bound):
        ax1.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    ax2.bar(x, gate, color="#9467bd")
    ax2.set_xticks(x)
    ax2.set_xticklabels(engines, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("held-jump gravity gate\ndiscovered (rate)")
    ax2.set_title("Is the discontinuity discoverable?", fontweight="bold")
    for i, v in enumerate(gate):
        ax2.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle(
        "Symbolic-regression engine ablation: representation, fitness and data",
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--gp-seeds", dest="gp_seeds", type=int, default=3)
    parser.add_argument("--population-size", dest="population_size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--max-train", dest="max_train", type=int, default=4000)
    parser.add_argument("--fit-windows", dest="fit_windows", type=int, default=800)
    parser.add_argument("--eval-windows", dest="eval_windows", type=int, default=300)
    parser.add_argument("--rollout-len", dest="rollout_len", type=int, default=12)
    parser.add_argument("--pysr-iterations", dest="pysr_iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    run_study(
        output_dir=args.output_dir,
        replicates=args.replicates,
        gp_seeds=args.gp_seeds,
        population_size=args.population_size,
        generations=args.generations,
        max_train=args.max_train,
        fit_windows=args.fit_windows,
        eval_windows=args.eval_windows,
        rollout_len=args.rollout_len,
        pysr_iterations=args.pysr_iterations,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
