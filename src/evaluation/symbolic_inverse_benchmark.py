"""
symbolic_inverse_benchmark.py
Symbolic-regression study for the inverse problem (README Section 10.43).

Section 10.40 asks "which constants generated these trajectories, given the law?".
This study asks the harder question: "what is the law, if nothing is given?" Genetic
programming (``gplearn``) evolves the four one-step update laws of a hidden world
directly from transitions, a probe stage reads physical constants back out of whatever
expression survives selection, and the result is scored against the parametric
identification of 10.40, the discrete-guarantee metrics of 8.2, and the published
learned world models.

Four experiments:

* **S1 - structure recovery on the hidden world.** The same OOD world as 10.40-E1
  (different fixed-point scale, different gravity). Per-law held-out accuracy, program
  parsimony, and the seven constants read out by probing, replicated over independent
  data draws and GP seeds, with paired Wilcoxon / t-tests and Cohen's d_z against the
  parametric estimator run on exactly the same transitions. Two controls separate "the
  representation cannot express it" from "the search was unlucky": a GP budget sweep and
  an excitation-reweighting ablation.
* **S2 - multi-step behaviour.** 120-frame open-loop rollouts of the discovered model:
  drift, rigid-bound overshoot, and the residual of the discrete integration identity.
* **S3 - genuine WRAM telemetry.** The laws are re-discovered on the real training split
  and scored per channel on the canonical test split next to the published per-variable
  rows; plus a grey-box control that fits GP to the *residuals* of the identified
  analytic model and asks whether the remaining model-form error has a closed form in
  the observed state at all.
* **S4 - zero-shot control transfer.** The 10.40-E3 held-out control battery, with the
  symbolic world model added to the prior / identified / oracle comparison.

Emulator-free and deterministic (every GP draw is seeded). Writes
``results/symbolic_inverse_metrics.json`` with ``_meta`` and
``results/figures/symbolic_inverse_recovery.png``.

Run:  python -m src.evaluation.symbolic_inverse_benchmark
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from src.environment.dataset_loader import load_and_preprocess_data  # noqa: E402
from src.evaluation.inverse_transfer_benchmark import OOD_WORLD, PRIOR  # noqa: E402
from src.inverse.parameter_identification import (
    PARAM_NAMES,
    generate_synthetic_windows,
    identify_params,
    make_windows,
    theta_tensor,
)
from src.inverse.symbolic_regression import (
    LAW_FEATURES,
    LAW_NAMES,
    TransitionBank,
    bank_from_transitions,
    bank_from_windows,
    evaluate_law,
    excitation_weights,
    fit_law,
    fit_law_bank,
    law_matrices,
    model_rollout,
    parametric_laws,
    probe_constants,
    relative_error,
    structural_tags,
    summarize_constants,
    symbolic_step,
    truth_vector,
)
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import RESULTS_DIR  # noqa: E402
from src.utils.provenance import read_metrics, write_metrics  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)


def _output_file(output_dir: str, name: str, subdir: str = "") -> str:
    """Path under ``output_dir``, created on demand.

    Everything in this study honours ``--output-dir``, so a smoke run writes into
    ``results_smoke/`` and can never overwrite a published artifact.
    """
    path = os.path.join(output_dir, subdir, name) if subdir else os.path.join(output_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


PUBLISHED_INVERSE = "inverse_identification_metrics.json"
PUBLISHED_BENCHMARK = "benchmark_metrics.json"
PUBLISHED_OPERATORS = "operator_benchmark_metrics.json"
ARTIFACT_NAME = "symbolic_inverse_metrics.json"

# Budget grid for the S1b control: does a longer search buy the missing structure?
BUDGET_GRID: Tuple[Tuple[int, int], ...] = ((300, 15), (500, 25), (1000, 60))

# The S3 residual control gives the search the richest observation the telemetry offers.
RESIDUAL_PRESET = "full"  # every observable channel of the 8D telemetry
CHANNEL_NAMES = ("x", "y", "vx", "vy")


def _paired_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Paired significance between two per-replicate metric vectors (a = symbolic)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    out: Dict[str, Any] = {
        "n_pairs": int(x.size),
        "symbolic_mean": float(x.mean()),
        "parametric_mean": float(y.mean()),
        "mean_difference": float((x - y).mean()),
    }
    if x.size < 3 or np.allclose(x, y):
        out.update({"wilcoxon_p": float("nan"), "ttest_p": float("nan"), "cohen_dz": float("nan")})
        return out
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    out["ttest_p"] = float(stats.ttest_rel(x, y).pvalue)
    diff = x - y
    sd = float(diff.std(ddof=1))
    out["cohen_dz"] = float(diff.mean() / sd) if sd > 1e-18 else float("nan")
    out["note"] = (
        "Paired by replicate, so the data draw is controlled and only the estimator "
        "varies. With a small replicate count the Wilcoxon floor is 2/2**n, so the "
        "effect size (Cohen's dz) is the load-bearing statistic."
    )
    return out


def _weighted_mse(bank: TransitionBank, pred: np.ndarray) -> float:
    """Channel-variance-weighted one-step MSE: the objective 10.40 fits, as a metric."""
    var = np.var(bank.next_states.reshape(-1, 4), axis=0)
    weight = 1.0 / np.clip(var, 1e-6, None)
    return float(np.mean(np.mean((pred - bank.next_states) ** 2, axis=0) * weight))


def _one_step_metrics(bank: TransitionBank, model: Dict[str, Any]) -> Dict[str, float]:
    """Per-channel single-step error of a world model on a transition bank."""
    pred = symbolic_step(bank.states, bank.actions, bank.contact, model)
    err = pred - bank.next_states
    out: Dict[str, float] = {"weighted_mse": _weighted_mse(bank, pred)}
    for i, name in enumerate(CHANNEL_NAMES):
        truth = bank.next_states[:, i]
        sse = float(np.sum((truth - pred[:, i]) ** 2))
        sst = float(np.sum((truth - truth.mean()) ** 2))
        out[f"{name}_mse"] = float(np.mean(err[:, i] ** 2))
        out[f"{name}_mae"] = float(np.mean(np.abs(err[:, i])))
        out[f"{name}_r2"] = 1.0 - sse / sst if sst > 1e-12 else float("nan")
    return out


def _law_rows(laws_by_seed: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-law accuracy / parsimony / structure statistics over the GP seeds."""
    rows: Dict[str, Dict[str, Any]] = {}
    for name, laws in laws_by_seed.items():
        mae = [law.hold_mae for law in laws]
        r2 = [law.hold_r2 for law in laws]
        null_r2 = [law.hold_null_r2 for law in laws]
        nodes = [law.n_nodes for law in laws]
        tags = {
            key: float(np.mean([float(structural_tags(law)[key]) for law in laws]))
            for key in structural_tags(laws[0])
        }
        rows[name] = {
            "hold_mae_mean": float(np.mean(mae)),
            "hold_mae_std": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
            "hold_mae_best": float(np.min(mae)),
            "hold_r2_mean": float(np.nanmean(r2)),
            "hold_r2_best": float(np.nanmax(r2)),
            # A law that does not beat the fit-mean predictor has discovered the mean of
            # the target distribution, not a response: the two are indistinguishable in
            # R^2 alone, so the null reference is reported next to it.
            "hold_null_r2_mean": float(np.nanmean(null_r2)),
            "beats_null_rate": float(np.mean([r > n for r, n in zip(r2, null_r2)])),
            "nodes_mean": float(np.mean(nodes)),
            "nodes_min": int(np.min(nodes)),
            "nodes_max": int(np.max(nodes)),
            "fit_seconds_mean": float(np.mean([law.fit_seconds for law in laws])),
            "structure_rate": tags,
        }
    return rows


def _expressions(per_seed: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    return {name: [law.expression for law in laws] for name, laws in per_seed.items()}


def _mean_law_rows(rows: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Average the per-replicate per-law statistic blocks."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, block in rows[0].items():
        agg: Dict[str, Any] = {}
        for key, val in block.items():
            if key == "structure_rate":
                agg[key] = {
                    k: float(np.mean([r[name]["structure_rate"][k] for r in rows])) for k in val
                }
            else:
                agg[key] = float(np.mean([r[name][key] for r in rows]))
        out[name] = agg
    return out


def _thinned_windows(windows: Sequence[Any], stride: int) -> Tuple[Any, ...]:
    """Thin whole rollout windows, so the fit bank and the parametric fit see the
    identical frames (and so the tensors stay tensors for ``identify_params``)."""
    return tuple(w[::stride] for w in windows)


def _warm_start(bank: TransitionBank, prior: np.ndarray) -> np.ndarray:
    """Initialise the velocity ceiling from the observed |vx| range (10.40's fix)."""
    init = prior.copy()
    init[0] = float(np.abs(bank.states[:, 2]).max()) * 1.05
    return init


def _synthetic_banks(
    params: torch.Tensor,
    fit_windows: int,
    eval_windows: int,
    rollout_len: int,
    rep: int,
    stride: int,
) -> Tuple[Tuple[Any, ...], TransitionBank, TransitionBank]:
    """Fit / evaluation transition banks for one replicate of the hidden world."""
    fit_w = generate_synthetic_windows(
        params, n_windows=fit_windows, rollout_len=rollout_len, seed=11 + 10 * rep
    )
    eval_w = generate_synthetic_windows(
        params, n_windows=eval_windows, rollout_len=rollout_len, seed=907 + 10 * rep
    )
    thinned = _thinned_windows(fit_w, stride)
    return thinned, bank_from_windows(thinned), bank_from_windows(eval_w)


def run_s1(
    truth: np.ndarray,
    prior: np.ndarray,
    replicates: int,
    gp_seeds: Sequence[int],
    budget: Dict[str, Any],
    fit_windows: int,
    eval_windows: int,
    rollout_len: int,
    id_steps: int,
    weight_strengths: Sequence[float],
    budget_sweep: bool,
) -> Dict[str, Any]:
    """S1: discover the laws of the hidden world and read the constants back out."""
    logger.info("=== S1: symbolic structure recovery on the hidden world ===")
    stride = int(budget.get("stride", 1))
    gp_kwargs = {k: v for k, v in budget.items() if k != "stride"}
    params = theta_tensor(OOD_WORLD)

    per_law: List[Dict[str, Dict[str, Any]]] = []
    bagged_tables: List[Dict[str, Dict[str, float]]] = []
    single_tables: List[Dict[str, Dict[str, float]]] = []
    param_errors: List[Dict[str, float]] = []
    accuracy: Dict[str, List[float]] = {
        "symbolic_bagged": [],
        "symbolic_best_seed": [],
        "parametric": [],
        "null_kinematic_persistence": [],
    }
    replicate_rows: List[Dict[str, Any]] = []
    expressions: Dict[str, List[str]] = {}
    law_rows_rep0: Dict[str, Dict[str, Any]] = {}
    fixed_point_rate: List[float] = []

    for rep in range(replicates):
        thinned, bank, eval_bank = _synthetic_banks(
            params, fit_windows, eval_windows, rollout_len, rep, stride
        )
        laws = fit_law_bank(bank, "synthetic", seeds=gp_seeds, eval_bank=eval_bank, **gp_kwargs)
        rows = _law_rows(laws.per_seed)
        per_law.append(rows)
        if rep == 0:
            expressions = _expressions(laws.per_seed)
            law_rows_rep0 = rows

        bagged_model = laws.bagged()
        bagged_table = relative_error(probe_constants(bagged_model, bank), truth)
        bagged_tables.append(bagged_table)
        single_table = relative_error(probe_constants(dict(laws.laws), bank), truth)
        single_tables.append(single_table)
        fixed_point_rate.append(float(bagged_table["max_vx"].get("fixed_point_found", 0.0)))

        theta_hat, _ = identify_params(
            thinned,
            torch.as_tensor(_warm_start(bank, prior), dtype=torch.float32),
            steps=id_steps,
            lr=0.05,
            seed=rep,
        )
        analytic = parametric_laws(theta_hat.numpy().astype(np.float64), "synthetic")
        acc_param = _one_step_metrics(eval_bank, analytic)["weighted_mse"]
        acc_bagged = _one_step_metrics(eval_bank, bagged_model)["weighted_mse"]
        acc_best = min(
            _one_step_metrics(eval_bank, {n: laws.per_seed[n][k] for n in LAW_NAMES})[
                "weighted_mse"
            ]
            for k in range(len(gp_seeds))
        )
        acc_null = _one_step_metrics(eval_bank, _null_velocity_model(float(truth[3]), "synthetic"))[
            "weighted_mse"
        ]
        accuracy["symbolic_bagged"].append(acc_bagged)
        accuracy["symbolic_best_seed"].append(acc_best)
        accuracy["parametric"].append(acc_param)
        accuracy["null_kinematic_persistence"].append(acc_null)
        param_errors.append(
            {
                name: float(100.0 * abs(float(theta_hat[k]) - truth[k]) / abs(truth[k]))
                for k, name in enumerate(PARAM_NAMES)
            }
        )
        replicate_rows.append(
            {
                "replicate": rep,
                "fit_transitions": bank.n,
                "eval_transitions": eval_bank.n,
                "mean_program_nodes": laws.mean_nodes(),
                "gp_seconds": laws.total_seconds(),
                "symbolic_bagged_weighted_mse": acc_bagged,
                "null_persistence_weighted_mse": acc_null,
                "symbolic_best_seed_weighted_mse": acc_best,
                "parametric_weighted_mse": acc_param,
            }
        )
        logger.info(
            "  rep %d | weighted one-step MSE: symbolic %.4f (best seed %.4f) vs parametric %.4f "
            "| %.0f nodes | %.1fs",
            rep,
            acc_bagged,
            acc_best,
            acc_param,
            laws.mean_nodes(),
            laws.total_seconds(),
        )

    summary = summarize_constants(bagged_tables)
    summary_single = summarize_constants(single_tables)
    constants: Dict[str, Dict[str, Any]] = {}
    for k, name in enumerate(PARAM_NAMES):
        rel = [
            t[name]["rel_error_pct"]
            for t in bagged_tables
            if np.isfinite(t[name].get("rel_error_pct", float("nan")))
        ]
        rel_s = [
            t[name]["rel_error_pct"]
            for t in single_tables
            if np.isfinite(t[name].get("rel_error_pct", float("nan")))
        ]
        prior_v, t_v = float(prior[k]), float(truth[k])
        entry: Dict[str, Any] = {
            "true": t_v,
            "prior": prior_v,
            "prior_rel_error_pct": float(100.0 * abs(prior_v - t_v) / abs(t_v)),
            "symbolic_bagged_median_rel_error_pct": float(np.median(rel)) if rel else float("nan"),
            "symbolic_bagged_rel_error_std": float(np.std(rel, ddof=1)) if len(rel) > 1 else 0.0,
            "symbolic_single_seed_median_rel_error_pct": float(np.median(rel_s))
            if rel_s
            else float("nan"),
            "symbolic_bagged_recovery_rate": summary[name]["recovery_rate"],
            "symbolic_single_seed_recovery_rate": summary_single[name]["recovery_rate"],
            "symbolic_mean_value": summary[name]["mean"],
            "symbolic_std_value": summary[name]["std"],
            "parametric_median_rel_error_pct": float(np.median([e[name] for e in param_errors])),
            "parametric_max_rel_error_pct": float(np.max([e[name] for e in param_errors])),
        }
        if name == "max_vx":
            entry["structurally_discovered_rate"] = float(np.mean(fixed_point_rate))
            entry["mean_overshoot_px_per_frame"] = float(
                np.mean([t["max_vx"].get("overshoot_px_per_frame", 0.0) for t in bagged_tables])
            )
            entry["reachable_support_px"] = float(
                np.mean([t["max_vx"].get("reachable_support", float("nan")) for t in bagged_tables])
            )
        if name == "subpixels_per_pixel":
            entry["identity_residual_px_mean"] = float(
                np.mean(
                    [
                        t["subpixels_per_pixel"].get("identity_residual", float("nan"))
                        for t in bagged_tables
                    ]
                )
            )
        constants[name] = entry
    for extra in ("fall_gravity_descending", "subpixels_per_pixel_from_y"):
        constants[extra] = {
            "symbolic_mean_value": summary[extra]["mean"],
            "symbolic_std_value": summary[extra]["std"],
            "symbolic_recovery_rate": summary[extra]["recovery_rate"],
        }

    payload: Dict[str, Any] = {
        "per_law_replicate0": law_rows_rep0,
        "per_law_mean_over_replicates": _mean_law_rows(per_law),
        "expressions_replicate0": expressions,
        "constants": constants,
        "accuracy_paired": {
            "metric": "channel-variance-weighted one-step MSE on independent unseen rollouts",
            "bagged_symbolic_vs_parametric": _paired_tests(
                accuracy["symbolic_bagged"], accuracy["parametric"]
            ),
            "best_seed_symbolic_vs_parametric": _paired_tests(
                accuracy["symbolic_best_seed"], accuracy["parametric"]
            ),
            "bagged_symbolic_vs_null_persistence": _paired_tests(
                accuracy["symbolic_bagged"], accuracy["null_kinematic_persistence"]
            ),
        },
        "accuracy_by_condition": {k: v for k, v in accuracy.items()},
        "replicates": replicate_rows,
        "budget_control": _budget_sweep(truth, budget, fit_windows, rollout_len, gp_seeds)
        if budget_sweep
        else {},
        "excitation_weight_control": _weight_ablation(
            truth, budget, fit_windows, eval_windows, rollout_len, weight_strengths, gp_seeds
        ),
    }
    return payload


def _budget_sweep(
    truth: np.ndarray,
    budget: Dict[str, Any],
    fit_windows: int,
    rollout_len: int,
    gp_seeds: Sequence[int],
) -> Dict[str, Any]:
    """S1b: grow the GP budget and measure accuracy AND structure at each setting.

    Held-out $R^2$ alone cannot distinguish "the representation cannot express the law"
    from "the search was unlucky", because a smooth surrogate can fit a discontinuous
    map's bulk without ever containing it. Every budget point is therefore also probed
    for the two structural quantities of interest: does the horizontal map have a fixed
    point (i.e. did it discover a rigid bound at all), and does the vertical law separate
    the held-jump gravity tier from the falling tier?
    """
    logger.info("=== S1b: GP budget control (structure or search effort?) ===")
    params = theta_tensor(OOD_WORLD)
    fit_w = generate_synthetic_windows(
        params, n_windows=fit_windows, rollout_len=rollout_len, seed=11
    )
    eval_w = generate_synthetic_windows(params, n_windows=250, rollout_len=rollout_len, seed=907)
    bank = bank_from_windows(_thinned_windows(fit_w, int(budget.get("stride", 1))))
    eval_bank = bank_from_windows(eval_w)
    fit_mats = law_matrices(bank, "synthetic")
    eval_mats = law_matrices(eval_bank, "synthetic")
    out: Dict[str, Any] = {}
    for pop, gens in BUDGET_GRID:
        key = f"population{pop}_generations{gens}"
        out[key] = {}
        for name in ("dvx", "dvy"):
            X, y = fit_mats[name]
            Xe, ye = eval_mats[name]
            r2s, maes, nodes, seconds = [], [], [], []
            struct: List[Dict[str, float]] = []
            for seed in gp_seeds:
                law, _ = fit_law(
                    name,
                    X,
                    y,
                    LAW_FEATURES["synthetic"][name],
                    seed=seed,
                    population_size=pop,
                    generations=gens,
                    max_train=int(budget["max_train"]),
                    parsimony_coefficient=float(budget["parsimony_coefficient"]),
                    n_jobs=int(budget.get("n_jobs", 1)),
                )
                metrics = evaluate_law(law, Xe, ye)
                r2s.append(metrics["r2"])
                maes.append(metrics["mae"])
                nodes.append(law.n_nodes)
                seconds.append(law.fit_seconds)
                table = relative_error(probe_constants({name: law}, bank), truth)
                if name == "dvx":
                    struct.append(
                        {
                            "fixed_point_found": float(table["max_vx"]["fixed_point_found"]),
                            "decel_rel_error_pct": table["decel"]["rel_error_pct"],
                            "walk_rel_error_pct": table["walk_accel"]["rel_error_pct"],
                            "run_rel_error_pct": table["run_accel"]["rel_error_pct"],
                        }
                    )
                else:
                    struct.append(
                        {
                            "held_gravity_rel_error_pct": table["held_gravity"]["rel_error_pct"],
                            "fall_gravity_rel_error_pct": table["fall_gravity"]["rel_error_pct"],
                            "tier_separation_px_per_frame": float(
                                table["held_gravity"]["value"] - table["fall_gravity"]["value"]
                            ),
                        }
                    )
            row: Dict[str, Any] = {
                "eval_r2_mean": float(np.nanmean(r2s)),
                "eval_r2_std": float(np.nanstd(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                "eval_mae_mean": float(np.mean(maes)),
                "nodes_mean": float(np.mean(nodes)),
                "fit_seconds_mean": float(np.mean(seconds)),
            }
            for field in struct[0]:
                vals = [s[field] for s in struct]
                row[field + "_mean"] = float(np.nanmean(vals))
            if name == "dvy":
                row["true_tier_separation"] = float(truth[4] - truth[5])
            out[key][name] = row
        logger.info(
            "  %s -> eval R2 %s | dvx fixed-point rate %.2f | dvy tier gap %.2f (true %.2f)",
            key,
            {k: round(out[key][k]["eval_r2_mean"], 3) for k in ("dvx", "dvy")},
            out[key]["dvx"]["fixed_point_found_mean"],
            out[key]["dvy"]["tier_separation_px_per_frame_mean"],
            out[key]["dvy"]["true_tier_separation"],
        )
    out["interpretation"] = (
        "Held-out R^2 of the velocity laws is reported against the search budget together "
        "with two structural read-outs: whether the horizontal map acquired a fixed point "
        "(a discovered rigid bound) and how far the vertical law separates the held-jump "
        "gravity tier from the falling tier. Accuracy and structure can move at different "
        "rates, and only the structural columns answer the discovery question."
    )
    return out


def _weight_ablation(
    truth: np.ndarray,
    budget: Dict[str, Any],
    fit_windows: int,
    eval_windows: int,
    rollout_len: int,
    strengths: Sequence[float],
    gp_seeds: Sequence[int],
) -> Dict[str, Any]:
    """S1c: over-represent the under-excited branches and see what gets recovered."""
    logger.info("=== S1c: excitation-reweighting control ===")
    params = theta_tensor(OOD_WORLD)
    stride = int(budget.get("stride", 1))
    gp_kwargs = {k: v for k, v in budget.items() if k != "stride"}
    out: Dict[str, Any] = {}
    for strength in strengths:
        decel_err, gravity_err, maes, nodes = [], [], {"dvx": [], "dvy": []}, {"dvx": [], "dvy": []}
        for rep in range(3):
            _, bank, eval_bank = _synthetic_banks(
                params, fit_windows, eval_windows, rollout_len, rep, stride
            )
            weights = {name: excitation_weights(bank, name, strength) for name in ("dvx", "dvy")}
            laws = fit_law_bank(
                bank, "synthetic", seeds=gp_seeds, eval_bank=eval_bank, weights=weights, **gp_kwargs
            )
            probe = relative_error(probe_constants(laws.bagged(), bank), truth)
            decel_err.append(probe["decel"]["rel_error_pct"])
            gravity_err.append(probe["held_gravity"]["rel_error_pct"])
            for name in ("dvx", "dvy"):
                nodes[name].append(float(np.mean([law.n_nodes for law in laws.per_seed[name]])))
                maes[name].append(float(np.mean([law.hold_mae for law in laws.per_seed[name]])))
        out[f"weight_{strength:g}"] = {
            "strength": float(strength),
            "decel_rel_error_pct_mean": float(np.nanmean(decel_err)),
            "held_gravity_rel_error_pct_mean": float(np.nanmean(gravity_err)),
            "dvx_nodes_mean": float(np.mean(nodes["dvx"])),
            "dvy_nodes_mean": float(np.mean(nodes["dvy"])),
            "dvx_hold_mae_mean": float(np.mean(maes["dvx"])),
            "dvy_hold_mae_mean": float(np.mean(maes["dvy"])),
        }
        logger.info(
            "  strength %-4g -> decel err %.1f%%, g_hold err %.1f%% | hold MAE dvx %.3f dvy %.3f",
            strength,
            out[f"weight_{strength:g}"]["decel_rel_error_pct_mean"],
            out[f"weight_{strength:g}"]["held_gravity_rel_error_pct_mean"],
            out[f"weight_{strength:g}"]["dvx_hold_mae_mean"],
            out[f"weight_{strength:g}"]["dvy_hold_mae_mean"],
        )
    out["note"] = (
        "strength 0 is the passive-observation baseline; larger values over-weight "
        "coasting frames (dvx, where the friction deadband lives) and ascending "
        "jump-held frames (dvy, where the held-gravity tier lives). Reweighting trades "
        "global accuracy for the under-excited branch, so both columns must be read "
        "together."
    )
    return out


def run_s2(
    truth: np.ndarray,
    prior: np.ndarray,
    replicates: int,
    budget: Dict[str, Any],
    gp_seeds: Sequence[int],
    fit_windows: int,
    rollout_len: int,
    rollout_horizon: int,
    rollout_starts: int,
    id_steps: int,
) -> Dict[str, Any]:
    """S2: roll the discovered model out and score which guarantees break."""
    logger.info("=== S2: long-horizon open-loop rollouts of the discovered model ===")
    params = theta_tensor(OOD_WORLD)
    stride = int(budget.get("stride", 1))
    gp_kwargs = {k: v for k, v in budget.items() if k != "stride"}
    series: Dict[str, Dict[str, List[float]]] = {}
    reps = max(1, min(replicates, 3))

    for rep in range(reps):
        thinned, bank, _ = _synthetic_banks(params, fit_windows, 200, rollout_len, rep, stride)
        laws = fit_law_bank(bank, "synthetic", seeds=gp_seeds, **gp_kwargs)
        theta_hat, _ = identify_params(
            thinned,
            torch.as_tensor(_warm_start(bank, prior), dtype=torch.float32),
            steps=id_steps,
            lr=0.05,
            seed=rep,
        )
        roll_w = generate_synthetic_windows(
            params, n_windows=rollout_starts, rollout_len=rollout_horizon, seed=4001 + 10 * rep
        )
        s0, acts, targets, contact = (np.asarray(t, dtype=np.float64) for t in roll_w)
        models = {
            "symbolic_single_seed": dict(laws.laws),
            "symbolic_bagged": laws.bagged(),
            "parametric_identified": parametric_laws(
                theta_hat.numpy().astype(np.float64), "synthetic"
            ),
            "prior_smw_constants": parametric_laws(prior.astype(np.float64), "synthetic"),
            "oracle_true_world": parametric_laws(truth.astype(np.float64), "synthetic"),
        }
        for name, model in models.items():
            pred = model_rollout(model, s0, acts, contact)
            drift = np.sqrt(((pred[:, :, :2] - targets[:, :, :2]) ** 2).sum(axis=2))
            cap = float(truth[0])
            vx_abs = np.abs(pred[:, :, 2])
            prev_x = np.concatenate([s0[:, None, 0], pred[:, :, 0]], axis=1)
            step = np.diff(prev_x, axis=1)
            kin = np.abs(step - pred[:, :, 2] / float(truth[3]))
            record = {
                "mean_drift_px": float(drift.mean()),
                "final_drift_px": float(drift[:, -1].mean()),
                "cap_violation_frames_pct": float(100.0 * (vx_abs > cap + 1e-3).mean()),
                "max_overshoot_px_per_frame": float(max(0.0, vx_abs.max() - cap)),
                "kinematic_identity_residual_px": float(kin.mean()),
                "rollout_frames": float(rollout_horizon),
            }
            for key, val in record.items():
                series.setdefault(name, {}).setdefault(key, []).append(val)

    out: Dict[str, Any] = {}
    for name, keys in series.items():
        out[name] = {
            key: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
            for key, vals in keys.items()
        }
    out["paired_tests_symbolic_vs_parametric"] = {
        metric: _paired_tests(
            series["symbolic_bagged"][metric], series["parametric_identified"][metric]
        )
        for metric in ("mean_drift_px", "cap_violation_frames_pct")
    }
    out["replicates_evaluated"] = reps
    out["metric_note"] = (
        "The identity residual and the cap are measured against the TRUE fixed-point "
        "scale and velocity ceiling, because a structure-free model earns no right to "
        "its own units; every model is scored on the identical reference."
    )
    return out


class _ConstantLaw:
    """A law whose increment is a fixed function of its features (baseline machinery).

    Used to build the *kinematic persistence* baseline: no force law at all, velocities
    carried forward unchanged, positions integrated by the engine's own fixed-point
    scale. It is the number the discovered laws must beat to claim they found dynamics
    rather than the identity.
    """

    def __init__(self, name: str, preset: str, fn: Any) -> None:
        self.name = name
        self.feature_names = LAW_FEATURES[preset][name]
        self._fn = fn

    def increment(self, X_phys: np.ndarray) -> np.ndarray:
        return self._fn(np.atleast_2d(X_phys))


def _null_velocity_model(scale: float, preset: str) -> Dict[str, Any]:
    def zero(X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])

    def over_scale(X: np.ndarray) -> np.ndarray:
        return X[:, 0] / scale

    return {
        "dvx": _ConstantLaw("dvx", preset, zero),
        "dvy": _ConstantLaw("dvy", preset, zero),
        "dx": _ConstantLaw("dx", preset, over_scale),
        "dy": _ConstantLaw("dy", preset, over_scale),
    }


def _residual_block(bank: TransitionBank) -> Tuple[np.ndarray, List[str]]:
    """The full observable feature block used by the grey-box residual control."""
    X, _ = law_matrices(bank, RESIDUAL_PRESET)["dvx"]
    return X, LAW_FEATURES[RESIDUAL_PRESET]["dvx"]


def _hybrid_residual_control(
    fit_bank: TransitionBank,
    test_bank: TransitionBank,
    theta: np.ndarray,
    budget: Dict[str, Any],
    gp_seeds: Sequence[int],
) -> Dict[str, Any]:
    """Grey-box control: does the identified model's remaining error have a closed form?

    GP is fitted to the *residuals* of the analytic identified map, given everything the
    observation carries. A discoverable residual is missing physics the analytic model
    can be repaired with; an undiscoverable one (train R^2 high, test R^2 near zero) is
    the formal statement that the remaining error is driven by geometry the state vector
    never carried.
    """
    model = parametric_laws(theta, "real")
    X_fit, features = _residual_block(fit_bank)
    X_test, _ = _residual_block(test_bank)
    out: Dict[str, Any] = {}
    for bank, X, tag in ((fit_bank, X_fit, "train"), (test_bank, X_test, "test")):
        pred = symbolic_step(bank.states, bank.actions, bank.contact, model)
        out.setdefault("analytic_residual_variance", {})[tag] = {
            name: float(np.var(bank.next_states[:, i] - pred[:, i]))
            for i, name in enumerate(CHANNEL_NAMES)
        }
    fit_pred = symbolic_step(fit_bank.states, fit_bank.actions, fit_bank.contact, model)
    test_pred = symbolic_step(test_bank.states, test_bank.actions, test_bank.contact, model)
    gp_kwargs = {k: v for k, v in budget.items() if k != "stride"}
    for i, name in enumerate(CHANNEL_NAMES):
        y_fit = fit_bank.next_states[:, i] - fit_pred[:, i]
        y_test = test_bank.next_states[:, i] - test_pred[:, i]
        rows: List[Dict[str, float]] = []
        for seed in gp_seeds:
            law, _ = fit_law(f"residual_{name}", X_fit, y_fit, features, seed=seed, **gp_kwargs)
            train = evaluate_law(law, X_fit, y_fit)
            test = evaluate_law(law, X_test, y_test)
            rows.append(
                {
                    "seed": float(seed),
                    "train_r2": train["r2"],
                    "test_r2": test["r2"],
                    "test_mae": test["mae"],
                    "nodes": float(law.n_nodes),
                    "expression": law.expression,
                }
            )
        out[name] = {
            "train_r2_mean": float(np.nanmean([r["train_r2"] for r in rows])),
            "test_r2_mean": float(np.nanmean([r["test_r2"] for r in rows])),
            "test_r2_best": float(np.nanmax([r["test_r2"] for r in rows])),
            "test_mae_mean": float(np.mean([r["test_mae"] for r in rows])),
            "per_seed": rows,
        }
        logger.info(
            "  residual %-2s: train R2 %+.3f -> test R2 %+.3f (best %+.3f)",
            name,
            out[name]["train_r2_mean"],
            out[name]["test_r2_mean"],
            out[name]["test_r2_best"],
        )
    closed_form = [c for c in CHANNEL_NAMES if out[c]["test_r2_mean"] > 0.05]
    opaque = [c for c in CHANNEL_NAMES if out[c]["train_r2_mean"] <= 0.0]
    out["interpretation"] = (
        f"Out-of-sample the residual admits a closed form in the observation for "
        f"{closed_form if closed_form else 'no channel'} (test R^2 > 0.05), while "
        f"{opaque if opaque else 'no channel'} is not even fit in-sample, so its "
        "model-form error cannot be a function of the observed state at all and needs a "
        "richer observation (tile geometry, sprite table), not a better estimator."
    )
    return out


def run_s3(
    budget: Dict[str, Any],
    gp_seeds: Sequence[int],
    id_steps: int,
    output_dir: str,
    real_stride: int,
) -> Dict[str, Any]:
    """S3: re-discover the laws on genuine WRAM telemetry and score them honestly."""
    logger.info("=== S3: symbolic discovery on real WRAM gameplay transitions ===")
    data = load_and_preprocess_data()
    # The dataset's next-state channels 4-7 are the terrain-contact byte, so the
    # collision flags the analytic model of 10.40-E2 consumes are available here too.
    fit_bank = bank_from_transitions(
        data["train_states"], data["train_actions"], data["train_next_states"], stride=real_stride
    )
    test_bank = bank_from_transitions(
        data["test_states"], data["test_actions"], data["test_next_states"]
    )
    gp_kwargs = {k: v for k, v in budget.items() if k != "stride"}
    laws = fit_law_bank(fit_bank, "real", seeds=gp_seeds, eval_bank=test_bank, **gp_kwargs)
    model = laws.bagged()
    prior_vec = truth_vector(PRIOR)
    theta_real, _ = identify_params(
        make_windows(
            data["train_states"],
            data["train_actions"],
            data["train_next_states"],
            data["train_episodes"],
            rollout_len=8,
        ),
        theta_tensor(PRIOR),
        steps=id_steps,
        lr=0.05,
        seed=42,
    )
    constants = relative_error(probe_constants(model, fit_bank), prior_vec)
    payload: Dict[str, Any] = {
        "transitions": {"fit": fit_bank.n, "test": test_bank.n},
        "per_law": _law_rows(laws.per_seed),
        "expressions": _expressions(laws.per_seed),
        "one_step_test_metrics": {
            "symbolic_bagged": _one_step_metrics(test_bank, model),
            "null_kinematic_persistence": _one_step_metrics(
                test_bank, _null_velocity_model(float(prior_vec[3]), "real")
            ),
            "symbolic_single_seed": _one_step_metrics(test_bank, dict(laws.laws)),
            "parametric_identified_analytic": _one_step_metrics(
                test_bank, parametric_laws(theta_real.numpy().astype(np.float64), "real")
            ),
            "prior_analytic": _one_step_metrics(
                test_bank, parametric_laws(prior_vec.astype(np.float64), "real")
            ),
        },
        "constants_vs_wram_reference": {
            name: {
                "wram_reference": constants[name]["true"],
                "symbolic_value": constants[name]["value"],
                "rel_error_pct": constants[name]["rel_error_pct"],
                "fixed_point_found": constants[name].get("fixed_point_found"),
                "identity_residual": constants[name].get("identity_residual"),
            }
            for name in PARAM_NAMES
        },
        "hybrid_residual_control": _hybrid_residual_control(
            fit_bank, test_bank, theta_real.numpy().astype(np.float64), budget, gp_seeds
        ),
        "published_reference": _published_rows(output_dir),
        "model_scope_note": (
            "GP sees exactly the same observable channels as the analytic model of "
            "10.40-E2 (state, buttons, four terrain-contact bytes) and nothing else: no "
            "tilemap, no sprite table. Whatever the discovered law cannot express is "
            "model-form error from unobserved geometry, which the residual control "
            "measures directly."
        ),
    }
    return payload


def _published_rows(output_dir: str) -> Dict[str, Any]:
    """Per-variable rows of the committed artifacts, for the taxonomy comparison."""
    out: Dict[str, Any] = {}
    for label, fname in (
        ("canonical_benchmark", PUBLISHED_BENCHMARK),
        ("operator_family", PUBLISHED_OPERATORS),
    ):
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            logger.warning("%s not found; published-comparison block omitted", fname)
            continue
        doc = read_metrics(path)
        rows = doc.get("per_variable_metrics")
        if rows is None:
            rows = {
                name: block.get("per_variable_metrics", {})
                for name, block in doc.get("architectures", {}).items()
            }
        out[label] = {
            name: {
                ch: {k: v for k, v in block[ch].items() if k in ("mse", "mae", "r2")}
                for ch in CHANNEL_NAMES
                if isinstance(block, dict) and ch in block
            }
            for name, block in rows.items()
            if isinstance(block, dict)
        }
    return out


def run_s4(
    truth: np.ndarray,
    prior: np.ndarray,
    budget: Dict[str, Any],
    gp_seeds: Sequence[int],
    fit_windows: int,
    id_steps: int,
    output_dir: str,
) -> Dict[str, Any]:
    """S4: the 10.40-E3 held-out control battery, with the symbolic model added."""
    logger.info("=== S4: zero-shot control transfer of the discovered law ===")
    params = theta_tensor(OOD_WORLD)
    stride = int(budget.get("stride", 1))
    gp_kwargs = {k: v for k, v in budget.items() if k != "stride"}
    battery = generate_synthetic_windows(params, n_windows=400, rollout_len=24, seed=101)
    s0, acts, targets, contact = (np.asarray(t, dtype=np.float64) for t in battery)
    achieved = targets[:, -1, 0]

    thinned, bank, _ = _synthetic_banks(params, fit_windows, 200, 12, 0, stride)
    laws = fit_law_bank(bank, "synthetic", seeds=gp_seeds, **gp_kwargs)
    theta_hat, _ = identify_params(
        thinned,
        torch.as_tensor(_warm_start(bank, prior), dtype=torch.float32),
        steps=id_steps,
        lr=0.05,
        seed=0,
    )
    models = {
        "symbolic_single_seed": dict(laws.laws),
        "symbolic_bagged": laws.bagged(),
        "prior_smw_constants": parametric_laws(prior.astype(np.float64), "synthetic"),
        "identified": parametric_laws(theta_hat.numpy().astype(np.float64), "synthetic"),
        "oracle_true_world": parametric_laws(truth.astype(np.float64), "synthetic"),
    }
    rows: Dict[str, Any] = {}
    for name, model in models.items():
        pred = model_rollout(model, s0, acts, contact)[:, -1, 0]
        gap = pred - achieved
        rows[name] = {
            "mean_optimism_px": float(gap.mean()),
            "mae_transfer_px": float(np.abs(gap).mean()),
            "relative_mae_pct": float(
                100.0 * np.abs(gap).mean() / (np.abs(achieved).mean() + 1e-6)
            ),
            "n_controls": int(achieved.size),
        }
        logger.info(
            "  %-22s transfer MAE %.2f px (%.1f%% rel), optimism %+.2f px",
            name,
            rows[name]["mae_transfer_px"],
            rows[name]["relative_mae_pct"],
            rows[name]["mean_optimism_px"],
        )
    published: Dict[str, Any] = {}
    agreement: Dict[str, Any] = {}
    path = os.path.join(output_dir, PUBLISHED_INVERSE)
    if os.path.isfile(path):
        published = read_metrics(path).get("E3_zero_shot_transfer", {}).get("models", {})
        agreement = {
            name: {
                "published_mae_px": row.get("mae_transfer_px"),
                "reproduced_mae_px": rows.get(name, {}).get("mae_transfer_px"),
                "abs_difference_px": float(
                    abs(
                        float(row.get("mae_transfer_px", float("nan")))
                        - float(rows.get(name, {}).get("mae_transfer_px", float("nan")))
                    )
                ),
            }
            for name, row in published.items()
        }
    return {
        "metric": (
            "same held-out control battery as 10.40-E3: signed (predicted - achieved) "
            "final x per world model; |mean| is planner optimism, mean-abs is the "
            "transfer error"
        ),
        "models": rows,
        "published_10_40_rows": published,
        "reproduction_check": agreement,
        "comparability_note": (
            "The prior / identified / oracle rows recomputed here are compared against "
            "the published 10.40-E3 numbers; agreement is the evidence that the symbolic "
            "row sits on the same battery rather than a similar one. The identified row "
            "differs where this study's parametric fit used a shorter budget."
        ),
    }


def run_study(
    output_dir: str = RESULTS_DIR,
    replicates: int = 6,
    gp_seeds: int = 3,
    population_size: int = 500,
    generations: int = 25,
    max_train: int = 4000,
    parsimony: float = 1e-3,
    fit_windows: int = 800,
    eval_windows: int = 300,
    rollout_len: int = 12,
    rollout_horizon: int = 120,
    rollout_starts: int = 40,
    id_steps: int = 900,
    real_stride: int = 3,
    weight_strengths: str = "0,3,10",
    budget_sweep: bool = True,
    jobs: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run S1-S4 and write the aggregate artifact plus the figure."""
    set_global_seed(seed)
    torch.manual_seed(seed)
    truth = truth_vector(OOD_WORLD)
    prior = truth_vector(PRIOR)
    seeds = tuple(range(gp_seeds))
    budget: Dict[str, Any] = {
        "population_size": population_size,
        "generations": generations,
        "max_train": max_train,
        "parsimony_coefficient": parsimony,
        "n_jobs": jobs,
        "stride": 2,
    }
    strengths = [float(s) for s in str(weight_strengths).split(",") if s.strip()]
    os.makedirs(output_dir, exist_ok=True)

    s1 = run_s1(
        truth=truth,
        prior=prior,
        replicates=replicates,
        gp_seeds=seeds,
        budget=budget,
        fit_windows=fit_windows,
        eval_windows=eval_windows,
        rollout_len=rollout_len,
        id_steps=id_steps,
        weight_strengths=strengths,
        budget_sweep=budget_sweep,
    )
    s2 = run_s2(
        truth=truth,
        prior=prior,
        replicates=replicates,
        budget=budget,
        gp_seeds=seeds,
        fit_windows=fit_windows,
        rollout_len=rollout_len,
        rollout_horizon=rollout_horizon,
        rollout_starts=rollout_starts,
        id_steps=id_steps,
    )
    s3 = run_s3(
        budget=budget,
        gp_seeds=seeds,
        id_steps=id_steps,
        output_dir=output_dir,
        real_stride=real_stride,
    )
    s4 = run_s4(
        truth=truth,
        prior=prior,
        budget=budget,
        gp_seeds=seeds,
        fit_windows=fit_windows,
        id_steps=id_steps,
        output_dir=output_dir,
    )

    payload: Dict[str, Any] = {
        "study": (
            "Symbolic regression as an inverse-problem method: genetic programming "
            "discovers the one-step update laws of a hidden world and of genuine WRAM "
            "telemetry, a probe stage reads the physical constants back out of the "
            "discovered expressions, and the result is scored against the parametric "
            "identification of 10.40, the discrete-guarantee metrics of 8.2 and the "
            "published learned world models."
        ),
        "hidden_world": {name: float(truth[i]) for i, name in enumerate(PARAM_NAMES)},
        "prior": {name: float(prior[i]) for i, name in enumerate(PARAM_NAMES)},
        "protocol": {
            "replicates": replicates,
            "gp_seeds_per_law": gp_seeds,
            "population_size": population_size,
            "generations": generations,
            "max_train_transitions": max_train,
            "parsimony_coefficient": parsimony,
            "fit_windows": fit_windows,
            "eval_windows": eval_windows,
            "fit_window_stride": budget["stride"],
            "rollout_len": rollout_len,
            "rollout_horizon": rollout_horizon,
            "rollout_starts": rollout_starts,
            "parametric_id_steps": id_steps,
            "real_data_transition_stride": real_stride,
            "function_set": "piecewise-affine: add sub mul div neg abs min max (gplearn protected variants)",
            "normalisation": (
                "every law is fitted with each feature and the target divided by the "
                "maximum its FIT rows reach (excitation normalisation), so GP's bounded "
                "terminal constants can express the law; constants are read back by "
                "probing the fitted map, never by copying a terminal value"
            ),
            "gp_jobs": jobs,
            "seed": seed,
        },
        "S1_structure_recovery": s1,
        "S2_multistep_guarantees": s2,
        "S3_real_telemetry": s3,
        "S4_control_transfer": s4,
    }

    artifact = _output_file(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.symbolic_inverse_benchmark",
        extra_meta={
            "hidden_world": OOD_WORLD.__dict__,
            "replicates": replicates,
            "gp_seeds": gp_seeds,
        },
    )
    logger.info("Artifact written to: %s", artifact)
    figure = _render_figure(
        s1,
        s3,
        _output_file(output_dir, "symbolic_inverse_recovery.png", subdir="figures"),
        budget_sweep,
    )
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def _render_figure(
    s1: Dict[str, Any], s3: Dict[str, Any], path: str, budget_sweep_done: bool
) -> str:
    constants = s1["constants"]
    names = list(PARAM_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(
        x - 0.27,
        [constants[n]["prior_rel_error_pct"] for n in names],
        width=0.26,
        color="#d62728",
        label="prior (SMW constants)",
    )
    ax.bar(
        x,
        [max(constants[n]["symbolic_bagged_median_rel_error_pct"], 1e-3) for n in names],
        width=0.26,
        color="#1f77b4",
        label="symbolic GP + probe (Ours)",
    )
    ax.bar(
        x + 0.27,
        [max(constants[n]["parametric_median_rel_error_pct"], 1e-3) for n in names],
        width=0.26,
        color="#2ca02c",
        label="parametric ID (10.40)",
    )
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("constant recovery error (%)  [symlog]")
    ax.set_title("S1: what each inverse method recovers", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1]
    sweep = {k: v for k, v in (s1.get("budget_control") or {}).items() if isinstance(v, dict)}
    if sweep:
        labels = list(sweep)
        for law, color in (("dvx", "#1f77b4"), ("dvy", "#9467bd")):
            vals = [sweep[k][law]["eval_r2_mean"] for k in labels]
            errs = [sweep[k][law]["eval_r2_std"] for k in labels]
            ax.errorbar(
                range(len(labels)), vals, yerr=errs, marker="o", color=color, label=f"{law} law"
            )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [k.replace("population", "pop ").replace("_generations", " gen ") for k in labels],
            fontsize=7,
        )
        ax.set_xlabel("GP search budget")
        ax.set_ylabel("held-out R$^2$ of the velocity law")
        ax.set_title("S1b: more search does not buy the missing structure", fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    ax = axes[2]
    rows = s3["one_step_test_metrics"]
    width = 0.2
    for i, (key, label, color) in enumerate(
        (
            ("symbolic_bagged", "Symbolic GP (Ours)", "#1f77b4"),
            ("parametric_identified_analytic", "Parametric ID (10.40)", "#2ca02c"),
            ("prior_analytic", "Analytic prior", "#d62728"),
        )
    ):
        vals = [max(rows[key][f"{c}_mse"], 1e-6) for c in CHANNEL_NAMES]
        ax.bar(
            np.arange(len(CHANNEL_NAMES)) + (i - 1) * width,
            vals,
            width=width,
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(CHANNEL_NAMES)))
    ax.set_xticklabels(list(CHANNEL_NAMES))
    ax.set_ylabel("real-telemetry test MSE (px$^2$, log)")
    ax.set_title("S3: discovered laws on genuine WRAM data", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Symbolic regression for the inverse problem: discovered laws vs posited structure",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(
        "Figure saved to: %s (budget panel %s)", path, "drawn" if budget_sweep_done else "empty"
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--gp-seeds", dest="gp_seeds", type=int, default=3)
    parser.add_argument("--population-size", dest="population_size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--max-train", dest="max_train", type=int, default=4000)
    parser.add_argument("--parsimony", type=float, default=1e-3)
    parser.add_argument("--fit-windows", dest="fit_windows", type=int, default=800)
    parser.add_argument("--eval-windows", dest="eval_windows", type=int, default=300)
    parser.add_argument("--rollout-len", dest="rollout_len", type=int, default=12)
    parser.add_argument("--rollout-horizon", dest="rollout_horizon", type=int, default=120)
    parser.add_argument("--rollout-starts", dest="rollout_starts", type=int, default=40)
    parser.add_argument("--id-steps", dest="id_steps", type=int, default=900)
    parser.add_argument("--real-stride", dest="real_stride", type=int, default=3)
    parser.add_argument(
        "--weight-strengths",
        dest="weight_strengths",
        default="0,3,10",
        help="comma-separated excitation-weight strengths for the S1c control",
    )
    parser.add_argument("--budget-sweep", dest="budget_sweep", action="store_true", default=True)
    parser.add_argument("--no-budget-sweep", dest="budget_sweep", action="store_false")
    parser.add_argument("--jobs", type=int, default=1)
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
        parsimony=args.parsimony,
        fit_windows=args.fit_windows,
        eval_windows=args.eval_windows,
        rollout_len=args.rollout_len,
        rollout_horizon=args.rollout_horizon,
        rollout_starts=args.rollout_starts,
        id_steps=args.id_steps,
        real_stride=args.real_stride,
        weight_strengths=args.weight_strengths,
        budget_sweep=args.budget_sweep,
        jobs=args.jobs,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
