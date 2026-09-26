"""
symbolic_tilemap_residual_benchmark.py
Does tile geometry explain what the state could not? (README Section 10.43.8)

Section 10.43.5 ended on a negative discovery result: genetic programming, given all
eleven observable channels of the 8D WRAM telemetry, could not even *fit* the horizontal
velocity residual of the identified analytic model (train R^2 < 0). The reading offered
there was that the residual is unobserved tile geometry rather than a missing closed form
- but that is an inference from one experiment, and the repository's own data already
contains the counterfactual: ``smw_tilemap_dataset.npz`` records the 7x7 local terrain
patch ($7E:C800) around Mario for every transition of the same stage.

So this experiment re-runs the residual discovery with the geometry available, in four
conditions that differ only in what the search may condition on:

* ``state`` - the eleven observable channels of the 8D telemetry (the 10.43.5 replication,
  on the tilemap recording);
* ``state+geom`` - those eleven plus seven compressed occupancy descriptors of the 7x7
  patch (below, ahead-left, ahead-right, ceiling, center, slope fraction, fill);
* ``state+full`` - those eleven plus all 49 raw patch cells;
* ``state+geom-shuffled`` - the compressed descriptors permuted across transitions, i.e. a
  geometry channel that carries no information about the frame. This is the placebo: any
  apparent gain that survives shuffling is overfitting, not physics.

An ordinary least-squares fit on the same design matrix is reported beside every GP row,
so a GP advantage can be told apart from "these terms are linearly predictive".

The verdict is computed from the measured numbers rather than asserted: the observation-gap
claim holds only if the geometry conditions raise held-out R^2 over the state-only condition
AND the shuffled-placebo condition does not.

Emulator-free, deterministic, CPU-only. Writes
``results/symbolic_tilemap_residual_metrics.json`` with ``_meta`` and
``results/figures/symbolic_tilemap_residual_gain.png``.

Run:  python -m src.evaluation.symbolic_tilemap_residual_benchmark
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

from src.environment.dataset_loader import load_and_preprocess_data  # noqa: E402
from src.inverse.parameter_identification import (
    PARAM_NAMES,
    identify_params,
    make_windows,
    theta_tensor,
)
from src.inverse.symbolic_regression import (
    TransitionBank,
    bank_from_transitions,
    evaluate_law,
    fit_law,
    parametric_laws,
    symbolic_step,
    truth_vector,
)
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import DATASET_TILEMAP, RESULTS_DIR  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

# The reverse-engineered WRAM constants are the prior the identification starts from,
# exactly as in 10.40-E2 / 10.43-S3.
from src.evaluation.inverse_transfer_benchmark import PRIOR  # noqa: E402

ARTIFACT_NAME = "symbolic_tilemap_residual_metrics.json"
FIGURE_NAME = "symbolic_tilemap_residual_gain.png"
CHANNELS: Tuple[str, ...] = ("x", "y", "vx", "vy")

# The eleven observable channels of the 8D telemetry: state, buttons, contact byte.
STATE_FEATURES: List[str] = [
    "x",
    "y",
    "vx",
    "vy",
    "dir",
    "run",
    "jump",
    "ground",
    "ceiling",
    "left",
    "right",
]
# Compressed descriptors of the 7x7 patch. Row 0 is above Mario and row 6 below (the
# recorder walks dy from -3 to +3), column 0 is left and column 6 right; the center cell
# is [3, 3].
GEOM_FEATURES: List[str] = [
    "solid_below",
    "solid_ahead_right",
    "solid_ahead_left",
    "ceiling_above",
    "center_solid",
    "slope_fraction",
    "patch_fill",
]
# Terrain x input products: the terms a wall-stopped horizontal law actually needs. They
# are handed over as a condition rather than assumed, because the question is whether the
# terrain carries the signal at all, not whether one particular search can build a product.
INTERACT_FEATURES: List[str] = [
    "dir_x_ahead_right",
    "dir_x_ahead_left",
    "vx_x_ahead",
    "vy_x_below",
]


def geometry_descriptors(patches: np.ndarray) -> np.ndarray:
    """Seven occupancy descriptors of a stack of ``[N, 7, 7]`` patches.

    Deliberately low-dimensional and interpretable: the search is handed the *shape* of
    the local terrain (is there ground under me, wall in front of me, ceiling over me,
    slope fraction nearby) rather than 49 anonymous pixels, so a positive result says the
    residual is explained by terrain occupancy and not by memorising a patch code.
    """
    p = np.asarray(patches, dtype=np.float64)
    solid = (p > 0).astype(np.float64)
    slope = (p == 3).astype(np.float64)
    return np.stack(
        [
            solid[:, 5:7, 2:5].mean(axis=(1, 2)),
            solid[:, 2:5, 5:7].mean(axis=(1, 2)),
            solid[:, 2:5, 0:2].mean(axis=(1, 2)),
            solid[:, 0:2, 2:5].mean(axis=(1, 2)),
            solid[:, 3, 3],
            slope.mean(axis=(1, 2)),
            solid.mean(axis=(1, 2)),
        ],
        axis=1,
    )


def state_block(bank: TransitionBank) -> np.ndarray:
    """The observable 11-channel design matrix of a transition bank."""
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
    }
    return np.stack([columns[name] for name in STATE_FEATURES], axis=1)


def interaction_block(bank: TransitionBank, geom: np.ndarray) -> np.ndarray:
    """Terrain x input products built from the compressed descriptors."""
    below, ahead_r, ahead_l = geom[:, 0], geom[:, 1], geom[:, 2]
    vx, vy, direction = bank.states[:, 2], bank.states[:, 3], bank.direction
    ahead = np.where(direction >= 0, ahead_r, ahead_l)
    return np.stack([direction * ahead_r, direction * ahead_l, vx * ahead, vy * below], axis=1)


def signal_diagnostics(
    bank: TransitionBank, residual: np.ndarray, patches: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Estimator-free check that terrain and residual co-vary at all.

    A correlation table answers what no fitted model can settle alone: if the residual does
    not vary with terrain occupancy in the data, then no search over these features could
    find it, and a null result is a statement about the observation rather than the estimator.
    """
    geom = geometry_descriptors(patches)
    out: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(GEOM_FEATURES):
        col = geom[:, j]
        row: Dict[str, float] = {}
        for k, channel in enumerate(CHANNELS):
            r = residual[:, k]
            row[channel] = (
                float(np.corrcoef(col, r)[0, 1]) if col.std() > 1e-9 and r.std() > 1e-12 else 0.0
            )
        out[name] = row
    return out


def condition_designs(
    bank: TransitionBank, patches: np.ndarray, rng: np.random.RandomState
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Design matrix and feature names for every conditioning condition."""
    state = state_block(bank)
    geom = geometry_descriptors(patches)
    inter = interaction_block(bank, geom)
    full = np.asarray(patches, dtype=np.float64).reshape(patches.shape[0], -1)
    cell_names = [f"tile_r{r}c{c}" for r in range(7) for c in range(7)]
    shuffled = geom[rng.permutation(geom.shape[0])]
    return {
        "state": (state, list(STATE_FEATURES)),
        "state+geom": (
            np.concatenate([state, geom], axis=1),
            list(STATE_FEATURES) + list(GEOM_FEATURES),
        ),
        "state+geom+inter": (
            np.concatenate([state, geom, inter], axis=1),
            list(STATE_FEATURES) + list(GEOM_FEATURES) + list(INTERACT_FEATURES),
        ),
        "state+full": (
            np.concatenate([state, full], axis=1),
            list(STATE_FEATURES) + cell_names,
        ),
        "state+geom-shuffled": (
            np.concatenate([state, shuffled, inter], axis=1),
            list(STATE_FEATURES)
            + [f"{g}_shuffled" for g in GEOM_FEATURES]
            + list(INTERACT_FEATURES),
        ),
    }


def load_tilemap_splits(
    dataset_path: str,
) -> Dict[str, Any]:
    """Split the tilemap recording with the canonical protocol and keep the patches aligned.

    ``load_and_preprocess_data`` performs the pit-death filter and the seeded episodic
    split but returns only the state arrays, so the split is reconstructed here from the
    episode sets it produced: the same mask and the same episode membership, which is
    asserted rather than assumed.
    """
    data = load_and_preprocess_data(dataset_path=dataset_path, seed=42)
    raw = np.load(dataset_path)
    states, actions, next_states, episodes = (
        raw["states"],
        raw["actions"],
        raw["next_states"],
        raw["episodes"],
    )
    patches, next_patches = raw["tile_patches"], raw["next_tile_patches"]
    valid = (
        (states[:, 1] >= 0)
        & (states[:, 1] <= 500)
        & (next_states[:, 1] >= 0)
        & (next_states[:, 1] <= 500)
    )
    kept = [arr[valid] for arr in (states, actions, next_states, episodes, patches, next_patches)]
    k_states, k_actions, k_next, k_eps, k_patches, k_next_patches = kept
    total = sum(data[f"{s}_states"].shape[0] for s in ("train", "val", "test"))
    assert k_states.shape[0] == total, "pit-death filter disagreed with the canonical loader"
    out: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        eps = set(np.unique(data[f"{split}_episodes"]).tolist())
        mask = np.isin(k_eps, list(eps))
        assert mask.sum() == data[f"{split}_states"].shape[0], f"{split} patch alignment mismatch"
        np.testing.assert_allclose(k_states[mask][:, :4], data[f"{split}_states"][:, :4])
        out[f"{split}_states"] = data[f"{split}_states"]
        out[f"{split}_actions"] = data[f"{split}_actions"]
        out[f"{split}_next_states"] = data[f"{split}_next_states"]
        out[f"{split}_episodes"] = data[f"{split}_episodes"]
        out[f"{split}_patches"] = k_patches[mask]
        out[f"{split}_next_patches"] = k_next_patches[mask]
    return out


def build_verdict(
    results: Dict[str, Dict[str, Any]], diagnostics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Any]:
    """Per-channel verdict: does terrain occupancy explain the analytic model's residual?

    Support needs the geometry condition to actually predict out of sample, to beat the
    state-only fit, and to beat the shuffled-geometry placebo by more than the placebo
    beats the state fit - "less catastrophic" is not "explained", and a gain that the
    placebo also achieves is overfitting.
    """
    per_channel: Dict[str, Any] = {}
    for channel in CHANNELS:
        base = results["state"][channel]
        gains = {
            c: results[c][channel]["gp_test_r2_median"] - base["gp_test_r2_median"]
            for c in results
            if c != "state"
        }
        geom_gain = max(v for c, v in gains.items() if "shuffled" not in c)
        best_condition = max(
            (c for c in results if "shuffled" not in c and c != "state"),
            key=lambda c: results[c][channel]["gp_test_r2_median"],
        )
        placebo_gain = gains["state+geom-shuffled"]
        per_channel[channel] = {
            "state_test_r2_median": base["gp_test_r2_median"],
            "best_geometry_condition": best_condition,
            "best_geometry_test_r2_median": results[best_condition][channel]["gp_test_r2_median"],
            "gain_over_state": geom_gain,
            "placebo_gain": placebo_gain,
            "gain_by_condition": gains,
            "ols_state_test_r2": results["state"][channel]["ols"]["test_r2"],
            "ols_best_geometry_test_r2": results[best_condition][channel]["ols"]["test_r2"],
            "max_abs_terrain_correlation": float(
                max(abs(diagnostics["test"][g][channel]) for g in GEOM_FEATURES)
            ),
            "terrain_explains_it": bool(
                results[best_condition][channel]["gp_test_r2_median"] > 0.05
                and geom_gain > 0.05
                and geom_gain > placebo_gain + 0.02
            ),
        }
    explained = [c for c in CHANNELS if per_channel[c]["terrain_explains_it"]]
    not_explained = [c for c in CHANNELS if not per_channel[c]["terrain_explains_it"]]
    verdict = {
        "per_channel": per_channel,
        "terrain_explains_channels": explained,
        "terrain_does_not_explain_channels": not_explained,
        "observation_gap_claim_supported_for_vx": per_channel["vx"]["terrain_explains_it"],
        "reading": (
            f"Local terrain occupancy is causally implicated in the residuals of "
            f"{explained if explained else 'no channel'} and not in those of "
            f"{not_explained}. The 10.43.5 reading - that the horizontal-velocity residual "
            "is unobserved tile geometry - is therefore only supported if handing the search "
            "the recorded 7x7 patch raises its held-out R^2 above the state-only fit AND "
            "above the shuffled-geometry placebo."
        ),
        "caveat": (
            "The recorded patch is binary in this stage: the slope class (3) never occurs, so "
            "a slope-driven horizontal residual would remain invisible even to this richer "
            "observation. The test bounds what the recording can express, not what the engine "
            "does."
        ),
    }
    for channel in CHANNELS:
        row = per_channel[channel]
        logger.info(
            "  residual %-2s: state %+.3f -> %s %+.3f (gain %+.3f, placebo %+.3f, "
            "|corr| %.3f) | terrain explains: %s",
            channel,
            row["state_test_r2_median"],
            row["best_geometry_condition"],
            row["best_geometry_test_r2_median"],
            row["gain_over_state"],
            row["placebo_gain"],
            row["max_abs_terrain_correlation"],
            row["terrain_explains_it"],
        )
    logger.info("  terrain explains: %s | not: %s", explained, not_explained)
    return verdict


def _ols_r2(X: np.ndarray, y: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
    """Least-squares control on the same design matrix (intercept included)."""
    A = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1) @ coef
    sse = float(np.sum((y_te - pred) ** 2))
    sst = float(np.sum((y_te - y_te.mean()) ** 2))
    train = A @ coef
    sse_tr = float(np.sum((y - train) ** 2))
    return {
        "train_r2": 1.0 - sse_tr / max(float(np.sum((y - y.mean()) ** 2)), 1e-18),
        "test_r2": 1.0 - sse / sst if sst > 1e-18 else float("nan"),
    }


def run_benchmark(
    output_dir: str = RESULTS_DIR,
    dataset_path: str = DATASET_TILEMAP,
    gp_seeds: Sequence[int] = (0, 1, 2),
    population_size: int = 500,
    generations: int = 25,
    max_train: int = 4000,
    parsimony: float = 1e-3,
    id_steps: int = 1200,
    stride: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Identify the analytic map on the tilemap recording and probe its residual per condition."""
    set_global_seed(seed)
    torch.manual_seed(seed)
    logger.info("=== Loading the tilemap recording with the canonical episodic split ===")
    data = load_tilemap_splits(dataset_path)
    fit_bank = bank_from_transitions(
        data["train_states"], data["train_actions"], data["train_next_states"], stride=stride
    )
    test_bank = bank_from_transitions(
        data["test_states"], data["test_actions"], data["test_next_states"]
    )
    rng = np.random.RandomState(seed)
    fit_patches = data["train_patches"][::stride]
    test_patches = data["test_patches"]
    logger.info(
        "fit transitions %d | test transitions %d | patch cells %dx%d",
        fit_bank.n,
        test_bank.n,
        *data["train_patches"].shape[1:],
    )

    logger.info("=== Identifying the analytic map on the tilemap recording (10.40-E2 recipe) ===")
    train_windows = make_windows(
        data["train_states"],
        data["train_actions"],
        data["train_next_states"],
        data["train_episodes"],
        rollout_len=8,
    )
    theta_hat, _ = identify_params(
        train_windows, theta_tensor(PRIOR), steps=id_steps, lr=0.05, seed=42
    )
    wram_ref = truth_vector(PRIOR)
    identified = {
        name: {
            "wram_reference": float(wram_ref[i]),
            "identified": float(theta_hat[i]),
            "rel_error_pct": float(
                100.0 * abs(float(theta_hat[i]) - float(wram_ref[i])) / abs(float(wram_ref[i]))
            ),
        }
        for i, name in enumerate(PARAM_NAMES)
    }

    model = parametric_laws(theta_hat.numpy().astype(np.float64), "real")
    fit_pred = symbolic_step(fit_bank.states, fit_bank.actions, fit_bank.contact, model)
    test_pred = symbolic_step(test_bank.states, test_bank.actions, test_bank.contact, model)
    residual_var = {
        tag: {c: float(np.var(bank.next_states[:, i] - pred[:, i])) for i, c in enumerate(CHANNELS)}
        for tag, bank, pred in (("train", fit_bank, fit_pred), ("test", test_bank, test_pred))
    }

    fit_designs = condition_designs(fit_bank, fit_patches, rng)
    test_designs = condition_designs(test_bank, test_patches, np.random.RandomState(seed + 1))
    diagnostics = {
        "train": signal_diagnostics(fit_bank, fit_bank.next_states - fit_pred, fit_patches),
        "test": signal_diagnostics(test_bank, test_bank.next_states - test_pred, test_patches),
    }

    results: Dict[str, Any] = {}
    for condition, (X_fit, features) in fit_designs.items():
        X_test, _ = test_designs[condition]
        block: Dict[str, Any] = {"n_features": int(X_fit.shape[1])}
        for i, channel in enumerate(CHANNELS):
            y_fit = fit_bank.next_states[:, i] - fit_pred[:, i]
            y_test = test_bank.next_states[:, i] - test_pred[:, i]
            rows: List[Dict[str, float]] = []
            exprs: List[str] = []
            for s in gp_seeds:
                law, _ = fit_law(
                    f"residual_{channel}",
                    X_fit,
                    y_fit,
                    features,
                    seed=s,
                    population_size=population_size,
                    generations=generations,
                    max_train=max_train,
                    parsimony_coefficient=parsimony,
                )
                tr = evaluate_law(law, X_fit, y_fit)
                te = evaluate_law(law, X_test, y_test)
                rows.append(
                    {
                        "seed": float(s),
                        "train_r2": tr["r2"],
                        "test_r2": te["r2"],
                        "nodes": float(law.n_nodes),
                    }
                )
                exprs.append(law.expression)
            tests = [r["test_r2"] for r in rows]
            block[channel] = {
                "gp_train_r2_mean": float(np.nanmean([r["train_r2"] for r in rows])),
                "gp_test_r2_mean": float(np.nanmean(tests)),
                "gp_test_r2_median": float(np.nanmedian(tests)),
                "gp_test_r2_best": float(np.nanmax(tests)),
                "gp_test_r2_worst": float(np.nanmin(tests)),
                "gp_test_r2_std": float(np.nanstd(tests)) if len(rows) > 1 else 0.0,
                "gp_seeds_with_positive_test_r2": float(
                    np.mean([r["test_r2"] > 0.0 for r in rows])
                ),
                "ols": _ols_r2(X_fit, y_fit, X_test, y_test),
                "per_seed": rows,
                "best_expression": exprs[int(np.nanargmax([r["test_r2"] for r in rows]))],
            }
        results[condition] = block
        logger.info(
            "  %-22s vx: GP test R2 %+.3f (best %+.3f, %d/%d seeds > 0) | OLS test R2 %+.3f",
            condition,
            block["vx"]["gp_test_r2_mean"],
            block["vx"]["gp_test_r2_best"],
            int(round(block["vx"]["gp_seeds_with_positive_test_r2"] * len(gp_seeds))),
            len(gp_seeds),
            block["vx"]["ols"]["test_r2"],
        )

    verdict = build_verdict(results, diagnostics)
    payload: Dict[str, Any] = {
        "study": (
            "Tilemap-conditioned residual discovery: the 10.43.5 grey-box control re-run with "
            "the 7x7 local terrain patch of the same stage available to the search, against a "
            "shuffled-geometry placebo and a least-squares control."
        ),
        "dataset": os.path.basename(dataset_path),
        "transitions": {"fit": fit_bank.n, "test": test_bank.n, "patch": [7, 7]},
        "protocol": {
            "gp_seeds": list(gp_seeds),
            "population_size": population_size,
            "generations": generations,
            "max_train_transitions": max_train,
            "parsimony_coefficient": parsimony,
            "parametric_id_steps": id_steps,
            "fit_transition_stride": stride,
            "split": "canonical seeded episodic split (seed 42), patches aligned by episode membership and asserted",
            "conditions": {
                "state": STATE_FEATURES,
                "geometry_descriptors": GEOM_FEATURES,
                "terrain_input_interactions": INTERACT_FEATURES,
                "state+full": "all 49 patch cells, row 0 above Mario / column 0 to his left",
                "state+geom-shuffled": (
                    "the compressed descriptors permuted across transitions (and the terrain x "
                    "input products kept), so any gain surviving it is overfitting"
                ),
            },
            "seed": seed,
        },
        "identified_constants_vs_wram_reference": identified,
        "analytic_residual_variance_px2": residual_var,
        "terrain_residual_correlation": diagnostics,
        "conditions": results,
        "verdict": verdict,
    }
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.symbolic_tilemap_residual_benchmark",
        extra_meta={"gp_seeds": list(gp_seeds), "conditions": list(results)},
    )
    logger.info("Artifact written to: %s", artifact)
    figure = _render_figure(results, os.path.join(output_dir, "figures", FIGURE_NAME))
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def _render_figure(results: Dict[str, Any], path: str) -> str:
    conditions = list(results)
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14.0, 4.0), sharex=True)
    for ax, channel in zip(np.atleast_1d(axes), CHANNELS):
        med = [results[c][channel]["gp_test_r2_median"] for c in conditions]
        lo = [
            results[c][channel]["gp_test_r2_median"] - results[c][channel]["gp_test_r2_worst"]
            for c in conditions
        ]
        hi = [
            results[c][channel]["gp_test_r2_best"] - results[c][channel]["gp_test_r2_median"]
            for c in conditions
        ]
        ols = [results[c][channel]["ols"]["test_r2"] for c in conditions]
        x = np.arange(len(conditions))
        ax.bar(x - 0.2, med, width=0.4, color="#1f77b4", label="GP median over seeds")
        ax.bar(x + 0.2, ols, width=0.4, color="#7f7f7f", label="OLS on same features")
        ax.errorbar(x - 0.2, med, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3)
        ax.axhline(0.0, color="red", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("state+", "").replace("geom-shuffled", "geom (sham)") for c in conditions],
            fontsize=7,
            rotation=20,
        )
        ax.set_title(f"residual {channel}: held-out $R^2$", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Does tile geometry explain what the state could not? (10.43.8)", fontweight="bold"
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
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    parser.add_argument("--dataset-path", dest="dataset_path", default=DATASET_TILEMAP)
    parser.add_argument("--gp-seeds", dest="gp_seeds", default="0,1,2")
    parser.add_argument("--population-size", dest="population_size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--max-train", dest="max_train", type=int, default=4000)
    parser.add_argument("--parsimony", type=float, default=1e-3)
    parser.add_argument("--id-steps", dest="id_steps", type=int, default=1200)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    seeds = tuple(int(s) for s in str(args.gp_seeds).split(",") if s.strip())
    run_benchmark(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        gp_seeds=seeds,
        population_size=args.population_size,
        generations=args.generations,
        max_train=args.max_train,
        parsimony=args.parsimony,
        id_steps=args.id_steps,
        stride=args.stride,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
