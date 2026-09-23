"""
inverse_transfer_benchmark.py
Physics parameter identification and zero-shot transfer (README Section 10.40).

The inverse problem counterpart to the forward-prediction benchmark: instead of asking
"given the physics, predict the future", it asks "given observed futures, recover the
physics", and then whether a model-based controller built on the recovered physics
transfers to a world whose discretisation it was never told about. It is the executable
answer to the deferred future-work note in README Section 10.16-6 and the "system
identification inside the graph" idea of ``src/models/pinn_gravity.py``.

Three experiments are run from one hidden target world ``theta_ood``:

* **E1 - synthetic recovery.** Transitions are generated from a simulator parameterised by
  ``theta_ood``; identification is started from the *wrong* prior (the hard-coded SMW
  constants) and must recover ``theta_ood``. A bootstrap over refits and a Laplace /
  Gauss-Newton posterior both give per-constant uncertainty, and the posterior's Fisher
  eigen-spectrum is a formal identifiability diagnostic.
* **E2 - real-data identification.** The same estimator is run on genuine WRAM gameplay
  transitions (``smw_gameplay_dataset.npz``). The analytic model now includes coast friction
  and a ground-contact velocity reset, but still omits collision response and wall checks, so
  the recovered constants remain approximate; the experiment reports how far they sit from
  the reverse-engineered values and how much identification lowers open-loop rollout error
  relative to the prior anyway - an honest measure of the model-misspecification ceiling.
* **E3 - zero-shot control transfer.** Predictive-control optimism is measured on a held-out
  control battery under three world models: the misspecified prior, the identified
  ``theta_hat``, and the oracle ``theta_ood``. The prior is systematically optimistic about
  held-out outcomes while the identified model matches the oracle; a posterior-predictive
  sample gives a credible interval on the identified model's transfer error.

Everything is emulator-free and deterministic under ``set_global_seed``. Writes
``results/inverse_identification_metrics.json`` (with ``_meta``) and
``results/figures/inverse_parameter_recovery.png``.

Run:  python -m src.evaluation.inverse_transfer_benchmark
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.environment.dataset_loader import load_and_preprocess_data  # noqa: E402
from src.inverse.parameter_identification import (  # noqa: E402
    PARAM_NAMES,
    EngineParams,
    bootstrap_ci,
    generate_synthetic_windows,
    identify_params,
    make_windows,
    per_variable_mse,
    posterior_laplace,
    sample_posterior,
    simulate_rollout,
    theta_tensor,
)
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import RESULTS_DIR, figure_file, results_file  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

# Hidden target world: a SMW-like game with a *different* fixed-point scale and gravity,
# i.e. the "unknown discretisation scheme" scenario named in README Section 10.16-6.
OOD_WORLD = EngineParams(
    max_vx=48.0,
    walk_accel=1.00,
    run_accel=1.80,
    subpixels_per_pixel=20.0,
    held_gravity=2.40,
    fall_gravity=5.20,
    decel=0.60,
)
PRIOR = EngineParams()  # the hard-coded SMW constants a naive zero-shot agent would carry in.


def run_e3_transfer(
    models: Dict[str, torch.Tensor],
    true_params: torch.Tensor,
    seed: int = 101,
    n_windows: int = 400,
    horizon: int = 24,
) -> Dict[str, Dict[str, float]]:
    """Zero-shot predictive-control transfer on a held-out control battery.

    For each ``(s0, action_seq)`` drawn from the *true* world, the achieved final ``x`` is the
    ground-truth window target. Each candidate world model predicts its own final ``x`` for the
    identical plan. The signed mean of (predicted - achieved) is the model's **optimism** - the
    systematic bias a planner inherits from a misspecified dynamics model - and its absolute
    mean is the **transfer error**. A model that identifies the physics (correct scale, ceiling,
    gravity) predicts held-out outcomes like the oracle; the naive SMW prior is systematically
    optimistic. Because the plans are fixed and held out, this metric is about the physics, not
    about search quality.
    """
    s0, acts, targets, gr = generate_synthetic_windows(true_params, n_windows, horizon, seed=seed)
    achieved = targets[:, -1, 0]  # true final x of each held-out control
    out: Dict[str, Dict[str, float]] = {}
    for name, mp in models.items():
        pred = simulate_rollout(s0, acts, mp, gr)[:, -1, 0]
        gap = pred - achieved
        out[name] = {
            "mean_optimism_px": float(gap.mean()),
            "mae_transfer_px": float(gap.abs().mean()),
            "relative_mae_pct": float(100.0 * gap.abs().mean() / (achieved.abs().mean() + 1e-6)),
            "n_controls": int(n_windows),
        }
    return out


def posterior_predictive_transfer(
    synth,
    theta_hat: torch.Tensor,
    post: Dict[str, object],
    ood_t: torch.Tensor,
    n_samples: int = 16,
    seed: int = 101,
    n_windows: int = 200,
    horizon: int = 24,
) -> Dict[str, float]:
    """Credible interval on the identified model's transfer error.

    Draws ``n_samples`` parameter vectors from the Laplace posterior and, for
    each, evaluates the held-out control-battery transfer error against the
    true world.  The spread is the posterior uncertainty propagated to the
    control-relevant quantity - the Bayesian-inverse analogue of the Deep
    Ensemble's predictive spread (README Section 10.9), here over *physics*
    constants rather than network weights.
    """
    samples = sample_posterior(theta_hat, post, n_samples, seed=7)
    s0, acts, targets, gr = generate_synthetic_windows(ood_t, n_windows, horizon, seed=seed)
    achieved = targets[:, -1, 0]
    maes = []
    for theta in samples:
        pred = simulate_rollout(s0, acts, theta, gr)[:, -1, 0]
        maes.append(float((pred - achieved).abs().mean()))
    arr = np.asarray(maes)
    return {
        "mean_mae_px": float(arr.mean()),
        "ci_low_px": float(np.percentile(arr, 2.5)),
        "ci_high_px": float(np.percentile(arr, 97.5)),
        "n_samples": int(n_samples),
    }


def run_benchmark(output_dir: str = RESULTS_DIR, n_boot: int = 16) -> Dict[str, object]:
    """Run all three experiments and write the aggregate artifact + figure."""
    set_global_seed(42)
    torch.manual_seed(42)
    prior_t = theta_tensor(PRIOR)
    ood_t = theta_tensor(OOD_WORLD)
    logger.info("=== E1: synthetic recovery of an unknown world ===")

    synth = generate_synthetic_windows(ood_t, n_windows=1200, rollout_len=12, seed=11)
    # System-identification warm start: the velocity ceiling is initialised from the observed
    # |vx| range of the data. If it were left at the SMW prior (72) while the true world
    # saturates at 48, the model's own rollout never reaches its (too-high) clamp, so
    # d(prediction)/d(max_vx) = 0 and gradient descent cannot lower it - the classic
    # inactive-constraint gradient pathology. Warm-starting the ceiling from the data range is
    # standard practice and leaves the other five constants at the (wrong) SMW prior.
    init_vec = prior_t.clone()
    init_vec[0] = float(synth[0][:, 2].abs().max()) * 1.05
    theta_hat, info1 = identify_params(synth, init_vec, steps=2000, lr=0.05, seed=11)
    boot = bootstrap_ci(synth, init_vec, n_boot=n_boot, steps=1500, lr=0.05, seed=11)
    post = posterior_laplace(synth, theta_hat)

    recovery: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(PARAM_NAMES):
        truth, est, prior_v = float(ood_t[k]), float(theta_hat[k]), float(prior_t[k])
        ci = boot[name]
        recovery[name] = {
            "true": truth,
            "prior": prior_v,
            "identified": est,
            "abs_error": abs(est - truth),
            "rel_error_pct": (100.0 * abs(est - truth) / abs(truth)) if truth else float("nan"),
            "prior_rel_error_pct": (100.0 * abs(prior_v - truth) / abs(truth))
            if truth
            else float("nan"),
            "boot_ci_low": ci["ci_low"],
            "boot_ci_high": ci["ci_high"],
            "boot_std": ci["std"],
            "truth_in_ci": bool(ci["ci_low"] <= truth <= ci["ci_high"]),
        }
    for name in PARAM_NAMES:
        r = recovery[name]
        logger.info(
            "  %-18s true=%.3f hat=%.3f (prior rel err %.1f%% -> id rel err %.1f%%)",
            name,
            r["true"],
            r["identified"],
            r["prior_rel_error_pct"],
            r["rel_error_pct"],
        )

    logger.info("=== E2: identification on real WRAM gameplay ===")
    data = load_and_preprocess_data()
    train_win = make_windows(
        data["train_states"],
        data["train_actions"],
        data["train_next_states"],
        data["train_episodes"],
        rollout_len=8,
    )
    test_win = make_windows(
        data["test_states"],
        data["test_actions"],
        data["test_next_states"],
        data["test_episodes"],
        rollout_len=8,
    )
    theta_real, info2 = identify_params(train_win, prior_t, steps=2000, lr=0.05, seed=42)
    truth_defaults = theta_tensor(PRIOR)  # reverse-engineered WRAM values = the reference
    real_prior_pvar = per_variable_mse(test_win, prior_t)
    real_id_pvar = per_variable_mse(test_win, theta_real)
    real_recovery: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(PARAM_NAMES):
        ref, est = float(truth_defaults[k]), float(theta_real[k])
        real_recovery[name] = {
            "wram_reference": ref,
            "identified": est,
            "abs_error": abs(est - ref),
            "rel_error_pct": (100.0 * abs(est - ref) / abs(ref)) if ref else float("nan"),
        }
    rollout_drift_px = {
        "prior": real_prior_pvar,
        "identified": real_id_pvar,
    }
    logger.info(
        "  test rollout MSE (px^2): prior x=%.3f identified x=%.3f",
        real_prior_pvar["x"],
        real_id_pvar["x"],
    )

    logger.info("=== E3: zero-shot predictive-control transfer ===")
    transfer = run_e3_transfer(
        {"prior_smw_constants": prior_t, "identified": theta_hat, "oracle_true_world": ood_t},
        ood_t,
    )
    pp = posterior_predictive_transfer(synth, theta_hat, post, ood_t)
    for name, row in transfer.items():
        logger.info(
            "  %-20s transfer MAE = %.2f px (%.1f%% rel), optimism = %+.2f px",
            name,
            row["mae_transfer_px"],
            row["relative_mae_pct"],
            row["mean_optimism_px"],
        )
    logger.info(
        "  posterior-predictive identified MAE = %.2f px [%.2f, %.2f] (95%% credible)",
        pp["mean_mae_px"],
        pp["ci_low_px"],
        pp["ci_high_px"],
    )

    payload: Dict[str, object] = {
        "study": "inverse_physics_identification",
        "hidden_world": {n: recovery[n]["true"] for n in PARAM_NAMES},
        "prior": {n: recovery[n]["prior"] for n in PARAM_NAMES},
        "identifiability_note": (
            "A constant is identifiable only if the observed data excite the term that uses"
            " it; the bootstrap width is the empirical identifiability measure (a very wide"
            " interval means the windows did not exercise that invariant)."
        ),
        "E1_synthetic_recovery": {
            "params": recovery,
            "final_loss": info1["final_loss"],
            "bootstrap_n": n_boot,
            "max_rel_error_pct": max(recovery[n]["rel_error_pct"] for n in PARAM_NAMES),
            "posterior_laplace": {
                "std_errors": dict(zip(PARAM_NAMES, post["std_errors"])),
                "relative_std": dict(zip(PARAM_NAMES, post["relative_std"])),
                "identified": dict(zip(PARAM_NAMES, post["identified"])),
                "condition_number": post["condition_number"],
                "fisher_eigenvalues": post["eigenvalues"],
                "noise_variance": post["noise_variance"],
                "note": (
                    "Gauss-Newton/Laplace posterior covariance sigma^2 (J^T J)^-1 at the point"
                    " estimate; the Fisher eigen-spectrum is the identifiability statement - a"
                    " near-zero eigenvalue is a parameter direction the transitions cannot"
                    " resolve, the same structural limit that caps any learned model."
                ),
            },
        },
        "E2_real_data_identification": {
            "params": real_recovery,
            "train_windows": int(train_win[0].shape[0]),
            "test_windows": int(test_win[0].shape[0]),
            "final_loss": info2["final_loss"],
            "test_rollout_mse_px2": rollout_drift_px,
            "model_scope_note": (
                "The analytic model now includes coast friction and a ground-contact velocity"
                " reset but still excludes collision response and wall checks, so real-data"
                " recovery is bounded by residual model misspecification; the honest result is"
                " how much identification lowers open-loop rollout error versus the prior."
            ),
        },
        "E3_zero_shot_transfer": {
            "metric": (
                "held-out control battery: signed (predicted - achieved) final x per world"
                " model; |mean| is planner optimism, mean-abs is the transfer error"
            ),
            "models": transfer,
            "posterior_predictive_identified": pp,
        },
    }

    artifact = results_file("inverse_identification_metrics.json")
    write_metrics(
        artifact,
        payload,
        seed=42,
        command="python -m src.evaluation.inverse_transfer_benchmark",
        extra_meta={"hidden_world": OOD_WORLD.__dict__, "n_bootstrap": n_boot},
    )
    logger.info("Artifact written to: %s", artifact)

    figure = _render_figure(recovery, transfer, figure_file("inverse_parameter_recovery.png"))
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def _render_figure(
    recovery: Dict[str, Dict[str, float]],
    transfer: Dict[str, Dict[str, float]],
    path: str,
) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    names = PARAM_NAMES
    rel = [recovery[n]["rel_error_pct"] for n in names]
    prior_rel = [recovery[n]["prior_rel_error_pct"] for n in names]
    x = np.arange(len(names))
    ax1.bar(x - 0.2, prior_rel, width=0.4, color="#d62728", label="prior (SMW constants)")
    ax1.bar(x + 0.2, rel, width=0.4, color="#2ca02c", label="identified")
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax1.set_ylabel("relative recovery error (%)  [symlog]")
    ax1.set_title("E1: recovery of the unknown world (lower = better)", fontweight="bold")
    ax1.legend()

    models = ["prior_smw_constants", "identified", "oracle_true_world"]
    errs = [transfer[m]["relative_mae_pct"] for m in models]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    ax2.bar(range(3), errs, color=colors)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["prior\nSMW", "identified\n(Ours)", "oracle\ntrue world"])
    ax2.set_ylabel("transfer error (% of achieved x, lower = better)")
    ax2.set_title("E3: zero-shot MPC transfer to the unknown world", fontweight="bold")

    fig.suptitle(
        "Physics parameter identification (inverse problem) enables control transfer",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    parser.add_argument("--bootstrap-n", dest="bootstrap_n", type=int, default=16)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    run_benchmark(output_dir=args.output_dir, n_boot=args.bootstrap_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
