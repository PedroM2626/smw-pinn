"""
piml_mfrl_study.py
Multi-seed, per-mechanism ablation study behind README Section 10.39 (PIML-MFRL).

Runs the *same* model-free PPO loop from ``src.training.piml_mfrl`` on the authentic
SNES console across five conditions - the canonical model-free baseline (all physics off)
plus each approach in isolation (A, B, C) and their combination (A+B+C) - over a set of
seeds, then aggregates the result into one artifact. The per-mechanism ablation is what
makes this a definitive benchmark: it attributes any change in return and in the executed
action's physics-violation rate to a specific coupling instead of a single black-box
"physics on/off" toggle.

Writes ``results/piml_mfrl_metrics.json`` (with a ``_meta`` provenance block) and
``results/figures/piml_mfrl_comparison.png``.

Requires the Libretro core + ROM. Run:
    python -m src.evaluation.piml_mfrl_study --seeds 42,43,44 --total-timesteps 10000
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.training.piml_mfrl import train_piml_mfrl  # noqa: E402
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import RESULTS_DIR, figure_file, results_file  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

logger = get_logger(__name__)

BASELINE = "model_free_ppo"

# Condition name -> the three PIML-MFRL switches. The baseline is genuine model-free PPO
# (all off); A/B/C isolate each coupling; full combines them.
CONDITIONS: Dict[str, Dict[str, bool]] = {
    "model_free_ppo": {
        "use_physics_critic": False,
        "use_cbf_filter": False,
        "use_action_penalty": False,
    },
    "piml_A_critic": {
        "use_physics_critic": True,
        "use_cbf_filter": False,
        "use_action_penalty": False,
    },
    "piml_B_cbf": {
        "use_physics_critic": False,
        "use_cbf_filter": True,
        "use_action_penalty": False,
    },
    "piml_C_action": {
        "use_physics_critic": False,
        "use_cbf_filter": False,
        "use_action_penalty": True,
    },
    "piml_full_A_B_C": {
        "use_physics_critic": True,
        "use_cbf_filter": True,
        "use_action_penalty": True,
    },
}

# Plot labels keyed by condition (short, for the figure x-axis).
PLOT_LABELS: Dict[str, str] = {
    "model_free_ppo": "PPO\n(baseline)",
    "piml_A_critic": "A\n(critic)",
    "piml_B_cbf": "B\n(CBF)",
    "piml_C_action": "C\n(penalty)",
    "piml_full_A_B_C": "A+B+C\n(full)",
}


def _mean_std(values: List[float]) -> Dict[str, float]:
    """Population mean / std / count over a list of samples (returns a plain dict)."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


def aggregate_runs(per_run: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group per-run metrics by condition into mean/std summaries (pure, testable).

    Args:
        per_run: records with keys ``condition``, ``seed``, ``mean_final_return``,
            ``mean_action_violation``, ``total_episodes``, ``total_real_steps`` and
            ``training_time_seconds``.

    Returns:
        ``{"conditions": [...], <condition>: {metric: {mean,std,n}, "seed_returns": {...}}}``.
    """
    conditions = sorted({str(r["condition"]) for r in per_run})
    summary: Dict[str, Any] = {"conditions": conditions}
    for cond in conditions:
        runs = [r for r in per_run if r["condition"] == cond]
        summary[cond] = {
            "mean_final_return": _mean_std([float(r["mean_final_return"]) for r in runs]),
            "mean_action_violation": _mean_std([float(r["mean_action_violation"]) for r in runs]),
            "total_episodes": _mean_std([float(r["total_episodes"]) for r in runs]),
            "training_time_seconds": _mean_std([float(r["training_time_seconds"]) for r in runs]),
            "seed_returns": {int(r["seed"]): float(r["mean_final_return"]) for r in runs},
        }
    return summary


def compare_vs_baseline(
    summary: Dict[str, Any], baseline: str = BASELINE
) -> Dict[str, Dict[str, float]]:
    """Relative-to-baseline effect of every other condition (return + violation)."""
    if baseline not in summary:
        return {}
    base_ret = float(summary[baseline]["mean_final_return"]["mean"])
    base_vio = float(summary[baseline]["mean_action_violation"]["mean"])
    comparison: Dict[str, Dict[str, float]] = {}
    for cond in summary["conditions"]:
        if cond == baseline:
            continue
        ret = float(summary[cond]["mean_final_return"]["mean"])
        vio = float(summary[cond]["mean_action_violation"]["mean"])
        comparison[cond] = {
            "return_delta": ret - base_ret,
            "return_change_pct": (100.0 * (ret - base_ret) / abs(base_ret)) if base_ret else 0.0,
            "violation_delta": vio - base_vio,
            "violation_change_pct": (100.0 * (vio - base_vio) / base_vio) if base_vio else 0.0,
        }
    return comparison


def _render_figure(summary: Dict[str, Any], path: str) -> None:
    conds = [c for c in summary["conditions"]]
    x = list(range(len(conds)))
    colors = ["#95a5a6"] + ["#2ecc71"] * (len(conds) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ret_means = [summary[c]["mean_final_return"]["mean"] for c in conds]
    ret_errs = [summary[c]["mean_final_return"]["std"] for c in conds]
    ax1.bar(x, ret_means, yerr=ret_errs, capsize=5, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PLOT_LABELS.get(c, c) for c in conds])
    ax1.set_ylabel("Mean final return (last 20 episodes)")
    ax1.set_title("Return (mean +/- std over seeds)", fontweight="bold")

    vio_means = [summary[c]["mean_action_violation"]["mean"] for c in conds]
    vio_errs = [summary[c]["mean_action_violation"]["std"] for c in conds]
    ax2.bar(x, vio_means, yerr=vio_errs, capsize=5, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels([PLOT_LABELS.get(c, c) for c in conds])
    ax2.set_ylabel("Mean executed-action physics violation")
    ax2.set_title("Physics-violation of executed actions", fontweight="bold")

    fig.suptitle("PIML-MFRL per-mechanism ablation on authentic SNES hardware", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)


def run_study(
    seeds: List[int],
    total_timesteps: int,
    output_dir: str = RESULTS_DIR,
    conditions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run every (condition, seed) pair on the console and write the aggregate artifact."""
    wanted = conditions or list(CONDITIONS)
    per_run: List[Dict[str, Any]] = []
    for cond_name in wanted:
        flags = CONDITIONS[cond_name]
        for seed in seeds:
            logger.info("=== %s | seed %d | %d steps ===", cond_name, seed, total_timesteps)
            metrics = train_piml_mfrl(
                env=None,
                total_timesteps=total_timesteps,
                seed=seed,
                output_dir=output_dir,
                write_artifact=False,
                **flags,
            )
            per_run.append(
                {
                    "condition": cond_name,
                    "seed": seed,
                    "mean_final_return": float(metrics["mean_final_return"]),
                    "mean_action_violation": float(metrics["mean_action_violation"]),
                    "total_episodes": int(metrics["total_episodes"]),
                    "total_real_steps": int(metrics["total_real_steps"]),
                    "training_time_seconds": float(metrics["training_time_seconds"]),
                }
            )

    summary = aggregate_runs(per_run)
    payload: Dict[str, Any] = {
        "study": "piml_mfrl_ablation",
        "seeds": seeds,
        "total_timesteps_per_run": total_timesteps,
        "baseline": BASELINE,
        "conditions": {k: CONDITIONS[k] for k in wanted},
        "summary": summary,
        "comparison_vs_baseline": compare_vs_baseline(summary),
        "per_run": per_run,
    }

    artifact_path = results_file("piml_mfrl_metrics.json")
    write_metrics(
        artifact_path,
        payload,
        seed=seeds[0] if seeds else None,
        command="python -m src.evaluation.piml_mfrl_study",
        extra_meta={"seeds": seeds, "total_timesteps_per_run": total_timesteps},
    )
    logger.info("Study artifact written to: %s", artifact_path)

    figure_path = figure_file("piml_mfrl_comparison.png")
    _render_figure(summary, figure_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default="42,43,44", help="comma-separated seed list")
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=10000)
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    parser.add_argument(
        "--conditions",
        default="",
        help="optional comma-separated subset of conditions (default: all five)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    conditions = [c for c in str(args.conditions).split(",") if c.strip()] or None
    run_study(
        seeds=seeds,
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        conditions=conditions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
