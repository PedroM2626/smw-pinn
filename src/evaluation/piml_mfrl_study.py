"""
piml_mfrl_study.py
Multi-seed comparison study behind README Section 10.39 (PIML-MFRL).

Runs the *same* model-free PPO loop from ``src.training.piml_mfrl`` twice - once with
all physics mechanisms off (the canonical model-free baseline) and once with Approach A
(critic), B (CBF filter) and C (action penalty) all on - across a set of seeds, directly
on the authentic SNES console, then aggregates the result into a single artifact. The
comparison is what turns the implementation into a statement: it isolates the effect of
the physics coupling on (a) episodic return and (b) how often the deployed policy
executes a physically-infeasible action.

Writes ``results/piml_mfrl_metrics.json`` (with a ``_meta`` provenance block) and
``results/figures/piml_mfrl_learning_curves.png``.

Requires the Libretro core + ROM. Run:
    python -m src.evaluation.piml_mfrl_study --seeds 42,43,44 --total-timesteps 8000
"""

from __future__ import annotations

import argparse
import statistics
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.planning.mpc_planner import ACTION_MATRIX  # noqa: E402
from src.training.piml_mfrl import train_piml_mfrl  # noqa: E402
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import RESULTS_DIR, figure_file, results_file  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

logger = get_logger(__name__)

# The two conditions compared. "baseline" is model-free PPO with every physics term off;
# "piml_full" enables Approaches A + B + C together.
CONDITIONS: Dict[str, Dict[str, bool]] = {
    "model_free_ppo": {
        "use_physics_critic": False,
        "use_cbf_filter": False,
        "use_action_penalty": False,
    },
    "piml_mfrl_full": {
        "use_physics_critic": True,
        "use_cbf_filter": True,
        "use_action_penalty": True,
    },
}


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate_runs(per_run: List[Dict[str, float]]) -> Dict[str, object]:
    """Group per-run metrics by condition into mean/std summaries (pure, testable).

    Args:
        per_run: list of records, each with keys ``condition``, ``mean_final_return``,
            ``mean_action_violation``, ``total_episodes``, ``total_real_steps`` and
            ``training_time_seconds``.

    Returns:
        ``{condition: {metric: {mean, std, n}, runs: [...]}}`` plus the condition list.
    """
    conditions = sorted({r["condition"] for r in per_run})
    summary: Dict[str, object] = {"conditions": conditions}
    for cond in conditions:
        runs = [r for r in per_run if r["condition"] == cond]
        summary[cond] = {
            "mean_final_return": _mean_std([r["mean_final_return"] for r in runs]),
            "mean_action_violation": _mean_std([r["mean_action_violation"] for r in runs]),
            "total_episodes": _mean_std([float(r["total_episodes"]) for r in runs]),
            "training_time_seconds": _mean_std([r["training_time_seconds"] for r in runs]),
            "seed_returns": {int(r["seed"]): float(r["mean_final_return"]) for r in runs},
        }
    return summary


def _comparison(summary: Dict[str, object]) -> Dict[str, float]:
    base = summary["model_free_ppo"]
    piml = summary["piml_mfrl_full"]
    base_ret = base["mean_final_return"]["mean"]
    piml_ret = piml["mean_final_return"]["mean"]
    base_vio = base["mean_action_violation"]["mean"]
    piml_vio = piml["mean_action_violation"]["mean"]
    return {
        "return_delta": piml_ret - base_ret,
        "return_change_pct": (100.0 * (piml_ret - base_ret) / abs(base_ret)) if base_ret else 0.0,
        "violation_delta": piml_vio - base_vio,
        "violation_reduction_pct": (100.0 * (base_vio - piml_vio) / base_vio) if base_vio else 0.0,
    }


def _render_figure(summary: Dict[str, object], path: str) -> None:
    labels = ["model_free_ppo", "piml_mfrl_full"]
    title = {"model_free_ppo": "Model-Free PPO\n(baseline)", "piml_mfrl_full": "PIML-MFRL\n(A+B+C)"}

    def bars(metric: str):
        means = [summary[c][metric]["mean"] for c in labels]
        errs = [summary[c][metric]["std"] for c in labels]
        return means, errs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    m, e = bars("mean_final_return")
    ax1.bar(range(2), m, yerr=e, capsize=6, color=["#95a5a6", "#2ecc71"])
    ax1.set_xticks(range(2))
    ax1.set_xticklabels([title[c] for c in labels])
    ax1.set_ylabel("Mean final return (last 20 episodes)")
    ax1.set_title("Return (mean +/- std over seeds)", fontweight="bold")

    m, e = bars("mean_action_violation")
    ax2.bar(range(2), m, yerr=e, capsize=6, color=["#95a5a6", "#e74c3c"])
    ax2.set_xticks(range(2))
    ax2.set_xticklabels([title[c] for c in labels])
    ax2.set_ylabel("Mean executed-action physics violation")
    ax2.set_title("Physics-violation of executed actions", fontweight="bold")

    fig.suptitle("PIML-MFRL vs. Model-Free PPO on authentic SNES hardware", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)


def run_study(
    seeds: List[int],
    total_timesteps: int,
    output_dir: str = RESULTS_DIR,
) -> Dict[str, object]:
    per_run: List[Dict[str, float]] = []
    for cond_name, flags in CONDITIONS.items():
        for seed in seeds:
            logger.info("=== %s | seed %d ===", cond_name, seed)
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
    comparison = _comparison(summary)
    payload: Dict[str, object] = {
        "study": "piml_mfrl",
        "action_space_size": len(ACTION_MATRIX),
        "seeds": seeds,
        "total_timesteps_per_run": total_timesteps,
        "conditions": {k: v for k, v in CONDITIONS.items()},
        "summary": summary,
        "comparison": comparison,
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

    figure_path = figure_file("piml_mfrl_learning_curves.png")
    _render_figure(summary, figure_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default="42,43,44", help="comma-separated seed list")
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=8000)
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    run_study(seeds=seeds, total_timesteps=args.total_timesteps, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
