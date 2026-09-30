"""
gate_excitation_benchmark.py
Does the held-jump gravity gate become measurable when the data is collected for it?

The one result this repository has never reversed is about the vertical tier: the engine
integrates ``g_hold`` while the jump button is held during ascent and ``g_fall`` otherwise,
a step of about 3 sub-pixels/frame^2, and no method has recovered it from telemetry.
Section 10.43.9 found neither tree search recovering it on the published recording;
Section 10.45 found PySR recovering it on the hidden world but not on real data; Section
10.47 found no learned model, in any of fifteen cells with any of three mechanisms, coming
closer than 0.86 of a px/frame to the step - and the strongest hint about why came from
Section 10.37.1, which showed the impulse is not identifiable from single transitions.

That is a coverage hypothesis, and 10.45 supplies the method for testing one: collect data
whose policy visits the missing region. The region here is narrow and specific - a frame
where Mario is **rising with the jump button already released**, the branch that separates
the two gravity tiers - and a policy that holds the button through the apex never produces
it. `scripts/record_jump_gameplay.py` records exactly that, and this study measures the
gate on the three recordings side by side: the published gameplay, the sprint excitation of
10.45, and the new jump-targeted one.

Two things are reported per recording. First the *strata*, measured directly from the
console: how many transitions fall in each of {ascending with the button held, ascending
with it released, descending} and what the median vertical increment is in each. That is
the gate as the data shows it, before any estimator. Second the three engines of 10.43.9,
run unchanged, so a recovery here is a replication-grade comparison and not a new
instrument.

Writes ``results/gate_excitation_metrics.json``. Needs the recorded ``.npz`` files only; the
recording step needs the Libretro core and ROM of Section 11.2.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.inverse_transfer_benchmark import PRIOR
from src.evaluation.sprint_excitation_benchmark import GP_BUDGET, analyse_dataset
from src.inverse.parameter_identification import PARAM_NAMES
from src.inverse.symbolic_regression import truth_vector
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, DATASET_JUMP, DATASET_SPRINT, RESULTS_DIR
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

ARTIFACT_NAME = "gate_excitation_metrics.json"
FIGURE_NAME = "gate_excitation_profile.png"

RECORDINGS = (
    ("published_gameplay", DATASET_GAMEPLAY),
    ("sprint_targeted", DATASET_SPRINT),
    ("jump_targeted", DATASET_JUMP),
)


def gate_strata(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> Dict[str, Any]:
    """The gravity tiers as the console shows them, counted and measured per stratum."""
    vy = np.asarray(next_states, dtype=np.float64)[:, 3]
    vy_prev = np.asarray(states, dtype=np.float64)[:, 3]
    jump = np.asarray(actions, dtype=np.float64)[:, 0] > 0.5
    ascent = vy_prev < 0.0
    delta = vy - vy_prev

    strata = {
        "ascent_held": ascent & jump,
        "ascent_released": ascent & ~jump,
        "descent": ~ascent,
    }
    out: Dict[str, Any] = {}
    for name, mask in strata.items():
        count = int(mask.sum())
        out[name] = {
            "transitions": count,
            "fraction": float(mask.mean()),
            "median_delta_vy": float(np.median(delta[mask])) if count else None,
            "mean_delta_vy": float(np.mean(delta[mask])) if count else None,
        }
    held = out["ascent_held"]["median_delta_vy"]
    released = out["ascent_released"]["median_delta_vy"]
    out["measured_tier_separation"] = (
        None if held is None or released is None else float(released - held)
    )
    out["both_tiers_present"] = bool(
        out["ascent_held"]["transitions"] >= 50 and out["ascent_released"]["transitions"] >= 50
    )
    return out


def analyse_recording(
    label: str,
    path: str,
    truth: np.ndarray,
    seed: int,
    gp_seeds: int,
    pysr_iterations: int,
    max_train: int,
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Strata from the console, then the three engines of 10.43.9 on the same split."""
    if not os.path.isfile(path):
        return {"available": False, "reason": f"{path} not recorded yet"}
    block: Dict[str, Any] = {"dataset": os.path.basename(path)}
    from src.environment.dataset_loader import load_and_preprocess_data

    data = load_and_preprocess_data(dataset_path=path, seed=seed)
    block["strata"] = gate_strata(
        data["train_states"], data["train_actions"], data["train_next_states"]
    )
    block["heldout_strata"] = gate_strata(
        data["test_states"], data["test_actions"], data["test_next_states"]
    )
    block.update(
        analyse_dataset(
            path=path,
            label=label,
            truth=truth,
            seed=seed,
            gp_seeds=gp_seeds,
            pysr_iterations=pysr_iterations,
            max_train=max_train,
            budget=budget,
        )
    )
    return block


def build_verdict(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare coverage against recovery, with every number taken from the blocks."""
    rows: Dict[str, Any] = {}
    for label, block in results.items():
        if not block.get("available", True):
            continue
        strata = block["strata"]
        structure = block.get("templates", {}).get("structure", {})

        def _gate(engine_key: str) -> Optional[bool]:
            engine = block.get(engine_key) or {}
            if not engine.get("available", True) and engine_key != "templates":
                return None
            inner = engine.get("structure", {})
            return bool(inner.get("gravity_gate_discovered"))

        rows[label] = {
            "ascent_released_transitions": strata["ascent_released"]["transitions"],
            "ascent_released_fraction": strata["ascent_released"]["fraction"],
            "measured_tier_separation": strata["measured_tier_separation"],
            "template_gate_recovered": _gate("templates"),
            "template_tier_separation": structure.get("tier_separation"),
            "pysr_gate_recovered": _gate("pysr"),
            "gplearn_gate_recovered": _gate("gplearn"),
        }
    if not rows:
        return {"per_recording": {}, "reading": "no recording was available to analyse"}
    with_gate = [label for label, row in rows.items() if row["template_gate_recovered"]]
    thin = [label for label, row in rows.items() if (row["ascent_released_transitions"] or 0) < 50]
    measured = [
        row["measured_tier_separation"]
        for row in rows.values()
        if row["measured_tier_separation"] is not None
    ]
    return {
        "per_recording": rows,
        "recordings_where_the_template_recovers_the_gate": with_gate,
        "recordings_without_the_released_ascent_branch": thin,
        "measured_separation_range": [min(measured), max(measured)] if measured else None,
        "reading": (
            "The released-ascent branch carries "
            + ", ".join(
                f"{rows[label]['ascent_released_transitions']} of {label} transitions"
                f" ({rows[label]['ascent_released_fraction']:.1%})"
                for label in rows
            )
            + "; the console's own median tier separation is "
            + ", ".join(
                f"{rows[label]['measured_tier_separation']:+.3f}"
                for label in rows
                if rows[label]["measured_tier_separation"] is not None
            )
            + f". The template engine recovers the gate on {with_gate or 'no recording'}"
            + (
                f", and the recordings that lack the branch entirely are {thin}."
                if thin
                else ", and every recording now contains the branch."
            )
        ),
    }


def _render_figure(results: Dict[str, Dict[str, Any]], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k, v in results.items() if v.get("available", True)]
    if not labels:
        return False
    widths = [results[k]["strata"]["ascent_released"]["fraction"] for k in labels]
    medians = [results[k]["strata"]["ascent_released"]["median_delta_vy"] or 0.0 for k in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(range(len(labels)), widths, color="#f6c445")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("fraction of transitions: rising, button released")
    ax2.bar(range(len(labels)), medians, color="#41d19a")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel(r"median $\Delta v_y$ in that stratum (sub-px/frame$^2$)")
    ax2.axhline(0.0, color="#888888", linewidth=0.8)
    fig.suptitle("Excitation of the held-jump gravity gate (README 10.52)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def run_gate_excitation_benchmark(
    output_dir: str = RESULTS_DIR,
    seed: int = 42,
    gp_seeds: int = 3,
    pysr_iterations: int = 120,
    max_train: int = 4000,
    recordings: tuple = RECORDINGS,
    only: Optional[Sequence[str]] = None,
    gp_population: int = 500,
    gp_generations: int = 25,
) -> Dict[str, Any]:
    """Measure the gate on every recording available."""
    if only:
        wanted = {str(name).strip() for name in only}
        recordings = tuple(pair for pair in recordings if pair[0] in wanted)
    results: Dict[str, Dict[str, Any]] = {}
    truth = truth_vector(PRIOR)
    budget = {"population_size": gp_population, "generations": gp_generations}
    for label, path in recordings:
        logger.info("=== Gate excitation on %s ===", label)
        results[label] = analyse_recording(
            label,
            path,
            np.asarray(truth),
            seed,
            gp_seeds,
            pysr_iterations,
            max_train,
            budget,
        )

    payload: Dict[str, Any] = {
        "study": (
            "Whether the held-jump gravity discontinuity becomes recoverable once the "
            "released-ascent branch is deliberately sampled, measured on three recordings "
            "with the engines of 10.43.9."
        ),
        "protocol": {
            "seed": seed,
            "gp_seeds": gp_seeds,
            "gp_budget": list(GP_BUDGET),
            "pysr_iterations": pysr_iterations,
            "max_train": max_train,
            "recordings": {label: os.path.basename(path) for label, path in recordings},
            "strata": "ascent means v_y < 0 at the current frame; the jump tier is action channel 0",
        },
        "results": results,
        "verdict": build_verdict(results),
        "constants": {"param_names": list(PARAM_NAMES)},
    }

    if _render_figure(results, os.path.join(output_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.gate_excitation_benchmark",
        extra_meta={
            "recordings": [label for label, v in results.items() if v.get("available", True)]
        },
    )
    logger.info("Metrics written to %s", artifact)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--output-dir", dest="output_dir", default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gp-seeds", dest="gp_seeds", type=int, default=3)
    parser.add_argument("--pysr-iterations", dest="pysr_iterations", type=int, default=120)
    parser.add_argument("--max-train", dest="max_train", type=int, default=4000)
    parser.add_argument("--gp-population", dest="gp_population", type=int, default=500)
    parser.add_argument("--gp-generations", dest="gp_generations", type=int, default=25)
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated subset of the recordings to analyse (default: all present)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    run_gate_excitation_benchmark(
        output_dir=args.output_dir,
        seed=args.seed,
        gp_seeds=args.gp_seeds,
        pysr_iterations=args.pysr_iterations,
        max_train=args.max_train,
        only=[s.strip() for s in str(args.only).split(",") if s.strip()],
        gp_population=args.gp_population,
        gp_generations=args.gp_generations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
