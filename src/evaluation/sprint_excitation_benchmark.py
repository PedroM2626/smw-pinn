"""
sprint_excitation_benchmark.py
Does the ceiling become measurable when the data is collected for it? (README Section 10.45)

Section 10.43.9 ended on a data statement rather than an estimator statement: on the
published gameplay recording, *zero* held-out transitions reach 90% of the training speed
support, so the frames a rigid clamp could be identified from simply do not exist in the
evaluation split. That is a claim about the collection policy, and the repository owns the
collection rig, so the claim is testable: ``scripts/record_sprint_gameplay.py`` records a
second dataset whose only purpose is to run Mario fast and far, and this study runs the same
three engines over both recordings under the same protocol.

Three questions are answered per dataset, in order of how much they cost:

1. **Excitation profile** - the observed speed support and the fraction of held-out frames
   at or above 90% / 95% of it. This is the quantity the ceiling is identified from.
2. **Structure recovery** - the nested-template engine, whose clamp level is the estimate of
   the ceiling, with BIC / held-out RMSE / tail criteria reported side by side.
3. **Search recovery** - gplearn and (when installed) PySR on the identical design, so the
   comparison with 10.43.9 is a replication rather than a new measurement.

If the ceiling is a property of the console, more excitation must move the estimate toward
it and must not change the sign of the conclusion. If the two recordings disagree, the
disagreement is itself the finding, and it is reported as such: the WRAM reference constant
of 72.0 sub-pixels/frame has never been reproduced from telemetry by any estimator in this
repository, and this study is the one designed to test whether that is a measurement
problem or a reference problem.

Requires the Libretro core and a ROM dump for the recording step (README 11.2); the analysis
itself only needs the two ``.npz`` files. Writes
``results/sprint_excitation_metrics.json`` with ``_meta`` and
``results/figures/sprint_excitation_profile.png``.

Run:  python -m src.evaluation.sprint_excitation_benchmark
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.environment.dataset_loader import load_and_preprocess_data  # noqa: E402
from src.evaluation.inverse_transfer_benchmark import PRIOR  # noqa: E402
from src.evaluation.symbolic_engine_ablation_benchmark import (  # noqa: E402
    pysr_available,
    run_gplearn,
    run_pysr,
    run_templates,
)
from src.inverse.parameter_identification import PARAM_NAMES  # noqa: E402
from src.inverse.symbolic_regression import bank_from_transitions, truth_vector  # noqa: E402
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import DATASET_GAMEPLAY, DATASET_SPRINT  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

ARTIFACT_NAME = "sprint_excitation_metrics.json"
FIGURE_NAME = "sprint_excitation_profile.png"
GP_BUDGET = {
    "population_size": 500,
    "generations": 25,
    "max_train": 4000,
    "parsimony_coefficient": 1e-3,
    "n_jobs": 1,
}


def _output_file(output_dir: str, name: str, subdir: str = "") -> str:
    """Honour --output-dir so a smoke run can never overwrite a published artifact."""
    path = os.path.join(output_dir, subdir, name) if subdir else os.path.join(output_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def excitation_profile(
    states: np.ndarray, next_states: np.ndarray, support: float
) -> Dict[str, float]:
    """How much of the speed range the held-out frames actually visit."""
    speed = np.abs(np.asarray(next_states, dtype=np.float64)[:, 2])
    return {
        "max_abs_vx": float(speed.max()),
        "fraction_at_90pct_of_support": float(np.mean(speed >= 0.90 * support)),
        "fraction_at_95pct_of_support": float(np.mean(speed >= 0.95 * support)),
        "fraction_at_99pct_of_support": float(np.mean(speed >= 0.99 * support)),
        "p99_abs_vx": float(np.percentile(speed, 99)),
        "support_used": float(support),
        "train_transitions": int(states.shape[0]),
    }


def analyse_dataset(
    path: str,
    label: str,
    truth: np.ndarray,
    seed: int,
    gp_seeds: int,
    pysr_iterations: int,
    max_train: int,
) -> Dict[str, Any]:
    """Run the excitation profile and all three engines over one recording."""
    data = load_and_preprocess_data(dataset_path=path, seed=seed)
    fit_bank = bank_from_transitions(
        data["train_states"], data["train_actions"], data["train_next_states"]
    )
    eval_bank = bank_from_transitions(
        data["test_states"], data["test_actions"], data["test_next_states"]
    )
    support = float(np.abs(fit_bank.next_states[:, 2]).max())
    out: Dict[str, Any] = {
        "dataset": os.path.basename(path),
        "profile": excitation_profile(fit_bank.states, fit_bank.next_states, support),
        "heldout_profile": excitation_profile(eval_bank.states, eval_bank.next_states, support),
    }
    out["templates"] = run_templates(
        fit_bank.states,
        fit_bank.actions,
        fit_bank.next_states,
        fit_bank.contact,
        eval_bank.states,
        eval_bank.actions,
        eval_bank.next_states,
        eval_bank.contact,
        truth,
        with_contact=True,
    )
    out["gplearn"] = run_gplearn(
        fit_bank.states,
        fit_bank.actions,
        fit_bank.next_states,
        fit_bank.contact,
        eval_bank,
        truth,
        tuple(range(gp_seeds)),
        GP_BUDGET,
        preset="real",
    )
    out["pysr"] = (
        run_pysr(fit_bank, eval_bank, truth, pysr_iterations, max_train, seed, preset="real")
        if pysr_available()
        else {"available": False, "reason": "the pysr package is not installed"}
    )
    logger.info(
        "  %-18s support %.2f | held-out >=90%%: %.4f | clamp %.3f (BIC %s / tail %s)",
        label,
        support,
        out["heldout_profile"]["fraction_at_90pct_of_support"],
        out["templates"]["structure"]["bound_value"],
        out["templates"]["horizontal"]["bic_selected"],
        out["templates"]["horizontal"]["tail_rmse_selected"],
    )
    return out


def _clamp_of(block: Dict[str, Any]) -> float:
    value = block.get("structure", {}).get("bound_value")
    return float("nan") if value is None else float(value)


def build_verdict(results: Dict[str, Dict[str, Any]], reference: float) -> Dict[str, Any]:
    """Compare the two recordings on excitation and on what the engines then recovered."""
    labels = list(results)
    held = {
        label: float(results[label]["heldout_profile"]["fraction_at_90pct_of_support"])
        for label in labels
    }
    clamps = {label: _clamp_of(results[label]["templates"]) for label in labels}
    support = {label: float(results[label]["profile"]["max_abs_vx"]) for label in labels}
    better_excited = max(labels, key=lambda label: held[label])
    newly_measurable = bool(
        held[better_excited] > 0.0 and min(held.values()) < held[better_excited]
    )
    return {
        "heldout_binding_fraction": held,
        "observed_support": support,
        "template_clamp_estimate": clamps,
        "wram_reference_max_vx": float(reference),
        "best_excited_dataset": better_excited,
        "ceiling_became_measurable": bool(newly_measurable),
        "reading": (
            "Held-out frames at or above 90% of the speed support: "
            + ", ".join(f"{label} {held[label] * 100:.2f}%" for label in labels)
            + f". The clamp estimate moves from {_clamp_of(results[labels[0]]['templates']):.3f} "
            f"to {_clamp_of(results[labels[-1]]['templates']):.3f} between the two recordings, "
            f"against a WRAM reference of {reference:.1f}. "
            + (
                "More excitation therefore does move the estimate, so the ceiling is a "
                "measurement problem rather than a reference problem."
                if abs(_clamp_of(results[labels[-1]]["templates"]) - reference)
                < abs(_clamp_of(results[labels[0]]["templates"]) - reference)
                else "More excitation does not move the estimate toward the WRAM reference, so "
                "the disagreement with the 72.0 constant is not explained by coverage."
            )
        ),
    }


def _render_figure(results: Dict[str, Dict[str, Any]], path: str) -> str:
    labels = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    colors = ["#7f7f7f", "#d62728"]
    for index, label in enumerate(labels):
        data = np.load(results[label]["dataset_file"], allow_pickle=False)
        speed = np.abs(np.asarray(data["next_states"], dtype=np.float64)[:, 2])
        ax1.hist(speed, bins=40, alpha=0.55, label=label, color=colors[index % len(colors)])
    ax1.set_xlabel("|v_x| (sub-pixels per frame)")
    ax1.set_ylabel("transitions")
    ax1.set_title("Where the speed actually lives", fontweight="bold")
    ax1.legend(fontsize=8)
    frac = [results[k]["heldout_profile"]["fraction_at_90pct_of_support"] for k in labels]
    clamp = [_clamp_of(results[k]["templates"]) for k in labels]
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, np.asarray(frac) * 100.0, width=0.38, color="#1f77b4", label="held-out >=90%")
    ax2.bar(x + 0.2, clamp, width=0.38, color="#ff7f0e", label="clamp estimate (px)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title("Excitation and what it bought", fontweight="bold")
    ax2.legend(fontsize=8)
    fig.suptitle(
        "Section 10.45 - the velocity ceiling under an excitation-targeted recording",
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)
    return path


def run_study(
    gameplay_dataset: str = DATASET_GAMEPLAY,
    sprint_dataset: str = DATASET_SPRINT,
    output_dir: str = "results",
    seed: int = 42,
    gp_seeds: int = 3,
    pysr_iterations: int = 40,
    max_train: int = 4000,
) -> Dict[str, Any]:
    """Measure the ceiling twice: on the published recording and on the targeted one."""
    set_global_seed(seed)
    truth = truth_vector(PRIOR)
    paths = {"published_gameplay": gameplay_dataset}
    if os.path.isfile(sprint_dataset):
        paths["sprint_targeted"] = sprint_dataset
    else:
        logger.warning(
            "%s not found - recording it is step 1 of section 11.5 command 50 "
            "(python scripts/record_sprint_gameplay.py). Only the published dataset is "
            "analysed, and the study reports that no comparison was possible.",
            sprint_dataset,
        )
    results: Dict[str, Dict[str, Any]] = {}
    for label, path in paths.items():
        block = analyse_dataset(path, label, truth, seed, gp_seeds, pysr_iterations, max_train)
        block["dataset_file"] = path
        results[label] = block
    verdict = (
        build_verdict(results, float(truth[0]))
        if len(results) > 1
        else {
            "available": False,
            "reason": "only one recording was available, so no excitation comparison exists",
        }
    )
    payload: Dict[str, Any] = {
        "study": (
            "The velocity ceiling measured twice: on the published gameplay recording, whose "
            "held-out frames never approach the speed support, and on a recording made for the "
            "purpose of saturating it. The question is whether 10.43.9's 'the ceiling is not "
            "identifiable here' is a statement about the console or about the collection policy."
        ),
        "protocol": {
            "split": "the canonical seeded episodic split of each file (seed 42)",
            "feature_design": "real (vx, dir, run, ground, ceiling, left, right | vy, jump, ground, ...)",
            "gp_seeds": list(range(gp_seeds)),
            "max_train_transitions": max_train,
            "pysr_iterations": pysr_iterations,
            "recording_policy_for_sprint_dataset": (
                "the 10.37 established-rules MPC, so coverage is competent but every recorded "
                "transition is read back from WRAM after the frame"
            ),
            "seed": seed,
        },
        "wram_reference": {n: float(truth[i]) for i, n in enumerate(PARAM_NAMES)},
        "datasets": results,
        "verdict": verdict,
    }
    artifact = _output_file(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.sprint_excitation_benchmark",
        extra_meta={"datasets": list(results)},
    )
    logger.info("Artifact written to: %s", artifact)
    figure = _render_figure(results, _output_file(output_dir, FIGURE_NAME, "figures"))
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--gameplay-dataset", dest="gameplay_dataset", default=DATASET_GAMEPLAY)
    parser.add_argument("--sprint-dataset", dest="sprint_dataset", default=DATASET_SPRINT)
    parser.add_argument("--gp-seeds", dest="gp_seeds", type=int, default=3)
    parser.add_argument("--pysr-iterations", dest="pysr_iterations", type=int, default=40)
    parser.add_argument("--max-train", dest="max_train", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    if not os.path.isfile(args.sprint_dataset):
        logger.error(
            "Neither recording is present: %s missing. Record it first with "
            "python scripts/record_sprint_gameplay.py (needs the Libretro core and ROM).",
            args.sprint_dataset,
        )
        return 1
    run_study(
        gameplay_dataset=args.gameplay_dataset,
        sprint_dataset=args.sprint_dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        gp_seeds=args.gp_seeds,
        pysr_iterations=args.pysr_iterations,
        max_train=args.max_train,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
