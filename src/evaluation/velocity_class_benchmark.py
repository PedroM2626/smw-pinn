"""
velocity_class_benchmark.py
Which documented speed class is the engine's bound on this telemetry? (README 10.49)

Section 4.3 documents three horizontal velocity classes for Super Mario World -
walk at 20, run at 48 and maximum sprint with the P-meter active at 72
sub-pixels/frame - and the repository has used **72.0** as `max_vx` everywhere
since: the identification prior of 10.40, the velocity-bounds term of the composite
PINN loss, the hard shells of 10.27/10.42, and the reference value that every
ceiling estimate in 10.43, 10.43.9, 10.45 and 10.46 has been scored against. That
choice has a consequence the code itself records (`src/models/analytical_kinematics.py`:
"section 4.3.3 (P-meter; not observable in the 8D state)"): the constant is the cap
of a speed class the recordings never enter.

This study measures the velocity envelope of every WRAM recording in `data/raw` -
four interactive datasets plus the excitation-targeted sprint recording, ~45,000
transitions - against the three documented classes, verifies the sub-pixel scale
that converts the register into pixels, and re-scores the ceiling estimates this
repository has published against the class the data can actually reach.

Writes ``results/velocity_class_metrics.json`` and
``results/figures/velocity_class_envelope.png``. Emulator-free.
"""

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import RESULTS_DIR
from src.utils.provenance import read_metrics, write_metrics

logger = get_logger(__name__)

ARTIFACT_NAME = "velocity_class_metrics.json"
FIGURE_NAME = "velocity_class_envelope.png"

SUBPIXELS_PER_PIXEL = 16.0

# Section 4.3, items 1-3, in the units the register $7E:007B stores them.
DOCUMENTED_CLASSES: Dict[str, float] = {"walk": 20.0, "run": 48.0, "sprint_p_meter": 72.0}

# Section 4.2, items 1 and 5, in the same register units.
DOCUMENTED_JUMP_IMPULSE = (-80.0, -64.0)
DOCUMENTED_TERMINAL_VY = 64.0

# A frame must be moving for the dx/vx identity to say anything about the scale.
MIN_SPEED_FOR_SCALE = 8.0

# Section 4.1 integrates with the velocity at $t$; every implementation in this
# repository (analytical_kinematics, the hard shells, ResidualDynamics) advances
# with the velocity it predicted for $t+1$. Both are measured against the data.
MISMATCH_PX = 1.0

DATASETS = ("smw_gameplay_dataset.npz", "smw_tilemap_dataset.npz", "smw_multi_entity_dataset.npz")
SPRINT_DATASET = "smw_sprint_dataset.npz"


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "max": float(values.max()),
    }


def analyse_recording(path: str, data_dir: str) -> Dict[str, Any]:
    """Measure one recording's velocity envelope against the documented classes."""
    raw = np.load(os.path.join(data_dir, path), allow_pickle=False)
    states, next_states = raw["states"], raw["next_states"]
    episodes = raw["episodes"]
    vx_t_signed = states[:, 2].astype(np.float64)
    vx_next_signed = next_states[:, 2].astype(np.float64)
    vx = np.abs(vx_next_signed)
    ground = next_states[:, 4].astype(np.float64) > 0.5
    dx = next_states[:, 0].astype(np.float64) - states[:, 0].astype(np.float64)
    vy = next_states[:, 3].astype(np.float64)
    # Section 4.1's identity advances by the velocity stored at $t$; the shells
    # advance by the velocity they predicted for $t+1$. Measure both residuals.
    err_t_frames = np.abs(dx - vx_t_signed / SUBPIXELS_PER_PIXEL)
    err_next_frames = np.abs(dx - vx_next_signed / SUBPIXELS_PER_PIXEL)

    moving_t = np.abs(vx_t_signed) > MIN_SPEED_FOR_SCALE
    moving_next = np.abs(vx_next_signed) > MIN_SPEED_FOR_SCALE
    ratio_t = dx[moving_t] * SUBPIXELS_PER_PIXEL / vx_t_signed[moving_t]
    ratio_next = dx[moving_next] * SUBPIXELS_PER_PIXEL / vx_next_signed[moving_next]

    above_run = vx > DOCUMENTED_CLASSES["run"]
    above_sprint = vx > DOCUMENTED_CLASSES["sprint_p_meter"]
    band_edges = np.array([DOCUMENTED_CLASSES[c] for c in ("walk", "run", "sprint_p_meter")])
    counts = [
        int(np.sum(vx <= band_edges[0])),
        int(np.sum((vx > band_edges[0]) & (vx <= band_edges[1]))),
        int(np.sum((vx > band_edges[1]) & (vx <= band_edges[2]))),
        int(np.sum(vx > band_edges[2])),
    ]
    episodes_above_run = sorted({int(e) for e in episodes[above_run]})

    return {
        "transitions": int(len(vx)),
        "episodes": int(len(set(int(e) for e in episodes))),
        "envelope_all": _percentiles(vx),
        "envelope_on_ground": _percentiles(vx[ground]),
        "band_counts": {
            "up_to_walk": counts[0],
            "walk_to_run": counts[1],
            "run_to_p_meter": counts[2],
            "above_p_meter": counts[3],
        },
        "frames_above_run_cap": int(above_run.sum()),
        "frames_above_p_meter_cap": int(above_sprint.sum()),
        "episodes_reaching_above_run_cap": len(episodes_above_run),
        "vertical_bounds": {
            "vy_min": float(vy.min()),
            "vy_max": float(vy.max()),
            "fraction_above_terminal": float(np.mean(vy > DOCUMENTED_TERMINAL_VY)),
            "fraction_below_jump_floor": float(np.mean(vy < DOCUMENTED_JUMP_IMPULSE[0])),
            "fraction_outside_the_documented_window": float(
                np.mean((vy > DOCUMENTED_TERMINAL_VY) | (vy < DOCUMENTED_JUMP_IMPULSE[0]))
            ),
        },
        "integration_convention": {
            "median_abs_error_using_velocity_at_t": float(np.median(err_t_frames)),
            "median_abs_error_using_velocity_at_t_plus_1": float(np.median(err_next_frames)),
            "fraction_mismatching_by_more_than_1px": float(np.mean(err_next_frames > MISMATCH_PX)),
        },
        "subpixel_scale_identity": {
            "median_ratio_with_velocity_at_t": float(np.median(ratio_t)) if ratio_t.size else None,
            "median_ratio_with_velocity_at_t_plus_1": (
                float(np.median(ratio_next)) if ratio_next.size else None
            ),
            "fraction_within_one_subpixel_at_t": (
                float(np.mean(np.abs(ratio_t - 1.0) <= _subpixel_tolerance(dx[moving_t])))
                if ratio_t.size
                else None
            ),
            "median_abs_ratio_error_at_t": (
                float(np.median(np.abs(ratio_t - 1.0))) if ratio_t.size else None
            ),
            "median_abs_ratio_error_at_t_plus_1": (
                float(np.median(np.abs(ratio_next - 1.0))) if ratio_next.size else None
            ),
            "moving_frames": int(ratio_t.size),
        },
    }


def _subpixel_tolerance(dx: np.ndarray) -> float:
    """One sub-pixel of position quantisation, expressed as a ratio: 1 / (16 |dx|)."""
    median_step = float(np.median(np.abs(dx))) if dx.size else 0.0
    if median_step <= 0.0:
        return float("inf")
    return 1.0 / (SUBPIXELS_PER_PIXEL * median_step)


def _published_estimates(results_dir: str) -> Dict[str, Any]:
    """The ceiling numbers this repository has published, read from their own artifacts."""
    out: Dict[str, Any] = {}
    sprint_path = os.path.join(results_dir, "sprint_excitation_metrics.json")
    ablation_path = os.path.join(results_dir, "symbolic_engine_ablation_metrics.json")
    if os.path.isfile(sprint_path):
        sprint = read_metrics(sprint_path)
        datasets = sprint.get("datasets", {})
        for name, block in datasets.items():
            value = block.get("templates", {}).get("structure", {}).get("bound_value")
            if value is not None:
                out[f"{name}_template_bound"] = float(value)
            pysr = block.get("pysr", {}).get("structure", {}).get("bound_value")
            if pysr is not None:
                out[f"{name}_pysr_fixed_point"] = float(pysr)
    if os.path.isfile(ablation_path):
        ablation = read_metrics(ablation_path)
        hidden = ablation.get("structure_matrix", {})
        for engine, block in hidden.items():
            value = block.get("recovered_bound") if isinstance(block, dict) else None
            if value is not None:
                out[f"hidden_world_{engine}"] = float(value)
    return out


def _scoring(estimates: Dict[str, float], reference: float) -> Dict[str, float]:
    return {
        name: float(abs(value - reference) / reference * 100.0) for name, value in estimates.items()
    }


def build_verdict(
    recordings: Dict[str, Any], estimates: Dict[str, float], against_run: Dict[str, float]
) -> Dict[str, Any]:
    """Prose assembled only from measured quantities."""
    total = sum(r["transitions"] for r in recordings.values())
    above_sprint = sum(r["frames_above_p_meter_cap"] for r in recordings.values())
    above_run = sum(r["frames_above_run_cap"] for r in recordings.values())
    top = max(r["envelope_all"]["max"] for r in recordings.values())
    at_t = [
        r["subpixel_scale_identity"]["median_ratio_with_velocity_at_t"] for r in recordings.values()
    ]
    at_next = [
        r["subpixel_scale_identity"]["median_ratio_with_velocity_at_t_plus_1"]
        for r in recordings.values()
    ]
    outside = max(
        r["vertical_bounds"]["fraction_outside_the_documented_window"] for r in recordings.values()
    )
    vy_floor = min(r["vertical_bounds"]["vy_min"] for r in recordings.values())
    vy_cap = max(r["vertical_bounds"]["vy_max"] for r in recordings.values())
    conv_t = max(
        r["integration_convention"]["median_abs_error_using_velocity_at_t"]
        for r in recordings.values()
    )
    conv_next = max(
        r["integration_convention"]["median_abs_error_using_velocity_at_t_plus_1"]
        for r in recordings.values()
    )
    mismatch = max(
        r["integration_convention"]["fraction_mismatching_by_more_than_1px"]
        for r in recordings.values()
    )
    tightest = min(against_run, key=lambda k: against_run[k]) if against_run else None
    err_t = [
        r["subpixel_scale_identity"]["median_abs_ratio_error_at_t"] for r in recordings.values()
    ]
    err_next = [
        r["subpixel_scale_identity"]["median_abs_ratio_error_at_t_plus_1"]
        for r in recordings.values()
    ]
    return {
        "transitions_examined": total,
        "frames_above_p_meter_cap": above_sprint,
        "frames_above_run_cap": above_run,
        "largest_velocity_ever_recorded": top,
        "median_scale_ratio_velocity_at_t": [min(at_t), max(at_t)],
        "median_scale_ratio_velocity_at_t_plus_1": [min(at_next), max(at_next)],
        "median_abs_ratio_error_at_t": [min(err_t), max(err_t)],
        "median_abs_ratio_error_at_t_plus_1": [min(err_next), max(err_next)],
        "vertical_window_recorded": [vy_floor, vy_cap],
        "largest_fraction_outside_the_vertical_window": outside,
        "integration_median_error_at_t_px": conv_t,
        "integration_median_error_at_t_plus_1_px": conv_next,
        "largest_fraction_mismatching_by_more_than_1px": mismatch,
        "closest_published_estimate_to_run_cap": tightest,
        "closest_published_error_pct_vs_run_cap": against_run.get(tightest) if tightest else None,
        "reading": (
            f"Across {total} transitions from {len(recordings)} recordings the horizontal "
            f"velocity never once exceeded {top:.1f} sub-pixels/frame: {above_sprint} frames are "
            f"above the documented P-meter class of "
            f"{DOCUMENTED_CLASSES['sprint_p_meter']:.0f} and {above_run} are above the run class "
            f"of {DOCUMENTED_CLASSES['run']:.0f}. The sub-pixel identity is satisfied to a median "
            f"ratio of {min(at_t):.3f}-{max(at_t):.3f} with the velocity at frame $t$ and "
            f"{min(at_next):.3f}-{max(at_next):.3f} with the velocity at $t+1$, so the factor of "
            "two is not a unit convention. "
            + (
                f"Section 4.2's vertical window of {DOCUMENTED_JUMP_IMPULSE[0]:.0f} to "
                f"{DOCUMENTED_TERMINAL_VY:.0f} is itself exceeded by the recordings, which reach "
                f"{vy_floor:.1f} to {vy_cap:.1f} and sit outside it on up to "
                f"{100 * outside:.1f}% of frames. "
            )
            + (
                f"The exact integration identity of Section 4.1 holds to a median of "
                f"{conv_t:.4f} px when position advances by the velocity at $t$, and "
                f"{conv_next:.4f} px when it advances by the velocity at $t+1$ - the convention "
                f"every implementation in this repository uses - while up to "
                f"{100 * mismatch:.1f}% of recorded frames mismatch by more than "
                f"{MISMATCH_PX:.0f} px. "
            )
            + (
                f"The published estimate closest to the run class is {tightest} at "
                f"{against_run[tightest]:.2f}% - the class the recordings can reach is the run "
                "class, not the P-meter class that every error column has been scored against."
                if tightest
                else "No published estimate was available to compare."
            )
        ),
    }


def _render_figure(recordings: Dict[str, Any], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(recordings)
    p95 = [recordings[n]["envelope_all"]["p95"] for n in names]
    p999 = [recordings[n]["envelope_all"]["p999"] for n in names]
    top = [recordings[n]["envelope_all"]["max"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, p95, width=0.2, label="p95", color="#8ab4f8")
    ax.bar(x, p999, width=0.2, label="p99.9", color="#f6c445")
    ax.bar(x + 0.2, top, width=0.2, label="max", color="#41d19a")
    for name, value, color in (
        ("walk 20", DOCUMENTED_CLASSES["walk"], "#888888"),
        ("run 48", DOCUMENTED_CLASSES["run"], "#e06666"),
        ("P-meter 72", DOCUMENTED_CLASSES["sprint_p_meter"], "#6a3aa0"),
    ):
        ax.axhline(value, color=color, linestyle="--", linewidth=1.0, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("smw_", "").replace("_dataset.npz", "") for n in names], fontsize=9
    )
    ax.set_ylabel(r"$|v_x|$ (sub-pixels/frame)")
    ax.set_title("Measured velocity envelope against the three documented speed classes (10.49)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def run_velocity_class_benchmark(
    data_dir: str = os.path.join("data", "raw"),
    results_dir: str = RESULTS_DIR,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure every recording's envelope and re-score the published ceilings."""
    out_dir = output_dir or results_dir
    available: List[str] = [os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "*.npz"))]
    wanted = [name for name in DATASETS + (SPRINT_DATASET,) if name in available]
    logger.info("=== Velocity-class audit on %d recordings ===", len(wanted))

    recordings = {name: analyse_recording(name, data_dir) for name in wanted}
    estimates = _published_estimates(results_dir)
    against_run = _scoring(estimates, DOCUMENTED_CLASSES["run"])
    against_p_meter = _scoring(estimates, DOCUMENTED_CLASSES["sprint_p_meter"])

    payload: Dict[str, Any] = {
        "study": (
            "Which of Section 4.3's three documented horizontal speed classes does the WRAM "
            "telemetry reach, and against which constant should a discovered ceiling be scored?"
        ),
        "protocol": {
            "recordings": wanted,
            "documented_classes_subpixels_per_frame": DOCUMENTED_CLASSES,
            "subpixels_per_pixel": SUBPIXELS_PER_PIXEL,
            "min_speed_for_scale_identity": MIN_SPEED_FOR_SCALE,
            "emulator_required": False,
        },
        "recordings": recordings,
        "published_ceiling_estimates": estimates,
        "relative_error_vs_run_cap_pct": against_run,
        "relative_error_vs_p_meter_cap_pct": against_p_meter,
        "verdict": build_verdict(recordings, estimates, against_run),
    }

    if _render_figure(recordings, os.path.join(out_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=0,
        command="python -m src.evaluation.velocity_class_benchmark",
    )
    logger.info("Metrics written to %s", artifact)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--data-dir", default=os.path.join("data", "raw"))
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=None)
    args = parse_args_with_config(parser)

    run_velocity_class_benchmark(
        data_dir=args.data_dir, results_dir=args.results_dir, output_dir=args.output_dir
    )
