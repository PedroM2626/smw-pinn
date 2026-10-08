r"""
sindy_identification_benchmark.py
The standard estimator on this repository's inverse problem (README 10.56).

Sections 10.37, 10.43 and 10.44 identify the engine's law with three instruments: a
gradient-based identification of seven constants inside a hand-written template, a genetic
search over compositions, and PySR with numerically-optimised unbounded constants. Two of the
three are this project's own machinery, and every published negative about the held-jump
gravity gate is stated against them. The community-standard estimator for the same inverse
problem - SINDy, sparse identification of nonlinear dynamics (Brunton, Proctor and Kutz,
PNAS 2016) - has never been run on WRAM telemetry, and it needs no structure posited from the
documentation and no evolutionary budget.

`src/inverse/sindy_identification.py` implements it canonically: a degree-2 polynomial
dictionary over the two velocities, the four contact flags and the six action channels; ridge
regression of the one-frame increment against it; sequential thresholding of the smallest
coefficients with a re-fit on the surviving support until the support settles. The derivative
is a forward difference over one frame, so every coefficient is already in WRAM units -
sub-pixels per frame - and Section 4's constants can be compared without rescaling.

Two protocol choices are measured rather than assumed, and both turned out to matter:

* **Duplicate dictionary columns are pruned and the duplicates are reported.** `c_right
  c_ground` is a copy of `c_right` on this data because a right contact only occurs on the
  floor, and ridge splits one coefficient across both: an unreadable law, not a wrong fit.
* **The loss is an axis.** A position increment has a large mean and rare, violent exceptions -
  wall stops, terminations, screen events - so the least-squares fit is owned by those frames.
  The same dictionary and the same thresholding with Huber weights answer a different question,
  and the two are reported side by side. This is the 10.49 lesson (median, not mean) applied to
  an identification method.

Writes ``results/sindy_identification_metrics.json``. Emulator-free, seconds per recording, and
every recording is read with the canonical split of `load_and_preprocess_data`, so the
transitions the identification sees are exactly the ones the models of 10.47/10.53/10.55 are
trained on.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment.dataset_loader import load_and_preprocess_data
from src.inverse.sindy_identification import (
    DYNAMIC_TARGETS,
    LIBRARY_INPUTS,
    SindyFit,
    evaluate_library,
    increments,
    library_inputs_of,
    library_terms,
    prune_columns,
    sindy,
    term_names,
)
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_GAMEPLAY,
    DATASET_JUMP,
    DATASET_MULTI_ENTITY,
    DATASET_SPRINT,
    DATASET_TILEMAP,
    RESULTS_DIR,
)
from src.utils.provenance import read_metrics, write_metrics

logger = get_logger(__name__)

ARTIFACT_NAME = "sindy_identification_metrics.json"
FIGURE_NAME = "sindy_identification.png"

RECORDINGS: Tuple[Tuple[str, str], ...] = (
    ("published_gameplay", DATASET_GAMEPLAY),
    ("tilemap", DATASET_TILEMAP),
    ("multi_entity", DATASET_MULTI_ENTITY),
    ("sprint_targeted", DATASET_SPRINT),
    ("jump_targeted", DATASET_JUMP),
)

DISPLAY = {
    "published_gameplay": "published gameplay",
    "tilemap": "tilemap (10.41)",
    "multi_entity": "multi-entity (10.31)",
    "sprint_targeted": "sprint-targeted (10.45)",
    "jump_targeted": "jump-targeted (10.52)",
}

# alpha is in the WRAM's own units: 0.05 means a term has to move the increment by five
# hundredths of a sub-pixel per frame to be kept.
ALPHA_SWEEP: Tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
SELECTED_ALPHA = 0.05
LOSSES: Tuple[str, ...] = ("least_squares", "robust")
MAX_DEGREE = 2

# Section 4's numbers, which 10.54 audits as text-and-code; here they are the reference an
# identification is scored against.
DOCUMENTED = {
    "subpixels_per_pixel": 16.0,
    "held_gravity": 3.0,
    "fall_gravity": 6.0,
}
IDENTIFICATION_ARTIFACT = "inverse_identification_metrics.json"


def _fmt(value: Optional[float], spec: str = ".3f", dash: str = "-") -> str:
    """Format a coefficient that a sparser dictionary may not contain at all."""
    return dash if value is None else format(value, spec)


def _coefficient(fit: SindyFit, target: str, term: str) -> Optional[float]:
    value = fit.coefficient_of(target, term)
    return value if term in [name for name, _ in fit.equations()[target]] else None


def next_state_r2(
    fit: SindyFit,
    theta: np.ndarray,
    states: np.ndarray,
    next_states: np.ndarray,
    mean_next: np.ndarray,
) -> Dict[str, float]:
    r"""$R^2$ of the identified law read forward one frame, against the recorded next state.

    The increments are what SINDy fits, but every other model in this repository is scored on
    the next state, so the two are only comparable through $s_t + \hat\Delta$. Without this
    row a low increment $R^2$ would read as a worse model than it is: the next velocity carries
    the current velocity, which the identification never had to predict.
    """
    predicted = np.asarray(states)[:, : len(DYNAMIC_TARGETS)] + theta @ fit.coefficients
    truth = np.asarray(next_states)[:, : len(DYNAMIC_TARGETS)]
    out: Dict[str, float] = {}
    for j, name in enumerate(fit.targets):
        total = float(np.sum((truth[:, j] - mean_next[j]) ** 2))
        out[name] = (
            float(1.0 - np.sum((predicted[:, j] - truth[:, j]) ** 2) / total)
            if total
            else float("nan")
        )
    total_pooled = float(np.sum((truth - mean_next) ** 2))
    out["pooled"] = (
        float(1.0 - np.sum((predicted - truth) ** 2) / total_pooled)
        if total_pooled
        else float("nan")
    )
    return out


def _position_scale(fit: SindyFit) -> Optional[float]:
    """The coefficient on $v_x$ in the $\\Delta x$ equation: the reciprocal sub-pixel scale."""
    return _coefficient(fit, "dx", "vx")


def _fit_one(
    label: str,
    path: str,
    seed: int,
    alpha: float,
    max_degree: int,
) -> Optional[Dict[str, Any]]:
    """Identify one recording under both losses, with the threshold sweep."""
    if not os.path.isfile(path):
        logger.warning("%s missing; the %s row is omitted.", path, label)
        return None
    data = load_and_preprocess_data(dataset_path=path, seed=seed)
    terms = library_terms(max_degree)
    names = term_names(terms)

    def pack(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = library_inputs_of(data[f"{split}_states"], data[f"{split}_actions"])
        theta = evaluate_library(inputs, terms)
        xdot = increments(data[f"{split}_states"], data[f"{split}_next_states"])[
            :, : len(DYNAMIC_TARGETS)
        ]
        return theta, xdot, inputs

    theta_train, xdot_train, _ = pack("train")
    theta_train, kept_names, dropped, keep = prune_columns(theta_train, names)
    theta_test_full, xdot_test, _ = pack("test")
    theta_test = theta_test_full[:, keep]
    held = {
        split: (
            np.asarray(data[f"{split}_states"], dtype=np.float64),
            np.asarray(data[f"{split}_next_states"], dtype=np.float64),
        )
        for split in ("train", "test")
    }
    mean_next = held["train"][1][:, : len(DYNAMIC_TARGETS)].mean(axis=0)
    actions = {
        split: np.asarray(data[f"{split}_actions"], dtype=np.float64) for split in ("train", "test")
    }
    stratum_source: Dict[str, SindyFit] = {}
    mean_train = xdot_train.mean(axis=0)

    out: Dict[str, Any] = {
        "dataset": os.path.basename(path),
        "n_train_transitions": int(theta_train.shape[0]),
        "n_test_transitions": int(theta_test.shape[0]),
        "dictionary_terms": len(names),
        "kept_terms": len(kept_names),
        "dropped_duplicates": dropped,
        "recovered_subpixels_per_pixel": {},
        "fits": {},
    }
    for loss in LOSSES:
        robust = loss == "robust"
        fit = sindy(
            xdot_train,
            theta_train,
            kept_names,
            alpha=alpha,
            targets=list(DYNAMIC_TARGETS),
            robust=robust,
        )
        scale = _position_scale(fit)
        block = {
            "alpha": fit.alpha,
            "iterations": fit.iterations,
            "sparsity": fit.sparsity(),
            "equations": {
                target: [
                    {"term": name, "coefficient": round(value, 6)} for name, value in terms_list
                ]
                for target, terms_list in fit.equations().items()
            },
            "train_relative_error": fit.relative_error(theta_train, xdot_train),
            "heldout_relative_error": fit.relative_error(theta_test, xdot_test),
            "train_r2": fit.variance_explained(theta_train, xdot_train, mean_train),
            "heldout_r2": fit.variance_explained(theta_test, xdot_test, mean_train),
            "heldout_next_state_r2": next_state_r2(
                fit, theta_test, held["test"][0], held["test"][1], mean_next
            ),
            "vertical_constant": _coefficient(fit, "dvy", "1"),
            "vertical_ground_coefficient": _coefficient(fit, "dvy", "c_ground"),
            "vertical_jump_coefficient": _coefficient(fit, "dvy", "a_jump"),
        }
        out["fits"][loss] = block
        out["recovered_subpixels_per_pixel"][loss] = (1.0 / scale) if scale else None
        stratum_source[loss] = fit

    sweep = []
    for trial in ALPHA_SWEEP:
        variant = sindy(
            xdot_train,
            theta_train,
            kept_names,
            alpha=trial,
            targets=list(DYNAMIC_TARGETS),
            robust=True,
        )
        sweep.append(
            {
                "alpha": trial,
                "total_terms": int(sum(len(s) for s in variant.support)),
                "sparsity": variant.sparsity(),
                # The displacement law is the one channel whose terms matter for the physics
                # prose, so the sweep carries them: "the threshold deleted the 1/16 term" has to
                # be readable from the artifact, not from a second fit run by hand.
                "terms_dx": [
                    {"term": name, "coefficient": round(value, 6)}
                    for name, value in sorted(
                        variant.equations()["dx"], key=lambda entry: -abs(entry[1])
                    )
                ],
                "heldout_relative_error": variant.relative_error(theta_test, xdot_test),
                "heldout_r2": variant.variance_explained(theta_test, xdot_test, mean_train),
            }
        )
    out["alpha_sweep_robust"] = sweep
    out["strata"] = {
        loss: stratum_agreement(fit, theta_test, held["test"][0], held["test"][1], actions["test"])
        for loss, fit in stratum_source.items()
    }
    out["train_strata"] = {
        loss: stratum_agreement(
            fit, theta_train, held["train"][0], held["train"][1], actions["train"]
        )
        for loss, fit in stratum_source.items()
    }
    return out


def stratum_agreement(
    fit: SindyFit,
    theta: np.ndarray,
    states: np.ndarray,
    next_states: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, Any]:
    r"""Score the discovered vertical law on the four strata the gravity gate lives on.

    10.52 measured the gate as the median vertical increment inside each stratum of the
    recording itself, so the comparison that matters for an identification method is not its
    coefficient list but those same medians *predicted* by the law it discovered. Held and
    released are read from the recorded action channel, which is the caveat 10.52 states: the
    engine's internal jump flag is a different byte and no recorder here reads both.
    """
    rising = states[:, 3] < 0.0
    falling = states[:, 3] > 0.0
    held = actions[:, 0] > 0.5
    airborne = states[:, 4] < 0.5
    dvy = (next_states - states)[:, 3]
    predicted = (theta @ fit.coefficients)[:, list(DYNAMIC_TARGETS).index("dvy")]
    out: Dict[str, Any] = {}
    for name, mask in (
        ("ascent_held", rising & held & airborne),
        ("ascent_released", rising & ~held & airborne),
        ("descent", falling & airborne),
        ("grounded", ~airborne),
    ):
        out[name] = {
            "transitions": int(mask.sum()),
            "observed_median_dvy": float(np.median(dvy[mask])) if mask.any() else None,
            "predicted_mean_dvy": float(np.mean(predicted[mask])) if mask.any() else None,
        }
    pair = [out[name]["observed_median_dvy"] for name in ("ascent_released", "ascent_held")]
    identified_pair = [
        out[name]["predicted_mean_dvy"] for name in ("ascent_released", "ascent_held")
    ]
    out["measured_tier_separation"] = (
        float(pair[0] - pair[1]) if all(value is not None for value in pair) else None
    )
    out["identified_tier_separation"] = (
        float(identified_pair[0] - identified_pair[1])
        if all(value is not None for value in identified_pair)
        else None
    )
    return out


def _cross_instrument_reference() -> Dict[str, Any]:
    """The constants 10.37 identified, read from its own artifact."""
    path = os.path.join(RESULTS_DIR, IDENTIFICATION_ARTIFACT)
    if not os.path.isfile(path):
        return {"available": False, "reason": f"{IDENTIFICATION_ARTIFACT} not found"}
    block = read_metrics(path)["E2_real_data_identification"]["params"]
    return {
        "available": True,
        "artifact": f"results/{IDENTIFICATION_ARTIFACT}",
        "identified": {name: float(values["identified"]) for name, values in block.items()},
        "wram_reference": {name: float(values["wram_reference"]) for name, values in block.items()},
    }


def _verdict(results: Dict[str, Any], cross: Dict[str, Any]) -> Dict[str, Any]:
    scales = {label: block["recovered_subpixels_per_pixel"] for label, block in results.items()}
    published = results.get("published_gameplay", {})
    fits = published.get("fits", {})
    gravity = {loss: fits.get(loss, {}).get("vertical_constant") for loss in LOSSES}
    gated = {
        label: block["fits"]["robust"]["vertical_jump_coefficient"] is not None
        for label, block in results.items()
    }
    separations = {
        label: {
            loss: {
                "measured": block["strata"][loss]["measured_tier_separation"],
                "identified": block["strata"][loss]["identified_tier_separation"],
            }
            for loss in LOSSES
        }
        for label, block in results.items()
    }
    # "Spurious" has to be a magnitude, not a non-zero float: the robust fit on a recording
    # with no gate returns 4.4e-16, which is exact to machine precision, and calling that a
    # manufactured structure would be reading noise as a finding.
    spurious = [
        label
        for label, per_loss in separations.items()
        if per_loss["robust"]["measured"] == 0.0
        and per_loss["robust"]["identified"] is not None
        and abs(per_loss["robust"]["identified"]) > SELECTED_ALPHA
    ]
    inverted = [
        label
        for label, per_loss in separations.items()
        if per_loss["least_squares"]["measured"] not in (None, 0.0)
        and per_loss["least_squares"]["identified"] is not None
        and per_loss["least_squares"]["identified"] * per_loss["least_squares"]["measured"] < 0
    ]
    recovered = {
        label: per_loss["robust"]["identified"]
        for label, per_loss in separations.items()
        if per_loss["robust"]["measured"] not in (None, 0.0)
        and per_loss["robust"]["identified"] is not None
    }
    return {
        "recordings_identified": len(results),
        "subpixels_per_pixel_recovered": scales,
        "subpixels_per_pixel_documented": DOCUMENTED["subpixels_per_pixel"],
        "published_vertical_constant": gravity,
        "jump_button_in_vertical_law": gated,
        "tier_separation_measured_and_identified": separations,
        "gate_recovered_fraction_of_measured": {
            label: value / separations[label]["robust"]["measured"]
            for label, value in recovered.items()
        },
        "recordings_where_least_squares_inverts_the_gate": inverted,
        "recordings_with_a_spurious_gate_after_thresholding": spurious,
        "gradient_identification_of_10_37": (
            cross.get("identified", {}) if cross.get("available") else None
        ),
        "reading": (
            f"{len(results)} recordings were identified with a degree-{MAX_DEGREE} dictionary at "
            f"alpha = {SELECTED_ALPHA:g} sub-pixels/frame under both losses. On the published "
            f"recording the least-squares fit puts the sub-pixel scale at "
            f"{_fmt(scales['published_gameplay']['least_squares'], '.2f')} and the vertical "
            f"constant at {_fmt(gravity['least_squares'])}; the Huber fit puts them at "
            f"{_fmt(scales['published_gameplay']['robust'], '.2f')} and "
            f"{_fmt(gravity['robust'])}, against the documented "
            f"{DOCUMENTED['subpixels_per_pixel']:g} and "
            f"{DOCUMENTED['held_gravity']:g}. The jump button enters the vertical law on "
            f"{[label for label, value in gated.items() if value] or 'no recording'}."
        ),
    }


def render_identification_table(results: Dict[str, Any]) -> List[str]:
    """One row per recording and loss: what survived, what scale it recovered, what it cost."""
    rows: List[str] = []
    for label, block in results.items():
        for loss in LOSSES:
            fit = block["fits"][loss]
            scale = block["recovered_subpixels_per_pixel"][loss]
            scale_cell = f"{scale:.2f}" if scale else "-"
            constant = fit["vertical_constant"]
            constant_cell = "-" if constant is None else f"{constant:.3f}"
            jump_cell = "**yes**" if fit["vertical_jump_coefficient"] is not None else "no"
            rows.append(
                rf"| {DISPLAY.get(label, label)} | `{loss}`"
                rf" | {fit['sparsity']['dx']} / {fit['sparsity']['dvx']}"
                rf" / {fit['sparsity']['dvy']}"
                rf" | {scale_cell}"
                rf" | {constant_cell}"
                rf" | {jump_cell}"
                rf" | {fit['heldout_relative_error']['dx']:.4f}"
                rf" | {fit['heldout_r2']['dvx']:.3f}"
                rf" | {fit['heldout_r2']['dvy']:.3f} |"
            )
    return rows


MAX_TERMS_SHOWN = 8


def render_equation_table(
    results: Dict[str, Any], labels: Optional[Sequence[str]] = None
) -> List[str]:
    """The discovered laws, longest ones ordered by coefficient size and cut at eight terms.

    The cut is stated in the cell rather than silent: a 23-term horizontal law is the finding
    as much as any eight of its terms are, so the row says how many it found.
    """
    rows: List[str] = []
    for label, block in results.items():
        if labels and label not in labels:
            continue
        fit = block["fits"]["robust"]
        for target in ("dx", "dvx", "dvy"):
            terms = sorted(fit["equations"][target], key=lambda entry: -abs(entry["coefficient"]))
            shown = terms[:MAX_TERMS_SHOWN]
            body = (
                " + ".join(
                    (
                        f"{entry['coefficient']:.4f}"
                        if entry["term"] == "1"
                        else f"{entry['coefficient']:.4f}·{entry['term']}"
                    )
                    for entry in shown
                )
                or "no term survives"
            )
            extra = (
                ""
                if len(terms) <= MAX_TERMS_SHOWN
                else f" + … ({len(terms) - MAX_TERMS_SHOWN} more terms)"
            )
            rows.append(
                rf"| {DISPLAY.get(label, label)} | ${target}$ | {body}{extra}"
                rf" | {len(terms)}"
                rf" | {fit['heldout_relative_error'][target]:.4f}"
                rf" | {fit['heldout_r2'][target]:.3f} |"
            )
    return rows


def render_constants_table(results: Dict[str, Any], cross: Dict[str, Any]) -> List[str]:
    """SINDy's constants against 10.37's gradient identification and Section 4's numbers."""
    published = results.get("published_gameplay")
    if published is None or not cross.get("available"):
        return []
    identified = cross["identified"]
    rows: List[str] = []
    scale = {loss: published["recovered_subpixels_per_pixel"][loss] for loss in LOSSES}
    rows.append(
        rf"| sub-pixels per pixel ($\Delta x$ on $v_x$) | {_fmt(scale['least_squares'], '.2f')}"
        rf" | {_fmt(scale['robust'], '.2f')}"
        rf" | {identified['subpixels_per_pixel']:.2f}"
        rf" | {DOCUMENTED['subpixels_per_pixel']:g} |"
    )
    vertical = {loss: published["fits"][loss]["vertical_constant"] for loss in LOSSES}
    rows.append(
        rf"| ascent gravity (constant in $\Delta v_y$) | {_fmt(vertical['least_squares'])}"
        rf" | {_fmt(vertical['robust'])}"
        rf" | {identified['held_gravity']:.3f}"
        rf" | {DOCUMENTED['held_gravity']:g} |"
    )
    ground = {loss: published["fits"][loss]["vertical_ground_coefficient"] for loss in LOSSES}
    ground_cells = {
        loss: "-" if value is None else f"{value:+.3f}" for loss, value in ground.items()
    }
    rows.append(
        rf"| ground coefficient in $\Delta v_y$ | {ground_cells['least_squares']}"
        rf" | {ground_cells['robust']} | - | not documented |"
    )
    horizontal = published["fits"]["robust"]["equations"]["dvx"]
    by_term = {entry["term"]: entry["coefficient"] for entry in horizontal}
    run_cell = _fmt(by_term.get("a_run"), "+.3f", dash="term absent")
    coast_cell = _fmt(by_term.get("vx"), "+.4f", dash="term absent")
    rows.append(
        rf"| run-button coefficient in $\Delta v_x$ | -"
        rf" | {run_cell}"
        rf" | {identified['run_accel']:.3f} | 1.5 |"
    )
    rows.append(
        rf"| coast coefficient on $v_x$ in $\Delta v_x$ | -"
        rf" | {coast_cell}"
        rf" | {identified['decel']:.3f} | 0.5 |"
    )
    return rows


def render_sweep_table(results: Dict[str, Any], labels: Sequence[str]) -> List[str]:
    r"""The displacement law at every threshold: the point where the $1/16$ term is deleted.

    $\alpha$ is in physical units, so a threshold above $1/16$ px/frame removes the engine's
    integration term by definition. The rows are the sweep the protocol runs, with the support
    of $\Delta x$ spelled out, because the finding is about which columns survive and not only
    about how many.
    """
    rows: List[str] = []
    for label in labels:
        block = results.get(label)
        if block is None:
            continue
        for entry in block["alpha_sweep_robust"]:
            support = (
                " + ".join(
                    (
                        f"{term['coefficient']:.4f}"
                        if term["term"] == "1"
                        else f"{term['coefficient']:.4f}·{term['term']}"
                    )
                    for term in entry["terms_dx"]
                )
                or "no term survives"
            )
            rows.append(
                rf"| {DISPLAY.get(label, label)} | {entry['alpha']:g}"
                rf" | {entry['sparsity']['dx']}"
                rf" | {support}"
                rf" | {entry['heldout_relative_error']['dx']:.4f}"
                rf" | {entry['heldout_r2']['dx']:.4f}"
                rf" | {entry['heldout_r2']['pooled']:.4f} |"
            )
    return rows


def render_gate_table(results: Dict[str, Any]) -> List[str]:
    """The gravity gate as the recording measures it, then as the identified law predicts it."""
    rows: List[str] = []
    for label, block in results.items():
        for loss in LOSSES:
            strata = block["strata"][loss]
            held, released = strata["ascent_held"], strata["ascent_released"]

            def cell(entry: Dict[str, Any]) -> str:
                if not entry["transitions"]:
                    return "-"
                return (
                    rf"{entry['transitions']:,}, {entry['observed_median_dvy']:.1f},"
                    rf" law {entry['predicted_mean_dvy']:.2f}"
                )

            separation = strata["measured_tier_separation"]
            identified = strata["identified_tier_separation"]
            rows.append(
                rf"| {DISPLAY.get(label, label)} | `{loss}` | {cell(held)} | {cell(released)}"
                rf" | {'-' if separation is None else f'{separation:+.1f}'}"
                rf" | {'-' if identified is None else f'{identified:+.3f}'} |"
            )
    return rows


def integration_convention(data: Dict[str, Any], split: str = "train") -> Dict[str, float]:
    r"""The least-squares second-order coefficient of the recorded transitions themselves.

    Section 10.55 fits $\hat x_{t+1} = x_t + v_{x,t}/16 + c\,\hat a/16$ to every arm and to
    the console. Fitted over all frames, the console's $c$ is not a clean reading: the wall
    stops, terminations and screen wraps carry both the largest displacement error and the
    largest acceleration, so they pull the slope toward the middle of the range. The same
    statistic on the contact-free subset, where no collision flag is set and the engine is
    doing nothing but integrating, is what the integrator question actually needs - and it is
    computed here because this study is seconds long and the grid that owns the question is
    an hour of training.
    """
    states = np.asarray(data[f"{split}_states"], dtype=np.float64)
    next_states = np.asarray(data[f"{split}_next_states"], dtype=np.float64)
    dxy = next_states[:, :2] - states[:, :2]
    accel = next_states[:, 2:4] - states[:, 2:4]
    contact_free = (states[:, 4:8] < 0.5).all(axis=1)
    out: Dict[str, float] = {}
    for name, mask in (
        ("all_frames", np.ones(states.shape[0], dtype=bool)),
        ("contact_free", contact_free),
    ):
        for channel, axis in ((0, "x"), (1, "y")):
            a = accel[mask, channel]
            residual = 16.0 * dxy[mask, channel] - states[mask, 2 + channel]
            denominator = float(np.dot(a, a))
            moving = np.abs(a) > 2.0
            out[f"c_{axis}_{name}"] = (
                float(np.dot(residual, a) / denominator) if denominator else None
            )
            out[f"exact_rate_{axis}_{name}"] = float(np.mean(np.abs(residual) < 0.5))
            out[f"n_accelerating_{axis}_{name}"] = int(moving.sum())
            # The three frames that decide the slope, named by count rather than listed: this
            # is the whole difference between "the engine truncates" and "the fit says 0.25".
            broken = moving & (np.abs(residual) >= 0.5)
            out[f"n_frames_holding_the_slope_{axis}_{name}"] = int(broken.sum())
            out[f"mean_residual_of_those_{axis}_{name}"] = (
                float(np.mean(residual[broken])) if broken.any() else None
            )
        out[f"n_{name}"] = int(mask.sum())
    return out


def run_sindy_identification_benchmark(
    output_dir: Optional[str] = None,
    seed: int = 42,
    alpha: float = SELECTED_ALPHA,
    max_degree: int = MAX_DEGREE,
    recordings: Sequence[Tuple[str, str]] = RECORDINGS,
) -> Dict[str, Any]:
    """Identify the engine's law on every recording available, with the canonical split."""
    out_dir = output_dir or RESULTS_DIR
    logger.info("=== SINDy identification on %d recordings ===", len(recordings))
    results: Dict[str, Any] = {}
    for label, path in recordings:
        block = _fit_one(label, path, seed=seed, alpha=alpha, max_degree=max_degree)
        if block is None:
            continue
        results[label] = block
        robust = block["fits"]["robust"]
        logger.info(
            "  %-18s | terms %d/%d/%d | scale %.2f | gravity %.3f | heldout r2 dvx %.3f dvy %.3f",
            label,
            robust["sparsity"]["dx"],
            robust["sparsity"]["dvx"],
            robust["sparsity"]["dvy"],
            block["recovered_subpixels_per_pixel"]["robust"] or float("nan"),
            robust["vertical_constant"]
            if robust["vertical_constant"] is not None
            else float("nan"),
            robust["heldout_r2"]["dvx"],
            robust["heldout_r2"]["dvy"],
        )
    cross = _cross_instrument_reference()
    convention = integration_convention(
        load_and_preprocess_data(dataset_path=DATASET_GAMEPLAY, seed=seed)
    )
    payload: Dict[str, Any] = {
        "study": (
            "Sparse identification of nonlinear dynamics (Brunton, Proctor and Kutz 2016) on "
            "WRAM telemetry: the standard estimator, run on every recording this repository has, "
            "under least-squares and Huber losses."
        ),
        "protocol": {
            "seed": seed,
            "alpha": alpha,
            "alpha_sweep": list(ALPHA_SWEEP),
            "ridge_lambda": 1e-3,
            "max_iterations": 20,
            "max_degree": max_degree,
            "losses": list(LOSSES),
            "library_inputs": list(LIBRARY_INPUTS),
            "targets": list(DYNAMIC_TARGETS),
            "derivative": "forward difference over one frame, so coefficients are sub-pixels/frame",
            "positions_in_dictionary": False,
            "duplicate_columns_pruned": True,
            "split": "load_and_preprocess_data canonical episodic split",
            "emulator_required": False,
        },
        "results": results,
        "console_integration_convention": convention,
        "cross_instrument": cross,
        "documented": DOCUMENTED,
        "verdict": _verdict(results, cross),
    }
    if _render_figure(results, os.path.join(out_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command=f"python -m src.evaluation.sindy_identification_benchmark --alpha {alpha}",
    )
    logger.info("Metrics written to %s", artifact)
    return payload


def _render_figure(results: Dict[str, Any], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return False
    labels = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for target, marker in (("dx", "o"), ("dvx", "s"), ("dvy", "^")):
        for loss, style in (("least_squares", "--"), ("robust", "-")):
            axes[0].plot(
                range(len(labels)),
                [results[label]["fits"][loss]["heldout_r2"][target] for label in labels],
                marker=marker,
                linestyle=style,
                label=f"{target}, {loss}",
            )
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("held-out $R^2$")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)

    for label in labels:
        sweep = results[label]["alpha_sweep_robust"]
        axes[1].plot(
            [entry["total_terms"] for entry in sweep],
            [entry["heldout_r2"]["pooled"] for entry in sweep],
            marker="o",
            label=DISPLAY.get(label, label),
        )
    axes[1].set_xlabel("number of surviving terms")
    axes[1].set_ylabel("pooled held-out $R^2$")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle("SINDy on WRAM telemetry: dictionary, loss and threshold (README 10.56)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sparse identification of nonlinear dynamics on SMW WRAM telemetry (10.56)."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=SELECTED_ALPHA)
    parser.add_argument("--max-degree", type=int, default=MAX_DEGREE)
    args = parse_args_with_config(parser)

    run_sindy_identification_benchmark(
        output_dir=args.output_dir,
        seed=args.seed,
        alpha=args.alpha,
        max_degree=args.max_degree,
    )
