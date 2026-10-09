r"""
physics_claim_audit.py
Machine-check the physics statements of README section 4 against code and telemetry (10.54).

Section 4 is the only part of this repository that is *quoted by physics* rather than by
measurement: §4.1 states the integration identity, §4.2 the jump window and the two gravity
tiers, §4.3 the three speed classes. Those statements are then implemented - a residual in
`DiscreteKinematicsLoss`, a predicate in `RolloutEvaluator`, clamps in every shell - and
nothing checked the implementations against each other or against the recorded transitions.

That gap is what 10.49 fell into: §4.3 publishes walk 20, run 48 and sprint-with-P-meter 72
as three separate classes, and four sections scored ceiling estimates against 72.0 as if the
README had said "the maximum velocity". The first version of this audit answered the same
question about four more claims and found the prose wrong in four places; the prose has since
been rewritten to what the recording supports, which changes what the audit is for. It now
holds §4 to three things at once:

1. the README text of every claim, quoted from the file, so a prose edit that changes the
   physics fails the gate;
2. the telemetry, with an explicit test per claim (`asserts` below) - a claim is satisfied
   only if the measurement the prose now states comes out the way the prose states it;
3. the code, through the literal that makes each declared file an implementation of the
   claim, and a divergence list for the files that implement a form §4 no longer claims.

So `claims_the_telemetry_does_not_satisfy` is expected to be empty - that is the correction -
and the load-bearing output is `prose_and_code_disagree`: the documentation says the ground
flag gates the gravity step and `GroundContactConsistencyLoss` still penalises a vertical
velocity of zero, the identity says the carried velocity and two shells advance with the
predicted next one.

Writes ``results/physics_claim_audit_metrics.json``. Emulator-free.

Run:  python -m src.evaluation.physics_claim_audit
"""

import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from src.environment.dataset_loader import load_and_preprocess_data
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import write_metrics
from src.utils.typography import demath_typographic, unemphasise_math

logger = get_logger(__name__)

SUBPIXELS_PER_PIXEL = 16.0
PUBLISHED_TOLERANCE_PX = 0.2
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README_PATH = os.path.join(REPO_ROOT, "README.md")

# Claims of README section 4, as the section now states them. `sites` are (file, literal)
# pairs: the literal is the piece of source that makes that file an implementation of the
# claim, and `convention` says which velocity the site advances position with where the claim
# is the integration identity. `asserts` names the test in SATISFACTION_TESTS that decides
# whether the recorded transitions agree with the sentence, and `disagrees_with` lists the
# sites that implement a different rule from the one §4 now states - documentation and code
# on the same subject, deliberately kept apart so the gap stays visible.
#
# `corrected_in` is new as of 10.57: the literal that makes the *corrected* rule reachable in the
# same file, and `corrected_by_default` says whether it is what a caller gets without asking.
# Every default is still the published one, because every artifact in the repository was recorded
# with it - so for each claim the honest state is "implemented, not default", and that is what the
# audit now prints instead of "no implementation exists".
CLAIMS: Dict[str, Dict[str, Any]] = {
    "4.1_integration_identity_carried_velocity": {
        "readme_marker": "### 4.1 Fixed-Point Arithmetic",
        "claim": r"$$X_{t+1} = X_t + \frac{v_{x, t}}{16.0}$$",
        "convention": "carried",
        "asserts": "carried_identity_is_exact",
        "sites": [
            ("src/losses/physics_losses.py", "expected_dx = vx_t / self.subpixels_per_pixel"),
            ("src/evaluation/rollout_evaluator.py", "curr_state[0, 2].item() / 16.0"),
            ("src/evaluation/rollout_diagnostics.py", "v_prev = pred_traj[:-1, 2]"),
            (
                "src/models/effective_velocity_dynamics.py",
                "used_vx, used_vy = vx_t + offset_vx, vy_t + offset_vy",
            ),
            ("src/models/neural_ode_dynamics.py", "z[:, 2:4] / self.subpixels_per_pixel"),
            ("src/models/residual_dynamics.py", "if self.position_velocity == CARRIED else"),
            ("src/models/output_projection.py", "if self.position_velocity == CARRIED else"),
            ("src/models/pinn_hard_residual.py", "if self.position_velocity == CARRIED else"),
            ("src/models/analytical_kinematics.py", "if self.position_velocity == CARRIED else"),
        ],
        "corrected_in": 'position_velocity="carried" in four shells',
        "corrected_by_default": False,
    },
    "4.1_violation_defined_with_carried_velocity": {
        "readme_marker": "### 4.1 Fixed-Point Arithmetic",
        "claim": r"\hat{X}_{t+1} \ne X_t + \frac{v_{x, t}}{16.0}",
        "convention": "carried",
        "asserts": "next_velocity_definition_would_flag_the_engine_itself",
        "sites": [
            ("src/losses/physics_losses.py", "expected_dx = vx_t / self.subpixels_per_pixel"),
            ("src/evaluation/rollout_evaluator.py", "curr_state[0, 2].item() / 16.0"),
        ],
        # The four shells still default to the next-frame form; the literal that makes it the
        # default is the constructor argument, which is precisely what a caller inherits.
        "disagrees_with": [
            ("src/models/residual_dynamics.py", "position_velocity: str = NEXT"),
            ("src/models/output_projection.py", "position_velocity: str = NEXT"),
            ("src/models/pinn_hard_residual.py", "position_velocity: str = NEXT"),
            ("src/models/analytical_kinematics.py", "position_velocity: str = NEXT"),
        ],
        "corrected_in": 'position_velocity="carried"',
        "corrected_by_default": False,
    },
    "4.2.1_jump_impulse_range": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"v_{y, 0} \in [-64, -80]\text{ subpixels/frame}",
        "asserts": "impulse_range_is_not_an_observed_bound",
        "sites": [
            ("src/models/analytical_kinematics.py", "MIN_VY = -80.0"),
            ("src/losses/physics_losses.py", "min_vy: float = -80.0"),
        ],
    },
    "4.2.3_4.2.4_gravity_tiers": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"g_{\text{held}} = +3.0\text{ subpixels/frame}^2",
        "asserts": "both_gravity_tiers_are_observed",
        "sites": [
            ("src/models/analytical_kinematics.py", "G_HOLD = 3.0"),
            ("src/models/analytical_kinematics.py", "G_FALL = 6.0"),
        ],
    },
    "4.2.4_descent_tier_needs_the_right_excitation": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"g_{\text{fall}} = +6.0\text{ subpixels/frame}^2",
        "asserts": "released_descent_is_the_only_stratum_where_the_upper_tier_is_common",
        "sites": [("src/models/analytical_kinematics.py", "G_FALL = 6.0")],
    },
    "4.2.5_terminal_velocity": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"v_{y} \le v_{y, \text{term}} = +64.0\text{ subpixels/frame}",
        "asserts": "terminal_velocity_is_broken_by_the_recording",
        "sites": [
            ("src/models/analytical_kinematics.py", "TERMINAL_VY = 64.0"),
            ("src/losses/physics_losses.py", "terminal_vy: float = 64.0"),
        ],
        "disagrees_with": [("src/evaluation/rollout_evaluator.py", "terminal_vy: float = 64.0")],
        "corrected_in": "RolloutEvaluator(terminal_vy=...)",
        "corrected_by_default": False,
    },
    "4.3.1_walk_speed": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"$|v_x| \le 20\text{ subpixels/frame}$",
        "asserts": "walk_cap_is_a_class_not_a_bound",
        "sites": [("src/models/analytical_kinematics.py", "VX_WALK = 20.0")],
    },
    "4.3.2_run_speed": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"$|v_x| \le 48\text{ subpixels/frame}$",
        "asserts": "run_cap_is_a_class_and_is_breached",
        "sites": [("src/models/analytical_kinematics.py", "VX_RUN = 48.0")],
    },
    "4.3.3_sprint_speed": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"$|v_x| \le 72\text{ subpixels/frame}$",
        "asserts": "top_class_is_never_reached",
        "sites": [
            ("src/models/analytical_kinematics.py", "VX_SPRINT = 72.0"),
            ("src/models/residual_dynamics.py", "max_vx: float = 72.0"),
            ("src/models/output_projection.py", "max_vx: float = 72.0"),
            ("src/losses/physics_losses.py", "max_vx: float = 72.0"),
        ],
        "disagrees_with": [("src/evaluation/rollout_evaluator.py", "max_vx: float = 72.0")],
        "corrected_in": "RolloutEvaluator(max_vx=48.0), the class 10.49 reaches",
        "corrected_by_default": False,
    },
    "4.3.5_ground_flag_gates_gravity": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"v_{y, t+1} = v_{y, t} + g \cdot (1 - c_{t, \text{ground}})",
        "asserts": "grounded_frames_are_never_stopped",
        # The rule now has an implementation in both places it matters - the closed-form engine
        # rules and the composite penalty - but neither is what a caller gets by default, because
        # every published artifact was recorded with the retracted rest-state form.
        "sites": [
            ("src/models/analytical_kinematics.py", "vy_next = torch.where(grounded, vy, vy_next)"),
            (
                "src/losses/physics_losses.py",
                "ground_vy = ground_vy - current_state[stationary_ground_mask, 3]",
            ),
        ],
        "disagrees_with": [
            (
                "src/losses/physics_losses.py",
                "def __init__(self, rule: str = CONTACT_ZERO_VELOCITY)",
            ),
            ("src/models/analytical_kinematics.py", "ground_rule: str = CONTACT_ZERO_VELOCITY"),
        ],
        "corrected_in": 'contact_rule="zero_increment" / ground_rule="zero_increment"',
        "corrected_by_default": False,
    },
}


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _section(text: str, marker: str) -> str:
    """The README lines of the section a claim lives in, up to the next header."""
    start = text.find(marker)
    if start < 0:
        return ""
    rest = text[start:]
    end = len(rest)
    for stop in ("\n### ", "\n## "):
        found = rest.find(stop, len(marker))
        if found >= 0:
            end = min(end, found)
    return rest[:end]


def audit_claims(claims: Dict[str, Dict[str, Any]] = CLAIMS) -> Dict[str, Any]:
    """Check each claim's README text, its declared implementation sites and its dissenters."""
    readme = _read(README_PATH)
    sources: Dict[str, str] = {}
    report: Dict[str, Any] = {}
    for name, spec in claims.items():
        section = _section(readme, spec["readme_marker"])
        found, missing = [], []
        for path, literal in spec["sites"]:
            if path not in sources:
                sources[path] = _read(path)
            (found if literal in sources[path] else missing).append(f"{path}: {literal}")
        dissenters, dissent_missing = [], []
        for path, literal in spec.get("disagrees_with", []):
            if path not in sources:
                sources[path] = _read(path)
            present = literal in sources[path]
            (dissenters if present else dissent_missing).append(f"{path}: {literal}")
        report[name] = {
            "claim": spec["claim"],
            "readme_marker": spec["readme_marker"],
            "claim_quoted_from_the_readme": spec["claim"] in section,
            "declared_sites": len(spec["sites"]),
            "sites_found": found,
            "sites_missing": missing,
            "implemented_by_every_declared_site": not missing,
            "dissenting_sites": dissenters,
            "dissenting_sites_missing": dissent_missing,
            "asserts": spec["asserts"],
            "convention": spec.get("convention"),
            # Reachability is verified through `sites` (the corrected line must be in the file);
            # these two fields only say how a caller reaches it, and whether that is the default.
            "corrected_form": spec.get("corrected_in"),
            "corrected_by_default": spec.get("corrected_by_default"),
        }
    return report


# Each test answers: does the recorded transition data do what the sentence now says?
# `evidence` is the same string the audit table prints, so the verdict and the quote of the
# measurement cannot drift apart.
def _exact_rate(block: Dict[str, float]) -> float:
    return float(block["exact_to_half_a_subpixel_rate"])


def _tier(telemetry: Dict[str, Any], stratum: str, key: str) -> float:
    """One cell of the stratum table, defaulted so a thin stratum reads as zero evidence."""
    value = telemetry["gravity_step_by_stratum"][stratum][key]
    return 0.0 if value is None else float(value)


SATISFACTION_TESTS: Dict[str, Callable[[Dict[str, Any]], Tuple[bool, str]]] = {
    "carried_identity_is_exact": lambda t: (
        t["identity_vs_carried_velocity"]["median_abs_residual_px"] == 0.0
        and _exact_rate(t["identity_vs_carried_velocity"]) > 0.9,
        f"median {t['identity_vs_carried_velocity']['median_abs_residual_px']:.4f} px, exact on "
        f"{t['identity_vs_carried_velocity']['exact_to_half_a_subpixel_rate']:.2%} of frames",
    ),
    "next_velocity_definition_would_flag_the_engine_itself": lambda t: (
        t["identity_vs_next_velocity"]["median_abs_residual_px"]
        > t["identity_vs_carried_velocity"]["median_abs_residual_px"],
        f"median {t['identity_vs_next_velocity']['median_abs_residual_px']:.4f} px against "
        f"$\\hat v_{{x,t+1}}$ - the console would violate its own definition on "
        f"{1 - _exact_rate(t['identity_vs_next_velocity']):.2%} of frames",
    ),
    "impulse_range_is_not_an_observed_bound": lambda t: (
        t["window_observed"]["vy_min"] < -80.0,
        f"{t['window_observed']['rate_below_jump_window']:.2%} of velocities below $-80$ "
        f"(min {t['window_observed']['vy_min']:.1f})",
    ),
    "both_gravity_tiers_are_observed": lambda t: (
        t["gravity_step_on_airborne_ascent"]["rate_at_3.0"] > 0.0
        and t["gravity_step_on_airborne_ascent"]["rate_at_6.0"] > 0.0,
        f"{t['gravity_step_on_airborne_ascent']['rate_at_3.0']:.2%} of "
        f"{t['gravity_step_on_airborne_ascent']['n_frames']:,} airborne-ascent frames step $3.0$, "
        f"{t['gravity_step_on_airborne_ascent']['rate_at_6.0']:.2%} step $6.0$",
    ),
    "terminal_velocity_is_broken_by_the_recording": lambda t: (
        t["window_observed"]["rate_above_terminal_velocity"] > 0.0,
        f"{t['window_observed']['rate_above_terminal_velocity']:.2%} above $+64$ "
        f"(max {t['window_observed']['vy_max']:.1f})",
    ),
    "released_descent_is_the_only_stratum_where_the_upper_tier_is_common": lambda t: (
        _tier(t, "descent_released", "rate_at_6.0")
        > max(_tier(t, name, "rate_at_6.0") for name in ("ascent_held", "ascent_released"))
        and _tier(t, "ascent_released", "rate_at_3.0") > _tier(t, "ascent_released", "rate_at_6.0"),
        "released descent is the only stratum where $6.0$ is common "
        f"({_tier(t, 'descent_released', 'rate_at_6.0'):.2%} of "
        f"{_tier(t, 'descent_released', 'n_frames'):,.0f} frames) - released *ascent* still steps "
        f"$3.0$ in {_tier(t, 'ascent_released', 'rate_at_3.0'):.2%} of "
        f"{_tier(t, 'ascent_released', 'n_frames'):,.0f} frames, so the gate is not separable by "
        f"button state in this recording",
    ),
    "walk_cap_is_a_class_not_a_bound": lambda t: (
        t["speed_classes_observed"]["rate_above_walk_cap"] > 0.0,
        f"{t['speed_classes_observed']['rate_above_walk_cap']:.2%} of frames above $20$",
    ),
    "run_cap_is_a_class_and_is_breached": lambda t: (
        t["speed_classes_observed"]["rate_above_run_cap"] > 0.0,
        f"{t['speed_classes_observed']['rate_above_run_cap']:.4%} of frames above $48$",
    ),
    "top_class_is_never_reached": lambda t: (
        t["speed_classes_observed"]["rate_above_sprint_cap"] == 0.0,
        f"{t['speed_classes_observed']['rate_above_sprint_cap']:.2%} above $72$, max observed "
        rf"$\lvert v_x \rvert = "
        f"{t['speed_classes_observed']['max_abs_vx']:.1f}$",
    ),
    "grounded_frames_are_never_stopped": lambda t: (
        t["ground_boundary_condition"]["claimed_rate_vy_next_exactly_zero"] is not None
        and t["ground_boundary_condition"]["claimed_rate_vy_next_exactly_zero"] < 0.05
        and t["ground_boundary_condition"]["claimed_rate_step_exactly_zero"] > 0.5,
        f"the recorded vertical velocity is exactly zero on "
        f"{t['ground_boundary_condition']['claimed_rate_vy_next_exactly_zero']:.2%} of "
        f"{t['ground_boundary_condition']['n_frames_claimed']:,} frames, median "
        rf"$\lvert v_y\rvert = "
        f"{t['ground_boundary_condition']['claimed_median_abs_vy_next']:.1f}$, while its "
        f"*increment* is zero on "
        f"{t['ground_boundary_condition']['claimed_rate_step_exactly_zero']:.2%} of them "
        f"(median step "
        f"{t['ground_boundary_condition']['claimed_median_step']:+.1f})",
    ),
}


def score_claims(report: Dict[str, Any], telemetry: Dict[str, Any]) -> None:
    """Attach each claim's verdict and evidence, in place."""
    for name, block in report.items():
        test = SATISFACTION_TESTS[block["asserts"]]
        satisfied, evidence = test(telemetry)
        block["satisfied_by_the_telemetry"] = bool(satisfied)
        block["telemetry_evidence"] = evidence


def telemetry_satisfaction(dataset_path: str = DATASET_GAMEPLAY, seed: int = 42) -> Dict[str, Any]:
    """Score README section 4's claims against the recorded WRAM transitions."""
    data = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    states = np.asarray(data["train_states"], dtype=np.float64)
    next_states = np.asarray(data["train_next_states"], dtype=np.float64)
    actions = np.asarray(data["train_actions"], dtype=np.float64)
    dx = next_states[:, 0] - states[:, 0]
    dy = next_states[:, 1] - states[:, 1]
    vx_t, vy_t = states[:, 2], states[:, 3]
    vx_next, vy_next = next_states[:, 2], next_states[:, 3]
    vx_all = np.concatenate([np.abs(vx_t), np.abs(vx_next)])
    vy_all = np.concatenate([vy_t, vy_next])

    ascent = vy_t < 0.0
    step = vy_next - vy_t
    airborne = next_states[:, 4] < 0.5
    gravity_steps = step[ascent & airborne]
    no_contact = (states[:, 4:8] < 0.5).all(axis=1)

    def residual_report(d: np.ndarray, velocity: np.ndarray) -> Dict[str, float]:
        residual = np.abs(d - velocity / SUBPIXELS_PER_PIXEL)
        return {
            "n_frames": int(d.size),
            "mean_abs_residual_px": float(residual.mean()),
            "median_abs_residual_px": float(np.median(residual)),
            "within_published_tolerance_rate": float(np.mean(residual <= PUBLISHED_TOLERANCE_PX)),
            "exact_to_half_a_subpixel_rate": float(
                np.mean(np.abs(SUBPIXELS_PER_PIXEL * d - velocity) < 0.5)
            ),
        }

    return {
        "n_transitions": int(states.shape[0]),
        "identity_vs_carried_velocity": residual_report(dx, vx_t),
        "identity_vs_next_velocity": residual_report(dx, vx_next),
        # The stratum where §4.1 is a rule rather than a median tendency: no collision flag is
        # set, so the engine is doing nothing except accumulating. 10.55 measures the same thing
        # as a fitted integration coefficient; this is the version the documentation quotes.
        "identity_vs_carried_velocity_without_collision_flag": residual_report(
            dx[no_contact], vx_t[no_contact]
        ),
        "vertical_identity_vs_carried_velocity": residual_report(dy, vy_t),
        "vertical_identity_vs_next_velocity": residual_report(dy, vy_next),
        "speed_classes_observed": {
            "max_abs_vx": float(vx_all.max()),
            "rate_above_walk_cap": float(np.mean(vx_all > 20.0)),
            "rate_above_run_cap": float(np.mean(vx_all > 48.0)),
            "rate_above_sprint_cap": float(np.mean(vx_all > 72.0)),
        },
        "window_observed": {
            "vy_min": float(vy_all.min()),
            "vy_max": float(vy_all.max()),
            "rate_above_terminal_velocity": float(np.mean(vy_all > 64.0)),
            "rate_below_jump_window": float(np.mean(vy_all < -80.0)),
        },
        "gravity_step_on_airborne_ascent": {
            "n_frames": int(gravity_steps.size),
            "median_step_subpixels": float(np.median(gravity_steps))
            if gravity_steps.size
            else None,
            "rate_at_3.0": (
                float(np.mean(np.abs(gravity_steps - 3.0) < 0.5)) if gravity_steps.size else None
            ),
            "rate_at_6.0": (
                float(np.mean(np.abs(gravity_steps - 6.0) < 0.5)) if gravity_steps.size else None
            ),
        },
        "gravity_step_by_stratum": _gravity_strata(states, next_states, actions),
        "ground_boundary_condition": _ground_condition_rate(states, actions, next_states),
    }


def _tier_table(step: np.ndarray) -> Dict[str, float]:
    """How one stratum's vertical steps distribute over the two documented tiers."""
    return {
        "n_frames": int(step.size),
        "median_step_subpixels": float(np.median(step)) if step.size else None,
        "rate_at_3.0": float(np.mean(np.abs(step - 3.0) < 0.5)) if step.size else None,
        "rate_at_6.0": float(np.mean(np.abs(step - 6.0) < 0.5)) if step.size else None,
        "rate_step_exactly_zero": float(np.mean(step == 0.0)) if step.size else None,
    }


def _gravity_strata(
    states: np.ndarray, next_states: np.ndarray, actions: np.ndarray
) -> Dict[str, Dict[str, float]]:
    r"""The four stratum §4.2 distinguishes, scored separately.

    Section 4.2.3/4 define the tiers by button state *and* by direction, so a single pooled
    airborne figure cannot test them - and on the gameplay recording it actively misleads,
    because the released-descent stratum is the only place the upper tier is common. This is
    the measurement that §10.52's targeted excitation and §10.56's identified coefficient
    agree with.
    """
    step = next_states[:, 3] - states[:, 3]
    airborne = next_states[:, 4] < 0.5
    held = actions[:, 0] > 0.5
    ascending = states[:, 3] < 0.0
    return {
        "ascent_held": _tier_table(step[airborne & held & ascending]),
        "ascent_released": _tier_table(step[airborne & ~held & ascending]),
        "descent_held": _tier_table(step[airborne & held & ~ascending]),
        "descent_released": _tier_table(step[airborne & ~held & ~ascending]),
    }


def _ground_condition_rate(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> Dict[str, float]:
    """How often the engine really forces $v_{y,t+1}=0$ on a grounded, un-jumping frame.

    Conditioned the way the penalty conditions it - the ground flag of the state the frame
    starts from, no jump commanded, the vertical velocity of the state it ends in - because
    the claim under test is now the one `GroundContactConsistencyLoss` disagrees with: the
    ground flag gates the gravity step rather than stopping the body.
    """
    commanded = (states[:, 4] > 0.5) & (actions[:, 0] < 0.5)
    landed = (next_states[:, 4] > 0.5) & (actions[:, 0] < 0.5)
    vy_next = next_states[:, 3]
    step = next_states[:, 3] - states[:, 3]
    out: Dict[str, float] = {"n_frames_claimed": int(commanded.sum())}
    for name, mask in (("claimed", commanded), ("landed_this_frame", landed)):
        out[f"{name}_rate_vy_next_exactly_zero"] = (
            float(np.mean(vy_next[mask] == 0.0)) if mask.any() else None
        )
        out[f"{name}_median_abs_vy_next"] = (
            float(np.median(np.abs(vy_next[mask]))) if mask.any() else None
        )
        # The positive half of the corrected claim: on the ground the gravity *step* is
        # suppressed, which is a different statement from the body being stopped.
        out[f"{name}_rate_step_exactly_zero"] = (
            float(np.mean(step[mask] == 0.0)) if mask.any() else None
        )
        out[f"{name}_median_step"] = float(np.median(step[mask])) if mask.any() else None
    # Scored against the velocity the frame starts with too, so the result cannot be
    # dismissed as an indexing artifact of reading the state after the update.
    out["claimed_rate_vy_current_exactly_zero"] = (
        float(np.mean(states[commanded, 3] == 0.0)) if commanded.any() else None
    )
    out["claimed_median_abs_vy_current"] = (
        float(np.median(np.abs(states[commanded, 3]))) if commanded.any() else None
    )
    return out


def _prose_and_code_disagreements(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Claims whose own file list contains a site implementing a different rule."""
    return [
        {
            "claim": name,
            "as_stated_in_section_4": block["claim"],
            "sites_implementing_a_different_rule": block["dissenting_sites"],
        }
        for name, block in report.items()
        if block["dissenting_sites"]
    ]


def _verdict(report: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    unimplemented = [
        name for name, block in report.items() if not block["implemented_by_every_declared_site"]
    ]
    unquoted = [name for name, block in report.items() if not block["claim_quoted_from_the_readme"]]
    unsatisfied = [
        name for name, block in report.items() if not block["satisfied_by_the_telemetry"]
    ]
    disagreements = _prose_and_code_disagreements(report)
    carried = telemetry["identity_vs_carried_velocity"]
    following = telemetry["identity_vs_next_velocity"]
    with_correction = [name for name, block in report.items() if block.get("corrected_form")]
    default_correction = [
        name for name, block in report.items() if block.get("corrected_by_default")
    ]
    dissent_files = sum(
        len(block["dissenting_sites"]) for block in report.values() if block["dissenting_sites"]
    )
    return {
        "claims_audited": len(report),
        "claims_with_a_missing_implementation": unimplemented,
        "claims_whose_readme_quote_was_not_found": unquoted,
        "claims_the_telemetry_does_not_satisfy": unsatisfied,
        "prose_and_code_disagree": disagreements,
        "claims_with_a_reachable_corrected_form": with_correction,
        "claims_whose_corrected_form_is_the_default": default_correction,
        "diverging_sites": dissent_files,
        "telemetry_prefers": "carried"
        if carried["median_abs_residual_px"] <= following["median_abs_residual_px"]
        else "next",
        "carried_median_residual_px": carried["median_abs_residual_px"],
        "next_median_residual_px": following["median_abs_residual_px"],
        "reading": (
            f"{len(report) - len(unimplemented)} of {len(report)} claims are implemented by every "
            f"file declared to implement them ({unimplemented or 'none missing'}), "
            f"{len(report) - len(unsatisfied)} of {len(report)} are satisfied by the recorded "
            f"transitions as the section now states them ({unsatisfied or 'none failing'}), and "
            f"{len(disagreements)} are still disagreed with by {dissent_files} default code paths: "
            f"the identity is written with the carried velocity "
            f"({carried['exact_to_half_a_subpixel_rate']:.2%} of frames exact, median "
            f"{carried['median_abs_residual_px']:.4f} px) while the shells advance position with "
            f"the predicted next velocity by default "
            f"(median {following['median_abs_residual_px']:.4f} px). The state of the fix is now "
            f"part of the record: {len(with_correction)} of {len(report)} claims have a corrected "
            f"form reachable in the same class or penalty, and "
            f"{len(with_correction) - len(default_correction)} of those are not the default, "
            f"because every artifact in this repository was recorded with the published form. "
            f"10.57 measures what the difference costs per family."
        ),
    }


def _cell_math(latex: str) -> str:
    r"""Quote a §4 claim as a table cell GitHub can render.

    Claims are stored exactly as §4 writes them - delimiters included, because
    `claim_quoted_from_the_readme` compares the stored string with the section - so here
    the delimiters are reduced to inline ones. The pipes then have to go: GitHub splits a
    table row on a raw `|` even inside `$…$`, and the usual escape `\|` is KaTeX's double
    bar, not an absolute value. `\lvert`/`\rvert` keep the glyph and the row.
    """
    body = latex
    if body.startswith("$$") and body.endswith("$$"):
        body = body[2:-2]
    elif body.startswith("$") and body.endswith("$"):
        body = body[1:-1]
    cells: List[str] = []
    left = True
    for char in body:
        if char != "|":
            cells.append(char)
            continue
        # The spaces are load-bearing: `\lvert` is a TeX control word, and `\lvertv_x`
        # would be read as one unknown command.
        cells.append(r"\lvert " if left else r" \rvert ")
        left = not left
    return "$" + " ".join("".join(cells).split()) + "$"


def render_audit_table(payload: Dict[str, Any]) -> List[str]:
    r"""The README rows for this audit, generated from the artifact.

    Section 10.54 quotes one line per physics claim. Because every cell is produced here,
    the citation gate re-runs this function and compares the result to the README, so a
    transcribed figure in the audit table is impossible: a re-run that changes a number
    changes the table text too, and the gate fails until the README is regenerated.
    """
    rows: List[str] = []
    for name, block in payload["claims"].items():
        sites = f"{len(block['sites_found'])}/{block['declared_sites']}"
        flag = "" if block["implemented_by_every_declared_site"] else " (site missing)"
        verdict = "satisfied" if block["satisfied_by_the_telemetry"] else "**not satisfied**"
        dissent = (
            "-"
            if not block["dissenting_sites"]
            else ", ".join(
                sorted({s.split(":")[0].split("/")[-1] for s in block["dissenting_sites"]})
            )
        )
        # The seventh column is the state of the fix, not of the measurement: how a caller reaches
        # the rule §4 states, and whether asking for it is the default. "not the default" is the
        # honest description of every one of them, because the published artifacts used the old
        # form; 10.57 is what the difference costs.
        corrected = block.get("corrected_form")
        reachable = (
            "n/a"
            if not corrected
            else f"`{corrected}`"
            + ("" if block.get("corrected_by_default") else " - not the default")  # noqa: E501
        )
        rows.append(
            rf"| §{name.split('_')[0]} | {_cell_math(block['claim'])} | "
            rf"{sites}{flag} | {block['telemetry_evidence']} "
            rf"| {verdict} | {dissent} | {reachable} |"
        )
    return [demath_typographic(row) for row in rows]


def render_section_4_qualifiers(payload: Dict[str, Any]) -> List[str]:
    r"""The measured qualifier lines that Section 4 itself now carries, one per claim.

    Section 4 used to state rules and leave their measurements in 10.54, three hundred lines
    away, which is how a refuted rule stayed in print. Each qualifier is emitted here and
    pasted into the section verbatim, so the citation gate re-reads §4 from the artifact and a
    rule that stops matching the recording fails CI rather than becoming folklore.
    """
    telemetry = payload["telemetry"]
    speeds = telemetry["speed_classes_observed"]
    window = telemetry["window_observed"]
    ground = telemetry["ground_boundary_condition"]
    strata = telemetry["gravity_step_by_stratum"]
    carried = telemetry["identity_vs_carried_velocity"]
    following = telemetry["identity_vs_next_velocity"]
    free = telemetry["identity_vs_carried_velocity_without_collision_flag"]
    released_descent = strata["descent_released"]
    released_ascent = strata["ascent_released"]
    n = telemetry["n_transitions"]
    rows = [
        rf"*Measured on the {n:,} training transitions of the gameplay recording (§10.54): exact "
        rf"on {carried['exact_to_half_a_subpixel_rate']:.2%} of them, median residual "
        rf"{carried['median_abs_residual_px']:.4f} px. Scored against $\hat v_{{x,t+1}}$ instead "
        rf"it is {following['exact_to_half_a_subpixel_rate']:.2%} and "
        rf"{following['median_abs_residual_px']:.4f} px, so a violation has to be defined with "
        rf"the frame's initial velocity - and on the {free['n_frames']:,} frames that carry no "
        rf"collision flag at all it is exact on {free['exact_to_half_a_subpixel_rate']:.2%} "
        rf"(§10.55 reads the same fact as a fitted integration coefficient).*",
        rf"*Measured: {window['rate_below_jump_window']:.2%} of the recorded vertical velocities "
        rf"lie below $-80$, reaching {window['vy_min']:.1f} - this is the impulse the engine "
        rf"injects at take-off, not a bound it enforces, and a model that clamps to it rewrites "
        rf"real states (§10.54).*",
        rf"*Measured on airborne-ascent frames: {telemetry['gravity_step_on_airborne_ascent']['rate_at_3.0']:.2%} "
        rf"of {telemetry['gravity_step_on_airborne_ascent']['n_frames']:,} step $+3.0$ and "
        rf"{telemetry['gravity_step_on_airborne_ascent']['rate_at_6.0']:.2%} step $+6.0$.*",
        rf"*Measured by stratum (§10.54): released descent is the only stratum where $+6.0$ is "
        rf"common ({released_descent['rate_at_6.0']:.2%} of {released_descent['n_frames']:,} "
        rf"frames), while released *ascent* still steps $+3.0$ in "
        rf"{released_ascent['rate_at_3.0']:.2%} of {released_ascent['n_frames']:,} - the "
        rf"button-state gate is not separable in this recording. §10.52 collected the excitation "
        rf"that shows it and §10.56 identifies its coefficient.*",
        rf"*Measured: {window['rate_above_terminal_velocity']:.2%} of the recorded vertical "
        rf"velocities exceed $+64$, reaching {window['vy_max']:.1f}, and "
        rf"{released_descent['rate_step_exactly_zero']:.2%} of released-descent frames show no "
        rf"vertical increment at all, which is where a clamp would show up if it were enforced "
        rf"here. It is a documented parameter this telemetry never shows the engine applying, so "
        rf"no model may treat $+64$ as an absolute bound on observed data.*",
        rf"*Measured on {n:,} transitions: {speeds['rate_above_walk_cap']:.2%} of frames exceed "
        rf"$20$ and {speeds['rate_above_run_cap']:.4%} exceed $48$, while nothing exceeds $72$ "
        rf"($\max |v_x| = {speeds['max_abs_vx']:.1f}$). These are three speed classes, not three "
        rf"successive bounds, and §10.49 is what re-scored the sections that read $72.0$ as the "
        rf"engine's maximum.*",
        rf"*Measured: on the {ground['n_frames_claimed']:,} grounded frames with no jump "
        rf"commanded the vertical increment is exactly zero on "
        rf"{ground['claimed_rate_step_exactly_zero']:.2%} of them, while the recorded $v_y$ is "
        rf"exactly zero on {ground['claimed_rate_vy_next_exactly_zero']:.2%} (median "
        rf"$|v_y| = {ground['claimed_median_abs_vy_next']:.1f}$, under all three ways of "
        rf"conditioning the stratum). The flag gates the gravity step; it does not stop the body. "
        rf"`GroundContactConsistencyLoss` and `AnalyticalKinematicsDynamics` both implement the "
        rf"retracted form, and the artifact records that as an open prose-and-code disagreement "
        rf"rather than smoothing it over.*",
    ]
    return [unemphasise_math(demath_typographic(row)) for row in rows]


def run_physics_claim_audit(
    dataset_path: str = DATASET_GAMEPLAY, output_dir: str = RESULTS_DIR
) -> Dict[str, Any]:
    """Audit README section 4 and write ``physics_claim_audit_metrics.json``."""
    report = audit_claims()
    telemetry = telemetry_satisfaction(dataset_path=dataset_path)
    score_claims(report, telemetry)
    payload: Dict[str, Any] = {
        "study": (
            "Every physics claim of README section 4 checked against the code that implements "
            "it and against the recorded transitions it describes."
        ),
        "protocol": {
            "readme": "README.md",
            "published_tolerance_px": PUBLISHED_TOLERANCE_PX,
            "subpixels_per_pixel": SUBPIXELS_PER_PIXEL,
            "telemetry_dataset": dataset_path,
            "telemetry_scope": "training split of the canonical gameplay recording",
            "emulator_required": False,
            "section_4_prose_state": "corrected to the measured rules, with the dissenting code recorded",
            "satisfaction_tests": sorted(SATISFACTION_TESTS),
        },
        "claims": report,
        "telemetry": telemetry,
        "verdict": _verdict(report, telemetry),
    }
    artifact = os.path.join(output_dir, "physics_claim_audit_metrics.json")
    write_metrics(
        artifact, payload, seed=42, command="python -m src.evaluation.physics_claim_audit"
    )
    for name, block in report.items():
        logger.info(
            "  [%s/%s] %s | %d/%d sites | %d dissenter(s)",
            "ok" if block["implemented_by_every_declared_site"] else "FAIL",
            "sat" if block["satisfied_by_the_telemetry"] else "NOT SAT",
            name,
            len(block["sites_found"]),
            block["declared_sites"],
            len(block["dissenting_sites"]),
        )
        for missing in block["sites_missing"]:
            logger.info("      missing implementation: %s", missing)
        for gone in block["dissenting_sites_missing"]:
            logger.info("      dissent no longer present: %s", gone)
    logger.info("  telemetry: %s", payload["verdict"]["reading"])
    logger.info("Metrics written to %s", artifact)
    return payload


if __name__ == "__main__":
    import argparse

    from src.utils.config import parse_args_with_config

    parser = argparse.ArgumentParser(
        description="Audit README section 4 against the code and the telemetry (README 10.54)."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parse_args_with_config(parser)

    run_physics_claim_audit(dataset_path=args.dataset_path, output_dir=args.output_dir)
