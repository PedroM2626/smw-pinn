r"""
physics_claim_audit.py
Machine-check the physics statements of README section 4 against code and telemetry (10.53).

Section 4 is the only part of this repository that is *quoted by physics* rather than by
measurement: §4.1 states the integration identity, §4.2 the jump window and the two gravity
tiers, §4.3 the three speed classes. Those statements are then implemented - a residual in
`DiscreteKinematicsLoss`, a predicate in `RolloutEvaluator`, clamps in every shell - and
nothing checked the implementations against each other or against the recorded transitions.

That gap is what 10.49 fell into: §4.3 publishes walk 20, run 48 and sprint-with-P-meter 72
as three separate classes, and four sections scored ceiling estimates against 72.0 as if the
README had said "the maximum velocity". The audit below is the automated version of the
question. For every claim it records:

1. the README text of the claim, so a prose edit that changes the physics fails the gate;
2. every code site declared to implement it, verified by the literal that makes it that
   implementation - a refactor of one of those lines invalidates the audit rather than
   silently passing;
3. what the telemetry says about the claim, scored rather than asserted.

The headline result is in the identity: §4.1 writes $X_{t+1} = X_t + v_{x,t}/16$ and, two
paragraphs later, defines a structural violation as
$\hat X_{t+1} \ne X_t + \hat v_{x,t+1}/16$. The two sentences name different velocities, and
the repository implements both - the penalty and the predicate follow the first, the hard
shells follow the second. The telemetry decides, which is what section 10.53 trains.

Writes ``results/physics_claim_audit_metrics.json``. Emulator-free.

Run:  python -m src.evaluation.physics_claim_audit
"""

import os
from typing import Any, Dict, List

import numpy as np

from src.environment.dataset_loader import load_and_preprocess_data
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

SUBPIXELS_PER_PIXEL = 16.0
PUBLISHED_TOLERANCE_PX = 0.2
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README_PATH = os.path.join(REPO_ROOT, "README.md")

# Claims of README section 4. `sites` are (file, literal) pairs: the literal is the piece of
# source that makes that file an implementation of the claim, and `convention` says which
# velocity the site advances position with, where the claim is the integration identity.
CLAIMS: Dict[str, Dict[str, Any]] = {
    "4.1_integration_identity_carried_velocity": {
        "readme_marker": "### 4.1 Fixed-Point Arithmetic",
        "claim": r"$$X_{t+1} = X_t + \frac{v_{x, t}}{16.0}$$",
        "convention": "carried",
        "sites": [
            ("src/losses/physics_losses.py", "expected_dx = vx_t / self.subpixels_per_pixel"),
            ("src/evaluation/rollout_evaluator.py", "curr_state[0, 2].item() / 16.0"),
            ("src/evaluation/rollout_diagnostics.py", "v_prev = pred_traj[:-1, 2]"),
            (
                "src/models/effective_velocity_dynamics.py",
                "used_vx, used_vy = vx_t + offset_vx, vy_t + offset_vy",
            ),
        ],
    },
    "4.1_violation_definition_next_velocity": {
        "readme_marker": "### 4.1 Fixed-Point Arithmetic",
        "claim": r"\hat{X}_{t+1} \ne X_t + \frac{\hat{v}_{x, t+1}}{16.0}",
        "convention": "next",
        "sites": [
            ("src/models/residual_dynamics.py", "hat_x = x_t + hat_vx / self.subpixels_per_pixel"),
            ("src/models/output_projection.py", "x = state[:, 0] + vx / self.subpixels_per_pixel"),
        ],
    },
    "4.2.1_jump_impulse_window": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"v_{y, 0} \in [-64, -80]",
        "sites": [
            ("src/models/analytical_kinematics.py", "MIN_VY = -80.0"),
            ("src/losses/physics_losses.py", "min_vy: float = -80.0"),
        ],
    },
    "4.2.3_4.2.4_gravity_tiers": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"g_{\text{held}} = +3.0",
        "sites": [
            ("src/models/analytical_kinematics.py", "G_HOLD = 3.0"),
            ("src/models/analytical_kinematics.py", "G_FALL = 6.0"),
        ],
    },
    "4.2.5_terminal_velocity": {
        "readme_marker": "### 4.2 Vertical Dynamics",
        "claim": r"v_{y} \le v_{y, \text{term}} = +64.0",
        "sites": [
            ("src/models/analytical_kinematics.py", "TERMINAL_VY = 64.0"),
            ("src/losses/physics_losses.py", "terminal_vy: float = 64.0"),
            ("src/evaluation/rollout_evaluator.py", "hat_vy > 64.0"),
        ],
    },
    "4.3.1_walk_speed": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"|v_x| \le 20",
        "sites": [("src/models/analytical_kinematics.py", "VX_WALK = 20.0")],
    },
    "4.3.2_run_speed": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"|v_x| \le 48",
        "sites": [("src/models/analytical_kinematics.py", "VX_RUN = 48.0")],
    },
    "4.3.3_sprint_speed": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"|v_x| \le 72",
        "sites": [
            ("src/models/analytical_kinematics.py", "VX_SPRINT = 72.0"),
            ("src/models/residual_dynamics.py", "max_vx: float = 72.0"),
            ("src/models/output_projection.py", "max_vx: float = 72.0"),
            ("src/losses/physics_losses.py", "max_vx: float = 72.0"),
            ("src/evaluation/rollout_evaluator.py", "abs(hat_vx) > 72.0"),
        ],
    },
    "4.3.5_ground_boundary_condition": {
        "readme_marker": "### 4.3 Horizontal Dynamics",
        "claim": r"$$v_{y, t+1} = 0$$",
        "sites": [
            ("src/losses/physics_losses.py", "stationary_ground_mask = (ground_contact > 0.5)"),
        ],
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
    """Check each claim's README text and every declared implementation site."""
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
        report[name] = {
            "claim": spec["claim"],
            "readme_marker": spec["readme_marker"],
            "claim_quoted_from_the_readme": spec["claim"] in section,
            "declared_sites": len(spec["sites"]),
            "sites_found": found,
            "sites_missing": missing,
            "implemented_by_every_declared_site": not missing,
            "convention": spec.get("convention"),
        }
    return report


def telemetry_satisfaction(dataset_path: str = DATASET_GAMEPLAY, seed: int = 42) -> Dict[str, Any]:
    """Score README section 4's claims against the recorded WRAM transitions."""
    data = load_and_preprocess_data(dataset_path=dataset_path, seed=seed)
    states = np.asarray(data["train_states"], dtype=np.float64)
    next_states = np.asarray(data["train_next_states"], dtype=np.float64)
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

    def residual_report(d: np.ndarray, velocity: np.ndarray) -> Dict[str, float]:
        residual = np.abs(d - velocity / SUBPIXELS_PER_PIXEL)
        return {
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
        "ground_boundary_condition": _ground_condition_rate(
            states, np.asarray(data["train_actions"], dtype=np.float64), next_states
        ),
    }


def _ground_condition_rate(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> Dict[str, float]:
    """How often the engine really forces $v_{y,t+1}=0$ on a grounded, un-jumping frame.

    Conditioned exactly the way §4.3.5 states it and the way `GroundContactConsistencyLoss`
    reads it - the ground flag of the state the frame starts from, no jump commanded, the
    vertical velocity of the state it ends in - so the claim and its penalty are scored on
    the same stratum.
    """
    commanded = (states[:, 4] > 0.5) & (actions[:, 0] < 0.5)
    landed = (next_states[:, 4] > 0.5) & (actions[:, 0] < 0.5)
    vy_next = next_states[:, 3]
    out: Dict[str, float] = {"n_frames_claimed": int(commanded.sum())}
    for name, mask in (("claimed", commanded), ("landed_this_frame", landed)):
        out[f"{name}_rate_vy_next_exactly_zero"] = (
            float(np.mean(vy_next[mask] == 0.0)) if mask.any() else None
        )
        out[f"{name}_median_abs_vy_next"] = (
            float(np.median(np.abs(vy_next[mask]))) if mask.any() else None
        )
    # Scored against the velocity the frame starts with too, so the result cannot be
    # dismissed as an indexing artifact of reading the state after the update.
    out["claimed_rate_vy_current_exactly_zero"] = (
        float(np.mean(states[commanded, 3] == 0.0)) if commanded.any() else None
    )
    out["claimed_median_abs_vy_current"] = (
        float(np.median(np.abs(states[commanded, 3]))) if commanded.any() else None
    )
    return out


def _conflicts(report: Dict[str, Any]) -> List[str]:
    """Which pairs of claims cannot both be the engine's rule."""
    conventions = {
        name: block["convention"] for name, block in report.items() if block.get("convention")
    }
    if len(set(conventions.values())) > 1:
        return [
            "integration identity: "
            + "; ".join(
                f"{name} -> {convention}" for name, convention in sorted(conventions.items())
            )
        ]
    return []


def _refuted_by_telemetry(telemetry: Dict[str, Any]) -> List[str]:
    """Claims the recorded transitions do not satisfy, judged by their own test."""
    out: List[str] = []
    carried = telemetry["identity_vs_carried_velocity"]["median_abs_residual_px"]
    following = telemetry["identity_vs_next_velocity"]["median_abs_residual_px"]
    if following > carried:
        out.append("4.1_violation_definition_next_velocity")
    window = telemetry["window_observed"]
    if window["rate_above_terminal_velocity"] > 0.0:
        out.append("4.2.5_terminal_velocity")
    if window["rate_below_jump_window"] > 0.0:
        out.append("4.2.1_jump_impulse_window")
    if telemetry["speed_classes_observed"]["rate_above_sprint_cap"] > 0.0:
        out.append("4.3.3_sprint_speed")
    ground = telemetry["ground_boundary_condition"]["claimed_rate_vy_next_exactly_zero"]
    if ground is not None and ground < 0.5:
        out.append("4.3.5_ground_boundary_condition")
    return sorted(out)


def _verdict(report: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    unimplemented = [
        name for name, block in report.items() if not block["implemented_by_every_declared_site"]
    ]
    unquoted = [name for name, block in report.items() if not block["claim_quoted_from_the_readme"]]
    carried = telemetry["identity_vs_carried_velocity"]
    following = telemetry["identity_vs_next_velocity"]
    refuted = _refuted_by_telemetry(telemetry)
    return {
        "claims_audited": len(report),
        "claims_with_a_missing_implementation": unimplemented,
        "claims_whose_readme_quote_was_not_found": unquoted,
        "claims_the_telemetry_does_not_satisfy": refuted,
        "contradictory_claim_pairs": _conflicts(report),
        "telemetry_prefers": "carried"
        if carried["median_abs_residual_px"] <= following["median_abs_residual_px"]
        else "next",
        "carried_median_residual_px": carried["median_abs_residual_px"],
        "next_median_residual_px": following["median_abs_residual_px"],
        "reading": (
            f"{len(report) - len(unimplemented)} of {len(report)} claims are implemented by every "
            f"file declared to implement them ({unimplemented or 'none missing'}). Two of the "
            f"claims describe the same integration identity with different velocities, and the "
            f"repository implements both: the penalty and the published predicate advance position "
            f"with the velocity the frame starts with, the hard shells with the velocity predicted "
            f"for the next. The recorded transitions settle it - median residual "
            f"{carried['median_abs_residual_px']:.4f} px for the carried velocity against "
            f"{following['median_abs_residual_px']:.4f} px for the next one, and "
            f"{carried['exact_to_half_a_subpixel_rate'] * 100:.2f}% of frames integrate with the "
            f"carried velocity exactly."
        ),
    }


def render_audit_table(payload: Dict[str, Any]) -> List[str]:
    r"""The README rows for this audit, generated from the artifact.

    Section 10.53 quotes one line per physics claim. Because every cell is produced here,
    the citation gate re-runs this function and compares the result to the README, so a
    transcribed figure in the audit table is impossible: a re-run that changes a number
    changes the table text too, and the gate fails until the README is regenerated.
    """
    claims, telemetry = payload["claims"], payload["telemetry"]
    identity = telemetry["identity_vs_carried_velocity"]
    following = telemetry["identity_vs_next_velocity"]
    window = telemetry["window_observed"]
    gravity = telemetry["gravity_step_on_airborne_ascent"]
    speeds = telemetry["speed_classes_observed"]
    ground = telemetry["ground_boundary_condition"]
    verdicts = {
        "4.1_integration_identity_carried_velocity": (
            rf"median {identity['median_abs_residual_px']:.4f} px, exact on "
            rf"{identity['exact_to_half_a_subpixel_rate']:.2%} of frames"
        ),
        "4.1_violation_definition_next_velocity": (
            rf"median {following['median_abs_residual_px']:.4f} px, exact on "
            f"{following['exact_to_half_a_subpixel_rate']:.2%}"
        ),
        "4.2.1_jump_impulse_window": (
            rf"{window['rate_below_jump_window']:.2%} of velocities below $-80$ "
            rf"(min {window['vy_min']:.1f})"
        ),
        "4.2.3_4.2.4_gravity_tiers": (
            rf"{gravity['rate_at_3.0']:.2%} of {gravity['n_frames']} airborne-ascent frames step "
            rf"$3.0$, {gravity['rate_at_6.0']:.2%} step $6.0$"
        ),
        "4.2.5_terminal_velocity": (
            rf"{window['rate_above_terminal_velocity']:.2%} above $+64$ "
            rf"(max {window['vy_max']:.1f})"
        ),
        "4.3.1_walk_speed": rf"{speeds['rate_above_walk_cap']:.2%} of frames above $20$",
        "4.3.2_run_speed": rf"{speeds['rate_above_run_cap']:.4%} of frames above $48$",
        "4.3.3_sprint_speed": (
            rf"{speeds['rate_above_sprint_cap']:.2%} above $72$, max observed $\|v_x\| = "
            rf"{speeds['max_abs_vx']:.1f}$"
        ),
        "4.3.5_ground_boundary_condition": (
            rf"the recorded vertical velocity is exactly zero on "
            rf"{ground['claimed_rate_vy_next_exactly_zero']:.2%} of "
            rf"{ground['n_frames_claimed']} frames, median $\|v_y\| = "
            rf"{ground['claimed_median_abs_vy_next']:.1f}$"
        ),
    }
    rows: List[str] = []
    for name, block in claims.items():
        sites = f"{len(block['sites_found'])}/{block['declared_sites']}"
        flag = "" if block["implemented_by_every_declared_site"] else " (site missing)"
        # The claim is quoted from section 4, where it is written as display math with bare
        # pipes; inside a table cell it has to be inline and escaped or it splits the row.
        quote = block["claim"].replace("$$", "$").replace("|", r"\|")
        rows.append(
            rf"| §{name.split('_')[0]} | {quote} | {sites}{flag} | "
            f"{verdicts[name]} |"
        )
    return rows


def run_physics_claim_audit(
    dataset_path: str = DATASET_GAMEPLAY, output_dir: str = RESULTS_DIR
) -> Dict[str, Any]:
    """Audit README section 4 and write ``physics_claim_audit_metrics.json``."""
    report = audit_claims()
    telemetry = telemetry_satisfaction(dataset_path=dataset_path)
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
            "  [%s] %s | %d/%d sites",
            "ok" if block["implemented_by_every_declared_site"] else "FAIL",
            name,
            len(block["sites_found"]),
            block["declared_sites"],
        )
        for missing in block["sites_missing"]:
            logger.info("      missing implementation: %s", missing)
    logger.info("  telemetry: %s", payload["verdict"]["reading"])
    logger.info("Metrics written to %s", artifact)
    return payload


if __name__ == "__main__":
    import argparse

    from src.utils.config import parse_args_with_config

    parser = argparse.ArgumentParser(
        description="Audit README section 4 against the code and the telemetry (README 10.53)."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parse_args_with_config(parser)

    run_physics_claim_audit(dataset_path=args.dataset_path, output_dir=args.output_dir)
