"""
learned_structure_probe_benchmark.py
Do learned dynamics models *contain* the engine's constraints? (README Section 10.46)

Sections 10.41 and 10.42 trained the neural-operator family and scored it the way this
repository scores everything: single-step error, multi-start rollout drift and a physical-
consistency rate. Sections 10.40 and 10.43 then asked a different question of the *inverse*
problem - can the engine's constants and constraints be recovered from transitions - and
found that recovery is budget-limited for tree search and, on real telemetry, limited by
what the recording ever excites.

Those two lines never met. This study puts the trained models under the same structural
interrogation the discovered laws were, with the same probes and the same acceptance rules:

* the **velocity ceiling** is the fixed point of the fully-driven map, ``v -> f(v)``,
  rejected when the map never accelerates, so a flat model is not misread as a bound;
* the **held-jump gravity gate** is the median tier separation between jump-held and
  jump-free ascending probes in free flight.

One distinction decides whether a positive result means anything, and it is recorded per
model rather than assumed: the Hard Residual PINN and the Physics-Constrained DeepONet
integrate their learned residual through a hard kinematic shell that *clamps* ``vx`` at a
constructor constant. For those two, a fixed point is a property of the architecture a
human wrote, not something the network learned, and the probe is really checking that the
clamp is where its author put it. The informative cases are the models with no clamp at
all - MLP, Soft PINN, plain DeepONet, FNO.

The second half answers a question the repository has been asked but never measured: why not
identify the constants *through* a trained operator, differentiating end to end? The test is
to run the published identification twice - once against real transitions, once against
pseudo-transitions rolled out by each learned model - and look at where the two estimates
land. A surrogate can only be as right as the data it approximates, so the gap between the
two arms is a direct measurement of that model's bias expressed in physical units of the
constants themselves.

Emulator-free: it loads committed checkpoints. Writes
``results/learned_structure_probe_metrics.json`` with ``_meta`` and
``results/figures/learned_structure_probe.png``.

Run:  python -m src.evaluation.learned_structure_probe_benchmark
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluation.inverse_transfer_benchmark import PRIOR  # noqa: E402
from src.evaluation.symbolic_engine_ablation_benchmark import _bound_fixed_point  # noqa: E402
from src.inverse.parameter_identification import (  # noqa: E402
    PARAM_NAMES,
    identify_params,
    make_windows,
    theta_tensor,
)
from src.models.deeponet import (  # noqa: E402
    DeepONetDynamics,
    PhysicsConstrainedDeepONetDynamics,
)
from src.models.fno import FNODynamics  # noqa: E402
from src.models.pinn_hard_residual import HardResidualPINNDynamics  # noqa: E402
from src.models.pinn_soft import SoftPINNDynamics  # noqa: E402
from src.models.statistical_mlp import StatisticalMLPDynamics  # noqa: E402
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

ARTIFACT_NAME = "learned_structure_probe_metrics.json"
FIGURE_NAME = "learned_structure_probe.png"

# (label, checkpoint, factory, clamp-imposed-by-construction, ceiling-the-shell-uses)
MODEL_REGISTRY: Tuple[Tuple[str, str, Callable[[], torch.nn.Module], bool, float], ...] = (
    (
        "mlp",
        "mlp_best.pt",
        lambda: StatisticalMLPDynamics(state_dim=8, action_dim=6),
        False,
        float("nan"),
    ),
    (
        "soft_pinn",
        "pinn_soft_best.pt",
        lambda: SoftPINNDynamics(state_dim=8, action_dim=6),
        False,
        float("nan"),
    ),
    (
        "hard_pinn",
        "pinn_hard_best.pt",
        lambda: HardResidualPINNDynamics(state_dim=8, action_dim=6),
        True,
        72.0,
    ),
    (
        "deeponet",
        "deeponet_best.pt",
        lambda: DeepONetDynamics(state_dim=8, action_dim=6, latent_dim=64),
        False,
        float("nan"),
    ),
    (
        "physics_constrained_deeponet",
        "operator_physicsconstrained_deeponet_best.pt",
        lambda: PhysicsConstrainedDeepONetDynamics(state_dim=8, action_dim=6, latent_dim=64),
        True,
        72.0,
    ),
    (
        "fno",
        "operator_fno_best.pt",
        lambda: FNODynamics(state_dim=8, action_dim=6, width=32, modes=6, n_layers=2),
        False,
        float("nan"),
    ),
)

# The action that drives Mario right at full sprint, and the two jump states of the gate
# probe, in the repository's [B, Y, UP, DOWN, LEFT, RIGHT] channel order.
SPRINT_ACTION = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# The engine's own top per-frame increment is 1.8 sub-pixels (walk 1.0 plus the run tier);
# 2.5 leaves room for a learned model's error without letting a 12-pixel artifact through.
PLAUSIBLE_MAX_ACCELERATION = 2.5


def load_model(checkpoint: str, factory: Callable[[], torch.nn.Module], device: torch.device):
    """Reload one committed dynamics checkpoint, or return None if it is absent."""
    path = os.path.join("results", "checkpoints", checkpoint)
    if not os.path.isfile(path):
        logger.warning("checkpoint %s missing; the model is recorded as unavailable", path)
        return None
    model = factory().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def device_of(model: torch.nn.Module) -> torch.device:
    """The device a loaded model lives on, so the probes never move tensors by hand."""
    return next(model.parameters()).device


def _step(model: torch.nn.Module, rows: np.ndarray, action: Sequence[float], device) -> np.ndarray:
    """One model call on physical numpy rows, returning the next-state rows."""
    with torch.no_grad():
        state = torch.as_tensor(np.asarray(rows, dtype=np.float32), device=device)
        act = torch.as_tensor(np.tile(np.asarray(action, dtype=np.float32), (state.shape[0], 1)))
        out = model(state, act.to(device))
    return out.detach().cpu().numpy().astype(np.float64)


def _state_with(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """8D rows carrying only the two velocity channels the probes read."""
    n = vx.shape[0]
    state = np.zeros((n, 8), dtype=np.float64)
    state[:, 0] = 100.0  # x, away from the level edges
    state[:, 1] = 200.0  # y, mid-screen
    state[:, 2] = vx
    state[:, 3] = vy
    return state


def probe_ceiling(
    model: torch.nn.Module, cap: float, device: torch.device, rows: int = 193
) -> Dict[str, float]:
    """Fixed point of the fully-driven map, classified rather than merely found.

    The 10.43 acceptance rule rejects a map that never accelerates, which is the right
    guard for a genetic program that collapsed to a constant. It is not enough for a
    learned model extrapolating past its recording: such a network can accelerate by
    twelve pixels a frame and then come back down through the diagonal at v = -0.5, and
    the 10.43 rule would report that crossing as a velocity ceiling. So the raw answer is
    kept for transparency and a stricter classification is computed alongside it. A
    crossing counts as a *ceiling* only if it is positive, lies within the observed
    support, is approached from below with the map still accelerating, and is stable - the
    map pulls back toward it from above as well.
    """

    def predict_next(v: np.ndarray) -> np.ndarray:
        return _step(model, _state_with(v, np.zeros_like(v)), SPRINT_ACTION, device)[:, 2]

    raw = _bound_fixed_point(predict_next, cap, rows=rows)
    # A thousandth of a pixel per frame: far below the engine's smallest real increment
    # (1.0), and far above the float32 round-trip noise these models are evaluated at.
    eps = 1e-3
    v = np.linspace(0.0, cap, rows)
    nxt = predict_next(v)
    delta = nxt - v
    accelerating = bool(np.any(delta > eps))
    crossings = np.where(delta <= eps)[0]
    ceiling: Optional[float] = None
    slope = float("nan")
    for i in crossings:
        level = float(nxt[i])
        if not 0.0 < level <= cap:
            continue
        below = v[v < level]
        if below.size:
            d_below = predict_next(below[-min(6, below.size) :]) - below[-min(6, below.size) :]
            if not bool(np.all(d_below > eps)):
                continue  # not approached from below while accelerating
        above = v[v > level]
        if above.size:
            d_above = predict_next(above[: min(6, above.size) :]) - above[: min(6, above.size) :]
            if not bool(np.all(d_above < eps)):
                continue  # unstable: the map runs away upward past its own crossing
        idx = max(min(int(round(level / cap * (rows - 1))), rows - 2), 1)
        slope = float((delta[idx] - delta[idx - 1]) / (v[idx] - v[idx - 1]))
        ceiling = level
        break
    return {
        "fixed_point_found_10_43_rule": raw["fixed_point_found"],
        "bound_value_10_43_rule": raw["bound_value"],
        "overshoot": raw["overshoot"],
        "accelerating": float(accelerating),
        "ceiling_like_fixed_point": float("nan") if ceiling is None else ceiling,
        "has_ceiling": float(ceiling is not None),
        "local_slope_at_ceiling": slope,
        "crossings_in_range": int(crossings.size),
    }


def probe_gravity_gate(
    model: torch.nn.Module, cap: float, device: torch.device
) -> Dict[str, float]:
    """Median tier separation between jump-held and jump-free ascending probes."""
    vy = np.linspace(-0.9 * cap, -0.1 * cap, 61)
    held = _step(model, _state_with(np.zeros_like(vy), vy), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], device)
    free = _step(model, _state_with(np.zeros_like(vy), vy), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device)
    inc_held = held[:, 3] - vy
    inc_free = free[:, 3] - vy
    return {
        "held_gravity": float(np.median(inc_held)),
        "fall_gravity": float(np.median(inc_free)),
        "tier_separation": float(np.median(inc_held) - np.median(inc_free)),
    }


def probe_acceleration_gain(model: torch.nn.Module, cap: float, device: torch.device) -> float:
    """How much the driven map actually accelerates - the flatness the probe must not reward."""
    v = np.linspace(0.0, cap, 61)
    nxt = _step(model, _state_with(v, np.zeros_like(v)), SPRINT_ACTION, device)[:, 2]
    return float(np.max(nxt - v))


def surrogate_windows(
    model: torch.nn.Module,
    states: np.ndarray,
    actions: np.ndarray,
    rollout_len: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll the learned model forward from the real initial states under the real actions.

    The result is pseudo-telemetry: same starting points and same inputs as the recorded
    windows, but the trajectory is the model's opinion. Identification against it therefore
    recovers the constants that best explain *the model*. Contact channels come along from
    the model's own output, since what is being substituted is the dynamics.

    Returns the pseudo next-state rows and, for each, the index of the real transition it
    stands in for - so the episode tags of the surrogate windows match the real ones.
    """
    n = int(states.shape[0])
    starts = np.arange(0, n - rollout_len, rollout_len)
    if starts.size == 0:
        raise ValueError(f"dataset of {n} rows cannot hold a rollout of {rollout_len}")
    out = np.zeros((starts.size * rollout_len, states.shape[1]), dtype=np.float64)
    source = np.zeros(starts.size * rollout_len, dtype=np.int64)
    for index, start in enumerate(starts):
        state = states[start].astype(np.float64)[None, :]
        for step in range(rollout_len):
            nxt = _step(model, state, actions[start + step], device)
            row = index * rollout_len + step
            out[row] = nxt[0]
            source[row] = start + step
            state = nxt
    return out, source


def run_surrogate_identification(
    data: Dict[str, np.ndarray],
    models: Dict[str, Any],
    reference: np.ndarray,
    steps: int,
    rollout_len: int,
    seed: int,
) -> Dict[str, Any]:
    """Identify the constants twice: against the recording, and against each model."""
    windows_real = make_windows(
        data["train_states"],
        data["train_actions"],
        data["train_next_states"],
        data["train_episodes"],
        rollout_len=rollout_len,
    )
    theta_real, info_real = identify_params(
        windows_real, theta_tensor(PRIOR), steps=steps, lr=0.05, seed=seed
    )
    real_errors = {
        n: float(100.0 * abs(float(theta_real[i]) - float(reference[i])) / abs(float(reference[i])))
        for i, n in enumerate(PARAM_NAMES)
    }
    out: Dict[str, Any] = {
        "reference_wram": {n: float(reference[i]) for i, n in enumerate(PARAM_NAMES)},
        "identification_on_real_transitions": {
            n: {"value": float(theta_real[i]), "rel_error_pct": real_errors[n]}
            for i, n in enumerate(PARAM_NAMES)
        },
        "real_max_relative_error_pct": max(real_errors.values()),
        "real_final_loss": float(info_real.get("final_loss", float("nan"))),
        "per_surrogate": {},
    }
    states = np.asarray(data["train_states"], dtype=np.float64)
    actions = np.asarray(data["train_actions"], dtype=np.float64)
    episodes = np.asarray(data["train_episodes"])
    for label, model in models.items():
        pseudo, source = surrogate_windows(model, states, actions, rollout_len, device_of(model))
        windows_s = make_windows(
            states[source], actions[source], pseudo, episodes[source], rollout_len=rollout_len
        )
        theta_s, info_s = identify_params(
            windows_s, theta_tensor(PRIOR), steps=steps, lr=0.05, seed=seed
        )
        drift = {
            n: {
                "value": float(theta_s[i]),
                "rel_error_pct": float(
                    100.0 * abs(float(theta_s[i]) - float(reference[i])) / abs(float(reference[i]))
                ),
                "distance_from_real_estimate": float(abs(float(theta_s[i]) - float(theta_real[i]))),
            }
            for i, n in enumerate(PARAM_NAMES)
        }
        out["per_surrogate"][label] = {
            "available": True,
            "constants": drift,
            "max_relative_error_pct": max(float(v["rel_error_pct"]) for v in drift.values()),
            "max_drift_from_real_identification": max(
                float(v["distance_from_real_estimate"]) for v in drift.values()
            ),
            "final_loss": float(info_s.get("final_loss", float("nan"))),
        }
    return out


def build_verdict(probes: Dict[str, Any], surrogate: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble both conclusions from what was measured, separating learned from imposed."""
    learned_bound, imposed_bound, no_bound, implausible = [], [], [], []
    for label, block in probes.items():
        if not block.get("available", False):
            continue
        if not block["ceiling"]["has_ceiling"]:
            no_bound.append(label)
        elif not block["ceiling_is_plausible"]:
            implausible.append(label)
        elif block["shell_clamp_binds_within_support"] and abs(
            float(block["ceiling"]["ceiling_like_fixed_point"]) - block["shell_clamp_constant"]
        ) <= 0.05 * abs(block["shell_clamp_constant"]):
            imposed_bound.append(label)
        else:
            learned_bound.append(label)
    worst = max(
        (
            (label, block["max_relative_error_pct"])
            for label, block in surrogate["per_surrogate"].items()
            if block.get("available")
        ),
        key=lambda pair: pair[1],
        default=("none", float("nan")),
    )
    best = min(
        (
            (label, block["max_relative_error_pct"])
            for label, block in surrogate["per_surrogate"].items()
            if block.get("available")
        ),
        key=lambda pair: pair[1],
        default=("none", float("nan")),
    )
    return {
        "models_with_a_fixed_point": sorted(learned_bound + imposed_bound),
        "fixed_point_learned": sorted(learned_bound),
        "fixed_point_imposed_by_architecture": sorted(imposed_bound),
        "models_without_a_fixed_point": sorted(no_bound),
        "crossing_without_plausible_traction": sorted(implausible),
        "surrogate_identification": {
            "worst_model": worst[0],
            "worst_max_relative_error_pct": worst[1],
            "best_model": best[0],
            "best_max_relative_error_pct": best[1],
        },
        "reading": (
            f"Of the {sum(1 for b in probes.values() if b.get('available'))} trained models "
            f"probed, {len(learned_bound) + len(imposed_bound) + len(implausible)} reach a "
            f"stable positive fixed point of the driven map, but only "
            f"{len(learned_bound) + len(imposed_bound)} of them do so with a traction the "
            f"engine could have: {len(implausible)} fold back only after accelerating past "
            f"{PLAUSIBLE_MAX_ACCELERATION} px/frame, which is an extrapolation artifact rather "
            f"than a bound ("
            + ", ".join(sorted(implausible) or ["none"])
            + f"), {len(imposed_bound)} sit at a clamp their own shell wrote inside the observed "
            f"support, and {len(no_bound)} show no crossing at all. A learned model having a "
            "plateau is therefore not evidence that it contains the engine's ceiling - the "
            "plateau's level, and the acceleration needed to reach it, are what decide that."
        ),
        "surrogate_reading": (
            "Identifying the constants through a learned surrogate instead of the recording "
            f"moves the worst-case relative error to {worst[1]:.1f}% ({worst[0]}) and the best "
            f"case to {best[1]:.1f}% ({best[0]}), against the direct fit's own worst case of "
            f"{float(surrogate['real_max_relative_error_pct']):.1f}%: "
            "the extra error is the surrogate's bias, and it is strictly additive because the "
            "exact forward model is already available and differentiable."
        ),
    }


def _render_figure(probes: Dict[str, Any], path: str) -> str:
    labels = [k for k, v in probes.items() if v.get("available")]
    found = [float(probes[k]["ceiling"]["has_ceiling"]) for k in labels]
    raw = [float(probes[k]["ceiling"]["fixed_point_found_10_43_rule"]) for k in labels]
    gain = [probes[k]["acceleration_gain_px_per_frame"] for k in labels]
    imposed = [float(bool(probes[k]["bound_imposed_by_construction"])) for k in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = np.arange(len(labels))
    ax1.bar(x, raw, width=0.38, x=x - 0.19, color="#b0b0b0", label="10.43 rule: any crossing")
    ax1.bar(
        x,
        found,
        width=0.38,
        x=x + 0.19,
        color=["#ff7f0e" if i else "#1f77b4" for i in imposed],
        label="classified ceiling",
    )
    ax1.legend(fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=7)
    ax1.set_ylim(0, 1.3)
    ax1.set_ylabel("driven map has a fixed point")
    ax1.set_title("Is the ceiling in the model?", fontweight="bold")
    for i, (v, imp) in enumerate(zip(found, imposed)):
        ax1.text(
            i, v + 0.05, "imposed" if imp and v else ("yes" if v else "no"), ha="center", fontsize=7
        )
    ax2.bar(x, gain, color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=7)
    ax2.set_ylabel("max driven acceleration (px/frame)")
    ax2.set_title("Does it accelerate at all?", fontweight="bold")
    for i, v in enumerate(gain):
        ax2.text(i, v + max(gain) * 0.02, f"{v:.2f}", ha="center", fontsize=7)
    fig.suptitle(
        "Section 10.46 - structural probes on the committed learned dynamics models",
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Figure saved to: %s", path)
    return path


def run_study(
    output_dir: str = "results",
    id_steps: int = 900,
    rollout_len: int = 8,
    probe_rows: int = 193,
    seed: int = 42,
) -> Dict[str, Any]:
    """Probe every committed learned model for the engine's two structural signatures."""
    set_global_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = np.array([float(v) for v in theta_tensor(PRIOR).tolist()])

    from src.environment.dataset_loader import load_and_preprocess_data

    data = load_and_preprocess_data(seed=seed)
    cap = float(np.abs(np.asarray(data["train_next_states"], dtype=np.float64)[:, 2]).max())
    logger.info("Observed velocity support on the training split: %.3f px/frame", cap)

    models: Dict[str, Any] = {}
    probes: Dict[str, Any] = {}
    for label, checkpoint, factory, imposed, shell_ceiling in MODEL_REGISTRY:
        model = load_model(checkpoint, factory, device)
        if model is None:
            probes[label] = {"available": False, "reason": f"checkpoint {checkpoint} absent"}
            continue
        models[label] = model
        ceiling = probe_ceiling(model, cap, device, rows=probe_rows)
        gate = probe_gravity_gate(model, cap, device)
        gain = probe_acceleration_gain(model, cap, device)
        probes[label] = {
            "available": True,
            "checkpoint": checkpoint,
            "bound_imposed_by_construction": bool(imposed),
            "shell_clamp_constant": shell_ceiling,
            # A clamp above the observed support never binds, so a crossing below it cannot
            # be attributed to the architecture that wrote it.
            "shell_clamp_binds_within_support": bool(
                imposed and np.isfinite(shell_ceiling) and shell_ceiling <= cap
            ),
            "ceiling": ceiling,
            "acceleration_gain_px_per_frame": gain,
            # A crossing is only a ceiling if the model reaches it with a plausible traction:
            # the engine's largest per-frame increment is 1.8, and a network that adds twelve
            # pixels a frame before folding back has an extrapolation artifact, not a bound.
            "ceiling_is_plausible": bool(
                ceiling["has_ceiling"] > 0.5 and gain <= PLAUSIBLE_MAX_ACCELERATION
            ),
            "gravity_gate": gate,
            "gravity_gate_found": bool(
                abs(gate["tier_separation"]) > 0.5 * abs(reference[4] - reference[5])
            ),
        }
        logger.info(
            "  %-30s ceiling %-8s (10.43 rule %s, imposed %s) | gain %.2f | gate sep %.2f",
            label,
            "none"
            if not np.isfinite(ceiling["ceiling_like_fixed_point"])
            else f"{ceiling['ceiling_like_fixed_point']:.3f}",
            bool(ceiling["fixed_point_found_10_43_rule"] > 0.5),
            imposed,
            gain,
            gate["tier_separation"],
        )

    logger.info("=== Identification through the learned surrogates ===")
    surrogate = run_surrogate_identification(data, models, reference, id_steps, rollout_len, seed)

    payload: Dict[str, Any] = {
        "study": (
            "The trained dynamics models of sections 8, 10.41 and 10.42 are interrogated with "
            "the structural probes of 10.43/10.43.9 - the fixed point of the driven map and the "
            "held-jump tier separation - and the identification of 10.40 is re-run through each "
            "learned model as a surrogate forward, to measure what that substitution costs."
        ),
        "protocol": {
            "velocity_support_cap": cap,
            "probe_rows": probe_rows,
            "acceptance_rule": "a crossing counts as a ceiling only if it is positive, inside the observed support, approached from below while the map still accelerates, and stable from above; the raw 10.43 rule (which asks only that the map accelerate somewhere) is reported beside it for transparency",
            "surrogate_identification": "Adam on the softplus-constrained 7-constant map, warm-started from the WRAM prior, same steps and lr as 10.40-E2",
            "surrogate_rollout_len": rollout_len,
            "seed": seed,
            "models": [label for label, *_ in MODEL_REGISTRY],
        },
        "structural_probes": probes,
        "surrogate_identification": surrogate,
        "verdict": build_verdict(probes, surrogate),
    }
    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command="python -m src.evaluation.learned_structure_probe_benchmark",
        extra_meta={"models": list(probes)},
    )
    logger.info("Artifact written to: %s", artifact)
    figure = _render_figure(probes, os.path.join(output_dir, "figures", FIGURE_NAME))
    payload["_artifact"] = artifact
    payload["_figure"] = figure
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--id-steps", dest="id_steps", type=int, default=900)
    parser.add_argument("--rollout-len", dest="rollout_len", type=int, default=8)
    parser.add_argument("--probe-rows", dest="probe_rows", type=int, default=193)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    run_study(
        output_dir=args.output_dir,
        id_steps=args.id_steps,
        rollout_len=args.rollout_len,
        probe_rows=args.probe_rows,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
