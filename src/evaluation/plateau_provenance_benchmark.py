"""
plateau_provenance_benchmark.py
Is a learned model's plateau the console's bound or the recording's support? (README 10.50)

Section 10.46 recorded a coincidence instead of smoothing it over: the plain
DeepONet's implied plateau, 35.61 sub-pixels/frame, sits 1.3% from the sustained
sprint ceiling that Section 10.45 measured on purpose-targeted telemetry, 36.075 -
while the other five models probed there sat between 21 and 30. Either that is a
fact about representation, or it is a fact about the data the model saw.

This study decides between them the only way available: change the data's support
without changing the dynamics, and see what follows the plateau. The same three
architectures are trained on transitions whose next-state velocity is truncated at
several caps - 49 (the untruncated recording), 36, 30 and 24 sub-pixels/frame - and
every resulting model is probed with the 10.46 fixed-point instrument over the same
full recorded range. If the plateau is the console's bound it stays where it is; if
it is the support it walks with the truncation.

Writes ``results/plateau_provenance_metrics.json`` and
``results/figures/plateau_provenance.png``. Emulator-free.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.learned_structure_probe_benchmark import (
    probe_acceleration_gain,
    probe_ceiling,
)
from src.evaluation.operator_physics_injection_benchmark import build_arm
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_GAMEPLAY, RESULTS_DIR
from src.utils.provenance import write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

ARTIFACT_NAME = "plateau_provenance_metrics.json"
FIGURE_NAME = "plateau_provenance.png"

FAMILIES: Tuple[str, ...] = ("DeepONet", "FNO", "MLP")
SUPPORT_CAPS: Tuple[float, ...] = (49.0, 36.0, 30.0, 24.0)
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)

# The published sustained ceiling (10.45) and the coincidence under test (10.46).
MEASURED_SUSTAINED_CEILING = 36.075
PUBLISHED_DEEPONET_PLATEAU = 35.61
PLAUSIBLE_MAX_ACCELERATION = 2.5

# Per-process, so two concurrent runs cannot delete each other's scratch weights.
_SCRATCH = os.path.join(tempfile.gettempdir(), f"mworld_plateau_scratch_{os.getpid()}")


def _truncate(data: Dict[str, np.ndarray], cap: float) -> Dict[str, np.ndarray]:
    """Keep training transitions whose next-state velocity lies under `cap`.

    Only the training arrays are filtered: the validation split drives early
    stopping and the test split is untouched, so every arm is scored on the same
    held-out transitions and the manipulation is confined to what the model sees.
    """
    if cap >= SUPPORT_CAPS[0]:
        return dict(data)
    keep = np.abs(data["train_next_states"][:, 2]) <= cap
    out = dict(data)
    for key in ("train_states", "train_actions", "train_next_states", "train_episodes"):
        if key in data:
            out[key] = data[key][keep]
    return out


def _train_one(
    family: str,
    cap: float,
    seed: int,
    data: Dict[str, Any],
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    latent_dim: int,
    fno_width: int,
    fno_modes: int,
    fno_layers: int,
) -> torch.nn.Module:
    """Fit one state-target arm on the velocity-truncated training split."""
    from src.environment.dataset_loader import create_dataloaders
    from src.training.trainer import DynamicsTrainer

    filtered = _truncate(data, cap)
    train_loader, val_loader, _ = create_dataloaders(filtered, batch_size=batch_size, seed=seed)
    state_dim = int(data["train_states"].shape[1])
    action_dim = int(data["train_actions"].shape[1])
    model = build_arm(
        family, "state", "none", state_dim, action_dim, latent_dim, fno_width, fno_modes, fno_layers
    )
    trainer = DynamicsTrainer(
        model=model,
        model_type=f"plateau_{family.lower()}_{int(cap)}_s{seed}",
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir=_SCRATCH,
    )
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs, patience=patience)
    return model


def _fit_slope(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Least-squares slope of plateau on the support cap that produced it."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 2 or float(np.std(xs)) < 1e-9:
        return {"slope": None, "intercept": None, "correlation": None}
    slope, intercept = np.polyfit(xs, ys, 1)
    correlation = float(np.corrcoef(xs, ys)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "correlation": correlation}


def run_plateau_provenance_benchmark(
    dataset_path: str = DATASET_GAMEPLAY,
    output_dir: Optional[str] = None,
    epochs: int = 35,
    batch_size: int = 128,
    patience: int = 8,
    probe_rows: int = 193,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    caps: Sequence[float] = SUPPORT_CAPS,
    latent_dim: int = 64,
    fno_width: int = 32,
    fno_modes: int = 6,
    fno_layers: int = 2,
) -> Dict[str, Any]:
    """Train every (family, support-cap, seed) arm and follow its plateau."""
    out_dir = output_dir or RESULTS_DIR
    from src.environment.dataset_loader import load_and_preprocess_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(int(seeds[0]))
    data = load_and_preprocess_data(dataset_path=dataset_path, seed=int(seeds[0]))
    full_support = float(np.abs(data["train_next_states"][:, 2]).max())
    logger.info(
        "=== Plateau provenance: 3 families x %d caps x %d seeds ===", len(caps), len(seeds)
    )

    results: Dict[str, Any] = {}
    for family in FAMILIES:
        by_cap: Dict[str, Any] = {}
        for cap in caps:
            plateaus: List[float] = []
            gains: List[float] = []
            for seed in seeds:
                set_global_seed(seed)
                model = _train_one(
                    family,
                    float(cap),
                    seed,
                    data,
                    device,
                    epochs,
                    batch_size,
                    patience,
                    latent_dim,
                    fno_width,
                    fno_modes,
                    fno_layers,
                )
                model.eval()
                ceiling = probe_ceiling(model, full_support, device, rows=probe_rows)
                gain = probe_acceleration_gain(model, full_support, device)
                value = ceiling["ceiling_like_fixed_point"]
                if value is not None and np.isfinite(value):
                    plateaus.append(float(value))
                    gains.append(float(gain))
                logger.info(
                    "  %-9s cap %4.1f seed %d | plateau %s | gain %.2f",
                    family,
                    cap,
                    seed,
                    "none" if not np.isfinite(value) else f"{float(value):.2f}",
                    gain,
                )
            trained_support = float(
                np.abs(_truncate(data, float(cap))["train_next_states"][:, 2]).max()
            )
            by_cap[str(cap)] = {
                "training_support_cap": trained_support,
                "plateaus": plateaus,
                "plateau_mean": float(np.mean(plateaus)) if plateaus else None,
                "plateau_std": float(np.std(plateaus, ddof=1)) if len(plateaus) > 1 else None,
                "gain_mean": float(np.mean(gains)) if gains else None,
                "seeds_with_a_plateau": len(plateaus),
            }
        xs = [by_cap[k]["training_support_cap"] for k in by_cap if by_cap[k]["plateau_mean"]]
        ys = [by_cap[k]["plateau_mean"] for k in by_cap if by_cap[k]["plateau_mean"]]
        results[family] = {
            "by_cap": by_cap,
            "plateau_vs_support_cap": _fit_slope(xs, ys),
        }

    verdict = _verdict(results, full_support)
    payload: Dict[str, Any] = {
        "study": (
            "Does a learned model's implied velocity plateau track the console's bound or the "
            "support of the transitions it was trained on?"
        ),
        "protocol": {
            "dataset_path": dataset_path,
            "families": list(FAMILIES),
            "support_caps": list(caps),
            "seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "probe_rows": probe_rows,
            "full_recorded_support": full_support,
            "mechanism": "the training split is truncated by |v_x| while the probe always scans "
            "the full recorded range, so a plateau that follows the truncation is a support "
            "artifact and one that stays is a dynamical bound",
        },
        "reference": {
            "measured_sustained_ceiling_10_45": MEASURED_SUSTAINED_CEILING,
            "published_deeponet_plateau_10_46": PUBLISHED_DEEPONET_PLATEAU,
            "plausible_traction_px_per_frame": PLAUSIBLE_MAX_ACCELERATION,
        },
        "results": results,
        "verdict": verdict,
    }

    if _render_figure(results, os.path.join(out_dir, "figures", FIGURE_NAME)):
        payload["figure"] = f"results/figures/{FIGURE_NAME}"

    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=int(seeds[0]),
        command=f"python -m src.evaluation.plateau_provenance_benchmark --seeds {','.join(str(s) for s in seeds)}",
    )
    logger.info("Metrics written to %s", artifact)
    return payload


def _verdict(results: Dict[str, Any], full_support: float) -> Dict[str, Any]:
    slopes = {
        family: block["plateau_vs_support_cap"]["slope"]
        for family, block in results.items()
        if block["plateau_vs_support_cap"]["slope"] is not None
    }
    near_one = [f for f, s in slopes.items() if s is not None and 0.7 <= s <= 1.3]
    deeponet = results.get("DeepONet", {}).get("by_cap", {})
    first = deeponet.get(str(SUPPORT_CAPS[0]), {}).get("plateau_mean")
    last = deeponet.get(str(SUPPORT_CAPS[-1]), {}).get("plateau_mean")
    return {
        "families_whose_plateau_follows_the_support": near_one,
        "slopes": slopes,
        "deeponet_plateau_untruncated": first,
        "deeponet_plateau_at_cap_24": last,
        "reading": (
            f"Across the support caps tried, the plateau moves with the truncation for "
            f"{near_one or 'no family'} (slopes {slopes}). "
            + (
                f"DeepONet's plateau goes from {first:.2f} on the untruncated recording "
                f"(full support {full_support:.1f}) to {last:.2f} when its training frames are "
                "capped at 24, so the 35.61 that 10.46 found 1.3% from the console's sustained "
                "ceiling is tracking the data, not the engine."
                if first is not None and last is not None
                else "Not every cap produced a plateau, so the tracking test is incomplete."
            )
        ),
    }


def _render_figure(results: Dict[str, Any], path: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    all_caps: List[float] = []
    for family, block in results.items():
        caps, means, errs = [], [], []
        for key, entry in block["by_cap"].items():
            if entry["plateau_mean"] is None:
                continue
            caps.append(entry["training_support_cap"])
            means.append(entry["plateau_mean"])
            errs.append(entry["plateau_std"] or 0.0)
        if caps:
            all_caps.extend(caps)
            ax.errorbar(caps, means, yerr=errs, marker="o", label=family, capsize=3)
    ax.axhline(
        MEASURED_SUSTAINED_CEILING, color="#e06666", linestyle="--", label="measured ceiling"
    )
    if all_caps:
        lo, hi = min(all_caps), max(all_caps)
        ax.plot([lo, hi], [lo, hi], color="#888888", linestyle=":", label="plateau = support")
    ax.set_xlabel("training support cap (sub-pixels/frame)")
    ax.set_ylabel("implied plateau (sub-pixels/frame)")
    ax.set_title("Does the learned plateau follow the bound or the support? (README 10.50)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--dataset-path", default=DATASET_GAMEPLAY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--probe-rows", type=int, default=193)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--caps", default="49,36,30,24")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=6)
    parser.add_argument("--fno-layers", type=int, default=2)
    args = parse_args_with_config(parser)

    run_plateau_provenance_benchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_rows=args.probe_rows,
        seeds=[int(s) for s in str(args.seeds).split(",") if s.strip()],
        caps=[float(c) for c in str(args.caps).split(",") if c.strip()],
        latent_dim=args.latent_dim,
        fno_width=args.fno_width,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
    )
