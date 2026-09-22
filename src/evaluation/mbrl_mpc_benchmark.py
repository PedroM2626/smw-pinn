"""
mbrl_mpc_benchmark.py
Empirical Model-Based Reinforcement Learning (MBRL) Benchmark:
Compares closed-loop Model Predictive Control (MPC) trajectory planning in Super Mario World
using the Hard Residual PINN World Model vs. Statistical MLP vs. Soft PINN vs. Random Actions.
Executes directly on the real headless SNES emulator (Snes9x core).

Running with ``--reproduction-check`` re-executes the identical published protocol
several times and reports the run-to-run spread against `mbrl_mpc_metrics.json`
instead of overwriting it (see README section 10.38).
"""

import argparse
import os
import statistics
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models import (
    HardResidualPINNDynamics,
    SoftPINNDynamics,
    StatisticalMLPDynamics,
)
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.logging import get_logger
from src.utils.paths import (
    CHECKPOINTS_DIR,
    CORE_PATH,
    RESULTS_DIR,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    results_file,
)
from src.utils.provenance import read_metrics, write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PUBLISHED_MPC_ARTIFACT = "mbrl_mpc_metrics.json"


def convert_action_vector_to_dict(action_vec: np.ndarray) -> Dict[str, bool]:
    """
    Converts 6D continuous/binary action vector [B, Y, UP, DOWN, LEFT, RIGHT]
    into SNES joypad button dictionary.
    """
    return {
        "B": bool(action_vec[0] > 0.5),  # Jump
        "Y": bool(action_vec[1] > 0.5),  # Run / Dash
        "UP": bool(action_vec[2] > 0.5),
        "DOWN": bool(action_vec[3] > 0.5),
        "LEFT": bool(action_vec[4] > 0.5),
        "RIGHT": bool(action_vec[5] > 0.5),
    }


def extract_state_vector(s_dict: dict) -> np.ndarray:
    return np.array(
        [
            s_dict["x"],
            s_dict["y"],
            s_dict["vx"],
            s_dict["vy"],
            s_dict["c_ground"],
            s_dict["c_ceiling"],
            s_dict["c_left"],
            s_dict["c_right"],
        ],
        dtype=np.float32,
    )


def run_mbrl_closed_loop_trial(
    controller: ModelPredictiveController,
    emu: SnesLibretroEmulator,
    initial_savestate: bytes | None,
    max_frames: int = 300,
    policy_name: str = "MPC_Agent",
) -> Dict:
    """
    Executes a closed-loop MPC control episode in the real SNES emulator.

    The episode starts through `SnesLibretroEmulator.start_episode`, which also
    forces interactive gameplay mode: the committed Yoshi's Island 1 savestate
    restores into engine mode $7E:0100 = 0x08, and planning against that mode
    costs ~3.4x progress (README section 10.38, `--preamble-probe`).

    Passing ``initial_savestate=None`` keeps whatever the caller already loaded,
    which is what the preamble probe below needs in order to vary the episode
    start without changing anything else.
    """
    if initial_savestate is not None:
        emu.start_episode(initial_savestate)

    trajectory_x = []
    trajectory_y = []
    trajectory_vx = []
    alignment_errors = []

    init_state_dict = emu.get_smw_state()
    x_init = init_state_dict["x"]

    t0 = time.time()
    survived_frames = 0

    for frame in range(max_frames):
        current_state_dict = emu.get_smw_state()
        curr_vec = extract_state_vector(current_state_dict)

        trajectory_x.append(curr_vec[0])
        trajectory_y.append(curr_vec[1])
        trajectory_vx.append(curr_vec[2])

        # Terminal conditions: fallen into pit or out of bounds
        if curr_vec[1] > 450.0 or curr_vec[1] < 0.0:
            break

        survived_frames += 1

        # 1. Plan optimal action via MPC using the World Model
        action_vec, plan_info = controller.plan(curr_vec)

        # 2. Inject action into SNES joypad
        action_dict = convert_action_vector_to_dict(action_vec)
        emu.set_input(action_dict)

        # 3. Step real SNES hardware
        emu.step_frame()

        # 4. Measure alignment between World Model's 1-step imagination and reality
        imagined_next_s = plan_info["imagined_trajectory"][0]
        actual_next_s = extract_state_vector(emu.get_smw_state())

        spatial_alignment_err = np.sqrt(
            (imagined_next_s[0] - actual_next_s[0]) ** 2
            + (imagined_next_s[1] - actual_next_s[1]) ** 2
        )
        alignment_errors.append(float(spatial_alignment_err))

    elapsed = time.time() - t0
    final_x = trajectory_x[-1] if trajectory_x else x_init
    max_x = max(trajectory_x) if trajectory_x else x_init
    total_progress = float(final_x - x_init)
    max_progress = float(max_x - x_init)
    mean_vx = float(np.mean(trajectory_vx)) if trajectory_vx else 0.0
    mean_alignment_error = float(np.mean(alignment_errors)) if alignment_errors else 0.0

    logger.info(
        f"[{policy_name:20s}] Survived: {survived_frames:3d}/{max_frames} frames | "
        f"Progress: {total_progress:+6.1f} px (Max: {max_progress:+6.1f} px) | "
        f"Mean vx: {mean_vx:+5.1f} | Alignment Error: {mean_alignment_error:5.2f} px | "
        f"Time: {elapsed:4.1f}s ({survived_frames / elapsed:4.1f} FPS)"
    )

    return {
        "policy_name": policy_name,
        "survived_frames": survived_frames,
        "total_progress": total_progress,
        "max_progress": max_progress,
        "mean_vx": mean_vx,
        "mean_alignment_error": mean_alignment_error,
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "trajectory_vx": trajectory_vx,
        "alignment_errors": alignment_errors,
    }


def run_random_baseline(
    emu: SnesLibretroEmulator,
    initial_savestate: bytes,
    max_frames: int = 300,
    seed: int = 42,
) -> Dict:
    """Executes a stochastic exploration baseline (Random Actions)."""
    set_global_seed(seed)
    emu.start_episode(initial_savestate)

    trajectory_x = []
    trajectory_y = []
    trajectory_vx = []

    init_state_dict = emu.get_smw_state()
    x_init = init_state_dict["x"]
    survived_frames = 0

    actions_pool = [
        {"RIGHT": True, "Y": True},
        {"RIGHT": True, "B": True},
        {"RIGHT": True},
        {"LEFT": True},
        {"B": True},
        {},
    ]

    for frame in range(max_frames):
        current_state_dict = emu.get_smw_state()
        curr_vec = extract_state_vector(current_state_dict)

        trajectory_x.append(curr_vec[0])
        trajectory_y.append(curr_vec[1])
        trajectory_vx.append(curr_vec[2])

        if curr_vec[1] > 450.0 or curr_vec[1] < 0.0:
            break

        survived_frames += 1
        act = np.random.choice(actions_pool)
        emu.set_input(act)
        emu.step_frame()

    final_x = trajectory_x[-1] if trajectory_x else x_init
    max_x = max(trajectory_x) if trajectory_x else x_init
    total_progress = float(final_x - x_init)
    max_progress = float(max_x - x_init)

    logger.info(
        f"[{'Random_Baseline':20s}] Survived: {survived_frames:3d}/{max_frames} frames | "
        f"Progress: {total_progress:+6.1f} px (Max: {max_progress:+6.1f} px) | "
        f"Mean vx: {np.mean(trajectory_vx):+5.1f} | Alignment Error: N/A"
    )

    return {
        "policy_name": "Random_Baseline",
        "survived_frames": survived_frames,
        "total_progress": total_progress,
        "max_progress": max_progress,
        "mean_vx": float(np.mean(trajectory_vx)),
        "mean_alignment_error": 0.0,
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "trajectory_vx": trajectory_vx,
        "alignment_errors": [],
    }


def load_world_models(
    checkpoints_dir: str = CHECKPOINTS_DIR,
    device: torch.device | None = None,
) -> Dict[str, torch.nn.Module]:
    """Restore the three 8D world models used by the published closed-loop protocol."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {
        "Hard_PINN_World_Model": ("pinn_hard_best.pt", HardResidualPINNDynamics),
        "Soft_PINN_World_Model": ("pinn_soft_best.pt", SoftPINNDynamics),
        "Statistical_MLP_World_Model": ("mlp_best.pt", StatisticalMLPDynamics),
    }
    models: Dict[str, torch.nn.Module] = {}
    for name, (ckpt_name, model_cls) in specs.items():
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found for %s at %s, skipping.", name, ckpt_path)
            continue
        model = model_cls(state_dim=8, action_dim=6).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        models[name] = model
    return models


def build_controller(model: torch.nn.Module, device: torch.device) -> ModelPredictiveController:
    """The single CEM/objective configuration every published MPC row was recorded with."""
    return ModelPredictiveController(
        world_model=model,
        device=device,
        horizon=15,
        num_candidates=256,
        cem_iterations=3,
        elite_ratio=0.1,
        objective=TrajectoryObjective(
            weight_progress=2.0,
            weight_velocity=0.5,
            pit_penalty=1000.0,
            death_y=450.0,
        ),
    )


def run_all_trials(
    emu: SnesLibretroEmulator,
    models: Dict[str, torch.nn.Module],
    initial_savestate: bytes,
    max_frames: int = 300,
    device: torch.device | None = None,
    random_seed: int = 42,
) -> Dict[str, Dict]:
    """One full pass of the benchmark: every MPC controller plus the random baseline."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials_results: Dict[str, Dict] = {}
    for name, model in models.items():
        trials_results[name] = run_mbrl_closed_loop_trial(
            controller=build_controller(model, device),
            emu=emu,
            initial_savestate=initial_savestate,
            max_frames=max_frames,
            policy_name=name,
        )
    trials_results["Random_Baseline"] = run_random_baseline(
        emu=emu,
        initial_savestate=initial_savestate,
        max_frames=max_frames,
        seed=random_seed,
    )
    return trials_results


def summarize_trials(trials_results: Dict[str, Dict]) -> Dict[str, Dict]:
    return {
        name: {
            "survived_frames": r["survived_frames"],
            "total_progress_pixels": r["total_progress"],
            "max_progress_pixels": r["max_progress"],
            "mean_vx": r["mean_vx"],
            "mean_alignment_error_pixels": r["mean_alignment_error"],
        }
        for name, r in trials_results.items()
    }


def run_mbrl_mpc_benchmark(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    output_dir: str = RESULTS_DIR,
    max_frames: int = 300,
    seed: int = 42,
):
    logger.info("====================================================================")
    logger.info("  MODEL-BASED REINFORCEMENT LEARNING (MBRL): MPC WORLD MODEL BENCHMARK")
    logger.info("====================================================================")

    # Seed before the MPC trials: historically only `run_random_baseline` seeded,
    # so the CEM sampling of every MPC row was a single unseeded draw.
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Planning Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"Planning GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Initialize emulator
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    # Load World Models and run one full pass of the protocol
    models = load_world_models(checkpoints_dir, device)
    trials_results = run_all_trials(
        emu=emu,
        models=models,
        initial_savestate=initial_savestate,
        max_frames=max_frames,
        device=device,
    )

    emu.close()

    # Save summary metrics to JSON
    summary_metrics = summarize_trials(trials_results)

    out_json = os.path.join(output_dir, PUBLISHED_MPC_ARTIFACT)
    write_metrics(
        out_json,
        summary_metrics,
        seed=seed,
        command="python -m src.evaluation.mbrl_mpc_benchmark",
    )
    logger.info(f"\nMBRL metrics saved to: {out_json}")

    # Generate comparative trajectory figure
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    colors = {
        "Hard_PINN_World_Model": "#2ecc71",  # Emerald Green
        "Statistical_MLP_World_Model": "#e74c3c",  # Alizarin Red
        "Soft_PINN_World_Model": "#f39c12",  # Orange
        "Random_Baseline": "#7f8c8d",  # Gray
    }

    # Plot 1: 2D World Trajectory (X vs Y) in SNES Level Space
    for name, r in trials_results.items():
        c = colors.get(name, "#3498db")
        ax1.plot(
            r["trajectory_x"], r["trajectory_y"], label=name, color=c, linewidth=2.5, alpha=0.9
        )
    ax1.invert_yaxis()
    ax1.set_title(
        "MBRL Closed-Loop Trajectory Execution in Super Mario World (SNES Console)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlabel("Level Horizontal Coordinate X (Pixels)")
    ax1.set_ylabel("Level Vertical Coordinate Y (Pixels - Inverted)")
    ax1.legend(loc="best", fontsize=10)

    # Plot 2: Forward Progress Over Time (Frames)
    for name, r in trials_results.items():
        c = colors.get(name, "#3498db")
        frames = np.arange(len(r["trajectory_x"]))
        prog = np.array(r["trajectory_x"]) - r["trajectory_x"][0]
        ax2.plot(frames, prog, label=name, color=c, linewidth=2.5)
    ax2.set_title("Cumulative Forward Progress vs. Execution Frame", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Simulation Frame (60 Hz)")
    ax2.set_ylabel("Forward Displacement ΔX (Pixels)")
    ax2.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "mbrl_mpc_trajectories.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Trajectory figure saved to: {fig_path}")

    # Summary table
    logger.info("\n====================================================================")
    logger.info("  MBRL MPC BENCHMARK SUMMARY")
    logger.info("====================================================================")
    logger.info(
        f"{'World Model Controller':30s} | {'Progress (px)':15s} | {'Alignment Err':15s} | {'Frames Alive':12s}"
    )
    logger.info("-" * 80)
    for name, s in summary_metrics.items():
        prog_str = f"{s['total_progress_pixels']:+6.1f} px"
        err_str = (
            f"{s['mean_alignment_error_pixels']:5.2f} px"
            if s["mean_alignment_error_pixels"] > 0
            else "N/A"
        )
        logger.info(
            f"{name:30s} | {prog_str:15s} | {err_str:15s} | {s['survived_frames']:4d}/{max_frames}"
        )
    logger.info("====================================================================")

    return summary_metrics


def run_reproduction_audit(
    repeats: int = 3,
    max_frames: int = 300,
    seed: int = 42,
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    published_path: str | None = None,
) -> Dict:
    """Re-execute the published closed-loop protocol and quantify its reproducibility.

    The published `mbrl_mpc_metrics.json` was recorded before either the gameplay-mode
    preamble or a seed guarded the MPC trials (only the random baseline seeded), so its
    MPC rows were single unseeded samples of a protocol that has since changed. Each
    repeat here is seeded from `base_seed`; the spread across repeats is the honest
    uncertainty of the closed-loop table in README 10.6.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    published = read_metrics(published_path or results_file(PUBLISHED_MPC_ARTIFACT))

    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()
    models = load_world_models(checkpoints_dir, device)

    runs = []
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    try:
        for repeat in range(repeats):
            set_global_seed(seed + repeat)
            logger.info(
                "Reproduction audit repeat %d/%d (seed %d)", repeat + 1, repeats, seed + repeat
            )
            trials = run_all_trials(
                emu=emu,
                models=models,
                initial_savestate=initial_savestate,
                max_frames=max_frames,
                device=device,
                random_seed=seed,
            )
            summary = summarize_trials(trials)
            runs.append(
                {
                    "seed": seed + repeat,
                    "progress_px": {n: s["total_progress_pixels"] for n, s in summary.items()},
                    "alignment_err_px": {
                        n: s["mean_alignment_error_pixels"] for n, s in summary.items()
                    },
                }
            )
    finally:
        emu.close()

    names = list(runs[0]["progress_px"])
    per_controller: Dict[str, Dict] = {}
    for name in names:
        values = [run["progress_px"][name] for run in runs]
        reference = published.get(name, {}).get("total_progress_pixels")
        mean = float(statistics.fmean(values))
        row: Dict[str, object] = {
            "audit_mean_progress_px": round(mean, 2),
            "audit_std_progress_px": round(float(statistics.stdev(values)), 2)
            if len(values) > 1
            else 0.0,
            "audit_min_progress_px": round(min(values), 2),
            "audit_max_progress_px": round(max(values), 2),
        }
        if isinstance(reference, (int, float)):
            row["published_progress_px"] = round(float(reference), 2)
            row["delta_px"] = round(mean - float(reference), 2)
            row["delta_percent"] = (
                round(100.0 * (mean - reference) / reference, 1) if reference else None
            )
        per_controller[name] = row

    return {
        "protocol": {
            "repeats": repeats,
            "base_seed": seed,
            "max_frames": max_frames,
            "source": PUBLISHED_MPC_ARTIFACT,
        },
        "per_controller": per_controller,
        "per_repeat": runs,
        "interpretation": (
            "Spread across seeded repeats is the sampling uncertainty of the CEM; a "
            "row whose published value lies outside the audit range was recorded under "
            "a different protocol (see --preamble-probe) rather than by unlucky draws."
        ),
    }


PREAMBLE_VARIANTS = ("load_only", "load_and_mode", "load_and_warmup", "standard")


def run_preamble_probe(
    frames: int = 300,
    seed: int = 42,
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    checkpoints_dir: str = CHECKPOINTS_DIR,
) -> Dict:
    """Quantify how much of closed-loop progress is decided by the episode preamble.

    Repository scripts disagreed on what happens between `load_state` and the first
    planned frame: some restore the savestate and act immediately, others also force
    interactive game mode ($7E:0100 = 0x14) and step a few warm-up frames so derived
    WRAM (collision flags, joypad latch) is refreshed. Nothing measured that gap - so
    the same checkpoint under the same planner could be reported twice with different
    numbers. Only the preamble varies here; model, objective and CEM budget are fixed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_world_models(checkpoints_dir, device)
    if "Hard_PINN_World_Model" not in models:
        raise RuntimeError("preamble probe needs results/checkpoints/pinn_hard_best.pt")
    controller = build_controller(models["Hard_PINN_World_Model"], device)

    with open(state_path, "rb") as fh:
        savestate = fh.read()

    per_variant: Dict[str, Dict] = {}
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    try:
        for variant in PREAMBLE_VARIANTS:
            set_global_seed(seed)
            emu.load_state(savestate)
            if variant in ("load_and_mode", "standard"):
                emu.enable_gameplay_mode()
            warmup = 5 if variant in ("load_and_warmup", "standard") else 0
            for _ in range(warmup):
                emu.set_input({})
                emu.step_frame()
            mode_at_start = hex(emu.get_game_mode())
            trial = run_mbrl_closed_loop_trial(
                controller=controller,
                emu=emu,
                initial_savestate=None,
                max_frames=frames,
                policy_name=f"preamble={variant}",
            )
            per_variant[variant] = {
                "progress_px": round(trial["total_progress"], 2),
                "survived_frames": trial["survived_frames"],
                "mean_alignment_error_px": round(trial["mean_alignment_error"], 2),
                "mode_at_first_planned_frame": mode_at_start,
                "warmup_frames": warmup,
                "gameplay_mode_forced": variant in ("load_and_mode", "standard"),
            }
    finally:
        emu.close()

    best = max(per_variant, key=lambda v: per_variant[v]["progress_px"])
    worst = min(per_variant, key=lambda v: per_variant[v]["progress_px"])
    return {
        "protocol": {
            "controller": "MPC + Hard Residual PINN",
            "frames_budget": frames,
            "seed": seed,
            "varied": "episode preamble only",
        },
        "per_variant": per_variant,
        "spread_px": round(per_variant[best]["progress_px"] - per_variant[worst]["progress_px"], 2),
        "interpretation": (
            f"The preamble alone moves progress by "
            f"{per_variant[best]['progress_px'] - per_variant[worst]['progress_px']:.1f} px "
            f"({worst} -> {best}). Closed-loop rows are only comparable when the "
            "episode start is identical, which is why every script now calls "
            "SnesLibretroEmulator.start_episode()."
        ),
    }


def main(argv: list[str] | None = None) -> Dict:
    parser = argparse.ArgumentParser(
        description="Closed-loop MPC world-model benchmark on the real SNES emulator."
    )
    parser.add_argument("--max-frames", type=int, default=300, help="frame budget per episode")
    parser.add_argument("--output-dir", default=RESULTS_DIR, help="directory for artifacts")
    parser.add_argument(
        "--reproduction-check",
        action="store_true",
        help="re-run the published protocol without overwriting mbrl_mpc_metrics.json",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="audit repetitions (with --reproduction-check)"
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed for the audit")
    parser.add_argument(
        "--preamble-probe",
        action="store_true",
        help="vary only the episode-start preamble and report the progress spread",
    )
    args = parser.parse_args(argv)

    if args.preamble_probe:
        probe = run_preamble_probe(frames=args.max_frames, seed=args.seed)
        target = results_file("mpc_preamble_probe_metrics.json")
        write_metrics(
            target,
            probe,
            seed=args.seed,
            command="python -m src.evaluation.mbrl_mpc_benchmark --preamble-probe",
        )
        logger.info("Preamble probe saved to: %s", target)
        for variant, row in probe["per_variant"].items():
            logger.info(
                "%-18s %7.2f px | mode %s | warmup %d | forced mode %s",
                variant,
                row["progress_px"],
                row["mode_at_first_planned_frame"],
                row["warmup_frames"],
                row["gameplay_mode_forced"],
            )
        logger.info("spread: %.2f px", probe["spread_px"])
        return probe

    if args.reproduction_check:
        audit = run_reproduction_audit(
            repeats=args.repeats, max_frames=args.max_frames, seed=args.seed
        )
        target = results_file("mpc_reproduction_metrics.json")
        write_metrics(
            target,
            audit,
            seed=args.seed,
            command="python -m src.evaluation.mbrl_mpc_benchmark --reproduction-check",
        )
        logger.info("Reproduction audit saved to: %s", target)
        for name, row in audit["per_controller"].items():
            logger.info(
                "%-30s published %+7.1f px | audit %7.1f +- %5.1f px | delta %+6.1f px",
                name,
                row.get("published_progress_px", float("nan")),
                row["audit_mean_progress_px"],
                row["audit_std_progress_px"],
                row.get("delta_px", float("nan")),
            )
        return audit

    return run_mbrl_mpc_benchmark(
        max_frames=args.max_frames, output_dir=args.output_dir, seed=args.seed
    )


if __name__ == "__main__":
    main()
