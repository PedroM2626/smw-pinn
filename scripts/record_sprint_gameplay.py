"""
record_sprint_gameplay.py
Records an excitation-targeted WRAM dataset whose purpose is to saturate the velocity bound.

README Section 10.43.9 measured why the engine's rigid velocity ceiling cannot be read off
the published gameplay recording: zero held-out frames reach 90% of the training speed
support, so no estimator - tree search, PySR or an explicit clamp template - has any
evidence about what happens there. That is a property of the *data collection policy*, not
of the console, and it is fixable: run Mario fast enough, for long enough, and far enough
into the level that the velocity law is exercised at its top.

The recording policy is the established-physics MPC of Section 10.37 (a competent runner
that reaches ~605 px in 300 frames), because a scripted "hold right and run" policy dies in
the first pit at x = 131 and never gets far enough to saturate. What the policy decides is
which states are *visited*; every recorded transition is genuine WRAM telemetry read from
the console after the frame, so the dynamics in the file are not the planner's opinion.

The published datasets are never touched: this writes its own file.

Run:  python scripts/record_sprint_gameplay.py --episodes 10 --frames-per-episode 900
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.dataset_loader import load_and_preprocess_data  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.evaluation.analytical_baselines import _to_tensors  # noqa: E402
from src.models.analytical_kinematics import AnalyticalKinematicsDynamics  # noqa: E402
from src.planning.mpc_planner import (  # noqa: E402
    ModelPredictiveController,
    TrajectoryObjective,
    action_vector_to_joypad,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import ROM_PATH, STATE_YOSHI_ISLAND_1  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUTPUT = os.path.join("data", "raw", "smw_sprint_dataset.npz")

# The published MBRL objective (README 10.6), with the velocity term raised: this policy
# exists to excite the speed channel, and at the published weight of 0.5 the planner's own
# peak |vx| was 37 sub-pixels/frame - below the 49 the old recording already contains, so it
# would have added coverage of *distance* without adding any coverage of *speed*.
OBJECTIVE = {
    "weight_progress": 2.0,
    "weight_velocity": 8.0,
    "pit_penalty": 1000.0,
    "death_y": 450.0,
}


def _vector(state: dict) -> np.ndarray:
    return np.array(
        [
            state["x"],
            state["y"],
            state["vx"],
            state["vy"],
            state["c_ground"],
            state["c_ceiling"],
            state["c_left"],
            state["c_right"],
        ],
        dtype=np.float32,
    )


def _joypad(action: np.ndarray) -> dict:
    return action_vector_to_joypad(np.asarray(action, dtype=np.float64))


def build_recording_controller(device: torch.device) -> ModelPredictiveController:
    """The established engine rules of 10.37, grid-fitted on the canonical training split."""
    data = load_and_preprocess_data(seed=42)
    model = AnalyticalKinematicsDynamics().to(device)
    model.fit_engine_rules(*_to_tensors(data, "train", device), device=device)
    return ModelPredictiveController(
        world_model=model,
        device=device,
        horizon=15,
        num_candidates=256,
        cem_iterations=3,
        objective=TrajectoryObjective(**OBJECTIVE),
    )


def record_sprint_dataset(
    output_path: str = DEFAULT_OUTPUT,
    episodes: int = 10,
    frames_per_episode: int = 900,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    seed: int = 42,
) -> dict:
    """Record ``episodes`` x ``frames_per_episode`` of fast forward play into ``output_path``."""
    if not os.path.isfile(rom_path):
        raise FileNotFoundError(f"ROM not found at {rom_path}; see README section 11.2")
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = build_recording_controller(device)
    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    episode_ids: List[int] = []
    deaths = 0
    peak_speed = 0.0

    emu = SnesLibretroEmulator()
    emu.load_rom(rom_path)
    started = time.time()
    try:
        for episode in range(episodes):
            # The standardized preamble (restore -> force gameplay mode -> warm up): a raw
            # load_state leaves the engine outside gameplay mode and Mario never moves.
            snapshot = emu.start_episode(initial_savestate)
            state = _vector(snapshot)
            for _ in range(frames_per_episode):
                action, _info = controller.plan(state)
                current = state.copy()
                emu.set_input(_joypad(action))
                emu.step_frame()
                snapshot = emu.get_smw_state()
                state = _vector(snapshot)
                peak_speed = max(peak_speed, float(abs(state[2])))
                if not 0.0 <= state[1] <= 450.0:
                    deaths += 1
                    break  # a pit ends the episode; the next one restores the savestate
                states.append(current)
                actions.append(np.asarray(action, dtype=np.float32))
                next_states.append(state)
                episode_ids.append(episode)
            logger.info(
                "  episode %2d/%d | %6d transitions | peak |vx| so far %.2f",
                episode + 1,
                episodes,
                len(states),
                peak_speed,
            )
    finally:
        emu.close()

    if not states:
        raise RuntimeError("the recording captured no valid transitions - check the savestate")
    arrays = {
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "next_states": np.stack(next_states).astype(np.float32),
        "episodes": np.asarray(episode_ids, dtype=np.int32),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays)
    logger.info(
        "Recorded %d transitions (%d deaths) in %.1f s -> %s",
        arrays["states"].shape[0],
        deaths,
        time.time() - started,
        output_path,
    )
    return arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames-per-episode", dest="frames_per_episode", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record_sprint_dataset(
        output_path=args.output,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
