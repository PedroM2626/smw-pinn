"""
record_jump_gameplay.py
Records an excitation-targeted WRAM dataset whose purpose is to sample the gravity gate.

Sections 10.43.9, 10.45 and 10.47 all end on the same negative: the held-jump gravity
discontinuity - ascent integrates ``g_hold`` while the jump button is held and ``g_fall``
otherwise, a step of -2.80 sub-pixels/frame^2 - is recovered by no engine and by no
learned model, from telemetry or from the hidden world. 10.45 established the method for
attacking exactly this kind of failure when the cause is coverage: the velocity bound
became measurable as soon as a recording was made whose policy *visits* the top of the
speed range. The gate needs the same treatment, and its missing region is narrower: the
console's vertical tier is decided by one button bit during ascent, so what has to be
sampled in volume is the frame where Mario is rising **and the jump button is already
released** - the early-release branch that a policy which holds the button through the
apex never produces.

The navigation policy is again the established-physics MPC of 10.37, because a scripted
jumper dies in the first pit. What the MPC decides is direction and survival; the jump
channel is overwritten by a schedule that alternates long holds (the ``g_hold`` branch,
through the apex) with deliberately short holds (the ``g_fall`` branch while still
ascending), which is the tier the published recordings barely contain.

Every transition is read back from WRAM after the frame, so the dynamics in the file are
the console's, not the policy's. The published datasets are never touched.

Run:  python scripts/record_jump_gameplay.py --episodes 12 --frames-per-episode 800
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

from scripts.record_sprint_gameplay import (  # noqa: E402
    _joypad,
    _vector,
    build_recording_controller,
)
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import ROM_PATH, STATE_YOSHI_ISLAND_1  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUTPUT = os.path.join("data", "raw", "smw_jump_dataset.npz")

JUMP_CHANNEL = 0  # action vector is [B, Y, UP, DOWN, LEFT, RIGHT]; B is the jump button

# A short hold releases the button well before the apex, which is the branch the gate
# needs and the recordings lack; a long hold keeps it through the apex, which is the
# branch every recording already contains. Both are kept so the tiers can be compared.
SHORT_HOLD_FRAMES = (1, 2, 3)
LONG_HOLD_FRAMES = (12, 30)
SHORT_HOLD_PROBABILITY = 0.55


def draw_hold(rng: np.random.Generator) -> int:
    """How many frames this jump keeps the button down."""
    if rng.random() < SHORT_HOLD_PROBABILITY:
        return int(rng.integers(SHORT_HOLD_FRAMES[0], SHORT_HOLD_FRAMES[1] + 1))
    return int(rng.integers(LONG_HOLD_FRAMES[0], LONG_HOLD_FRAMES[1] + 1))


def record_jump_dataset(
    output_path: str = DEFAULT_OUTPUT,
    episodes: int = 12,
    frames_per_episode: int = 800,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    seed: int = 42,
) -> dict:
    """Record jump-excited WRAM telemetry into ``output_path``."""
    if not os.path.isfile(rom_path):
        raise FileNotFoundError(f"ROM not found at {rom_path}; see README section 11.2")
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = build_recording_controller(device)
    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    episode_ids: List[int] = []
    deaths = 0
    ascending_released = 0
    ascending_held = 0
    falling = 0

    emu = SnesLibretroEmulator()
    emu.load_rom(rom_path)
    started = time.time()
    try:
        for episode in range(episodes):
            snapshot = emu.start_episode(initial_savestate)
            state = _vector(snapshot)
            hold_remaining = 0
            was_grounded = True
            for _ in range(frames_per_episode):
                action = np.asarray(controller.plan(state)[0], dtype=np.float32)
                grounded = state[4] > 0.5
                if grounded and not was_grounded:
                    hold_remaining = 0
                if grounded and hold_remaining == 0:
                    hold_remaining = draw_hold(rng)
                action[JUMP_CHANNEL] = 1.0 if hold_remaining > 0 else 0.0
                if hold_remaining > 0:
                    hold_remaining -= 1
                was_grounded = grounded

                current = state.copy()
                emu.set_input(_joypad(action))
                emu.step_frame()
                snapshot = emu.get_smw_state()
                state = _vector(snapshot)
                if state[3] < 0.0:
                    if action[JUMP_CHANNEL] > 0.5:
                        ascending_held += 1
                    else:
                        ascending_released += 1
                else:
                    falling += 1
                if not 0.0 <= state[1] <= 450.0:
                    deaths += 1
                    break
                states.append(current)
                actions.append(action)
                next_states.append(state)
                episode_ids.append(episode)
            logger.info(
                "  episode %2d/%d | %6d transitions | ascent-held %d ascent-released %d fall %d",
                episode + 1,
                episodes,
                len(states),
                ascending_held,
                ascending_released,
                falling,
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
        "Recorded %d transitions (%d deaths; %d ascending-released) in %.1f s -> %s",
        arrays["states"].shape[0],
        deaths,
        ascending_released,
        time.time() - started,
        output_path,
    )
    return arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--frames-per-episode", dest="frames_per_episode", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record_jump_dataset(
        output_path=args.output,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
