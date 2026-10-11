"""
record_pixel_gameplay.py
Records paired (RGB frame, 8D WRAM state, action) transitions for visual
state-estimation training. Frames are native SNES RGB (usually 256x224).

Storage warning: raw frames are large. Use --frame-stride and --downscale to
bound the .npz size (both default to light subsampling).

The default path is the published one and is left exactly as it was: the recorder boots the ROM,
steps 410 frames, writes `0x14` into whatever state that leaves (the forced-mode preamble 10.38.1
documents) and records from there. `--state` restores a given savestate instead, which is how the
level-entry recording of 10.62 was made, and `--clear_boxes` pulses the A button at a player whose
8D record has stopped changing - the message box 10.60 identifies - so a genuine level does not
contribute two hundred frames of a frozen scene to a training set.
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.episode_hygiene import PlayerHold  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.utils.paths import CORE_PATH, DATASET_PIXEL, ROM_PATH  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

#: consecutive identical 8D records before the walk assumes the engine is holding the player
PAUSE_TRIGGER = 10
#: how long to keep pulsing A at a held player before giving up on the episode
OPERATOR_A_LIMIT = 240


def extract_vector(state_dict: dict) -> np.ndarray:
    return np.array(
        [
            state_dict["x"],
            state_dict["y"],
            state_dict["vx"],
            state_dict["vy"],
            state_dict["c_ground"],
            state_dict["c_ceiling"],
            state_dict["c_left"],
            state_dict["c_right"],
        ],
        dtype=np.float32,
    )


def extract_action_vector(action_dict: dict) -> np.ndarray:
    return np.array(
        [
            1.0 if action_dict.get("B", False) else 0.0,
            1.0 if action_dict.get("Y", False) else 0.0,
            1.0 if action_dict.get("UP", False) else 0.0,
            1.0 if action_dict.get("DOWN", False) else 0.0,
            1.0 if action_dict.get("LEFT", False) else 0.0,
            1.0 if action_dict.get("RIGHT", False) else 0.0,
        ],
        dtype=np.float32,
    )


def record_pixel_dataset(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    output_path: str = DATASET_PIXEL,
    num_episodes: int = 10,
    frames_per_episode: int = 600,
    frame_stride: int = 2,
    downscale: int = 2,
    state_path: str | None = None,
    clear_boxes: bool = False,
):
    print("==========================================================")
    print("  RECORDING PAIRED PIXEL + WRAM DATASET (RGB FRAMES)      ")
    print("==========================================================")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    emu.enable_frame_capture(True)

    if state_path is not None:
        with open(state_path, "rb") as fh:
            initial_savestate = fh.read()
        print(f"  restoring {state_path} for every episode")
    else:
        initial_savestate = emu.forced_mode_start_state()

    action_behaviors = [
        {"RIGHT": True, "Y": True},
        {"RIGHT": True, "Y": True, "B": True},
        {"RIGHT": True, "B": True},
        {"RIGHT": True},
        {"LEFT": True, "Y": True},
        {"LEFT": True},
        {},
        {"B": True},
    ]

    frames_t, states_t, actions_t, held_t, camera_t = [], [], [], [], []
    total, t0 = 0, time.time()
    boxes_cleared = 0
    holds: List[PlayerHold] = []

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        set_global_seed(5000 + ep)  # parity: the published recording's stream
        curr_pattern, pattern_duration, pattern_timer = 0, 30, 0
        hold = PlayerHold()
        holds.append(hold)

        for f in range(frames_per_episode):
            if pattern_timer >= pattern_duration:
                curr_pattern = np.random.randint(len(action_behaviors))
                pattern_duration = np.random.randint(10, 50)
                pattern_timer = 0
            action_dict = dict(action_behaviors[curr_pattern])
            vector = extract_vector(emu.get_smw_state())
            held = hold.observe(vector) if clear_boxes else False
            if clear_boxes:
                if hold.exhausted():
                    break
                if hold.run == PAUSE_TRIGGER:
                    boxes_cleared += 1
                action_dict = hold.buttons(action_dict)
            emu.set_input(action_dict)
            emu.step_frame()
            next_state_dict = emu.get_smw_state()

            if f % frame_stride == 0:
                frame = emu.get_frame()
                if frame is not None and 0 <= next_state_dict["y"] <= 500:
                    if downscale > 1:
                        frame = frame[::downscale, ::downscale]
                    frames_t.append(frame)
                    states_t.append(extract_vector(next_state_dict))
                    actions_t.append(extract_action_vector(action_dict))
                    held_t.append(1.0 if held else 0.0)
                    camera_t.append(emu.get_camera_x())
                    total += 1

            pattern_timer += 1
            if next_state_dict["y"] > 500 or next_state_dict["y"] < 0:
                break
        print(f"Episode {ep + 1}/{num_episodes} | transitions: {total}")

    hold_total = sum(h.operator_frames for h in holds)
    emu.close()
    elapsed = time.time() - t0
    np.savez_compressed(
        output_path,
        frames=np.array(frames_t, dtype=np.uint8),
        states=np.array(states_t, dtype=np.float32),
        actions=np.array(actions_t, dtype=np.float32),
        held=np.array(held_t, dtype=np.float32),
        camera=np.array(camera_t, dtype=np.float32),
    )
    print(f"Saved {output_path} | N={total} | {elapsed:.1f}s")
    print(
        f"  held frames: {int(np.asarray(held_t).sum())} of {total} "
        f"| boxes cleared: {boxes_cleared} | operator A frames: {hold_total}"
    )
    return {
        "transitions": total,
        "held_frames": int(np.asarray(held_t).sum()),
        "boxes_cleared": boxes_cleared,
        "operator_A_frames": hold_total if clear_boxes else 0,
        "seconds": round(elapsed, 1),
        "output": output_path,
        "source_state": state_path,
        "frame_shape": list(np.array(frames_t[:1]).shape[1:]),
        "camera_travel_px": round(float(np.ptp(camera_t)) if camera_t else 0.0, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--state",
        default=None,
        help="savestate to restore; default is the published boot-and-force preamble",
    )
    parser.add_argument("--output", default=DATASET_PIXEL)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument(
        "--clear-boxes",
        action="store_true",
        help="pulse A at a held player "
        "(see 10.60); off by default so the published recording is reproducible",
    )
    args = parser.parse_args()
    record_pixel_dataset(
        output_path=args.output,
        num_episodes=args.episodes,
        frames_per_episode=args.frames,
        frame_stride=args.frame_stride,
        downscale=args.downscale,
        state_path=args.state,
        clear_boxes=args.clear_boxes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
