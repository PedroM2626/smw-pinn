"""
record_pixel_gameplay.py
Records paired (RGB frame, 8D WRAM state, action) transitions for visual
state-estimation training. Frames are native SNES RGB (usually 256x224).

Storage warning: raw frames are large. Use --frame-stride and --downscale to
bound the .npz size (both default to light subsampling).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator
from src.utils.paths import CORE_PATH, DATASET_PIXEL, ROM_PATH
from src.utils.seed import set_global_seed


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
):
    print("==========================================================")
    print("  RECORDING PAIRED PIXEL + WRAM DATASET (RGB FRAMES)      ")
    print("==========================================================")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    emu.enable_frame_capture(True)

    for _ in range(410):
        emu.step_frame()
    emu.enable_gameplay_mode()
    for _ in range(30):
        emu.step_frame()
    initial_savestate = emu.save_state()

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

    frames_t, states_t, actions_t = [], [], []
    total, t0 = 0, time.time()

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        set_global_seed(5000 + ep)
        curr_pattern, pattern_duration, pattern_timer = 0, 30, 0

        for f in range(frames_per_episode):
            if pattern_timer >= pattern_duration:
                curr_pattern = np.random.randint(len(action_behaviors))
                pattern_duration = np.random.randint(10, 50)
                pattern_timer = 0
            action_dict = action_behaviors[curr_pattern]
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
                    total += 1

            pattern_timer += 1
            if next_state_dict["y"] > 500 or next_state_dict["y"] < 0:
                break
        print(f"Episode {ep + 1}/{num_episodes} | transitions: {total}")

    emu.close()
    elapsed = time.time() - t0
    np.savez_compressed(
        output_path,
        frames=np.array(frames_t, dtype=np.uint8),
        states=np.array(states_t, dtype=np.float32),
        actions=np.array(actions_t, dtype=np.float32),
    )
    print(f"Saved {output_path} | N={total} | {elapsed:.1f}s")


if __name__ == "__main__":
    record_pixel_dataset()
