"""
record_tilemap_gameplay.py
Records 100% genuine interactive transitions for Mario kinematics (8D)
and local WRAM 7x7 level tilemap patches ($7E:C800) directly from Super Mario World.

State Vector (8D): [x, y, vx, vy, c_ground, c_ceiling, c_left, c_right]
Tilemap Patch (7x7): Integer matrix of tile classes (0: Air, 1: Solid/Pipes, 2: Hazard, 3: Slope)
Action Vector (6D): [B, Y, UP, DOWN, LEFT, RIGHT]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator
from src.utils.paths import CORE_PATH, DATASET_TILEMAP, ROM_PATH, STATE_YOSHI_ISLAND_1
from src.utils.seed import set_global_seed


def extract_vector(state_dict: dict) -> np.ndarray:
    """Converts state dictionary into an 8-dimensional kinematic numpy vector."""
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
    """Converts action dictionary into 6D vector [B, Y, UP, DOWN, LEFT, RIGHT]."""
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


def record_tilemap_dataset(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    output_path: str = DATASET_TILEMAP,
    num_episodes: int = 30,
    frames_per_episode: int = 500,
):
    print("==========================================================")
    print("  RECORDING GENUINE TILEMAP + KINEMATICS DATASET (WRAM)   ")
    print("==========================================================")
    print(f"ROM: {rom_path}")
    print(f"Core: {core_path}")
    print(f"Savestate: {state_path}")
    print(f"Episodes: {num_episodes} | Max frames per episode: {frames_per_episode}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    states_t = []
    tilemaps_t = []
    actions_t = []
    states_tp1 = []
    tilemaps_tp1 = []
    episode_ids = []

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    action_behaviors = [
        {"RIGHT": True, "Y": True},  # Continuous run
        {"RIGHT": True, "Y": True, "B": True},  # Running jump
        {"RIGHT": True, "B": True},  # Walking jump
        {"RIGHT": True},  # Walk
        {"LEFT": True, "Y": True},  # Left run
        {"LEFT": True, "B": True},  # Left jump
        {"LEFT": True},  # Left walk
        {},  # Idle / deceleration
        {"B": True},  # High vertical jump
        {"RIGHT": True, "DOWN": True},  # Crouch slide
    ]

    total_transitions = 0
    t0 = time.time()

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        emu.enable_gameplay_mode()  # Ensure interactive mode
        for _ in range(5):
            emu.step_frame()

        set_global_seed(3000 + ep)
        curr_pattern = 0
        pattern_duration = np.random.randint(15, 60)
        pattern_timer = 0
        prev_b = False

        curr_state_dict = emu.get_smw_state()
        curr_tilemap = emu.get_local_tilemap_patch(
            curr_state_dict["x"], curr_state_dict["y"], radius=3
        )

        for f in range(frames_per_episode):
            if pattern_timer >= pattern_duration:
                # 70% forward / jump exploration, 30% retreat / maneuvering
                if np.random.rand() < 0.70:
                    curr_pattern = np.random.choice([0, 1, 2, 3, 8])
                else:
                    curr_pattern = np.random.choice([4, 5, 6, 7, 9])
                pattern_duration = np.random.randint(15, 60)
                pattern_timer = 0

            action_dict = dict(action_behaviors[curr_pattern])
            # Handle jump release for re-jumping
            if action_dict.get("B", False):
                if prev_b:
                    action_dict["B"] = False  # release button for edge trigger
                prev_b = action_dict.get("B", False)
            else:
                prev_b = False

            emu.set_input(action_dict)

            s_vec = extract_vector(curr_state_dict)
            a_vec = extract_action_vector(action_dict)

            emu.step_frame()

            next_state_dict = emu.get_smw_state()
            next_s_vec = extract_vector(next_state_dict)
            next_tilemap = emu.get_local_tilemap_patch(
                next_state_dict["x"], next_state_dict["y"], radius=3
            )

            # Record transition within valid level coordinates
            if 0 <= next_s_vec[1] <= 500:
                states_t.append(s_vec)
                tilemaps_t.append(curr_tilemap)
                actions_t.append(a_vec)
                states_tp1.append(next_s_vec)
                tilemaps_tp1.append(next_tilemap)
                episode_ids.append(ep)
                total_transitions += 1

            curr_state_dict = next_state_dict
            curr_tilemap = next_tilemap
            pattern_timer += 1

            # Reset if fell into pit
            if next_s_vec[1] > 500 or next_s_vec[1] < 0:
                break

        if (ep + 1) % 5 == 0 or ep == num_episodes - 1:
            print(
                f"Episode {ep + 1:2d}/{num_episodes} completed | "
                f"Accumulated transitions: {total_transitions}"
            )

    emu.close()

    elapsed = time.time() - t0
    fps = total_transitions / max(1e-5, elapsed)

    print("\nDataset recording completed:")
    print(f"Total authentic transitions: {total_transitions}")
    print(f"Recording throughput: {fps:.1f} FPS (elapsed: {elapsed:.2f}s)")

    # Save to disk
    np.savez_compressed(
        output_path,
        states=np.array(states_t, dtype=np.float32),
        tile_patches=np.array(tilemaps_t, dtype=np.int64),
        actions=np.array(actions_t, dtype=np.float32),
        next_states=np.array(states_tp1, dtype=np.float32),
        next_tile_patches=np.array(tilemaps_tp1, dtype=np.int64),
        episodes=np.array(episode_ids, dtype=np.int32),
    )
    print(f"Compressed genuine dataset saved to: {output_path}")


if __name__ == "__main__":
    record_tilemap_dataset()
