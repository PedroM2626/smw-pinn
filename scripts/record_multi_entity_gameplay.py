"""
record_multi_entity_gameplay.py
Records 100% genuine interactive transitions for both Mario kinematics (8D)
and dynamic stage entities/sprites (4D) directly from Super Mario World WRAM.

State Vector (12D):
[0..7]: [x, y, vx, vy, c_ground, c_ceiling, c_left, c_right]
[8..11]: [delta_x_enemy, delta_y_enemy, vx_enemy, hazard_active]

Action Vector (6D):
[B, Y, UP, DOWN, LEFT, RIGHT]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator
from src.utils.paths import CORE_PATH, DATASET_MULTI_ENTITY, ROM_PATH, STATE_YOSHI_ISLAND_1
from src.utils.seed import set_global_seed


def extract_12d_vector(state_dict: dict) -> np.ndarray:
    """Converts the extended state dictionary into a 12-dimensional numpy vector."""
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
            state_dict["delta_x_enemy"],
            state_dict["delta_y_enemy"],
            state_dict["vx_enemy"],
            state_dict["hazard_active"],
        ],
        dtype=np.float32,
    )


def extract_action_vector(action_dict: dict) -> np.ndarray:
    """Converts the action dictionary into a 6-dimensional numpy vector [B, Y, UP, DOWN, LEFT, RIGHT]."""
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


def record_multi_entity_dataset(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    output_path: str = DATASET_MULTI_ENTITY,
    num_episodes: int = 35,
    frames_per_episode: int = 600,
):
    print("==========================================================")
    print("  RECORDING GENUINE MULTI-ENTITY GAMEPLAY (12D WRAM)      ")
    print("==========================================================")
    print(f"ROM: {rom_path}")
    print(f"Core: {core_path}")
    print(f"Savestate: {state_path}")
    print(f"Episodes: {num_episodes} | Max frames per episode: {frames_per_episode}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    states_t = []
    actions_t = []
    states_tp1 = []
    episode_ids = []

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    # Diverse action primitives designed to encounter, interact with, leap over, and stomp Rex
    action_behaviors = [
        {"RIGHT": True, "Y": True},  # Continuous run
        {"RIGHT": True, "Y": True, "B": True},  # Running jump
        {"RIGHT": True, "B": True},  # Walking jump
        {"RIGHT": True},  # Simple walk
        {"LEFT": True, "Y": True},  # Left run
        {"LEFT": True},  # Left walk
        {},  # Idle / deceleration
        {"B": True},  # Stationary vertical jump
        {"RIGHT": True, "DOWN": True},  # Crouch slide
        {"RIGHT": True, "Y": True, "B": True},  # Sustained running jump
        {"LEFT": True, "B": True},  # Backward retreat jump
    ]

    total_transitions = 0
    rex_encounters = 0
    t0 = time.time()

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        emu.enable_gameplay_mode()
        for _ in range(5):
            emu.step_frame()

        set_global_seed(2000 + ep)

        curr_pattern = 0
        pattern_duration = np.random.randint(10, 45)
        pattern_timer = 0
        prev_b = False

        curr_ext_s = emu.get_smw_extended_state()

        for f in range(frames_per_episode):
            # Dynamic pattern switching:
            # When approaching hazard, introduce evasive jumping actions
            dx_hazard = curr_ext_s["delta_x_enemy"]
            is_active = curr_ext_s["hazard_active"] > 0.5

            if is_active and 0.0 < dx_hazard < 80.0:
                rex_encounters += 1
                # Mix of jumping over Rex, running into Rex, stomping Rex, and dodging back
                p_choice = np.random.rand()
                if p_choice < 0.45:
                    # Running leap attempt
                    action_dict = {"RIGHT": True, "Y": True, "B": not prev_b}
                elif p_choice < 0.70:
                    # High walking leap attempt
                    action_dict = {"RIGHT": True, "B": not prev_b}
                elif p_choice < 0.85:
                    # Retreat / pause
                    action_dict = {"LEFT": True}
                else:
                    # Straight approach
                    action_dict = {"RIGHT": True}
            else:
                if pattern_timer >= pattern_duration:
                    if np.random.rand() < 0.70:
                        curr_pattern = np.random.choice([0, 1, 2, 3, 9])
                    else:
                        curr_pattern = np.random.choice([4, 5, 6, 7, 8, 10])
                    pattern_duration = np.random.randint(10, 40)
                    pattern_timer = 0
                action_dict = action_behaviors[curr_pattern].copy()
                # Edge trigger handling on B
                if action_dict.get("B", False) and prev_b and np.random.rand() < 0.3:
                    action_dict["B"] = False

            prev_b = action_dict.get("B", False)

            # Inject input
            emu.set_input(action_dict)

            s_vec = extract_12d_vector(curr_ext_s)
            a_vec = extract_action_vector(action_dict)

            # Step frame
            emu.step_frame()

            next_ext_s = emu.get_smw_extended_state()
            next_s_vec = extract_12d_vector(next_ext_s)

            # Valid coordinate check
            if 0.0 <= next_s_vec[1] <= 450.0:
                states_t.append(s_vec)
                actions_t.append(a_vec)
                states_tp1.append(next_s_vec)
                episode_ids.append(ep)
                total_transitions += 1

            curr_ext_s = next_ext_s
            pattern_timer += 1

            # Termination on pit death or death animation ($7E:0071 == 9)
            if curr_ext_s["y"] > 450.0 or curr_ext_s["y"] < 0.0 or curr_ext_s["air_state"] == 9:
                break

        if (ep + 1) % 5 == 0 or (ep + 1) == num_episodes:
            print(
                f"Episode {ep + 1:2d}/{num_episodes} | "
                f"Mario X={curr_ext_s['x']:.1f}, Rex dx={curr_ext_s['delta_x_enemy']:.1f} (Active={curr_ext_s['hazard_active']:.0f}) | "
                f"Total transitions: {total_transitions}"
            )

    emu.close()
    elapsed = time.time() - t0

    states_arr = np.array(states_t, dtype=np.float32)
    actions_arr = np.array(actions_t, dtype=np.float32)
    next_states_arr = np.array(states_tp1, dtype=np.float32)
    episodes_arr = np.array(episode_ids, dtype=np.int32)

    np.savez_compressed(
        output_path,
        states=states_arr,
        actions=actions_arr,
        next_states=next_states_arr,
        episodes=episodes_arr,
    )

    print("\n==========================================================")
    print(f"Genuine 12D Multi-Entity dataset successfully recorded: {output_path}")
    print(f"Total transitions: {len(states_arr)}")
    print(f"State shape: {states_arr.shape}")
    print(f"Rex interaction frames: {rex_encounters}")
    print(f"Emulation time: {elapsed:.2f}s ({total_transitions / elapsed:.1f} FPS)")
    print("==========================================================")


if __name__ == "__main__":
    record_multi_entity_dataset()
