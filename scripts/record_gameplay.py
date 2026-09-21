"""
record_gameplay.py
Records 100% genuine, interactive telemetry transitions from Super Mario World WRAM.
The emulator boots the game, enters Interactive Level Mode (Game Mode 0x14)
where controller inputs directly drive Mario in real-time through the level.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.environment.snes_emulator import SnesLibretroEmulator


def extract_vector(state_dict: dict) -> np.ndarray:
    """Converts the state dictionary into an 8-dimensional numpy vector."""
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


def record_interactive_trajectories(
    rom_path: str = "data/raw/smw_usa.sfc",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    output_path: str = "data/raw/smw_gameplay_dataset.npz",
    num_episodes: int = 15,
    frames_per_episode: int = 1000,
):
    print("==========================================================")
    print("  RECORDING GENUINE INTERACTIVE GAMEPLAY (MODE 0x14)      ")
    print("==========================================================")
    print(f"ROM: {rom_path}")
    print(f"Core: {core_path}")
    print(f"Episodes: {num_episodes} | Frames per episode: {frames_per_episode}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    states_t = []
    actions_t = []
    states_tp1 = []
    episode_ids = []

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    # 1. Advance boot routine and enable Interactive Gameplay Mode (0x14)
    print("Initializing emulator and enabling interactive level mode (0x14)...")
    for _ in range(410):
        emu.step_frame()

    emu.wram_buffer[0x0100] = 0x14
    for _ in range(30):
        emu.step_frame()

    # Capture initial savestate where Mario is 100% controllable by player I/O
    initial_savestate = emu.save_state()
    initial_state = emu.get_smw_state()
    print(f"Interactive savestate captured successfully! Initial state: {initial_state}")

    # Diverse action behaviors to cover the complete dynamic envelope
    action_behaviors = [
        {"RIGHT": True, "Y": True},                 # Continuous run
        {"RIGHT": True, "Y": True, "B": True},       # Running jump
        {"RIGHT": True, "B": True},                 # Walking jump
        {"RIGHT": True},                            # Simple walk
        {"LEFT": True, "Y": True},                  # Left run / skidding
        {"LEFT": True},                             # Left walk
        {},                                         # Idle / natural surface friction deceleration
        {"B": True},                                # Stationary vertical jump
        {"RIGHT": True, "DOWN": True},              # Crouched slide
        {"RIGHT": True, "Y": True, "A": True},       # Running Spin Jump
    ]

    total_transitions = 0
    t0 = time.time()

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        np.random.seed(100 + ep)

        curr_pattern = 0
        pattern_duration = np.random.randint(15, 60)
        pattern_timer = 0

        curr_state_dict = emu.get_smw_state()

        for f in range(frames_per_episode):
            # Dynamic pattern switching biased toward forward exploration
            if pattern_timer >= pattern_duration:
                # 65% forward progress, 35% maneuvers/jumps/friction
                if np.random.rand() < 0.65:
                    curr_pattern = np.random.choice([0, 1, 2, 3, 9])
                else:
                    curr_pattern = np.random.choice([4, 5, 6, 7, 8])
                pattern_duration = np.random.randint(10, 50)
                pattern_timer = 0

            action_dict = action_behaviors[curr_pattern]

            # Inject control command into SNES joypad
            emu.set_input(action_dict)

            s_vec = extract_vector(curr_state_dict)
            a_vec = extract_action_vector(action_dict)

            # Advance 1 frame of hardware simulation
            emu.step_frame()

            next_state_dict = emu.get_smw_state()
            next_s_vec = extract_vector(next_state_dict)

            # Record only if player is in valid level coordinate bounds
            if 0 <= next_s_vec[1] <= 500:
                states_t.append(s_vec)
                actions_t.append(a_vec)
                states_tp1.append(next_s_vec)
                episode_ids.append(ep)
                total_transitions += 1

            curr_state_dict = next_state_dict
            pattern_timer += 1

            # End episode on pit fall death
            if next_s_vec[1] > 500 or next_s_vec[1] < 0:
                break

        print(
            f"Episode {ep+1:2d}/{num_episodes} completed | "
            f"Final Mario: X={curr_state_dict['x']:.1f}, Y={curr_state_dict['y']:.1f}, "
            f"vx={curr_state_dict['vx']:.1f}, vy={curr_state_dict['vy']:.1f} | "
            f"Accumulated transitions: {total_transitions}"
        )

    emu.close()
    elapsed = time.time() - t0

    states_t_arr = np.array(states_t, dtype=np.float32)
    actions_t_arr = np.array(actions_t, dtype=np.float32)
    states_tp1_arr = np.array(states_tp1, dtype=np.float32)
    episode_ids_arr = np.array(episode_ids, dtype=np.int32)

    np.savez_compressed(
        output_path,
        states=states_t_arr,
        actions=actions_t_arr,
        next_states=states_tp1_arr,
        episodes=episode_ids_arr,
    )

    print("\n==========================================================")
    print(f"100% genuine interactive dataset saved at: {output_path}")
    print(f"Total transitions recorded: {len(states_t_arr)}")
    print(f"State dimensions: {states_t_arr.shape}")
    print(f"Action dimensions: {actions_t_arr.shape}")
    print(f"Emulation time: {elapsed:.2f}s ({total_transitions/elapsed:.1f} FPS)")
    print("==========================================================")


if __name__ == "__main__":
    record_interactive_trajectories()
