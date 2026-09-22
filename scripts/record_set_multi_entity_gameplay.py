"""
record_set_multi_entity_gameplay.py
Records full 12-slot sprite sets: per frame, Mario 8D + [12, 5] entity rows
([dx, dy, vx, vy, active], egocentric, slot-ordered, zero-padded) + action.
Unlike the 12D single-hazard dataset, this preserves *all* active sprites.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator
from src.environment.sprite_sets import sprites_to_entity_rows
from src.utils.paths import CORE_PATH, DATASET_SET_MULTI_ENTITY, ROM_PATH, STATE_YOSHI_ISLAND_1
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


def record_set_dataset(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    output_path: str = DATASET_SET_MULTI_ENTITY,
    num_episodes: int = 10,
    frames_per_episode: int = 400,
    max_entities: int = 12,
):
    print("==========================================================")
    print("  RECORDING FULL 12-SLOT SPRITE-SET DATASET              ")
    print("==========================================================")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    behaviors = [
        {"RIGHT": True, "Y": True},
        {"RIGHT": True, "Y": True, "B": True},
        {"RIGHT": True, "B": True},
        {"RIGHT": True},
        {"LEFT": True},
        {},
        {"B": True},
    ]

    mario_t, ent_t, act_t, mario_n, ent_n, eps = [], [], [], [], [], []
    total, active_frames = 0, 0

    for ep in range(num_episodes):
        emu.load_state(initial_savestate)
        emu.enable_gameplay_mode()
        for _ in range(5):
            emu.step_frame()
        set_global_seed(7000 + ep)
        pattern, duration, timer = 0, 25, 0

        for _ in range(frames_per_episode):
            if timer >= duration:
                pattern = np.random.randint(len(behaviors))
                duration = np.random.randint(10, 40)
                timer = 0
            action_dict = behaviors[pattern]
            emu.set_input(action_dict)
            s = emu.get_smw_state()
            ent = sprites_to_entity_rows(emu.get_active_sprites(), s["x"], s["y"], max_entities)
            emu.step_frame()
            s2 = emu.get_smw_state()
            ent2 = sprites_to_entity_rows(emu.get_active_sprites(), s2["x"], s2["y"], max_entities)
            if 0 <= s2["y"] <= 500:
                mario_t.append(extract_vector(s))
                ent_t.append(ent)
                act_t.append(extract_action_vector(action_dict))
                mario_n.append(extract_vector(s2))
                ent_n.append(ent2)
                eps.append(ep)
                total += 1
                active_frames += int(ent[:, 4].sum() > 0)
            timer += 1
            if s2["y"] > 500 or s2["y"] < 0:
                break
        print(f"Episode {ep + 1}/{num_episodes} | transitions: {total}")

    emu.close()
    np.savez_compressed(
        output_path,
        mario=np.array(mario_t, dtype=np.float32),
        entities=np.array(ent_t, dtype=np.float32),
        actions=np.array(act_t, dtype=np.float32),
        next_mario=np.array(mario_n, dtype=np.float32),
        next_entities=np.array(ent_n, dtype=np.float32),
        episodes=np.array(eps, dtype=np.int32),
    )
    print(f"Saved {output_path} | N={total} | active-sprite frames={active_frames}")


if __name__ == "__main__":
    record_set_dataset()
