"""
record_boundary_gameplay.py - record the state, the terrain AND the camera, so a boundary is observable.

10.58 found that 75.4-99.3% of the exceptions to section 4.1 are positions that do not move while
the velocity byte holds, and predicted that the missing observable is a constraint the 8D state does
not carry. This recording adds it: the layer 1 scroll at $7E:001A, identified by
`scripts/scan_scroll_address.py`, next to the same 8D state, 6D action and 7x7 tile patch the
tilemap recording stores. The behaviour vocabulary is the tilemap recorder's, so the two recordings
measure the same kind of driving and their identity rates are comparable.

State Vector (8D): [x, y, vx, vy, c_ground, c_ceiling, c_left, c_right]
Camera: layer 1 horizontal scroll in pixels; screen position = x - camera
Mode: the engine mode byte at $7E:0100 (0x14 is interactive), recorded because a stopped simulation
      and a stopped body look the same in every other channel
Tilemap Patch (7x7): 0 air, 1 solid, 2 hazard, 3 slope, centred on Mario's tile
Action Vector (6D): [B, Y, UP, DOWN, LEFT, RIGHT]

Run:  python scripts/record_boundary_gameplay.py
      python scripts/record_boundary_gameplay.py --episodes 10 --frames 200
"""

import argparse
import os
import sys
import time
import zlib
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CORE_PATH,
    DATASET_BOUNDARY,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
)
from src.utils.seed import set_global_seed  # noqa: E402

#: the tilemap recorder's behaviour vocabulary, so this is the same kind of driving
ACTION_BEHAVIORS: List[Dict[str, bool]] = [
    {"RIGHT": True, "Y": True},
    {"RIGHT": True, "Y": True, "B": True},
    {"RIGHT": True, "B": True},
    {"RIGHT": True},
    {"LEFT": True, "Y": True},
    {"LEFT": True, "B": True},
    {"LEFT": True},
    {},
    {"B": True},
    {"RIGHT": True, "DOWN": True},
]
FORWARD_CHOICES = [0, 1, 2, 3, 8]
RETREAT_CHOICES = [4, 5, 6, 7, 9]


def extract_vector(state: Dict[str, float]) -> np.ndarray:
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


def extract_action(action: Dict[str, bool]) -> np.ndarray:
    return np.array(
        [
            1.0 if action.get("B", False) else 0.0,
            1.0 if action.get("Y", False) else 0.0,
            1.0 if action.get("UP", False) else 0.0,
            1.0 if action.get("DOWN", False) else 0.0,
            1.0 if action.get("LEFT", False) else 0.0,
            1.0 if action.get("RIGHT", False) else 0.0,
        ],
        dtype=np.float32,
    )


def observe(emu: SnesLibretroEmulator) -> Dict[str, Any]:
    """One observation: the published 8D state, the camera, the terrain, and the engine mode.

    The mode byte is the channel that tells a stopped simulation from a stopped body: 10.59 measures
    that frames outside interactive mode carry no player dynamics at all, so a recording without this
    channel cannot tell the two apart. The CRC of all 128 KB of WRAM is the second half of that test:
    a frame whose player record repeats while the CRC moves is the console running and the player
    object not being processed, which is a mechanism; a frame whose CRC does not move is the harness
    sampling the same emulated frame twice, which is an artifact.
    """
    state = emu.get_smw_state()
    return {
        "vector": extract_vector(state),
        "camera": emu.get_camera_x(),
        "mode": float(emu.get_game_mode()),
        "wram_crc": float(zlib.crc32(bytes(emu.wram_buffer))),
        "patch": emu.get_local_tilemap_patch(state["x"], state["y"], radius=3),
    }


def record(
    episodes: int,
    frames: int,
    rom_path: str,
    core_path: str,
    state_path: str,
    output_path: str,
    seed_base: int = 4000,
) -> Dict[str, Any]:
    print("=" * 60)
    print("  RECORDING STATE + CAMERA + TERRAIN (WRAM $0094/$001A/$C800)")
    print("=" * 60)
    print(f"ROM: {rom_path}\nCore: {core_path}\nSavestate: {state_path}")
    print(f"Episodes: {episodes} | Max frames per episode: {frames}")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(state_path, "rb") as fh:
        initial_savestate = fh.read()

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    states: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    cameras: List[float] = []
    next_cameras: List[float] = []
    modes: List[float] = []
    next_modes: List[float] = []
    crcs: List[float] = []
    next_crcs: List[float] = []
    patches: List[np.ndarray] = []
    next_patches: List[np.ndarray] = []
    episode_ids: List[int] = []
    travel: List[float] = []
    t0 = time.time()

    for ep in range(episodes):
        emu.start_episode(initial_savestate)
        set_global_seed(seed_base + ep)
        pattern = 0
        duration = int(np.random.randint(15, 60))
        timer = 0
        prev_b = False
        obs = observe(emu)
        xs: List[float] = []

        for f in range(frames):
            if timer >= duration:
                forward = np.random.rand() < 0.70
                pattern = int(np.random.choice(FORWARD_CHOICES if forward else RETREAT_CHOICES))
                duration = int(np.random.randint(15, 60))
                timer = 0
            action = dict(ACTION_BEHAVIORS[pattern])
            if action.get("B", False):
                if prev_b:
                    action["B"] = False
                prev_b = action["B"]
            else:
                prev_b = False

            emu.set_input(action)
            emu.step_frame()
            nxt = observe(emu)
            xs.append(float(nxt["vector"][0]))
            if 0.0 <= float(nxt["vector"][1]) <= 500.0:
                states.append(obs["vector"])
                next_states.append(nxt["vector"])
                actions.append(extract_action(action))
                cameras.append(float(obs["camera"]))
                next_cameras.append(float(nxt["camera"]))
                modes.append(float(obs["mode"]))
                next_modes.append(float(nxt["mode"]))
                crcs.append(float(obs["wram_crc"]))
                next_crcs.append(float(nxt["wram_crc"]))
                patches.append(obs["patch"])
                next_patches.append(nxt["patch"])
                episode_ids.append(ep)
            obs = nxt
            timer += 1
            if nxt["vector"][1] > 500.0 or nxt["vector"][1] < 0.0:
                break
        travel.append(float(max(xs) - min(xs)) if xs else 0.0)
        if (ep + 1) % 5 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep + 1:2d}/{episodes} | transitions {len(states):6d} | "
                f"x travelled this episode {travel[-1]:7.2f} px"
            )
    emu.close()

    elapsed = max(1e-5, time.time() - t0)
    arrays = {
        "states": np.asarray(states, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "camera": np.asarray(cameras, dtype=np.float32),
        "next_camera": np.asarray(next_cameras, dtype=np.float32),
        "mode": np.asarray(modes, dtype=np.float32),
        "next_mode": np.asarray(next_modes, dtype=np.float32),
        "wram_crc": np.asarray(crcs, dtype=np.float64),
        "next_wram_crc": np.asarray(next_crcs, dtype=np.float64),
        "tile_patches": np.asarray(patches, dtype=np.int64),
        "next_tile_patches": np.asarray(next_patches, dtype=np.int64),
        "episodes": np.asarray(episode_ids, dtype=np.int32),
    }
    np.savez_compressed(output_path, **arrays)
    print("\nRecording complete:")
    print(f"Transitions: {len(states)} over {episodes} episodes")
    print(f"Throughput: {len(states) / elapsed:.1f} FPS (elapsed {elapsed:.1f}s)")
    print(f"Camera range: {arrays['camera'].min():.0f} to {arrays['camera'].max():.0f} px")
    print(f"Saved to: {output_path}")
    return {"transitions": len(states), "output": output_path, "elapsed_s": elapsed}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--rom", default=ROM_PATH)
    parser.add_argument("--core", default=CORE_PATH)
    parser.add_argument("--state", default=STATE_YOSHI_ISLAND_1)
    parser.add_argument("--output", default=DATASET_BOUNDARY)
    args = parser.parse_args(argv)
    record(
        episodes=args.episodes,
        frames=args.frames,
        rom_path=args.rom,
        core_path=args.core,
        state_path=args.state,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
