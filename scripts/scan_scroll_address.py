"""
scan_scroll_address.py - capture WRAM while Mario runs the level and identify the camera.

`src/environment/scroll_scan.py` holds the ranking and the axioms; this script is the part that
needs the console. It restores the Yoshi's Island 1 savestate, runs right with a jump every
`cadence` frames for a fixed list of cadences, keeps the trace that travelled furthest, and writes
``results/scroll_address_scan_metrics.json`` - the evidence behind ``ADDR_CAMERA_X``, which is the
address the boundary-channel recording and README section 10.59 then read.

Hardware-dependent by construction: it drives the emulator and reads its memory.

Run:  python scripts/scan_scroll_address.py
      python scripts/scan_scroll_address.py --frames 600 --output results/scratch_scan.json
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.scroll_scan import ARTIFACT_NAME, identify  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.environment.wram import WRAM_SIZE_BYTES  # noqa: E402
from src.utils.paths import CORE_PATH, RESULTS_DIR, ROM_PATH, STATE_YOSHI_ISLAND_1  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

CADENCES: List[int] = [24, 32, 16, 44, 8, 12, 20, 28]
INTERACTIVE_MODE = 0x14
DEATH_Y = 450.0


def trace(
    emu: SnesLibretroEmulator, savestate: bytes, cadence: int, frames: int, back: int
) -> Dict[str, Any]:
    """One episode: run right with a jump every `cadence` frames, then walk back."""
    emu.start_episode(savestate)
    ram: List[np.ndarray] = []
    xs: List[float] = []
    ys: List[float] = []
    vxs: List[float] = []
    modes: List[int] = []

    def snap() -> None:
        state = emu.get_smw_state()
        ram.append(np.frombuffer(emu.wram_buffer, np.uint8, count=WRAM_SIZE_BYTES).copy())
        xs.append(float(state["x"]))
        ys.append(float(state["y"]))
        vxs.append(float(state["vx"]))
        modes.append(int(emu.get_game_mode()))

    snap()
    for i in range(frames + back):
        if i < frames:
            action = {"RIGHT": True, "Y": True, "B": (i % cadence == 0)}
        else:
            action = {"LEFT": True, "B": ((i - frames) % cadence == 0)}
        emu.set_input(action)
        emu.step_frame()
        snap()
        if modes[-1] != INTERACTIVE_MODE or ys[-1] > DEATH_Y:
            break
    x = np.asarray(xs, dtype=np.float64)
    return {
        "cadence": cadence,
        "frames": len(ram),
        "ram": np.asarray(ram, dtype=np.uint8),
        "x": x,
        "vx": np.asarray(vxs, dtype=np.float64),
        "travel": float(x.max() - x.min()),
        "end_mode": modes[-1],
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--frames", type=int, default=420, help="forward frames per attempt")
    parser.add_argument("--back", type=int, default=120, help="frames spent walking back left")
    parser.add_argument("--cadences", default=",".join(str(c) for c in CADENCES))
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, ARTIFACT_NAME))
    args = parser.parse_args(argv)

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    with open(STATE_YOSHI_ISLAND_1, "rb") as fh:
        savestate = fh.read()

    attempts: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for cadence in [int(c) for c in args.cadences.split(",")]:
        got = trace(emu, savestate, cadence, args.frames, args.back)
        attempts.append(
            {"cadence": got["cadence"], "frames": got["frames"], "travel_px": got["travel"]}
        )
        print(
            f"cadence {cadence:3d}: {got['frames']:5d} frames, travelled {got['travel']:8.3f} px, "
            f"x reached {float(got['x'].max()):8.3f}"
        )
        if best is None or got["travel"] > best["travel"]:
            best = got
    emu.close()

    assert best is not None
    payload = identify(best["ram"], best["x"])
    payload["study"] = (
        "Identify the horizontal camera in WRAM from the console rather than from a memory map: "
        "rank every 16-bit word by how well it follows Mario's world coordinate, then require the "
        "winner to satisfy the axioms of a layer scroll - never ahead of the body, the screen "
        "position inside the window on every frame, still while he walks the dead zone, moving both "
        "ways, zero at the level start."
    )
    payload["protocol"] = {
        "savestate": os.path.basename(STATE_YOSHI_ISLAND_1),
        "forward_frames": args.frames,
        "back_frames": args.back,
        "cadences_tried": attempts,
        "kept_cadence": best["cadence"],
        "kept_frames": best["frames"],
        "kept_x_px": [float(best["x"].min()), float(best["x"].max())],
        "wram_bytes": WRAM_SIZE_BYTES,
        "deterministic": "the emulator is fed a fixed button schedule, so the trace repeats",
    }
    write_metrics(
        args.output,
        payload,
        command="python scripts/scan_scroll_address.py",
        extra_meta={"dataset": "in-memory WRAM dumps; nothing but the artifact is written"},
    )
    print(f"ranked {payload['words_ranked']} words; passing: {payload['passing']}")
    print(f"chosen: {payload['chosen']}  parallax sibling: {payload['parallax_sibling']}")
    print(f"artifact written to {args.output}")
    return 0 if payload["chosen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
