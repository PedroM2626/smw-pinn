"""
verify_level_entry.py - A/B the two boot schedules, and prove the good one renders a scrolling level.

`navigate_to_level.py --level 1` used to save a savestate it had never verified. When the
control-response gate built for stage 2 (§10.36.2) was applied to stage 1, the old schedule failed
it: the mode byte reaches 0x14, the PPU draws a level, and Mario stops responding after one stride -
the title screen's attract demo, whose player is scripted. The schedule that works was taken from
the route the MIT-licensed araujoviana/burn-mario-rl documents: wait for the *world map* (0x0E),
enter and leave Yoshi's House, then warp the map cursor and press A.

This script runs both arms on the console and writes
``results/level_entry_verification.json``, which README 10.36.3 quotes. The measurement that
distinguishes them is not the mode byte - both arms reach 0x14 - but whether the rendered picture
translates when the recorded position does, the same test `src/evaluation/pixel_frame_probe.py`
applies to the committed pixel recording.

Needs the core and the ROM, like every recorder here.

Run:  python scripts/verify_level_entry.py
      python scripts/verify_level_entry.py --probe-frames 600
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.navigate_to_level import (  # noqa: E402
    MAP_X_FIRST_LEVEL,
    MAP_Y_FIRST_LEVEL,
    _hold,
    _plausible,
    _right_response,
    walk_to_overworld,
    warp_on_map,
)
from src.environment import wram  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.evaluation.pixel_frame_probe import best_shift  # noqa: E402
from src.utils.paths import CORE_PATH, ROM_PATH  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

ARTIFACT_NAME = "level_entry_verification.json"
MODE_IN_LEVEL = wram.GAME_MODE_INTERACTIVE
MODE_WORLD_MAP = wram.GAME_MODE_WORLD_MAP


def probe_session(emu: SnesLibretroEmulator, frames: int) -> Dict[str, Any]:
    """Walk right with a jump every 24 frames: how far does the state go, and does the picture?"""
    xs: List[float] = []
    cams: List[float] = []
    shots: List[np.ndarray] = []
    for i in range(frames):
        emu.set_input({"RIGHT": True, "Y": True, "B": (i % 24 == 0)})
        emu.step_frame()
        frame = emu.get_frame()
        if frame is None:
            continue
        state = emu.get_smw_state()
        xs.append(state["x"])
        cams.append(emu.get_camera_x())
        shots.append(frame)
    if len(shots) < 20:
        return {"frames_captured": len(shots)}
    grid = np.asarray(shots, dtype=np.float32).mean(axis=3)[:, 14:, :]
    shifts = np.array(
        [best_shift(grid[t], grid[t + 1], 4)[0] for t in range(len(grid) - 1)], dtype=np.float64
    )
    dxs = np.diff(np.asarray(xs, dtype=np.float64))[: len(shifts)]
    return {
        "frames_captured": int(len(shots)),
        "x_start": float(xs[0]),
        "x_end": float(xs[-1]),
        "x_travel_px": float(xs[-1] - xs[0]),
        "camera_start": float(cams[0]),
        "camera_end": float(cams[-1]),
        "share_of_pairs_picture_shifts": float(np.mean(np.abs(shifts) > 0)),
        "mean_abs_picture_shift_px": float(np.abs(shifts).mean()),
        "mean_abs_dx_px": float(np.abs(dxs).mean()),
        "corr_picture_shift_with_dx": (
            float(np.corrcoef(shifts, dxs)[0, 1])
            if float(np.std(shifts)) > 1e-9 and float(np.std(dxs)) > 1e-9
            else None
        ),
    }


def arm_naive(emu: SnesLibretroEmulator, probe_frames: int) -> Dict[str, Any]:
    """The schedule the navigator used before: pulse START and B, never wait for the map."""
    _hold(emu, {}, 220)
    for _ in range(15):
        if emu.get_game_mode() not in (wram.GAME_MODE_TITLE, wram.GAME_MODE_FILE_SELECT):
            break
        _hold(emu, {"START": True}, 20)
        _hold(emu, {"B": True}, 20)
    _hold(emu, {}, 60)
    for _ in range(180):
        _hold(emu, {"B": True}, 1)
    for _ in range(60):
        _hold(emu, {"LEFT": True, "B": True}, 1)
    reached = -1
    for _ in range(40):
        if emu.get_game_mode() == MODE_IN_LEVEL:
            reached = 1
            break
        _hold(emu, {"START": True}, 8)
        _hold(emu, {}, 12)
        _hold(emu, {"B": True}, 8)
        _hold(emu, {}, 20)
    state = emu.get_smw_state()
    dx = _right_response(emu) if reached else None
    out: Dict[str, Any] = {
        "final_mode": f"0x{emu.get_game_mode():02X}",
        "reached_level_mode": bool(reached == 1),
        "x_at_entry": float(state["x"]),
        "y_at_entry": float(state["y"]),
        "right_response_px": None if dx is None else float(dx),
        "plausible": _plausible(emu),
    }
    if reached == 1:
        out["session"] = probe_session(emu, probe_frames)
    return out


def arm_measured(emu: SnesLibretroEmulator, probe_frames: int) -> Dict[str, Any]:
    """The route that works: world map, house in and out, then the cursor warp."""
    trace: Dict[str, Any] = {"target_level": 1}
    on_map = walk_to_overworld(emu, trace)
    out: Dict[str, Any] = {
        "reached_world_map": bool(on_map),
        "world_map_frames": trace.get("overworld_frames"),
        "house_entry_frames": trace.get("house_entry_frames"),
        "house_exit_frames": trace.get("house_exit_frames"),
    }
    if not on_map:
        return out
    entered = warp_on_map(emu, MAP_X_FIRST_LEVEL, MAP_Y_FIRST_LEVEL)
    out.update(
        {
            "entered_node": bool(entered),
            "map_cursor": [MAP_X_FIRST_LEVEL, MAP_Y_FIRST_LEVEL],
            "translevel": f"0x{emu.read_wram_u8(wram.ADDR_TRANSLEVEL):02X}",
            "final_mode": f"0x{emu.get_game_mode():02X}",
        }
    )
    if not entered:
        return out
    state = emu.get_smw_state()
    out["x_at_entry"] = float(state["x"])
    out["y_at_entry"] = float(state["y"])
    # The control probe walks Mario down the stage, so the session starts from where the level
    # starts, not from where the probe left him.
    level_start = emu.save_state()
    out["right_response_px"] = float(_right_response(emu))
    emu.load_state(level_start)
    out["session"] = probe_session(emu, probe_frames)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--probe-frames", type=int, default=400)
    parser.add_argument("--output", default=os.path.join("results", ARTIFACT_NAME))
    args = parser.parse_args(argv)

    results: Dict[str, Any] = {}
    for name, arm in (("naive_schedule", arm_naive), ("measured_route", arm_measured)):
        emu = SnesLibretroEmulator(CORE_PATH)
        emu.load_rom(ROM_PATH)
        emu.enable_frame_capture(True)
        results[name] = arm(emu, args.probe_frames)
        emu.close()
        print(
            f"{name}: {results[name].get('final_mode')} session={results[name].get('session', {})}"
        )

    naive_session = results["naive_schedule"].get("session", {})
    measured_session = results["measured_route"].get("session", {})
    payload: Dict[str, Any] = {
        "study": (
            "Which boot schedule lands on a rendered, controllable level, measured by whether the "
            "picture translates when the recorded position does."
        ),
        "protocol": {
            "rom": os.path.basename(ROM_PATH),
            "probe_frames": args.probe_frames,
            "first_node_cursor": [MAP_X_FIRST_LEVEL, MAP_Y_FIRST_LEVEL],
            "picture_shift_search_px": 4,
            "hud_rows_excluded": 14,
            "published_pixel_recording_reference": "results/pixel_frame_probe_metrics.json",
        },
        "arms": results,
        "comparison": {
            "naive_share_of_pairs_picture_shifts": naive_session.get(
                "share_of_pairs_picture_shifts"
            ),
            "measured_share_of_pairs_picture_shifts": measured_session.get(
                "share_of_pairs_picture_shifts"
            ),
            "naive_right_response_px": results["naive_schedule"].get("right_response_px"),
            "measured_right_response_px": results["measured_route"].get("right_response_px"),
        },
    }
    write_metrics(args.output, payload, command="python scripts/verify_level_entry.py")
    print(f"artifact written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
