"""
survey_world_map.py - enumerate the levels the world map will actually hand over.

The repository has one entered level (Yoshi's Island 1) and every picture-based claim in it has been
resting on that single stage. This walks the world-map cursor from the node 10.36.3 measured, records
every position the D-pad reaches, and then tries to enter each one: a node is a level when the engine
leaves the map (0x0E), loads a nonzero timer, and hands control to a player that walks when RIGHT is
held - the same behavioural gate that refuses a frozen attract demo.

The cursor is moved the way a player moves it: by holding a direction, from a map state replayed from
the beginning of the route. Writing $7E:1F17 and pressing A was tried first and is not a teleport -
three written positions entered the same level, translevel 0x2A, because the engine keeps its own
node index and the two bytes are what it reports, not what it decides. Every probe here therefore
replays a direction sequence, and the position read back after the replay is what proves the probe
stood where it intended to.

Writes `results/world_map_survey.json` and, for each node that passes the gate, a savestate under
`data/raw/` named after the node it came from.

Run:  python -m scripts.survey_world_map
      python -m scripts.survey_world_map --max-nodes 8 --no-save
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.navigate_to_level import (  # noqa: E402
    MAP_X_FIRST_LEVEL,
    MAP_Y_FIRST_LEVEL,
    _plausible,
    _right_response,
    walk_to_overworld,
)
from src.environment import wram  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.utils.paths import CORE_PATH, ROM_PATH, data_dir, results_file  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
#: how long one cursor step is held, and how long the map is left alone to finish moving it
STEP_FRAMES = 16
SETTLE_FRAMES = 30
#: frames spent waiting for the engine to reach interactive mode after A
ENTRY_BUDGET = 900
#: frames of nothing after the mode flips, before the level is asked to behave
ENTRY_SETTLE_FRAMES = 180

Position = Tuple[int, int]
Path = List[str]


def cursor(emu: SnesLibretroEmulator) -> Position:
    """The world-map cursor, in map pixels ($7E:1F17, $7E:1F19)."""
    return (
        int(emu.read_wram_u16_le(wram.ADDR_MAP_X)),
        int(emu.read_wram_u16_le(wram.ADDR_MAP_Y)),
    )


def hold(emu: SnesLibretroEmulator, buttons: Dict[str, bool], frames: int) -> None:
    for _ in range(frames):
        emu.set_input(buttons)
        emu.step_frame()


def walk_cursor(emu: SnesLibretroEmulator, direction: str) -> Position:
    """Pulse one D-pad direction on the map and report where the cursor ended up."""
    hold(emu, {direction: True}, STEP_FRAMES)
    hold(emu, {}, SETTLE_FRAMES)
    return cursor(emu)


def replay(emu: SnesLibretroEmulator, map_state: bytes, path: Path) -> Position:
    """Return to the saved map and walk the recorded direction sequence: stand where we stood."""
    emu.load_state(map_state)
    hold(emu, {}, 30)
    position = cursor(emu)
    for direction in path:
        position = walk_cursor(emu, direction)
    return position


def press_a_until_level(emu: SnesLibretroEmulator) -> Tuple[int, List[int]]:
    """Pulse A until the engine is interactive, reporting the frame it happened and the modes seen."""
    modes: List[int] = []
    for frame in range(ENTRY_BUDGET):
        emu.set_input({"A": True} if frame % 12 < 6 else {})
        emu.step_frame()
        mode = emu.get_game_mode()
        if not modes or modes[-1] != mode:
            modes.append(mode)
        if mode == wram.GAME_MODE_INTERACTIVE:
            return frame, modes
    return -1, modes


def discover_nodes(
    emu: SnesLibretroEmulator, map_state: bytes, start: Position, max_nodes: int
) -> Dict[str, Any]:
    """Breadth-first the map cursor, keeping the direction sequence that reaches each position.

    A direction that leaves the cursor where it started is an edge that does not exist. A position
    reached twice is the map's own confirmation that the walk is not inventing nodes, and the first
    sequence to reach it is the one every later probe replays.
    """
    paths: Dict[Position, Path] = {start: []}
    edges: List[Dict[str, Any]] = []
    queue: List[Position] = [start]
    truncated = False
    while queue:
        if len(paths) >= max_nodes:
            truncated = True
            break
        here = queue.pop(0)
        for direction in DIRECTIONS:
            replay(emu, map_state, paths[here])
            there = walk_cursor(emu, direction)
            if there == here:
                continue
            edges.append({"from": list(here), "to": list(there), "direction": direction})
            if there not in paths:
                paths[there] = paths[here] + [direction]
                queue.append(there)
    return {
        "start": list(start),
        "positions": [list(p) for p in paths],
        "paths": {"|".join(str(v) for v in key): value for key, value in paths.items()},
        "edges": edges,
        "max_nodes": max_nodes,
        "truncated": truncated,
    }


def terrain_width(emu: SnesLibretroEmulator, subscreens: int = 40) -> Dict[str, Any]:
    """How far the level's own terrain goes, read from the buffer rather than inferred from a run.

    The buffer at $7E:C800 stores one subscreen (16 columns x 27 rows) after another, so the last
    subscreen holding anything but air bounds the stage's width. This is the measurement the
    clearance question needed: a controller stopped at 4340 px has not met the end of a level that
    is 20 subscreens - 5120 px - wide, and it was the frame budget that ended the run.

    The count is the leading run of non-empty subscreens, so a stage with a genuinely empty
    subscreen in the middle would read short; the per-subscreen profile is returned beside it for
    exactly that reason.
    """
    cells = wram.TILEMAP_ROWS_PER_SUBSCREEN * wram.TILEMAP_COLUMNS_PER_SUBSCREEN
    profile: List[int] = []
    for sub in range(subscreens):
        base = wram.ADDR_TILEMAP_BASE + sub * wram.TILEMAP_SUBSCREEN_STRIDE
        if base + cells > wram.WRAM_SIZE_BYTES:
            break
        ids = [emu.read_wram_u8(base + i) for i in range(cells)]
        profile.append(sum(1 for tile in ids if tile != wram.TILE_AIR))
    leading = 0
    while leading < len(profile) and profile[leading] > 0:
        leading += 1
    return {
        "terrain_subscreens": leading,
        "terrain_px": leading * 256,
        "non_air_cells_by_subscreen": profile,
    }


def try_enter(
    emu: SnesLibretroEmulator, map_state: bytes, path: Path, node: Position
) -> Dict[str, Any]:
    """Stand on one node and press A: report what the engine did, and whether it gave a level.

    The settle after entry is not politeness. The mode byte flips to 0x14 while the level is still
    fading in and before the engine is stepping Mario's physics, which is exactly the frozen frame
    the handoff gate exists to refuse, so the walk test runs only after the settle.
    """
    landed = replay(emu, map_state, path)
    frame, modes = press_a_until_level(emu)
    out: Dict[str, Any] = {
        "node": list(node),
        "path": path,
        "cursor_before_A": list(landed),
        "cursor_reached": list(landed) == list(node),
        "mode_sequence": [f"0x{m:02X}" for m in modes],
        "entry_frame": frame,
        "translevel": f"0x{emu.read_wram_u8(wram.ADDR_TRANSLEVEL):02X}",
        "timer_hundreds": int(emu.read_wram_u8(wram.ADDR_TIMER_HUNDREDS)),
        "interactive": frame >= 0,
    }
    if not out["interactive"]:
        out.update({"level": False, "reason": "the engine never reached interactive mode"})
        return out
    hold(emu, {}, ENTRY_SETTLE_FRAMES)
    out["terrain"] = terrain_width(emu)
    state = emu.get_smw_state()
    dx = _right_response(emu)
    out.update(
        {
            "x_after_settle": float(state["x"]),
            "y_after_settle": float(state["y"]),
            "right_response_px": float(dx),
            "plausible": bool(_plausible(emu)),
            "walkable": bool(_plausible(emu) and dx > 6.0),
        }
    )
    out["level"] = bool(out["walkable"] and out["timer_hundreds"] > 0)
    if not out["walkable"]:
        out["reason"] = f"plausible={out['plausible']} right dx={dx:.1f}"
    elif not out["level"]:
        out["reason"] = "walks, but the timer reads zero: the blank-level case of 10.36.3"
    return out


def capture_level(
    emu: SnesLibretroEmulator, map_state: bytes, path: Path, node: Position, out_path: str
) -> Dict[str, Any]:
    """Re-enter one node and save the state the engine was playing after the fade in.

    The handoff probes of 10.36.3 walk Mario down the stage, so the saved bytes are taken before them.
    The probes are then run as *evidence about* the state rather than as a veto on saving it: a stage
    whose entrance slides Mario leftward fails a rightward-press gate without being unplayable, and
    the picture the dashboard draws from such a state will say for itself what the engine does there.
    """
    replay(emu, map_state, path)
    frame, _ = press_a_until_level(emu)
    if frame < 0:
        return {"saved": None, "reason": f"mode 0x{emu.get_game_mode():02X} after A"}
    hold(emu, {}, ENTRY_SETTLE_FRAMES)
    state_bytes = emu.save_state()
    with open(out_path, "wb") as handle:
        handle.write(state_bytes)
    settled = emu.get_smw_state()
    dx = _right_response(emu)
    return {
        "saved": os.path.basename(out_path),
        "bytes": len(state_bytes),
        "x_at_entry": float(settled["x"]),
        "y_at_entry": float(settled["y"]),
        "right_response_px": float(dx),
        "handoff_gate_of_10_36_3": bool(_plausible(emu) and dx > 6.0),
    }


def state_name(node: Position, translevel: Optional[str]) -> str:
    tag = (translevel or "unknown").lower().replace("0x", "t")
    return f"smw_map_node_{node[0]}_{node[1]}_{tag}.state"


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Survey the world map for enterable levels.")
    parser.add_argument("--max-nodes", type=int, default=40)
    parser.add_argument(
        "--no-save", action="store_true", help="survey the map without capturing states"
    )
    parser.add_argument("--output", default=results_file("world_map_survey.json"))
    parser.add_argument(
        "--from-map-state",
        default=None,
        help=(
            "survey a saved world-map state instead of booting to one - what 10.67 needs, because "
            "the map after a cleared stage is not the map a fresh file opens on"
        ),
    )
    args = parser.parse_args()

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    trace: Dict[str, Any] = {"target_level": 1, "node_round": None, "entry_frame": None}
    for _ in range(220):
        emu.step_frame()
    if args.from_map_state:
        with open(args.from_map_state, "rb") as handle:
            map_state = handle.read()
        emu.load_state(map_state)
        hold(emu, {}, 60)
        if emu.get_game_mode() != wram.GAME_MODE_WORLD_MAP:
            raise RuntimeError(
                f"{args.from_map_state} is not a world map "
                f"(mode 0x{emu.get_game_mode():02X}); the survey reads the cursor from 0x0E"
            )
        map_state = emu.save_state()
        route = f"loaded from {os.path.basename(args.from_map_state)}"
    else:
        if not walk_to_overworld(emu, trace):
            raise RuntimeError(
                f"the world map was never reached (mode 0x{emu.get_game_mode():02X})"
            )
        hold(emu, {}, 60)
        map_state = emu.save_state()
        route = "walk_to_overworld of 10.36.3, from a power-on boot"
    start = cursor(emu)
    print(f"World map live ({route}); cursor at {start}.")

    discovery = discover_nodes(emu, map_state, start, args.max_nodes)
    print(f"Cursor walk reached {len(discovery['positions'])} positions.")

    results: List[Dict[str, Any]] = []
    for position in discovery["positions"]:
        node = (int(position[0]), int(position[1]))
        path = discovery["paths"]["|".join(str(v) for v in node)]
        report = try_enter(emu, map_state, path, node)
        print(
            f"node {node} via {path or ['-']}: translevel {report['translevel']} "
            f"timer {report['timer_hundreds']} level={report.get('level')}"
        )
        if report.get("walkable") and not args.no_save:
            path_out = str(data_dir() / state_name(node, report.get("translevel")))
            report["capture"] = capture_level(emu, map_state, path, node, path_out)
            print(f"  capture: {report['capture']}")
        results.append(report)

    emu.close()
    payload = {
        "study": (
            "Which nodes of the Super Mario World map the engine will actually hand over: the cursor "
            "positions reachable from the first node, and for each one whether A leaves the map, "
            "loads a level with a timer, and gives control to a player that walks."
        ),
        "protocol": {
            "route": route,
            "from_map_state": (
                os.path.basename(args.from_map_state) if args.from_map_state else None
            ),
            "start_node": [MAP_X_FIRST_LEVEL, MAP_Y_FIRST_LEVEL],
            "cursor_at_start": list(start),
            "step_frames": STEP_FRAMES,
            "settle_frames": SETTLE_FRAMES,
            "entry_budget_frames": ENTRY_BUDGET,
            "entry_settle_frames": ENTRY_SETTLE_FRAMES,
            "gate": (
                "walkable = interactive mode + plausible coordinates + right-press dx > 6 px; "
                "level = walkable and a nonzero timer"
            ),
            "savestate_dir": "data/raw",
            "saved_states": bool(not args.no_save),
            "cursor_write_is_a_teleport": False,
        },
        "discovery": discovery,
        "nodes": results,
        "walkable": [r for r in results if r.get("walkable")],
        "levels": [r for r in results if r.get("level")],
        "translevels_seen": sorted({str(r.get("translevel")) for r in results}),
    }
    write_metrics(args.output, payload, command="python -m scripts.survey_world_map")
    print(f"{len(payload['levels'])} of {len(results)} nodes passed the level gate.")
    return payload


if __name__ == "__main__":
    main()
