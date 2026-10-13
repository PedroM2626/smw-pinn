"""
probe_tile_behaviors.py - measure what every Map16 tile does to the engine, off the engine itself.

The dashboard's tile window used to classify cells from a hardcoded list of tile ids guessed for
one stage, which painted bushes as pipes and walls, ignored half of every tile number (the buffer
stores 16-bit Map16 ids: low byte at $7E:C800, high byte at $7F:C800), and called every
non-blank tile solid. This probe replaces the guesses with measurement: for each tile id
0x000-0x1FF it writes that id into the level buffer at a fabricated, controlled position, drops
Mario onto it and runs Mario into it, and reads back what the engine itself decided - the
collision flags at $7E:0077, the resting altitude, the death mode, the coin counter, the buffer
cell after contact and whether the engine froze the player (a message box, the signature of
10.63).

The rig carves a landing zone several tiles wide around the candidate so the player hitbox
($7E:0094 is the left edge; the box spans x+2..x+13 and the small player's body the lower 16 of
the 32-pixel frame) can never catch a neighbouring cell's shoulder, which is what a narrower rig
does: a one-column pit still offers the next column's ledge. Both probes were validated against
cells whose behaviour is known from live play before this script trusts itself: the blank tile
lets the drop through, the grass top holds it, and a wall of the tile the published episode
brushed mid-air (0x124) stops the run while walls of bush and scenery tiles do not.

What is measured per tile id:

* the drop: does the engine hold Mario up (solid from above), where does he rest (inside the
  cell's vertical span means a slope), does contact kill a small Mario, does the cell pay a coin
  or vanish, does the engine freeze the player (an interaction);
* the wall run: does a running Mario stop against a column of the tile (solid from the side),
  and does that contact kill.

Writes `results/tile_behavior_probe_metrics.json` - the measured table every other tile reader
in this repository then classifies through (`src.environment.tile_behavior`).

Run:  python -m scripts.probe_tile_behaviors
      python -m scripts.probe_tile_behaviors --ids 0x25,0x100,0x2B,0x149
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import wram  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CORE_PATH,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1_LEVEL,
    results_file,
)
from src.utils.provenance import write_metrics  # noqa: E402

logger = get_logger(__name__)

#: the rig's candidate column, in tiles; far from the level's own walls and pits
RIG_COLUMN = 4
#: rows the drop rig clears to air around and below the candidate (row 27 does not exist)
DROP_ZONE_ROWS = (24, 26)
#: rows the wall rig clears, plus the running floor row below them
WALL_ZONE_ROWS = (18, 23)
WALL_FLOOR_ROW = 24
#: frames the drop is given to fall, land and settle on the candidate
DROP_FRAMES = 150
#: further frames standing on the candidate, watching for a late death or a frozen player
SETTLE_FRAMES = 90
#: frames the wall run is given
WALL_FRAMES = 90
#: the tile used as the running floor of the wall rig: the grass top, itself probed first
FLOOR_TILE = 0x0100
AIR = 0x0025
#: the coin counter is one byte at $7E:0DBF (0..99), not a word
ADDR_COINS = 0x0DBF
#: self-check anchors: the probe only trusts itself when these come out as live play shows
ANCHOR_AIR = 0x0025
ANCHOR_TOP = 0x0100
ANCHOR_WALL = 0x0124


def read_tile(emu: SnesLibretroEmulator, tx: int, ty: int) -> Optional[int]:
    """The full 16-bit Map16 id of one buffer cell: low byte at $7E:C800, high at $7F:C800."""
    addr = wram.tilemap_addr(tx, ty)
    if addr is None:
        return None
    return emu.read_wram_u8(addr) | (emu.read_wram_u8(addr + 0x10000) << 8)


def write_tile(emu: SnesLibretroEmulator, tx: int, ty: int, tile_id: int) -> None:
    addr = wram.tilemap_addr(tx, ty)
    assert addr is not None, f"the rig asked for a cell the buffer does not have: ({tx}, {ty})"
    emu.write_wram_u8(addr, tile_id & 0xFF)
    emu.write_wram_u8(addr + 0x10000, (tile_id >> 8) & 0xFF)


def fill(
    emu: SnesLibretroEmulator, tx_lo: int, tx_hi: int, ty_lo: int, ty_hi: int, tile_id: int
) -> None:
    """Write `tile_id` over the inclusive rectangle, all four bounds exclusive on the high side."""
    for ty in range(ty_lo, ty_hi):
        for tx in range(tx_lo, tx_hi):
            write_tile(emu, tx, ty, tile_id)


def clear_sprites(emu: SnesLibretroEmulator) -> None:
    for slot in range(wram.NUM_SPRITE_SLOTS):
        emu.write_wram_u8(wram.ADDR_SPRITE_STATUS + slot, 0)


def teleport(emu: SnesLibretroEmulator, x: int, y: int) -> None:
    emu.write_wram_u16_le(wram.ADDR_PLAYER_X, x)
    emu.write_wram_u16_le(wram.ADDR_PLAYER_Y, y)
    emu.write_wram_u8(wram.ADDR_PLAYER_X_SUB, 0)
    emu.write_wram_u8(wram.ADDR_PLAYER_Y_SUB, 0)
    emu.write_wram_u8(wram.ADDR_VX, 0)
    emu.write_wram_u8(wram.ADDR_VY, 0)


def start_rig(emu: SnesLibretroEmulator, state_bytes: bytes, tile_id: int, mode: str) -> int:
    """Reset to the level entry, clear sprites, carve the rig, place the candidate.

    Returns the coin count before the probe, so a probe can report what changed.
    """
    emu.load_state(state_bytes)
    for _ in range(10):
        emu.set_input({})
        emu.step_frame()
    clear_sprites(emu)
    if mode == "drop":
        fill(emu, RIG_COLUMN - 2, RIG_COLUMN + 3, DROP_ZONE_ROWS[0], DROP_ZONE_ROWS[1] + 1, AIR)
        write_tile(emu, RIG_COLUMN, DROP_ZONE_ROWS[0], tile_id)
    else:
        fill(emu, RIG_COLUMN - 2, RIG_COLUMN + 3, WALL_ZONE_ROWS[0], WALL_ZONE_ROWS[1] + 1, AIR)
        fill(emu, RIG_COLUMN - 2, RIG_COLUMN + 3, WALL_FLOOR_ROW, WALL_FLOOR_ROW + 1, FLOOR_TILE)
        fill(emu, RIG_COLUMN + 1, RIG_COLUMN + 2, WALL_ZONE_ROWS[0], WALL_ZONE_ROWS[1] + 1, tile_id)
    return emu.read_wram_u8(ADDR_COINS)


def frozen(emu: SnesLibretroEmulator, last: Optional[tuple]) -> bool:
    """True when the whole telemetry record has not moved: the engine is holding the player."""
    state = emu.get_smw_state()
    record = tuple(round(float(state[k]), 4) for k in ("x", "y", "vx", "vy"))
    moved = last is not None and record != last
    return (not moved, record)


def drop_probe(emu: SnesLibretroEmulator, state_bytes: bytes, tile_id: int) -> Dict[str, Any]:
    """Drop a small Mario onto one candidate cell, let him settle, and read what happened.

    The landing answers solidity from above, the resting altitude answers slope-vs-flat, the
    settle answers whether standing there kills or freezes, and the coin counter plus the cell
    itself answer what contact consumed.
    """
    coins_before = start_rig(emu, state_bytes, tile_id, "drop")
    row_top = DROP_ZONE_ROWS[0] * 16
    teleport(emu, RIG_COLUMN * 16 + 4, row_top - 160)
    for _ in range(DROP_FRAMES):
        emu.set_input({})
        emu.step_frame()
        if emu.get_game_mode() != wram.GAME_MODE_INTERACTIVE:
            break
    state = emu.get_smw_state()
    held = float(state["c_ground"]) > 0.5
    died = emu.get_game_mode() != wram.GAME_MODE_INTERACTIVE
    last: Optional[tuple] = None
    still = 0
    if held and not died:
        for _ in range(SETTLE_FRAMES):
            moved, last = frozen(emu, last)
            still = still + 1 if not moved else 0
            emu.set_input({})
            emu.step_frame()
            if emu.get_game_mode() != wram.GAME_MODE_INTERACTIVE:
                died = True
                break
            if still >= SETTLE_FRAMES:
                break
    final = emu.get_smw_state()
    feet = float(final["y"]) + 32.0
    return {
        "held_up": bool(float(final["c_ground"]) > 0.5) and not died,
        "rest_feet_px": round(feet, 1),
        "rests_inside_cell": bool(row_top + 2.0 <= feet <= row_top + 14.0),
        "fell_past": bool(feet > float(row_top) + 16.0),
        "died": bool(died),
        "held_frozen": bool(still >= SETTLE_FRAMES),
        "coins_gained": int(emu.read_wram_u8(ADDR_COINS)) - int(coins_before),
        "cell_after": read_tile(emu, RIG_COLUMN, DROP_ZONE_ROWS[0]),
        "cell_consumed": bool(read_tile(emu, RIG_COLUMN, DROP_ZONE_ROWS[0]) != tile_id),
    }


def wall_probe(emu: SnesLibretroEmulator, state_bytes: bytes, tile_id: int) -> Dict[str, Any]:
    """Run Mario along a fabricated floor into a column of the candidate."""
    start_rig(emu, state_bytes, tile_id, "wall")
    floor_top = WALL_FLOOR_ROW * 16
    teleport(emu, (RIG_COLUMN - 2) * 16 + 4, floor_top - 40)
    for _ in range(30):
        emu.set_input({})
        emu.step_frame()
    start_x = float(emu.get_smw_state()["x"])
    blocked = False
    died = False
    for frame in range(WALL_FRAMES):
        emu.set_input({"RIGHT": True, "Y": True})
        emu.step_frame()
        state = emu.get_smw_state()
        blocked = blocked or float(state["c_right"]) > 0.5
        if emu.get_game_mode() != wram.GAME_MODE_INTERACTIVE:
            died = True
            break
    final = emu.get_smw_state()
    return {
        "side_blocked": bool(blocked),
        "advance_px": round(float(final["x"]) - start_x, 1),
        "stopped_short": bool(abs(float(final["x"]) - start_x) < 40.0),
        "died": bool(died),
    }


def classify(drop: Dict[str, Any], wall: Dict[str, Any]) -> str:
    """The behaviour class the panel and the models use, derived only from what was measured.

    `solid` blocks from the side; `platform` only holds from above (the grass tops behave this
    way, which is why the old classifier painted every raised ledge as a wall the engine would
    have let the player walk behind); `slope` holds from above and rests the player inside the
    cell's own span; `harmful` kills on contact; `coin` pays the counter or is consumed by it;
    `interact` freezes the player without killing him; `decor` is everything the engine ignores.
    """
    if drop["died"] or wall["died"]:
        return "harmful"
    if drop["held_frozen"]:
        return "interact"
    if drop["coins_gained"] > 0 or drop["cell_consumed"]:
        return "coin"
    if drop["held_up"] and drop["rests_inside_cell"]:
        return "slope"
    if wall["side_blocked"] or wall["stopped_short"]:
        return "solid"
    if drop["held_up"]:
        return "platform"
    return "decor"


def parse_ids(text: str) -> List[int]:
    return [int(part, 0) for part in text.split(",") if part.strip()]


TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "environment", "tile_behavior_table.py"
)


def emit_table(records: List[Dict[str, Any]]) -> str:
    """Write the measured table as the Python module the runtime classifies through.

    The panel and the models read this table on every frame, so it is baked into a module instead
    of parsed from JSON at import time; `--emit-table` after a probe run is what keeps the two in
    step, and `tests/test_tile_behavior.py` gates the module against the artifact.
    """
    lines = [
        '"""tile_behavior_table.py - the measured Map16 behaviour table.',
        "",
        "Generated by `python -m scripts.probe_tile_behaviors --emit-table`; do not edit by hand.",
        "Every standard tile id (0x000-0x1FF) with the behaviour class the engine itself showed",
        "when the probe wrote that id into the level buffer: see",
        "`results/tile_behavior_probe_metrics.json` for the raw measurements and the protocol.",
        '"""',
        "",
        "BEHAVIOR = {",
    ]
    for record in sorted(records, key=lambda r: int(r["tile"], 16)):
        lines.append(f'    {record["tile"]}: "{record["class"]}",')
    lines.append("}")
    lines.append("")
    path = os.path.normpath(TABLE_PATH)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Measure every Map16 tile's engine behaviour.")
    parser.add_argument(
        "--ids", type=parse_ids, default=None, help="comma-separated tile ids; default: all 512"
    )
    parser.add_argument("--output", default=results_file("tile_behavior_probe_metrics.json"))
    parser.add_argument(
        "--emit-table",
        action="store_true",
        help="also rewrite src/environment/tile_behavior_table.py from the measurements",
    )
    args = parser.parse_args()

    with open(STATE_YOSHI_ISLAND_1_LEVEL, "rb") as fh:
        state_bytes = fh.read()
    ids: List[int] = args.ids if args.ids else list(range(0x200))

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)

    # the probe only trusts itself when its anchors behave the way live play shows they do
    anchors = {}
    for tile_id in (ANCHOR_AIR, ANCHOR_TOP, ANCHOR_WALL):
        drop = drop_probe(emu, state_bytes, tile_id)
        wall = wall_probe(emu, state_bytes, tile_id)
        anchors[f"{tile_id:#06x}"] = classify(drop, wall)
    ok = (
        anchors[f"{ANCHOR_AIR:#06x}"] == "decor"
        and anchors[f"{ANCHOR_TOP:#06x}"] == "platform"
        and anchors[f"{ANCHOR_WALL:#06x}"] == "solid"
    )
    if not ok:
        emu.close()
        raise RuntimeError(f"the probe's anchors disagreed with live play: {anchors}")
    logger.info(f"anchors hold: {anchors}")

    records: List[Dict[str, Any]] = []
    for tile_id in ids:
        drop = drop_probe(emu, state_bytes, tile_id)
        wall = wall_probe(emu, state_bytes, tile_id)
        records.append(
            {
                "tile": f"{tile_id:#06x}",
                "drop": drop,
                "wall": wall,
                "class": classify(drop, wall),
            }
        )
    emu.close()

    counts: Dict[str, int] = {}
    for record in records:
        counts[record["class"]] = counts.get(record["class"], 0) + 1
    payload = {
        "study": (
            "What the engine itself does with every standard Map16 tile id, measured by writing "
            "each id into the level buffer at a controlled rig position and reading the engine's "
            "own collision, death, coin and freeze responses."
        ),
        "protocol": {
            "savestate": os.path.basename(STATE_YOSHI_ISLAND_1_LEVEL),
            "rig_column": RIG_COLUMN,
            "drop_zone_rows": list(DROP_ZONE_ROWS),
            "wall_zone_rows": list(WALL_ZONE_ROWS),
            "wall_floor_row": WALL_FLOOR_ROW,
            "drop_frames": DROP_FRAMES,
            "settle_frames": SETTLE_FRAMES,
            "wall_frames": WALL_FRAMES,
            "floor_tile": f"{FLOOR_TILE:#06x}",
            "anchors": anchors,
            "ids_probed": len(ids),
            "class_rule": (
                "harmful: contact killed a small Mario; interact: the engine froze the player "
                "without killing him; coin: contact paid the coin counter or consumed the cell; "
                "slope: held up with the rest inside the cell's vertical span; platform: held up "
                "from above without blocking the side; solid: the side flag fired or the run "
                "stopped short; decor: the engine ignored the tile"
            ),
        },
        "tiles": records,
        "class_counts": counts,
    }
    write_metrics(args.output, payload, command="python -m scripts.probe_tile_behaviors")
    if args.emit_table:
        logger.info(f"tile behavior table: {emit_table(records)}")
    logger.info(f"tile behavior probe: {counts}")
    return payload


if __name__ == "__main__":
    main()
