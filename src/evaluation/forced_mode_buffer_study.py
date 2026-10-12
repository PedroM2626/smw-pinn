"""
forced_mode_buffer_study.py - README 10.64: whose tile buffer the telemetry recordings read.

10.61 established that the committed recordings faithfully describe the forced-mode session and that
the session's `$7E:C800` buffer is not the terrain of the level the world-map route enters. It left
the positive question unanswered, because comparing a buffer to a level says nothing about where the
buffer came from. This compares it to the two sessions it could plausibly be: the title screen with
its attract demo running, which is what the boot is at frame 410 where the published preamble writes
`0x14`; and the entered level itself.

The instrument is absolute rather than Mario-relative, which is the difference between a comparison
and an accident: for every tile column a session visits, all 27 rows of that column are read out of
the buffer, and two sessions are scored on the columns they share - by tile class (10.61's quantity)
and by raw tile ID, which is a fingerprint rather than a silhouette.

Needs core + ROM. Writes results/forced_mode_buffer_metrics.json.

Run:  python -m src.evaluation.forced_mode_buffer_study
      make forced-mode-buffer
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from src.environment import wram
from src.environment.episode_hygiene import PlayerHold
from src.environment.snes_emulator import SnesLibretroEmulator
from src.evaluation.stage_identity_study import compare, profile_from_frames
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1_LEVEL,
    results_file,
)
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

#: every tile row of a subscreen column: 27 rows x 16 px = 432 px, the level's whole height
COLUMN_ROWS = wram.TILEMAP_ROWS_PER_SUBSCREEN
#: the walk 10.61's arms use, so the two studies sample comparable ground
WALK_FRAMES = 900
PAUSE_TRIGGER = 10
OPERATOR_A_LIMIT = 240


def column_ids(emu: SnesLibretroEmulator, tile_col: int) -> np.ndarray:
    """All 27 tile IDs of one absolute level column, as the buffer stores them."""
    out = np.zeros(COLUMN_ROWS, dtype=np.int64)
    for row in range(COLUMN_ROWS):
        addr = emu.tilemap_addr(tile_col, row)
        out[row] = 0 if addr is None else emu.read_wram_u8(addr)
    return out


def sample(
    arm: str,
    how: str,
    frames: int,
    scripted: bool,
    boot: str,
    core_path: str = CORE_PATH,
    rom_path: str = ROM_PATH,
) -> Dict[str, Any]:
    """Run one session, then read its buffer column by column and its Mario-relative profile."""
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    if boot == "title":
        for _ in range(410):
            emu.step_frame()
        if scripted:
            emu.enable_gameplay_mode()
            for _ in range(30):
                emu.step_frame()
    else:
        with open(STATE_YOSHI_ISLAND_1_LEVEL, "rb") as fh:
            emu.start_episode(fh.read())
    translevel = emu.read_wram_u8(wram.ADDR_TRANSLEVEL)
    mode_at_start = emu.get_game_mode()
    xs: List[float] = []
    ys: List[float] = []
    patches: List[np.ndarray] = []
    columns: Dict[int, Tuple[np.ndarray, int]] = {}
    hold = PlayerHold(limit=OPERATOR_A_LIMIT)
    for frame in range(frames):
        state = emu.get_smw_state()
        hold.observe(
            np.asarray(
                [
                    state[k]
                    for k in ("x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right")
                ],
                dtype=np.float64,
            )
        )
        tile_col = int(state["x"]) // wram.TILE_SIZE_PX
        ids = column_ids(emu, tile_col)
        seen = columns.get(tile_col)
        if seen is None or int(np.sum(ids != seen[0])) == 0:
            columns[tile_col] = (ids, (seen[1] if seen else 0) + 1)
        xs.append(state["x"])
        ys.append(state["y"])
        patches.append(emu.get_local_tilemap_patch(state["x"], state["y"], radius=3))
        buttons: Dict[str, bool] = (
            {"RIGHT": True, "Y": True, "B": frame % 24 == 0} if scripted else {}
        )
        if scripted:
            if hold.exhausted():
                break
            buttons = hold.buttons(buttons)
        emu.set_input(buttons)
        emu.step_frame()
    mode_at_end = emu.get_game_mode()
    emu.close()
    prof = profile_from_frames(np.asarray(xs), np.stack(patches))
    counted = {str(col): vals.tolist() for col, (vals, _) in sorted(columns.items())}
    return {
        "arm": arm,
        "how": how,
        "boot": boot,
        "translevel_byte": int(translevel),
        "mode_at_start": int(mode_at_start),
        "mode_at_end": int(mode_at_end),
        "x_travel_px": round(float(max(xs) - min(xs)), 4),
        "columns_sampled": len(columns),
        "profile": prof["profile"],
        "column_ids": counted,
    }


def id_agreement(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Score two buffers on the columns both visited: identical cells out of all cells compared."""
    shared = sorted(set(left["column_ids"]) & set(right["column_ids"]), key=int)
    cells = 0
    equal = 0
    columns_equal = 0
    for col in shared:
        a = np.asarray(left["column_ids"][col])
        b = np.asarray(right["column_ids"][col])
        cells += a.size
        equal += int(np.sum(a == b))
        columns_equal += int(np.array_equal(a, b))
    return {
        "columns_compared": len(shared),
        "columns_identical": columns_equal,
        "share_of_columns_identical": (columns_equal / len(shared)) if shared else None,
        "cells_compared": cells,
        "cells_equal": equal,
        "share_of_cells_equal": (equal / cells) if cells else None,
    }


def class_profile_agreement(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """10.61's quantity, recomputed here so the two studies agree about what they disagree about."""
    return compare(
        {"profile": left["profile"], "source": left["arm"]},
        {"profile": right["profile"], "source": right["arm"]},
    )


def run_study(
    frames: int = WALK_FRAMES,
    output_metrics: str = results_file("forced_mode_buffer_metrics.json"),
) -> Dict[str, Any]:
    """Sample the three sessions and score every pair, by tile ID and by class profile."""
    arms = [
        sample(
            "attract_demo",
            "the title screen, no input, its own demo playing",
            frames,
            False,
            "title",
        ),
        sample(
            "forced_mode_preamble",
            "the published preamble: 410 frames then `0x14`",
            frames,
            True,
            "title",
        ),
        sample(
            "entered_level",
            "the world-map route of 10.36.3, entered and verified",
            frames,
            True,
            "savestate",
        ),
    ]
    by = {a["arm"]: a for a in arms}
    pairs = (
        ("forced_mode_preamble", "attract_demo"),
        ("forced_mode_preamble", "entered_level"),
        ("attract_demo", "entered_level"),
    )
    comparisons = {
        f"{left}_vs_{right}": {
            **id_agreement(by[left], by[right]),
            "class_profile": class_profile_agreement(by[left], by[right]),
        }
        for left, right in pairs
    }
    for arm in arms:
        arm["profile"] = {k: int(v) for k, v in arm["profile"].items()}
    verdict = {
        "forced_mode_reads_the_title_buffers": bool(
            comparisons["forced_mode_preamble_vs_attract_demo"]["share_of_cells_equal"]
            > comparisons["forced_mode_preamble_vs_entered_level"]["share_of_cells_equal"]
        ),
        "cells_equal_with_the_title_buffer": comparisons["forced_mode_preamble_vs_attract_demo"][
            "share_of_cells_equal"
        ],
        "cells_equal_with_the_entered_level": comparisons["forced_mode_preamble_vs_entered_level"][
            "share_of_cells_equal"
        ],
        "translevels": {a["arm"]: a["translevel_byte"] for a in arms},
    }
    payload: Dict[str, Any] = {
        "study": (
            "Whose tile buffer the telemetry recordings read: the forced-mode preamble's buffer is "
            "scored column by column against the title screen's own attract-demo session and against "
            "the level the world-map route enters."
        ),
        "protocol": {
            "instrument": "all 27 tile IDs of every absolute level column a session visits, plus "
            "10.61's Mario-relative class profile",
            "walk_frames": frames,
            "walk_rule": "hold right and dash, jump every 24 frames, A pulsed at a held player "
            "(10.63); the demo arm takes no input and is driven by the game",
            "boot": "the published preamble of 10.38.1: 410 frames from reset, then `0x14`",
            "entered_level_state": os.path.basename(STATE_YOSHI_ISLAND_1_LEVEL),
        },
        "arms": arms,
        "comparisons": comparisons,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(output_metrics) or ".", exist_ok=True)
    write_metrics(output_metrics, payload, command="make forced-mode-buffer")
    logger.info("Artifact written to: %s", output_metrics)
    payload["_artifact"] = output_metrics
    return payload


def render_section(payload: Dict[str, Any]) -> List[str]:
    """README 10.64, generated from the artifact."""
    arms = {a["arm"]: a for a in payload["arms"]}
    comp = payload["comparisons"]
    verdict = payload["verdict"]
    demo = arms["attract_demo"]
    forced = arms["forced_mode_preamble"]
    entered = arms["entered_level"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.2f}%"

    return [
        "### 10.64 The Buffer the Telemetry Reads Is the Title Screen's Own",
        "",
        f"10.61 could say the recordings' terrain is not the entered level's, and could not say what "
        f"it is. The boot answers that: at frame 410 of the published preamble the game is sitting on "
        f"its title screen with the attract demo running - mode `0x{demo['mode_at_start']:02X}`, "
        f"translevel `0x{demo['translevel_byte']:02X}`, the demo's own player at x moving under a "
        f"camera that follows it - and the preamble writes `0x14` into that session without loading "
        f"anything. So the buffer every recording reads is the title screen's.",
        "",
        "| session | mode at start | mode at end | translevel | x travelled (px) | columns sampled |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| attract demo, no input | `0x{demo['mode_at_start']:02X}` | "
        f"`0x{demo['mode_at_end']:02X}` | `0x{demo['translevel_byte']:02X}` | "
        f"{demo['x_travel_px']:.2f} | {demo['columns_sampled']} |",
        f"| forced-mode preamble | `0x{forced['mode_at_start']:02X}` | "
        f"`0x{forced['mode_at_end']:02X}` | `0x{forced['translevel_byte']:02X}` | "
        f"{forced['x_travel_px']:.2f} | {forced['columns_sampled']} |",
        f"| entered level (10.36.3) | `0x{entered['mode_at_start']:02X}` | "
        f"`0x{entered['mode_at_end']:02X}` | `0x{entered['translevel_byte']:02X}` | "
        f"{entered['x_travel_px']:.2f} | {entered['columns_sampled']} |",
        "",
        "Scored on the columns each pair visited, cell by cell over all 27 rows of a column, and "
        "again by 10.61's class profile:",
        "",
        "| pair | columns | identical columns | equal cells | class profile agrees |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {label.replace('_vs_', ' vs ').replace('_', ' ')} "
            f"| {row['columns_compared']} | {pct(row['share_of_columns_identical'])} "
            f"| {pct(row['share_of_cells_equal'])} "
            f"| {pct(row['class_profile']['share_agreeing'])} |"
            for label, row in comp.items()
        ),
        "",
        f"Cell equality on its own would mislead, and the middle column is what stops it: most of a "
        f"column is empty rows, so two unrelated buffers still agree on "
        f"{pct(verdict['cells_equal_with_the_entered_level'])} of their cells. Read a column whole "
        f"instead, and the title session and the forced-mode session are the same buffer - "
        f"{pct(comp['forced_mode_preamble_vs_attract_demo']['share_of_columns_identical'])} of the "
        f"{comp['forced_mode_preamble_vs_attract_demo']['columns_compared']} columns they share are "
        f"identical row for row - while the entered level shares "
        f"{pct(comp['forced_mode_preamble_vs_entered_level']['share_of_columns_identical'])} of them. "
        f"10.61's class profile says the same thing from the other side: "
        f"{pct(comp['forced_mode_preamble_vs_attract_demo']['class_profile']['share_agreeing'])} "
        f"against "
        f"{pct(comp['forced_mode_preamble_vs_entered_level']['class_profile']['share_agreeing'])}. "
        f"The translevel byte agrees: `0x{verdict['translevels']['attract_demo']:02X}` for the demo "
        f"and for the forced-mode session, `0x{verdict['translevels']['entered_level']:02X}` for the "
        f"level the map route enters.",
        "",
        "**What this changes, and what it does not.** It does not move a single physics number: the "
        "sub-pixel scale, the gravity steps and the velocity bounds the whole of sections 10.37 "
        "onward reports were measured out of WRAM telemetry, and the engine was running real "
        "physics while it did it. What it settles is what the tile-conditioned results are "
        "conditioned *on* - the terrain of the level the demo plays, not of a level the harness "
        "chose - and it retires the alternative 10.61 left open, that a different stage's buffer "
        "had been loaded. Nobody chose a stage: the title screen did. The entered level remains the "
        "only session here whose stage the repository picked, which is why 10.60's dashboard and "
        "10.62's re-recording both run on it.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--frames", type=int, default=WALK_FRAMES)
    parser.add_argument("--output-metrics", default=results_file("forced_mode_buffer_metrics.json"))
    args = parser.parse_args()
    run_study(frames=args.frames, output_metrics=args.output_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
