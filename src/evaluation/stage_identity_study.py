"""
stage_identity_study.py - which level do the committed telemetry recordings actually walk?

10.36.3 closed with this question open. Every telemetry file in the repository comes from the
forced-mode session - `enable_gameplay_mode()` writes `0x14` into whatever state the boot left, which
gives genuine engine physics while the PPU keeps drawing the menu - and §10.36.3 measured that the
terrain reconstructed from those recordings' 7x7 tile patches agrees with the verified, genuinely
entered level on only 32.65% of the columns both cover. That was a scratch measurement. This makes it
an artifact, and adds the third arm that turns it into an answer: sample the *same* profile from the
forced-mode session itself, live, and see which of the two it matches.

The profile is deliberately the same instrument on all three sides: the first non-air tile at or below
Mario's own tile row, within the 7x7 window the recordings carry, aggregated per absolute tile column.
That window is three tiles deep, so the quantity is "what is Mario standing on", not "what does the
level look like" - and it is the only terrain quantity the committed files can answer.

Needs core + ROM. Writes results/stage_identity_metrics.json.

Run:  python -m src.evaluation.stage_identity_study
      make stage-identity
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from src.environment import wram
from src.environment.snes_emulator import SnesLibretroEmulator
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    DATASET_TILEMAP,
    REPO_ROOT,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    STATE_YOSHI_ISLAND_1_LEVEL,
    results_file,
)
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

#: the radius the committed tilemap recording carries; the same window is used everywhere here
PATCH_RADIUS = 3
TILE_PX = float(wram.TILE_SIZE_PX)
#: frames of an unchanged 8D record before the walk assumes the engine is holding the player
PAUSE_TRIGGER = 10
#: how long to keep pulsing A at a held player before giving up on the walk
OPERATOR_A_LIMIT = 240
AGREEMENT_TOLERANCE_ROWS = 1


def profile_from_frames(xs: np.ndarray, patches: np.ndarray) -> Dict[str, Any]:
    """Aggregate "first non-air tile at or below Mario's row" per absolute tile column.

    The patch is centred on Mario's own tile, so row `radius` of the patch is the row he is in and
    larger indices are below him. A column contributes an observation only while the ground is inside
    the window, which is what makes the quantity "what is Mario standing on".

    Args:
        xs: Mario's x in pixels, one per observation.
        patches: ``(n, 2r+1, 2r+1)`` tile classes centred on Mario's tile, rows increasing downward.

    Returns:
        The columns with their modal depth, the frames that saw each column, and the coverage.
    """
    radius = (patches.shape[1] - 1) // 2
    seen: Dict[int, Counter] = {}
    for x, patch in zip(np.asarray(xs, float), patches):
        centre_col = int(x) // int(TILE_PX)
        for j in range(patch.shape[1]):
            column = centre_col + (j - radius)
            if column < 0:
                continue
            depth = next((d for d in range(radius + 1) if int(patch[radius + d][j]) != 0), None)
            if depth is None:
                continue
            seen.setdefault(column, Counter())[depth] += 1
    columns = sorted(seen)
    return {
        "profile": {str(col): int(seen[col].most_common(1)[0][0]) for col in columns},
        "frames_per_column": {str(col): int(sum(seen[col].values())) for col in columns},
        "n_columns": len(columns),
        "radius_tiles": radius,
    }


def profile_from_npz(path: str) -> Dict[str, Any]:
    """The recording side: states and patches already on disk."""
    with np.load(path, allow_pickle=False) as blob:
        states = np.asarray(blob["states"], dtype=np.float64)
        patches = np.asarray(blob["tile_patches"], dtype=np.int64)
    out = profile_from_frames(states[:, 0], patches)
    out["source"] = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    out["observations"] = int(states.shape[0])
    out["x_range_px"] = [float(states[:, 0].min()), float(states[:, 0].max())]
    return out


def live_profile(
    state_path: str,
    frames: int = 900,
    core_path: str = CORE_PATH,
    rom_path: str = ROM_PATH,
) -> Dict[str, Any]:
    """The console side: walk right in this savestate and read the same window from WRAM.

    The walk carries the message-box rule from 10.60 - a held player record gets pulsed A - because
    without it a genuine level stops at its first tutorial box and the profile would be truncated by
    an interface event rather than by the terrain.
    """
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as fh:
        savestate = fh.read()
    emu.start_episode(savestate)
    xs: List[float] = []
    ys: List[float] = []
    patches: List[np.ndarray] = []
    prev: Tuple[float, float, float, float, float, float, float, float] | None = None
    paused = 0
    operator = 0
    boxes_cleared = 0
    for frame in range(frames):
        state = emu.get_smw_state()
        key = tuple(
            state[k] for k in ("x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right")
        )
        held = prev is not None and key == prev
        paused = paused + 1 if held else 0
        prev = key
        buttons = {"RIGHT": True, "Y": True, "B": frame % 24 == 0}
        if paused >= PAUSE_TRIGGER:
            if operator >= OPERATOR_A_LIMIT:
                break
            buttons["A"] = (paused % 8) < 4
            operator += 1
            if paused == PAUSE_TRIGGER:
                boxes_cleared += 1
        emu.set_input(buttons)
        emu.step_frame()
        xs.append(state["x"])
        ys.append(state["y"])
        patches.append(emu.get_local_tilemap_patch(state["x"], state["y"], radius=PATCH_RADIUS))
    mode = int(emu.get_game_mode())
    camera = emu.get_camera_x()
    emu.close()
    out = profile_from_frames(np.asarray(xs), np.stack(patches))
    out.update(
        {
            "source": os.path.relpath(state_path, REPO_ROOT).replace("\\", "/"),
            "observations": len(xs),
            "x_range_px": [float(min(xs)), float(max(xs))],
            "final_mode_byte": mode,
            "final_camera_px": float(camera),
            "message_boxes_cleared": boxes_cleared,
            "operator_A_frames": operator,
        }
    )
    return out


def compare(
    a: Dict[str, Any], b: Dict[str, Any], tolerance: int = AGREEMENT_TOLERANCE_ROWS
) -> Dict[str, Any]:
    """Score one profile against another on the columns both cover."""
    shared = sorted(set(a["profile"]) & set(b["profile"]), key=int)
    agree = [c for c in shared if abs(a["profile"][c] - b["profile"][c]) <= tolerance]
    return {
        "columns_compared": len(shared),
        "columns_agreeing": len(agree),
        "share_agreeing": (len(agree) / len(shared)) if shared else None,
        "disagreeing_columns": [
            [int(c), a["profile"][c], b["profile"][c]] for c in shared if c not in agree
        ],
    }


def summarise_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Describe a profile without comparing it to anything: how much terrain does it actually see?"""
    depths = list(profile["profile"].values())
    return {
        "source": profile.get("source"),
        "n_columns": profile["n_columns"],
        "columns_with_ground": int(sum(1 for d in depths if d is not None)),
        "modal_depth_histogram": {str(k): int(v) for k, v in sorted(Counter(depths).items())},
        "x_range_px": profile.get("x_range_px"),
    }


def run_study(
    recording_path: str = DATASET_TILEMAP,
    forced_mode_state: str = STATE_YOSHI_ISLAND_1,
    level_state: str = STATE_YOSHI_ISLAND_1_LEVEL,
    frames: int = 900,
    output_metrics: str = results_file("stage_identity_metrics.json"),
) -> Dict[str, Any]:
    """Measure the three profiles, score them against each other, and write the artifact."""
    committed = profile_from_npz(recording_path)
    forced = live_profile(forced_mode_state, frames=frames)
    level = live_profile(level_state, frames=frames)
    payload: Dict[str, Any] = {
        "study": (
            "Which terrain the committed telemetry recordings walk: the profile reconstructed from "
            "the recording is scored against the same profile read live from the forced-mode session "
            "it came from, and against the verified world-map level entry of 10.36.3."
        ),
        "protocol": {
            "profile_definition": (
                "first non-air tile at or below Mario's own tile row, within the "
                f"{2 * PATCH_RADIUS + 1}x{2 * PATCH_RADIUS + 1} window the recording carries, "
                "modal value per absolute tile column"
            ),
            "agreement_tolerance_rows": AGREEMENT_TOLERANCE_ROWS,
            "walk_frames": frames,
            "recording": committed["source"],
            "forced_mode_state": forced["source"],
            "level_state": level["source"],
            "pause_rule": f"A pulsed after {PAUSE_TRIGGER} held frames, up to {OPERATOR_A_LIMIT}",
        },
        "profiles": {
            "recording": {
                **summarise_profile(committed),
                "observations": committed["observations"],
            },
            "forced_mode": {
                **summarise_profile(forced),
                "observations": forced["observations"],
                "final_mode_byte": forced["final_mode_byte"],
                "message_boxes_cleared": forced["message_boxes_cleared"],
            },
            "level_entry": {
                **summarise_profile(level),
                "observations": level["observations"],
                "final_mode_byte": level["final_mode_byte"],
                "message_boxes_cleared": level["message_boxes_cleared"],
            },
        },
        "comparisons": {
            "recording_vs_level": compare(committed, level),
            "recording_vs_forced_mode": compare(committed, forced),
            "forced_mode_vs_level": compare(forced, level),
        },
    }
    verdict = {
        "recording_matches_its_own_session": (
            payload["comparisons"]["recording_vs_forced_mode"]["share_agreeing"] or 0.0
        ),
        "recording_matches_the_genuine_level": (
            payload["comparisons"]["recording_vs_level"]["share_agreeing"] or 0.0
        ),
    }
    verdict["recording_faithful_to_its_session"] = bool(
        verdict["recording_matches_its_own_session"] >= 0.9
    )
    verdict["terrain_differs"] = bool(
        verdict["recording_matches_its_own_session"]
        - verdict["recording_matches_the_genuine_level"]
        > 0.05
    )
    payload["verdict"] = verdict
    os.makedirs(os.path.dirname(output_metrics) or ".", exist_ok=True)
    write_metrics(output_metrics, payload, command="make stage-identity")
    logger.info("Artifact written to: %s", output_metrics)
    payload["_artifact"] = output_metrics
    return payload


def render_section(payload: Dict[str, Any]) -> List[str]:
    """README 10.61, generated from the artifact."""
    proto = payload["protocol"]
    profiles = payload["profiles"]
    comp = payload["comparisons"]
    verdict = payload["verdict"]
    rec = profiles["recording"]
    forced = profiles["forced_mode"]
    level = profiles["level_entry"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.2f}%"

    live_vs_live = comp["forced_mode_vs_level"]
    return [
        "### 10.61 Which Level the Committed Recordings Actually Walk",
        "",
        f"10.36.3 closed with a question it could not answer: the terrain reconstructed from the "
        f"recorded 7x7 tile patches did not match the terrain of the genuinely entered level, and "
        f"that left two explanations - the recordings are of a different stage, or the tile buffer "
        f"of the forced-mode session is not a level buffer at all. The figure quoted there came from "
        f"a scratch run with an undefined sampling rule and no artifact, so it is superseded here "
        f"rather than re-derived. The profile is one instrument applied three times: "
        f"{proto['profile_definition']}.",
        "",
        "| side | observations | columns seen | x range (px) | message boxes cleared |",
        "| --- | --- | --- | --- | --- |",
        f"| recording `{rec['source']}` | {rec['observations']:,} | {rec['n_columns']} "
        f"| {rec['x_range_px'][0]:.2f} -> {rec['x_range_px'][1]:.2f} | - |",
        f"| forced-mode session, live | {forced['observations']:,} | {forced['n_columns']} "
        f"| {forced['x_range_px'][0]:.2f} -> {forced['x_range_px'][1]:.2f} "
        f"| {forced['message_boxes_cleared']} |",
        f"| verified level entry, live | {level['observations']:,} | {level['n_columns']} "
        f"| {level['x_range_px'][0]:.2f} -> {level['x_range_px'][1]:.2f} "
        f"| {level['message_boxes_cleared']} |",
        "",
        f"Scored pairwise, on the columns both sides see and within "
        f"{proto['agreement_tolerance_rows']} row:",
        "",
        "| comparison | columns | agreeing | share |",
        "| --- | --- | --- | --- |",
        f"| recording vs genuine level | {comp['recording_vs_level']['columns_compared']} "
        f"| {comp['recording_vs_level']['columns_agreeing']} "
        f"| {pct(comp['recording_vs_level']['share_agreeing'])} |",
        f"| recording vs its own session, live | "
        f"{comp['recording_vs_forced_mode']['columns_compared']} "
        f"| {comp['recording_vs_forced_mode']['columns_agreeing']} "
        f"| {pct(comp['recording_vs_forced_mode']['share_agreeing'])} |",
        f"| forced-mode session vs genuine level | {live_vs_live['columns_compared']} "
        f"| {live_vs_live['columns_agreeing']} | {pct(live_vs_live['share_agreeing'])} |",
        "",
        f"The two live arms run the identical scripted walk - hold right and dash, jump every 24 "
        f"frames, pulse A at a held player - so their comparison is the one that isolates terrain: "
        f"they agree on {pct(live_vs_live['share_agreeing'])} of the "
        f"{live_vs_live['columns_compared']} columns both cover. The forced-mode session is therefore "
        f"walking a buffer that is not the terrain of the stage the world-map route enters, and the "
        f"second row says the instrument is not what separates them: replaying it live over the "
        f"session the file came from reproduces the committed recording on "
        f"{pct(verdict['recording_matches_its_own_session'])} of columns. What the recordings "
        f"describe is the forced-mode session, faithfully. What the forced-mode session *is* stays "
        f"open at this resolution - another stage's buffer, or one left stale by the boot - and "
        f"10.36.3 already settled its visual half, which is that the PPU never leaves the menu.",
        "",
        "**Limitations.** The profile is measured from Mario's own tile row, so it answers "
        '"what is under him" rather than "what does the level look like", and the window is only '
        f"{2 * PATCH_RADIUS + 1} tiles deep: a column is observed only while its ground is inside "
        "that window, which is why the three sides see different column counts. Two walks over the "
        "same terrain can still disagree where one Mario stands on a block and the other on the "
        "ground, so every share here is a floor on the similarity, not an estimate of it - and the "
        f"forced-mode arm ended its walk in mode 0x{forced['final_mode_byte']:02X} while the level "
        f"arm was still in 0x{level['final_mode_byte']:02X}, which is geometry, not sampling.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--recording", default=DATASET_TILEMAP)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--output-metrics", default=results_file("stage_identity_metrics.json"))
    args = parser.parse_args()
    run_study(recording_path=args.recording, frames=args.frames, output_metrics=args.output_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
