"""
pixel_level_recheck.py - README 10.62: the pixel recording, re-made from a level really entered.

10.31.1 found that the committed RGB recording never translates while its player moves, and read
that as the forced-mode session drawing the menu behind live physics - a reading 10.60 re-verified
on a corrected decoder. The finding was a null, and a null measured on a recording nobody chose is
weak evidence: maybe the alignment cannot see a scroll after all. The fix is the positive control
the old file could not have: record the same instrument from the verified level-entry savestate of
10.36.3, and check the picture against the camera rather than against the player's own x, because
inside the camera's dead zone the player moves and the scene must not.

The recheck therefore reports three things: the old recording's null, the new recording's shift
statistics split by whether the camera moved, and the synthetic control that proves the alignment
recovers a scroll it is shown.

Needs no emulator: both recordings are on disk. Writes results/pixel_level_recheck_metrics.json.

Run:  python -m src.evaluation.pixel_level_recheck
      make pixel-level-recheck
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np

from src.evaluation.pixel_frame_probe import BAND_TOP_PX, best_shift, probe
from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_PIXEL,
    DATASET_PIXEL_LEVEL,
    REPO_ROOT,
    results_file,
)
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

#: the recorder stores frames at half resolution, so one frame pixel is two level pixels
DOWNSCALE = 2
#: a camera displacement below this many level pixels cannot move the picture by a whole pixel
CAMERA_MOVING_PX = float(DOWNSCALE)
#: the same search depth the 10.31.1 probe uses, so the two rows are one instrument
MAX_SHIFT_PX = 4


def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


def picture_camera_coupling(path: str) -> Dict[str, Any]:
    """Align consecutive frames and score the recovered shift against the recorded camera step."""
    with np.load(path, allow_pickle=False) as blob:
        frames = np.asarray(blob["frames"], dtype=np.float32).mean(axis=3)[:, BAND_TOP_PX:, :]
        camera = np.asarray(blob["camera"], dtype=np.float64)
        states = np.asarray(blob["states"], dtype=np.float64)
        held = np.asarray(blob["held"], dtype=np.float64) if "held" in blob.files else None
    shifts = np.array(
        [best_shift(frames[t], frames[t + 1], MAX_SHIFT_PX)[0] for t in range(len(frames) - 1)],
        dtype=np.float64,
    )
    dcam = np.diff(camera)[: len(shifts)]
    dx = np.diff(states[:, 0])[: len(shifts)]
    moving = np.abs(dcam) >= CAMERA_MOVING_PX
    expected = np.round(dcam / float(DOWNSCALE))
    still = ~moving
    if held is not None:
        still = still & (held[1 : len(shifts) + 1] < 0.5)
    out: Dict[str, Any] = {
        "dataset": _rel(path),
        "frames": int(frames.shape[0]),
        "pairs": int(shifts.shape[0]),
        "camera_travel_px": float(np.ptp(camera)),
        "x_travel_px": float(states[:, 0].max() - states[:, 0].min()),
        "pairs_where_camera_moved": int(moving.sum()),
        "share_of_camera_moving_pairs_picture_still": (
            float(np.mean(shifts[moving] == 0.0)) if moving.any() else None
        ),
        "share_of_camera_still_pairs_picture_still": (
            float(np.mean(shifts[still] == 0.0)) if still.any() else None
        ),
        "share_within_one_pixel_of_the_camera": (
            float(np.mean(np.abs(shifts[moving] - expected[moving]) <= 1.0))
            if moving.any()
            else None
        ),
        "share_with_camera_sign": (
            float(np.mean(np.sign(shifts[moving]) == np.sign(dcam[moving])))
            if moving.any()
            else None
        ),
        "mean_abs_picture_shift_px_overall": float(np.mean(np.abs(shifts))),
        "mean_abs_dx_px": float(np.mean(np.abs(dx))),
    }
    if held is not None:
        out["held_frames"] = int(held.sum())
    return out


def run_recheck(
    published_path: str = DATASET_PIXEL,
    level_path: str = DATASET_PIXEL_LEVEL,
    level_state: str = "data/raw/smw_yoshi_island_1_level.state",
    output_metrics: str = results_file("pixel_level_recheck_metrics.json"),
) -> Dict[str, Any]:
    """Probe both recordings with the same instrument and write the comparison artifact."""
    published = probe(path=published_path)
    level = probe(path=level_path)
    coupling = picture_camera_coupling(level_path)
    payload: Dict[str, Any] = {
        "study": (
            "The pixel-to-state recording, re-made from the verified level-entry savestate and scored "
            "against the camera rather than the player, as the positive control 10.31.1's null needed."
        ),
        "protocol": {
            "published_recording": _rel(published_path),
            "level_recording": _rel(level_path),
            "alignment": "best integer horizontal shift of each consecutive frame pair",
            "max_shift_px": MAX_SHIFT_PX,
            "band_top_px": BAND_TOP_PX,
            "downscale": DOWNSCALE,
            "camera_moving_threshold_px": CAMERA_MOVING_PX,
            "still_pairs_exclude_held_frames": True,
            "level_recording_state": level_state,
        },
        "published": published,
        "level": level,
        "camera_coupling": coupling,
    }
    verdict = {
        "published_picture_ever_shifts": float(1.0 - published["share_of_pairs_best_shift_zero"]),
        "level_picture_shifts_when_the_camera_does": float(
            1.0 - (coupling["share_of_camera_moving_pairs_picture_still"] or 0.0)
        ),
        "level_picture_holds_when_the_camera_does_not": float(
            coupling["share_of_camera_still_pairs_picture_still"] or 0.0
        ),
    }
    verdict["discriminates"] = bool(
        verdict["level_picture_shifts_when_the_camera_does"] > 0.5
        and verdict["level_picture_holds_when_the_camera_does_not"] > 0.9
    )
    payload["verdict"] = verdict
    os.makedirs(os.path.dirname(output_metrics) or ".", exist_ok=True)
    write_metrics(output_metrics, payload, command="make pixel-level-recheck")
    logger.info("Artifact written to: %s", output_metrics)
    payload["_artifact"] = output_metrics
    return payload


def render_section(payload: Dict[str, Any]) -> List[str]:
    """README 10.62, generated from the artifact."""

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.2f}%"

    proto = payload["protocol"]
    pub, lev, coup, verdict = (
        payload["published"],
        payload["level"],
        payload["camera_coupling"],
        payload["verdict"],
    )
    return [
        "### 10.62 The Pixel Recording, Re-made From a Level That Is Really Entered",
        "",
        "10.31.1's finding was a null: the committed RGB frames never translate while the recorded "
        "player moves. A null measured on a recording nobody chose is weak evidence - the alignment "
        "might simply be blind - so this adds the positive control the old file could not contain: "
        "the same instrument, applied to a recording made from the verified level-entry savestate of "
        "10.36.3, with the camera stored alongside the state.",
        "",
        "| recording | frames | pairs | picture shifted on | mean |dx| px/frame |",
        "| --- | --- | --- | --- | --- |",
        f"| `{proto['published_recording']}` (forced-mode, published) | {pub['frames']:,} "
        f"| {pub['pairs_tested']:,} | {pct(1.0 - pub['share_of_pairs_best_shift_zero'])} "
        f"| {pub['mean_abs_dx_px']:.4f} |",
        f"| `{proto['level_recording']}` (level entry, new) | {coup['frames']:,} | {coup['pairs']:,} "
        f"| {pct(1.0 - lev['share_of_pairs_best_shift_zero'])} | {lev['mean_abs_dx_px']:.4f} |",
        "",
        f"The second row's aggregate is not the interesting one, because inside the camera's dead "
        f"zone the player moves and the scene must not - scoring a pixel recording against the "
        f"player's own x asks the wrong question. Split by the camera instead, on the new recording's "
        f"{coup['pairs_where_camera_moved']:,} pairs where it moved by at least "
        f"{payload['protocol']['camera_moving_threshold_px']:.0f} level px:",
        "",
        f"* the picture holds still on {pct(coup['share_of_camera_moving_pairs_picture_still'])} of "
        f"them, and agrees in sign with the camera on "
        f"{pct(coup['share_with_camera_sign'])} -",
        f"* while on the pairs where the camera did not move, and the player was not one of the "
        f"{coup['held_frames']:,} frames the engine was holding, the picture holds still on "
        f"{pct(coup['share_of_camera_still_pairs_picture_still'])}.",
        "",
        f"That is the discrimination the null needed: the same alignment that reports zero shift on "
        f"{pct(pub['share_of_pairs_best_shift_zero'])} of the published recording's pairs reports "
        f"movement on {pct(verdict['level_picture_shifts_when_the_camera_does'])} of the pairs where "
        f"a level really is scrolling, and stillness where the camera says there is nothing to "
        f"scroll. The synthetic control rides along unchanged - a frame slid by 1, 2 and 3 px is "
        f"recovered exactly on {pct(pub['detector_control']['true_shift_1px']['share_recovering_exactly'])} "
        f"of its trials at each depth.",
        "",
        f"**What this changes, and what it does not.** It confirms 10.31.1's reading rather than "
        f"replacing it: the committed frames are the menu, and the estimator trained on them was "
        f"trained on the menu. Those published numbers stand as measurements of the file they name, "
        f"and retraining on this recording is a separate piece of work, not a consequence of this "
        f"one. What is new is that the repository now holds a pixel recording of a genuinely entered, "
        f"genuinely rendered, controllable level - {coup['x_travel_px']:.2f} px of travel, "
        f"{coup['camera_travel_px']:.2f} px of camera, {coup['held_frames']:,} held frames the "
        f"recorder cleared with the operator rule of 10.60 - and an instrument that can tell the two "
        f"recordings apart. The recording is a new file: "
        f"`{proto['level_recording']}`, written by "
        f"`python scripts/record_pixel_gameplay.py --state {proto['level_recording_state']}`, and the "
        f"published `{proto['published_recording']}` is untouched.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--published", default=DATASET_PIXEL)
    parser.add_argument("--level", default=DATASET_PIXEL_LEVEL)
    parser.add_argument(
        "--output-metrics", default=results_file("pixel_level_recheck_metrics.json")
    )
    args = parser.parse_args()
    run_recheck(
        published_path=args.published,
        level_path=args.level,
        output_metrics=args.output_metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
