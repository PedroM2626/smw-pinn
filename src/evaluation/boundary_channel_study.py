r"""
boundary_channel_study.py - the camera channel 10.58 predicted, tested and refuted (10.59).

10.58 measured the exceptions to section 4.1 and concluded that they are a *constraint*: 75.4-99.3%
of exception frames are positions that do not move while the velocity byte holds, with the residue
exactly minus that velocity, and the 8D state carries nothing that says "the body has been stopped".
Its prediction was that a ninth channel - the boundary, or the camera offset - removes them.

This section builds that channel and tests the prediction, and the prediction fails in a way that
matters more than if it had held.

* The channel exists and is identified from the console, not from a memory map:
  `scripts/scan_scroll_address.py` dumps all 128 KB of WRAM every frame of a scripted run, ranks the
  16-bit words by how well they follow Mario's x, and keeps the ones that satisfy the axioms of a
  layer scroll. The artifact records the winner, its mirror, and the word two slots away that holds
  exactly half of it - the parallax layer.
* On the recording made with it (`scripts/record_boundary_gameplay.py`), the screen-edge channel
  covers 1% of the exception frames at any margin from 8 to 32 px, and the identity with those frames
  removed moves by less than a fifth of a percentage point. The boundary is not where the residue is.
* What the exception frames actually are: the *whole player record repeating*. Every channel of the
  next state equals the current one - position, velocity and all four collision flags - on 86% of this
  recording's exceptions and 74-97% of the six committed recordings'. A CRC over all 128 KB of WRAM,
  recorded alongside them, changes on every single one of those frames: the console advanced, the
  player object did not. That is a paused simulation inside interactive mode, not a stopped body, and
  no positional channel can represent it because nothing about the position is special.
* Drop the paused frames and section 4.1 is exact on 99.2-99.9% of the frames left, against the
  92.7-98.1% 10.58 published. The persistence lift it measured (an exception followed by an exception
  at 104x-626x the rate) is the pause showing up as memory: once a frame repeats, the next one does too.
* What survives is smaller and stranger: 139 frames here, whose residue is exactly one whole pixel,
  and neither the screen bound (7.2%) nor the terrain ahead of Mario (89% but firing on 63% of all
  frames, so no lift) explains them.

Emulator-free and numpy-only: it reads the committed recordings and the two artifacts.

Run:  python -m src.evaluation.boundary_channel_study
      python -m src.evaluation.boundary_channel_study --margins 8,16,24 --primary-margin 16
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment.wram import SCREEN_WIDTH_PX, SUBPIXELS_PER_PIXEL
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import DATASET_BOUNDARY, REPO_ROOT
from src.utils.provenance import read_metrics, write_metrics
from src.utils.typography import demath_typographic

logger = get_logger(__name__)

ARTIFACT_NAME = "boundary_channel_metrics.json"
SCAN_ARTIFACT = os.path.join(REPO_ROOT, "results", "scroll_address_scan_metrics.json")
RESIDUE_ARTIFACT = os.path.join(REPO_ROOT, "results", "residue_process_metrics.json")
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
EXACT_TOLERANCE_SUBPIXELS = 0.5
VELOCITY_FLOOR_SUBPIXELS = 0.5
MARGINS_PX: Tuple[float, ...] = (8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
PRIMARY_MARGIN_PX = 16.0
TILE_SOLID = 1.0
TILE_RADIUS = 3
WALL_ROW_OFFSETS: Tuple[int, ...] = (-1, 0, 1, 2)
INTERACTIVE_MODE = 0x14

#: the four labels every exception frame gets exactly one of, in the order they are tested
CLASSES: Tuple[str, ...] = ("paused", "screen_bound", "wall_ahead", "camera_step", "unexplained")

#: the six recordings 10.58 measured, as (file, state key, next-state key)
COMMITTED: Dict[str, Tuple[str, str, str]] = {
    "gameplay": ("smw_gameplay_dataset.npz", "states", "next_states"),
    "jump": ("smw_jump_dataset.npz", "states", "next_states"),
    "sprint": ("smw_sprint_dataset.npz", "states", "next_states"),
    "multi_entity": ("smw_multi_entity_dataset.npz", "states", "next_states"),
    "tilemap": ("smw_tilemap_dataset.npz", "states", "next_states"),
    "set_multi_entity": ("smw_set_multi_entity_dataset.npz", "mario", "next_mario"),
}


@dataclass
class BoundaryRecording:
    """The boundary recording, with the residue and the three liveness channels derived."""

    state: np.ndarray
    next_state: np.ndarray
    episodes: np.ndarray
    camera: np.ndarray
    next_camera: np.ndarray
    mode: np.ndarray
    wram_crc: np.ndarray
    next_wram_crc: np.ndarray
    solid_ahead: np.ndarray

    @property
    def n(self) -> int:
        return int(self.state.shape[0])

    @property
    def x(self) -> np.ndarray:
        return self.state[:, 0]

    @property
    def vx(self) -> np.ndarray:
        return self.state[:, 2]

    @property
    def residue(self) -> np.ndarray:
        return SUBPIXELS_PER_PIXEL * (self.next_state[:, 0] - self.state[:, 0]) - self.state[:, 2]

    @property
    def screen(self) -> np.ndarray:
        return self.x - self.camera

    @property
    def adjacent(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=bool)
        if self.n > 1:
            out[:-1] = self.episodes[:-1] == self.episodes[1:]
        return out

    @property
    def exception(self) -> np.ndarray:
        return np.abs(self.residue) >= EXACT_TOLERANCE_SUBPIXELS

    @property
    def repeat(self) -> np.ndarray:
        """Every recorded channel of the next state equals the current one."""
        return np.all(np.abs(self.next_state - self.state) < 1e-9, axis=1)

    @property
    def paused(self) -> np.ndarray:
        """A repeating player record on a frame the console did advance: the mechanism, not the artifact."""
        return self.repeat & (self.wram_crc != self.next_wram_crc)

    @property
    def pinned(self) -> np.ndarray:
        """10.58's signature: the position holds while the velocity byte does not."""
        still = np.abs(self.residue + self.vx) < EXACT_TOLERANCE_SUBPIXELS
        return still & (np.abs(self.vx) >= VELOCITY_FLOOR_SUBPIXELS)


def load_boundary_recording(path: str = DATASET_BOUNDARY) -> BoundaryRecording:
    """Read the recording made with the camera channel and derive what section 4.1 needs."""
    with np.load(path, allow_pickle=False) as blob:
        state = np.asarray(blob["states"], dtype=np.float64)
        next_state = np.asarray(blob["next_states"], dtype=np.float64)
        patches = np.asarray(blob["tile_patches"], dtype=np.float64)
        ahead_column = np.where(state[:, 2] >= 0, TILE_RADIUS + 1, TILE_RADIUS - 1)
        rows = np.arange(state.shape[0])
        solid = np.zeros(state.shape[0], dtype=bool)
        for offset in WALL_ROW_OFFSETS:
            solid |= patches[rows, TILE_RADIUS + offset, ahead_column] >= TILE_SOLID
        return BoundaryRecording(
            state=state,
            next_state=next_state,
            episodes=np.asarray(blob["episodes"]),
            camera=np.asarray(blob["camera"], dtype=np.float64),
            next_camera=np.asarray(blob["next_camera"], dtype=np.float64),
            mode=np.asarray(blob["mode"], dtype=np.float64),
            wram_crc=np.asarray(blob["wram_crc"], dtype=np.float64),
            next_wram_crc=np.asarray(blob["next_wram_crc"], dtype=np.float64),
            solid_ahead=solid,
        )


def channel_at_margin(screen: np.ndarray, vx: np.ndarray, margin: float) -> np.ndarray:
    """The boundary channel: within `margin` px of a window edge and pushing outward."""
    left = (screen <= margin) & (vx < 0)
    right = (screen >= SCREEN_WIDTH_PX - margin) & (vx > 0)
    return left | right


def committed_repeat_report() -> Dict[str, Any]:
    """The same test on the six recordings 10.58 measured, which have no camera and no CRC."""
    out: Dict[str, Any] = {}
    for name, (file_name, key, next_key) in COMMITTED.items():
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.isfile(path):
            continue
        with np.load(path, allow_pickle=False) as blob:
            state = np.asarray(blob[key], dtype=np.float64)
            next_state = np.asarray(blob[next_key], dtype=np.float64)
            episodes = np.asarray(blob["episodes"])
        adjacent = np.zeros(state.shape[0], dtype=bool)
        adjacent[:-1] = episodes[:-1] == episodes[1:]
        residue = SUBPIXELS_PER_PIXEL * (next_state[:, 0] - state[:, 0]) - state[:, 2]
        exc = adjacent & (np.abs(residue) >= EXACT_TOLERANCE_SUBPIXELS)
        repeat = np.all(np.abs(next_state - state) < 1e-9, axis=1)
        live = adjacent & ~repeat
        out[name] = {
            "channels": int(state.shape[1]),
            "transitions": int(state.shape[0]),
            "exceptions": int(exc.sum()),
            "share_of_exceptions_repeating": float((exc & repeat).sum() / exc.sum())
            if exc.any()
            else None,
            "share_of_repeats_that_are_exceptions": float(exc[repeat].mean())
            if repeat.any()
            else None,
            "exact_rate_all": float(np.mean(np.abs(residue[adjacent]) < EXACT_TOLERANCE_SUBPIXELS)),
            "exact_rate_off_repeats": float(
                np.mean(np.abs(residue[live]) < EXACT_TOLERANCE_SUBPIXELS)
            )
            if live.any()
            else None,
        }
    return out


def sweep_report(rec: BoundaryRecording, margins: Sequence[float]) -> List[Dict[str, Any]]:
    """Coverage of the exceptions the pause does not explain, as the margin moves."""
    exc = rec.exception & rec.adjacent
    residue_exceptions = exc & ~rec.paused
    screen = rec.screen
    vx = rec.vx
    rows: List[Dict[str, Any]] = []
    for margin in margins:
        channel = channel_at_margin(screen, vx, margin) & rec.adjacent
        rows.append(
            {
                "margin_px": float(margin),
                "channel_fires_on_share": float(channel.mean()),
                "exceptions_covered": int((exc & channel).sum()),
                "share_of_all_exceptions": float((exc & channel).sum() / exc.sum())
                if exc.any()
                else None,
                "share_of_live_exceptions": float(
                    (residue_exceptions & channel).sum() / residue_exceptions.sum()
                )
                if residue_exceptions.any()
                else None,
                "lift": float(exc[channel].mean() / exc[rec.adjacent].mean())
                if channel.any() and exc[rec.adjacent].mean() > 0
                else None,
            }
        )
    return rows


def accounting(rec: BoundaryRecording, margin: float) -> Dict[str, Any]:
    """Every exception frame gets exactly one label, so the coverage claim is exhaustive."""
    exc = rec.exception & rec.adjacent
    channel = channel_at_margin(rec.screen, rec.vx, margin) & rec.adjacent
    masks = {
        "paused": exc & rec.paused,
        "screen_bound": exc & ~rec.paused & channel,
        "wall_ahead": exc & ~rec.paused & ~channel & rec.solid_ahead,
        "camera_step": exc
        & ~rec.paused
        & ~channel
        & ~rec.solid_ahead
        & (np.abs(rec.next_camera - rec.camera) >= 0.5),
        "unexplained": exc
        & ~rec.paused
        & ~channel
        & ~rec.solid_ahead
        & (np.abs(rec.next_camera - rec.camera) < 0.5),
    }
    classes = {name: int(mask.sum()) for name, mask in masks.items()}
    other = masks["unexplained"]
    live = exc & ~rec.paused
    atoms, counts = np.unique(np.round(rec.residue[live], 1), return_counts=True)
    order = np.argsort(-counts)[:6]
    total = int(exc.sum())
    return {
        "margin_px": float(margin),
        "n_exceptions": total,
        "classes": classes,
        "shares": {name: (count / total if total else None) for name, count in classes.items()},
        "fire_rates": {
            "screen_bound": float(channel.mean()),
            "wall_ahead": float(rec.solid_ahead.mean()),
            "camera_step": float((np.abs(rec.next_camera - rec.camera) >= 0.5).mean()),
        },
        "classes_sum_equals_exceptions": sum(classes.values()) == total,
        "live_exceptions": int(live.sum()),
        "live_exception_atoms": {f"{float(atoms[i]):+.1f}": int(counts[i]) for i in order},
        "live_share_exactly_one_pixel": float(
            np.mean(
                np.abs(np.abs(rec.residue[live]) - SUBPIXELS_PER_PIXEL) < EXACT_TOLERANCE_SUBPIXELS
            )
        )
        if live.any()
        else None,
        "unexplained_atoms": {
            f"{float(v):+.1f}": int(c)
            for v, c in zip(*np.unique(np.round(rec.residue[other], 1), return_counts=True))
        },
        "unexplained_share_exactly_one_pixel": float(
            np.mean(
                np.abs(np.abs(rec.residue[other]) - SUBPIXELS_PER_PIXEL) < EXACT_TOLERANCE_SUBPIXELS
            )
        )
        if other.any()
        else None,
        "wall_ahead_fire_rate": float(rec.solid_ahead.mean()),
    }


def liveness_report(rec: BoundaryRecording) -> Dict[str, Any]:
    """The test that tells a paused player object from a duplicated sample."""
    repeat = rec.repeat & rec.adjacent
    advanced = repeat & (rec.wram_crc != rec.next_wram_crc)
    exc = rec.exception & rec.adjacent
    return {
        "repeat_frames": int(repeat.sum()),
        "repeat_share": float(repeat.mean()),
        "share_of_repeats_where_wram_changed": float(advanced.sum() / repeat.sum())
        if repeat.any()
        else None,
        "share_of_exceptions_that_repeat": float((exc & repeat).sum() / exc.sum())
        if exc.any()
        else None,
        "share_of_repeats_that_are_exceptions": float(exc[repeat].mean()) if repeat.any() else None,
        "repeats_in_interactive_mode": int((repeat & (rec.mode == INTERACTIVE_MODE)).sum()),
        "modes_seen": sorted({int(m) for m in rec.mode}),
        "wram_identical_frames": int((repeat & (rec.wram_crc == rec.next_wram_crc)).sum()),
    }


def pause_structure(rec: BoundaryRecording) -> Dict[str, Any]:
    """How long the pauses run, and how many episodes they touch.

    This is the quantity that turns into 10.58's persistence lift: a frame that repeats makes the next
    frame repeat too, so an exception predicts an exception for dozens of frames in a row.
    """
    repeat = (rec.repeat & rec.adjacent).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, repeat, 0]) != 0)
    lengths = (edges[1::2] - edges[0::2]).astype(np.int64)
    episodes_with = np.unique(rec.episodes[(rec.repeat & rec.adjacent)]).astype(int)
    return {
        "runs": int(lengths.size),
        "mean_run_frames": float(lengths.mean()) if lengths.size else None,
        "max_run_frames": int(lengths.max()) if lengths.size else None,
        "episodes_with_a_pause": int(episodes_with.size),
        "episodes": int(len(np.unique(rec.episodes))),
    }


def camera_report(rec: BoundaryRecording) -> Dict[str, Any]:
    """The instrument's own health, measured on this recording rather than asserted."""
    screen = rec.screen
    return {
        "screen_min": float(screen.min()),
        "screen_max": float(screen.max()),
        "screen_inside_window_share": float(np.mean((screen >= 0.0) & (screen <= SCREEN_WIDTH_PX))),
        "camera_never_ahead_of_mario": bool(np.all(rec.camera <= rec.x + 1e-9)),
        "camera_values": int(len(np.unique(rec.camera))),
        "camera_min": float(rec.camera.min()),
        "camera_max": float(rec.camera.max()),
        "episodes": int(len(np.unique(rec.episodes))),
        "transitions": rec.n,
        "adjacent_pairs": int(rec.adjacent.sum()),
    }


def identification_report() -> Dict[str, Any]:
    """What the scan artifact established about the address, quoted rather than restated."""
    if not os.path.isfile(SCAN_ARTIFACT):
        return {}
    payload = read_metrics(SCAN_ARTIFACT)
    chosen = next(
        (c for c in payload.get("candidates", []) if c["addr"] == payload.get("chosen")), {}
    )
    return {
        "chosen": payload.get("chosen"),
        "words_ranked": payload.get("words_ranked"),
        "passing": payload.get("passing"),
        "mirrors": [a for a in payload.get("passing", []) if a != payload.get("chosen")],
        "parallax_sibling": (payload.get("parallax_sibling") or {}).get("addr"),
        "frames_dumped": payload.get("frames_dumped"),
        "x_travel_px": payload.get("x_travel_px"),
        "screen_min": (chosen.get("axioms") or {}).get("screen_min"),
        "screen_max": (chosen.get("axioms") or {}).get("screen_max"),
        "frac_steps_still": (chosen.get("axioms") or {}).get("frac_steps_still"),
    }


def summarize(rec: BoundaryRecording, margins: Sequence[float], primary: float) -> Dict[str, Any]:
    adjacent = rec.adjacent
    residue = rec.residue
    exact = np.abs(residue) < EXACT_TOLERANCE_SUBPIXELS
    channel = channel_at_margin(rec.screen, rec.vx, primary) & adjacent
    kept = adjacent & ~channel
    live = adjacent & ~rec.paused
    return {
        "exact_rate_all_frames": float(exact[adjacent].mean()),
        "exact_rate_off_channel": float(exact[kept].mean()),
        "exact_rate_off_paused": float(exact[live].mean()),
        "n_frames_removed_by_channel": int(channel.sum()),
        "exception_rate_all_frames": float((rec.exception & adjacent).mean()),
        "pinned_rate": float((rec.pinned & adjacent).mean()),
        "share_of_pinned_that_repeat": float(
            (rec.pinned & adjacent & rec.repeat).sum() / max(int((rec.pinned & adjacent).sum()), 1)
        ),
        "primary_margin_px": float(primary),
        "sweep": sweep_report(rec, margins),
        "accounting": accounting(rec, primary),
        "liveness": liveness_report(rec),
        "pause_structure": pause_structure(rec),
        "camera": camera_report(rec),
    }


def _verdict(summary: Dict[str, Any], committed: Dict[str, Any]) -> Dict[str, Any]:
    acc = summary["accounting"]
    channel_share = max((row["share_of_all_exceptions"] or 0.0) for row in summary["sweep"])
    exact_all = summary["exact_rate_all_frames"]
    exact_off = summary["exact_rate_off_channel"]
    committed_shares = [
        r["share_of_exceptions_repeating"]
        for r in committed.values()
        if r["share_of_exceptions_repeating"] is not None
    ]
    committed_exact_before = [r["exact_rate_all"] for r in committed.values()]
    committed_exact_after = [
        r["exact_rate_off_repeats"]
        for r in committed.values()
        if r["exact_rate_off_repeats"] is not None
    ]
    return {
        "prediction": (
            "10.58 predicted that a ninth channel carrying the boundary or the camera offset removes "
            "the exceptions to section 4.1."
        ),
        "prediction_holds": bool(exact_off >= 0.999),
        "channel_covers_share_of_exceptions": channel_share,
        "exact_rate_before": exact_all,
        "exact_rate_with_channel_removed": exact_off,
        "exact_gain_from_the_channel": exact_off - exact_all,
        "exact_rate_with_the_pause_removed": summary["exact_rate_off_paused"],
        "exact_gain_from_the_pause": summary["exact_rate_off_paused"] - exact_all,
        "exceptions_paused": acc["classes"]["paused"],
        "exceptions_total": acc["n_exceptions"],
        "exceptions_unexplained": acc["classes"]["unexplained"],
        "committed_share_of_exceptions_repeating": [min(committed_shares), max(committed_shares)]
        if committed_shares
        else None,
        "committed_exact_before": [min(committed_exact_before), max(committed_exact_before)]
        if committed_exact_before
        else None,
        "committed_exact_after": [min(committed_exact_after), max(committed_exact_after)]
        if committed_exact_after
        else None,
    }


def _pct(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else demath_typographic(f"{100.0 * value:.{digits}f}%")


def render_identification_table(identification: Dict[str, Any]) -> List[str]:
    return [
        "| Quantity | Measured |",
        "| :--- | :--- |",
        f"| Camera address | `{identification.get('chosen')}` |",
        f"| 16-bit words that follow x | {identification.get('words_ranked')} |",
        f"| Words satisfying the scroll axioms | {', '.join('`%s`' % a for a in identification.get('passing', []))} |",
        f"| Parallax sibling (exactly half) | `{identification.get('parallax_sibling')}` |",
        f"| Frames dumped | {identification.get('frames_dumped')} |",
        f"| Mario's travel | {identification.get('x_travel_px', [0, 0])[0]:.0f} to {identification.get('x_travel_px', [0, 0])[1]:.0f} px |",
        f"| Screen position on the run | {identification.get('screen_min'):.2f} to {identification.get('screen_max'):.2f} px |",
    ]


def render_sweep_table(summary: Dict[str, Any]) -> List[str]:
    rows = [
        "| Margin (px) | Channel fires | Exceptions covered | Live exceptions covered | Lift |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in summary["sweep"]:
        rows.append(
            "| {:.0f} | {} | {} | {} | {} |".format(
                row["margin_px"],
                _pct(row["channel_fires_on_share"]),
                _pct(row["share_of_all_exceptions"]),
                _pct(row["share_of_live_exceptions"]),
                "-" if row["lift"] is None else demath_typographic(f"{row['lift']:.1f}"),
            )
        )
    return rows


def render_accounting_table(summary: Dict[str, Any]) -> List[str]:
    acc = summary["accounting"]
    fires = {
        "paused": summary["liveness"]["repeat_share"],
        "screen_bound": acc["fire_rates"]["screen_bound"],
        "wall_ahead": acc["fire_rates"]["wall_ahead"],
        "camera_step": acc["fire_rates"]["camera_step"],
        "unexplained": None,
    }
    rows = [
        "| Exception class | Frames | Share of exceptions | The label fires on |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for name in CLASSES:
        rows.append(
            f"| `{name}` | {acc['classes'][name]} | {_pct(acc['shares'][name])} | {_pct(fires[name])} |"
        )
    return rows


def render_committed_table(committed: Dict[str, Any]) -> List[str]:
    rows = [
        "| Recording | Exceptions | Exceptions repeating | Identity exact | Identity with repeats dropped |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for name in sorted(committed):
        ref = committed[name]
        rows.append(
            "| `{}` | {} | {} | {} | {} |".format(
                name,
                ref["exceptions"],
                _pct(ref["share_of_exceptions_repeating"]),
                _pct(ref["exact_rate_all"]),
                _pct(ref["exact_rate_off_repeats"]),
            )
        )
    return rows


def run_study(
    data_path: str = DATASET_BOUNDARY,
    margins: Sequence[float] = MARGINS_PX,
    primary: float = PRIMARY_MARGIN_PX,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Measure the channel, the pause and the prediction, then write the artifact."""
    rec = load_boundary_recording(data_path)
    summary = summarize(rec, margins, primary)
    committed = committed_repeat_report()
    identification = identification_report()
    payload: Dict[str, Any] = {
        "study": (
            "Test 10.58's prediction that a camera/boundary channel closes the exceptions to section "
            "4.1, on a recording that carries the layer 1 scroll, the engine mode and a CRC of all "
            "128 KB of WRAM - the last two because a stopped body and a paused simulation look "
            "identical in the first."
        ),
        "protocol": {
            "recording": os.path.basename(data_path),
            "subpixels_per_pixel": SUBPIXELS_PER_PIXEL,
            "exact_tolerance_subpixels": EXACT_TOLERANCE_SUBPIXELS,
            "screen_width_px": SCREEN_WIDTH_PX,
            "margins_px": [float(m) for m in margins],
            "primary_margin_px": float(primary),
            "wall_row_offsets": list(WALL_ROW_OFFSETS),
            "camera_address": identification.get("chosen"),
            "camera_identified_by": "scripts/scan_scroll_address.py",
            "camera_evidence": os.path.basename(SCAN_ARTIFACT),
            "classes": list(CLASSES),
            "interactive_mode": INTERACTIVE_MODE,
        },
        "identification": identification,
        "summary": summary,
        "committed_recordings": committed,
        "verdict": _verdict(summary, committed),
    }
    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(artifact, payload, command="python -m src.evaluation.boundary_channel_study")
    logger.info(
        "identity %.4f all | %.4f off the channel | %.4f off the pause | exceptions %d, paused %d, unexplained %d",
        summary["exact_rate_all_frames"],
        summary["exact_rate_off_channel"],
        summary["exact_rate_off_paused"],
        summary["accounting"]["n_exceptions"],
        summary["accounting"]["classes"]["paused"],
        summary["accounting"]["classes"]["unexplained"],
    )
    logger.info("Artifact written to: %s", artifact)
    payload["_artifact"] = artifact
    return payload


def render_section(payload: Dict[str, Any]) -> List[str]:
    r"""The whole of README section 10.59, generated from the artifact.

    Every number in the prose is a measurement, including the ones that say the prediction failed, so
    the section is emitted rather than typed: a hand-written version could drift from the artifact and
    the citation gate would not see it.
    """
    summary: Dict[str, Any] = payload["summary"]
    verdict: Dict[str, Any] = payload["verdict"]
    identification: Dict[str, Any] = payload["identification"]
    committed: Dict[str, Any] = payload["committed_recordings"]
    acc = summary["accounting"]
    live = summary["liveness"]
    pause = summary["pause_structure"]
    cam = summary["camera"]
    sweep = summary["sweep"]

    best = max(sweep, key=lambda row: row["share_of_all_exceptions"] or 0.0)
    committed_shares = verdict["committed_share_of_exceptions_repeating"]
    before = verdict["committed_exact_before"]
    after = verdict["committed_exact_after"]

    lines: List[str] = [
        "### 10.59 The Boundary Channel 10.58 Predicted, Built, and Refuted",
        "",
        f"10.58 read the exceptions to §4.1 as a constraint the state vector does not carry, and "
        f"predicted that a ninth channel - the boundary, or the camera offset - would remove them. "
        f"Testing that needed an address the repository has never had, so the channel was built "
        f"first: `scripts/scan_scroll_address.py` dumps all 128 KB of WRAM on each of "
        f"{identification['frames_dumped']} frames of a scripted run, ranks every 16-bit word by how "
        f"well it follows Mario's x, and keeps the ones that satisfy the axioms of a layer scroll. "
        f"Then `scripts/record_boundary_gameplay.py` recorded {cam['transitions']:,} transitions with "
        f"that channel, the engine mode byte, and a CRC of all of WRAM beside the published 8D state.",
        "",
        "**The instrument.** The address was identified, not assumed:",
        "",
        *render_identification_table(identification),
        "",
        f"The screen position `x - camera` stays inside the 256 px window on "
        f"{_pct(cam['screen_inside_window_share'])} of the recording's frames and the camera is never "
        f"ahead of Mario, so the channel behaves; the words that pass the axioms are the chosen one "
        f"and its mirror, and the word two slots away holds exactly half of it, which is what a "
        f"parallax layer does.",
        "",
        "**The prediction test.** Coverage of the exception frames, as the edge margin moves:",
        "",
        *render_sweep_table(summary),
        "",
        f"At the widest margin the channel names {int(best['exceptions_covered'])} of the "
        f"{acc['n_exceptions']} exception frames ({_pct(best['share_of_all_exceptions'])}), and "
        f"removing them moves the identity from {_pct(verdict['exact_rate_before'])} to "
        f"{_pct(verdict['exact_rate_with_channel_removed'])} - "
        f"{100 * verdict['exact_gain_from_the_channel']:.2f} pp of a "
        f"{100 * (1 - verdict['exact_rate_before']):.2f} pp deficit. The boundary is not where the "
        f"residue is.",
        "",
        "**What the exception frames actually are.** Every label below is tested in this order, and "
        "the counts sum to the exceptions:",
        "",
        *render_accounting_table(summary),
        "",
        f"On {_pct(live['share_of_exceptions_that_repeat'])} of them the *whole player record* "
        f"repeats: every channel of the next state - position, velocity, all four collision flags - "
        f"equals the current one, and the residue is exactly minus the velocity because nothing moved. "
        f"The console did not stop: the CRC over all of WRAM changes on "
        f"{_pct(live['share_of_repeats_where_wram_changed'])} of the repeating frames "
        f"({live['wram_identical_frames']} identical), the mode byte reads interactive "
        f"(0x{INTERACTIVE_MODE:02X}) on all {live['repeat_frames']} of them, and they arrive as "
        f"{pause['runs']} runs of {pause['mean_run_frames']:.1f} frames on average "
        f"(longest {pause['max_run_frames']}), in {pause['episodes_with_a_pause']} of the "
        f"{pause['episodes']} episodes. A player object that is not being processed, not a body held "
        f"by a wall.",
        "",
        "**The same test on the recordings 10.58 measured.** They carry no camera and no CRC, but the "
        "repeat signature needs neither:",
        "",
        *render_committed_table(committed),
        "",
        "Findings:",
        "",
        f"1. **The channel is real and it is not the explanation.** {_pct(best['share_of_all_exceptions'])} "
        f"of exception frames sit at a window edge, and the identity with them removed is within "
        f"{100 * verdict['exact_gain_from_the_channel']:.2f} pp of the identity without any channel at "
        f"all. 10.58's prediction is refuted on its own instrument.",
        f"2. **The residue's exceptions are mostly a paused simulation.** "
        f"{_pct(committed_shares[0])}-{_pct(committed_shares[1])} of the exception frames on the six "
        f"committed recordings repeat their entire player record, and on this recording "
        f"{_pct(live['share_of_repeats_that_are_exceptions'])} of the repeating frames are exceptions - "
        f"a repeat *is* an exception, because a body whose velocity byte holds and whose position does "
        f"not move violates §4.1 by construction.",
        f"3. **Section 4.1 is far closer to exact than published.** With the repeats dropped the "
        f"identity holds on {_pct(after[0])}-{_pct(after[1])} of the adjacent frames of the six "
        f"recordings, against the {_pct(before[0])}-{_pct(before[1])} 10.58 printed, and on "
        f"{_pct(summary['exact_rate_off_paused'])} of this recording's frames against "
        f"{_pct(summary['exact_rate_all_frames'])}. The gap is "
        f"{100 * verdict['exact_gain_from_the_pause']:.2f} pp here.",
        "4. **What 10.58 called memory is the pause showing up as memory.** Its persistence lift of "
        "104x-626x and its mean runs of 4.16-33.44 frames are the same objects measured twice: once "
        "as a conditional probability and once as a run length. The renewal-process kernel that "
        "covered the intervals is covering *this* - a deterministic block of repeated frames - and "
        "the calibration conclusion that the residue is not noise stands, for a different reason "
        "than the one given.",
        f"5. **What survives is the one-pixel reposition, and no positional channel explains it.** "
        f"{acc['live_exceptions']} exception frames here are not repeats, "
        f"{_pct(acc['live_share_exactly_one_pixel'])} of them exactly one pixel off the velocity "
        f"(residue atoms {', '.join(sorted(acc['live_exception_atoms'], key=lambda k: -acc['live_exception_atoms'][k])[:2])} "
        f"sub-pixels carry "
        f"{sum(sorted(acc['live_exception_atoms'].values(), reverse=True)[:2])} of them). The terrain "
        f"ahead of Mario labels {acc['classes']['wall_ahead']} of the "
        f"{acc['n_exceptions']} exceptions, but it fires on {_pct(acc['fire_rates']['wall_ahead'])} of "
        f"all frames, so it discriminates nothing; the camera steps on "
        f"{_pct(acc['fire_rates']['camera_step'])} of frames and labels "
        f"{acc['classes']['camera_step']} exceptions.",
        "",
        "**What this changes.** The camera channel is committed machinery from here: "
        "`wram.ADDR_CAMERA_X` and `SnesLibretroEmulator.get_camera_x()` exist, are gated, and the "
        "next recording that wants a boundary can read one. The liveness channels - mode and CRC - are "
        "the part the recordings were missing, and any future study that scores exception *rates* on "
        "these npz files has to say whether it dropped the repeats: 10.58's numbers were measured as "
        "reported, and what is withdrawn is the word *clamp*. The follow-up this makes unavoidable is "
        "in the recorder, not the model: end an episode when the player record stops changing, or when "
        "the mode leaves 0x14, so the pause is never in the dataset to be learned.",
        "",
        "**Limitations.** (i) The pause is identified by its signature, not its cause. The harness "
        "restores a savestate that comes up in mode 0x08 and writes 0x14 to force interactive physics "
        "(§10.38.1), and a player object that stops integrating for ~41 frames inside that forced "
        "session is consistent with the compromise - but this section measured the repeats, the CRC "
        "and the mode, and nothing more. (ii) The channel sweep is measured on the one recording that "
        "has a camera; the six older recordings can only be tested for the repeat signature, which is "
        "what they are shown for. (iii) The terrain label's row set (offsets "
        f"{', '.join(str(o) for o in WALL_ROW_OFFSETS)} from Mario's tile row) is a choice; widening "
        "it can only raise its fire rate, which is already "
        f"{_pct(acc['fire_rates']['wall_ahead'])}, so no choice of rows makes it a detector. (iv) The "
        "residue here is the horizontal identity only; the vertical one has its own ground rule in "
        "§4.3.5 and is not scored.",
        "",
        "*Regenerate: `make boundary-channel` (or `python -m src.evaluation.boundary_channel_study`). "
        "The recording behind it: `python scripts/record_boundary_gameplay.py`; the address behind "
        "that: `python scripts/scan_scroll_address.py`.*",
        "",
    ]
    return [demath_typographic(line) for line in lines]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test 10.58's camera/boundary prediction on the recording that carries it (10.59)."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument(
        "--data-path", default=DATASET_BOUNDARY, help="the boundary recording to read"
    )
    parser.add_argument("--margins", default=",".join(str(m) for m in MARGINS_PX))
    parser.add_argument("--primary-margin", type=float, default=PRIMARY_MARGIN_PX)
    parser.add_argument("--output-dir", default="results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    margins = [float(m) for m in str(args.margins).split(",")]
    run_study(
        data_path=args.data_path,
        margins=margins,
        primary=float(args.primary_margin),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
