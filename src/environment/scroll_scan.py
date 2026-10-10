r"""
scroll_scan.py - identify the engine's horizontal camera from WRAM dumps, without assuming it.

The residue study (10.58) found that the exceptions to section 4.1 are positions that do not move
while the velocity byte holds, and predicted that the missing observable is a boundary: the state
carries no channel that says "the screen will not scroll further". Reading that channel needs an
address, and `src/environment/wram.py` has never had one. A scroll address is not something to be
recalled from a memory map - a wrong one would silently become the project's definition of a
boundary - so it is identified here from the console: dump all 128 KB of WRAM every frame while
Mario runs the level, rank every 16-bit word by how well it follows his world coordinate, and then
require the winner to satisfy the properties that belong to a layer scroll and to nothing else:

* it never exceeds Mario's own x (a camera cannot be ahead of the body it is following);
* the screen position `x - camera` stays inside the 256 px window on every frame;
* it sits still while Mario walks inside the dead zone, and moves at most a few pixels per frame;
* it moves both ways, and it is 0 at the level start.

Mirrors of the same series are expected (SNES WRAM has them) and are reported, because an address
that agrees with its own mirror in a different memory page is a weaker claim than one that does not.

CPU-only: the functions take the dumps as arrays, so the identification is unit-testable without an
emulator. The capture that produces the dumps lives in `scripts/scan_scroll_address.py`.

Run:  python -m src.environment.scroll_scan   (reports the committed artifact)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment.wram import WRAM_SIZE_BYTES
from src.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_NAME = "scroll_address_scan_metrics.json"
SCREEN_WIDTH_PX = 256.0
CORR_MIN = 0.97
STEP_MAX_PX = 5
VALUE_MAX_PX = 2000
SCAN_BLOCK_WORDS = 4096


@dataclass
class Candidate:
    """One WRAM word that follows Mario's world coordinate closely enough to be worth testing."""

    addr: int
    corr: float
    first: int
    last: int
    vmax: int
    frac_still: float
    max_step: int
    median_lag: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "addr": f"$7E:{self.addr:04X}",
            "addr_int": self.addr,
            "corr_with_x": round(self.corr, 6),
            "first": self.first,
            "last": self.last,
            "max": self.vmax,
            "frac_steps_still": round(self.frac_still, 4),
            "max_step_px": self.max_step,
            "median_x_minus_value": round(self.median_lag, 3),
        }


def u16_series(dumps: np.ndarray, addr: int) -> np.ndarray:
    """The little-endian 16-bit series at one byte offset of a (frames, bytes) WRAM dump."""
    if addr + 1 >= dumps.shape[1]:
        raise ValueError(f"address $7E:{addr:04X} runs past the dumped WRAM")
    return dumps[:, addr].astype(np.float64) + dumps[:, addr + 1].astype(np.float64) * 256.0


def _series_stats(
    values: np.ndarray, x: np.ndarray
) -> Optional[Tuple[float, float, float, float, float]]:
    """(correlation with x, share of steps that leave the value alone, largest step, median lag, max)."""
    sd = float(values.std())
    if sd <= 1e-9:
        return None
    corr = float(np.corrcoef(values, x)[0, 1])
    steps = np.diff(values)
    return (
        corr,
        float(np.mean(steps == 0)),
        float(np.max(np.abs(steps))) if steps.size else 0.0,
        float(np.median(x - values)),
        float(values.max()),
    )


def rank_candidates(
    dumps: np.ndarray,
    x: np.ndarray,
    *,
    corr_min: float = CORR_MIN,
    step_max: float = STEP_MAX_PX,
    value_max: float = VALUE_MAX_PX,
    block_words: int = SCAN_BLOCK_WORDS,
) -> List[Candidate]:
    """Every even-offset 16-bit word that tracks `x` closely, in order of correlation.

    The scan is deliberately generous: it ranks by correlation and only rejects words that move in
    jumps bigger than the engine's own top speed (4.5 px/frame), because a scroll is bounded by that.
    The axioms in `camera_axioms` do the identifying.
    """
    if dumps.ndim != 2 or dumps.shape[1] != WRAM_SIZE_BYTES:
        raise ValueError(
            f"expected WRAM dumps of shape (frames, {WRAM_SIZE_BYTES}), got {dumps.shape}"
        )
    if x.ndim != 1 or x.size != dumps.shape[0]:
        raise ValueError("the x series must have one entry per dumped frame")
    out: List[Candidate] = []
    n_words = (WRAM_SIZE_BYTES - 2) // 2
    for start in range(0, n_words, block_words):
        stop = min(start + block_words, n_words)
        lo, hi = start * 2, stop * 2
        block = dumps[:, lo:hi]
        words = block[:, 0::2].astype(np.float64) + block[:, 1::2].astype(np.float64) * 256.0
        centred = words - words.mean(axis=0, keepdims=True)
        sd = words.std(axis=0)
        xc = x - x.mean()
        denom = np.maximum(sd * float(np.sqrt((xc**2).mean())) * float(x.size) ** 0.5, 1e-9)
        corr = (centred * xc[:, None]).sum(axis=0) / denom
        steps = np.abs(np.diff(words, axis=0))
        max_step = steps.max(axis=0) if steps.size else np.zeros(words.shape[1])
        still = (steps == 0).mean(axis=0) if steps.size else np.ones(words.shape[1])
        vmax = words.max(axis=0)
        lag = np.median(x[:, None] - words, axis=0)
        keep = np.flatnonzero(
            (corr >= corr_min) & (sd > 1.0) & (max_step <= step_max) & (vmax <= value_max)
        )
        for j in keep:
            out.append(
                Candidate(
                    addr=lo + 2 * int(j),
                    corr=float(corr[j]),
                    first=int(words[0, j]),
                    last=int(words[-1, j]),
                    vmax=int(vmax[j]),
                    frac_still=float(still[j]),
                    max_step=int(max_step[j]),
                    median_lag=float(lag[j]),
                )
            )
    out.sort(key=lambda c: (-c.corr, c.addr))
    return out


def camera_axioms(
    camera: np.ndarray,
    x: np.ndarray,
    *,
    screen_width: float = SCREEN_WIDTH_PX,
    start_tolerance: float = 1.0,
) -> Dict[str, Any]:
    """The four properties a layer scroll has and no other tracked word has, measured not assumed."""
    screen = x - camera
    steps = np.diff(camera)
    moving = steps != 0
    still = steps == 0
    dx = np.diff(x)
    return {
        "never_ahead_of_mario": bool(np.all(camera <= x + 1e-9)),
        "screen_inside_window_share": float(np.mean((screen >= 0.0) & (screen <= screen_width))),
        "screen_min": float(screen.min()),
        "screen_max": float(screen.max()),
        "starts_at_zero": bool(abs(float(camera[0])) <= start_tolerance),
        "frac_steps_still": float(np.mean(still)),
        "px_moved_while_still_mean": float(np.abs(dx[still]).mean()) if still.any() else 0.0,
        "px_moved_while_still_max": float(np.abs(dx[still]).max()) if still.any() else 0.0,
        "max_step_px": int(np.abs(steps).max()) if steps.size else 0,
        "moves_both_ways": bool(np.any(steps > 0) and np.any(steps < 0)),
        "frac_steps_forward": float(np.mean(steps > 0)),
        "frac_steps_backward": float(np.mean(steps < 0)),
        "camera_max": int(camera.max()),
        "n_moving_steps": int(moving.sum()),
    }


def axioms_pass(report: Dict[str, Any]) -> bool:
    """A candidate is the camera when every measured property of a scroll holds."""
    return bool(
        report["never_ahead_of_mario"]
        and report["screen_inside_window_share"] >= 0.999
        and report["starts_at_zero"]
        and report["moves_both_ways"]
        and report["max_step_px"] <= STEP_MAX_PX
        and report["frac_steps_still"] > 0.02
    )


def find_parallax_sibling(dumps: np.ndarray, camera_addr: int) -> Optional[Dict[str, Any]]:
    """The word that holds exactly half the camera - SMW's second layer scrolls at half the rate.

    This is a corroboration, not a requirement: a background layer moving at half the rate of the
    terrain is what parallax *is*, so finding one two bytes away from the identified address makes a
    wrong pick much less likely.
    """
    camera = u16_series(dumps, camera_addr)
    half = np.floor(camera / 2.0)
    n_words = (WRAM_SIZE_BYTES - 2) // 2
    best: Optional[Tuple[int, int]] = None
    for start in range(0, n_words, SCAN_BLOCK_WORDS):
        stop = min(start + SCAN_BLOCK_WORDS, n_words)
        lo, hi = start * 2, stop * 2
        block = dumps[:, lo:hi]
        words = block[:, 0::2].astype(np.float64) + block[:, 1::2].astype(np.float64) * 256.0
        mismatch = np.abs(words - half[:, None])
        exact_frames = (mismatch < 0.5).mean(axis=0)
        for j in np.flatnonzero(exact_frames >= 1.0):
            addr = lo + 2 * int(j)
            if addr == camera_addr or abs(addr - camera_addr) <= 1:
                continue
            if best is None or addr < best[0]:
                best = (addr, int(words[:, j].max()))
    if best is None:
        return None
    return {"addr": f"$7E:{best[0]:04X}", "addr_int": best[0], "max": best[1]}


def identify(
    dumps: np.ndarray,
    x: np.ndarray,
    *,
    max_reported: int = 12,
) -> Dict[str, Any]:
    """Rank, test and pick. The artifact this writes is the evidence behind `ADDR_CAMERA_X`."""
    candidates = rank_candidates(dumps, x)
    tested: List[Dict[str, Any]] = []
    chosen: Optional[Dict[str, Any]] = None
    for cand in candidates[:max_reported]:
        report = camera_axioms(u16_series(dumps, cand.addr), x)
        entry = {**cand.as_dict(), "axioms": report, "passes": axioms_pass(report)}
        tested.append(entry)
        if entry["passes"] and chosen is None:
            chosen = entry
    passing = [e for e in tested if e["passes"]]
    out: Dict[str, Any] = {
        "frames_dumped": int(dumps.shape[0]),
        "wram_bytes": int(dumps.shape[1]),
        "x_travel_px": [float(x.min()), float(x.max())],
        "words_ranked": len(candidates),
        "candidates": tested,
        "passing": [e["addr"] for e in passing],
        "chosen": None if chosen is None else chosen["addr"],
        "chosen_addr_int": None if chosen is None else chosen["addr_int"],
        "mirrors": [
            e["addr"]
            for e in passing
            if chosen is not None
            and e["median_x_minus_value"] == chosen["median_x_minus_value"]
            and e["addr"] != chosen["addr"]
        ],
        "parallax_sibling": None,
        "screen_width_px": SCREEN_WIDTH_PX,
        "thresholds": {
            "corr_min": CORR_MIN,
            "step_max_px": STEP_MAX_PX,
            "value_max_px": VALUE_MAX_PX,
        },
    }
    if chosen is not None:
        out["parallax_sibling"] = find_parallax_sibling(dumps, chosen["addr_int"])
    return out


def read_artifact(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the committed identification, so the address can be checked without a capture."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "results", ARTIFACT_NAME)
    payload = read_artifact(os.path.abspath(path))
    logger.info("camera address identified as %s", payload.get("chosen"))
    for entry in payload.get("candidates", []):
        logger.info(
            "%s corr=%.4f passes=%s screen=[%.1f, %.1f] still=%.3f",
            entry["addr"],
            entry["corr_with_x"],
            entry["passes"],
            entry["axioms"]["screen_min"],
            entry["axioms"]["screen_max"],
            entry["axioms"]["frac_steps_still"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
