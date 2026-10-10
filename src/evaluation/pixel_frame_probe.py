r"""
pixel_frame_probe.py - do the committed RGB frames show the level the state describes (10.31 audit)?

Section 10.31 records 2,720 paired (frame, WRAM) transitions and reads the trained estimator's
position R² of 0.948 as "scroll position is visible in the background". That reading is testable
without any model: if the frames are the level Mario is walking through, the picture must translate
as he moves. So measure it directly - for every consecutive pair, find the integer horizontal shift
that best aligns the two frames, and compare it with the change in the recorded x.

The probe needs no emulator and no network: it reads the committed npz. It writes
``results/pixel_frame_probe_metrics.json`` and the README paragraph that quotes it.

Run:  python -m src.evaluation.pixel_frame_probe
      python -m src.evaluation.pixel_frame_probe --max-shift 6 --stride 1
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import REPO_ROOT
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

ARTIFACT_NAME = "pixel_frame_probe_metrics.json"
DATASET = os.path.join(REPO_ROOT, "data", "raw", "smw_pixel_dataset.npz")
MARGIN_PX = 6
BAND_TOP_PX = 14  # the HUD band at the top of the 112-row frame is excluded from the alignment


def frame_series(path: str = DATASET) -> Tuple[np.ndarray, np.ndarray]:
    """Grayscale frames and the recorded x, in the units the dataset stores them."""
    with np.load(path, allow_pickle=False) as blob:
        frames = np.asarray(blob["frames"], dtype=np.float32).mean(axis=3)
        states = np.asarray(blob["states"], dtype=np.float32)
    return frames[:, BAND_TOP_PX:, :], states[:, 0].astype(np.float64)


def best_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> Tuple[int, float, float]:
    """The integer shift of `b` relative to `a` that minimises mean squared difference.

    Returns (shift, score at the best shift, score at zero shift). A frame pair in which the scene
    translated by `s` pixels scores lowest at `s`; a pair in which nothing moved scores lowest at 0
    however much the *state* changed.
    """
    height, width = a.shape
    best = (0, np.inf, 0.0)
    zero = float(
        np.mean((a[:, MARGIN_PX : width - MARGIN_PX] - b[:, MARGIN_PX : width - MARGIN_PX]) ** 2)
    )
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            left = a[:, MARGIN_PX + shift : width - MARGIN_PX]
            right = b[:, MARGIN_PX : width - MARGIN_PX - shift]
        else:
            left = a[:, MARGIN_PX : width - MARGIN_PX + shift]
            right = b[:, MARGIN_PX - shift : width - MARGIN_PX]
        diff = left - right
        score = float(np.mean(diff * diff))
        if score < best[1]:
            best = (shift, score, zero)
    return best[0], best[1], zero


def detector_control(frames: np.ndarray, shifts: Sequence[int] = (1, 2, 3)) -> Dict[str, Any]:
    """Prove the alignment sees a scroll when there is one, by making one.

    A probe that reports "the picture never moves" is only worth printing if the same code reports
    the right number on a picture that does. Each synthetic pair takes one frame and slides it by a
    whole number of pixels, which is exactly what a one-pixel camera step draws.
    """
    out: Dict[str, Any] = {}
    for truth in shifts:
        recovered: List[int] = []
        for t in range(0, frames.shape[0] - 1, 20):
            a = frames[t]
            # A camera moving right by `truth` draws the scene shifted left by `truth`.
            b = np.roll(frames[t], -truth, axis=1).astype(np.float32)
            shift, _, _ = best_shift(a, b, max_shift=4)
            recovered.append(shift)
        values = np.array(recovered, dtype=np.float64)
        out[f"true_shift_{truth}px"] = {
            "recovered_mode": int(np.bincount((values + 4).astype(int)).argmax() - 4),
            "share_recovering_exactly": float(np.mean(values == truth)),
            "trials": int(values.size),
        }
    return out


def probe(path: str = DATASET, max_shift: int = 4, stride: int = 1) -> Dict[str, Any]:
    """Align every consecutive frame pair and compare the picture's motion with the state's."""
    frames, x = frame_series(path)
    pairs: List[Tuple[int, float, float]] = []
    for t in range(0, frames.shape[0] - 1, stride):
        pairs.append(best_shift(frames[t], frames[t + 1], max_shift))
    shifts = np.array([p[0] for p in pairs], dtype=np.float64)
    dx = np.diff(x)[::stride]
    moved = np.abs(dx) > 0.5
    corr: Optional[float] = None
    if moved.sum() > 10 and np.std(shifts[moved]) > 1e-9 and np.std(dx[moved]) > 1e-9:
        corr = float(np.corrcoef(shifts[moved], dx[moved])[0, 1])
    return {
        "frames": int(frames.shape[0]),
        "frame_shape": [int(frames.shape[1] + BAND_TOP_PX), int(frames.shape[2])],
        "pairs_tested": len(pairs),
        "max_shift_px": int(max_shift),
        "mean_abs_dx_px": float(np.abs(dx).mean()),
        "mean_abs_picture_shift_px": float(np.abs(shifts).mean()),
        "share_of_pairs_best_shift_zero": float(np.mean(shifts == 0)),
        "pairs_where_picture_shifted": int(np.sum(shifts != 0)),
        "corr_shift_with_dx_on_moving_pairs": corr,
        "moving_pairs": int(moved.sum()),
        "shift_histogram": {
            str(int(value)): int((shifts == value).sum()) for value in np.unique(shifts)
        },
        "detector_control": detector_control(frames),
    }


def render_probe(payload: Dict[str, Any]) -> List[str]:
    """The README paragraph for 10.31.1, generated from the probe."""
    measured = payload["probe"]
    control = measured["detector_control"]
    key = "true_shift_2px"
    recovered = control[key]["share_recovering_exactly"]
    return [
        "**An audit of the frames themselves, added later.** Whether the imagery is the level Mario "
        "is walking through is testable without a model: align each consecutive pair by the integer "
        "horizontal shift that best matches them, and compare that shift with the change in the "
        f"recorded x. The alignment does see a scroll when there is one - sliding a frame by two "
        f"pixels is recovered as two pixels on {100 * recovered:.2f}% of {control[key]['trials']} "
        f"synthetic trials - and on this file it finds nothing: the best shift is zero on "
        f"{100 * measured['share_of_pairs_best_shift_zero']:.2f}% of the "
        f"{measured['pairs_tested']:,} consecutive pairs, the picture moves "
        f"{measured['mean_abs_picture_shift_px']:.2f} px per frame while the state moves "
        f"{measured['mean_abs_dx_px']:.2f} px, and the correlation between the two is "
        + (
            f"{measured['corr_shift_with_dx_on_moving_pairs']:.3f}"
            if measured["corr_shift_with_dx_on_moving_pairs"] is not None
            else "undefined, because the picture never shifts at all"
        )
        + ". The frames are therefore not the scrolling level: the savestate this repository records "
        "from restores into engine mode `0x08` and the harness writes `0x14` to force interactive "
        "physics (§10.38.1), so the WRAM player is simulated while the PPU keeps drawing the title "
        "and file-select composite - which is what the exported frames show. The estimator numbers "
        "above are what that model achieved on that imagery and they stand as measurements of it; "
        "what is withdrawn is the reading beside the first row, position recoverable from the scroll "
        "of the background, because there is no background scroll in these frames to recover. "
        "`python -m src.evaluation.pixel_frame_probe` re-derives this paragraph from "
        "`results/pixel_frame_probe_metrics.json`.",
    ]


def run_probe(
    path: str = DATASET,
    max_shift: int = 4,
    stride: int = 1,
    output_dir: str = "results",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "study": (
            "Do the committed RGB frames translate when the recorded player moves? If they are the "
            "level, they must; the alignment is measured per consecutive pair with no model involved."
        ),
        "protocol": {
            "dataset": os.path.basename(path),
            "margin_px": MARGIN_PX,
            "band_top_px": BAND_TOP_PX,
            "max_shift_px": int(max_shift),
            "stride": int(stride),
            "frame_pairs": "consecutive frames of the recording",
        },
        "probe": probe(path=path, max_shift=max_shift, stride=stride),
    }
    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(artifact, payload, command="python -m src.evaluation.pixel_frame_probe")
    logger.info(
        "pairs %d | best shift zero on %.2f%% | mean |dx| %.2f px",
        payload["probe"]["pairs_tested"],
        100 * payload["probe"]["share_of_pairs_best_shift_zero"],
        payload["probe"]["mean_abs_dx_px"],
    )
    logger.info("Artifact written to: %s", artifact)
    payload["_artifact"] = artifact
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether the committed pixel recording's frames follow the player (10.31 audit)."
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--data-path", default=DATASET)
    parser.add_argument("--max-shift", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    run_probe(
        path=args.data_path,
        max_shift=args.max_shift,
        stride=args.stride,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
