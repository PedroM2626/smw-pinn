r"""Gates for README 10.62: the re-made pixel recording, and the discrimination it exists to provide.

10.31.1's finding was a null, and a null is only evidence if the instrument can report the other
answer. These gates therefore refuse the section if the alignment stops discriminating on the new
recording - the picture must move when the camera does and hold when it does not - re-derive the
detector's positive control from the artifact rather than trusting the prose, check the new file's
frames are full-width pictures (the defect 10.60 fixed), and assert the published recording this
study compares itself against was not touched.

CPU-only, numpy-only; skipped when the recheck has not been run.

Run:  pytest tests/test_pixel_level_recheck.py -q
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.pixel_level_recheck import render_section
from src.utils.paths import DATASET_PIXEL, DATASET_PIXEL_LEVEL

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "pixel_level_recheck_metrics.json"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the pixel level recheck has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def _new_recording():
    if not Path(DATASET_PIXEL_LEVEL).is_file():
        pytest.skip("the level-entry pixel recording has not been captured")
    with np.load(DATASET_PIXEL_LEVEL, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def test_readme_10_62_is_the_generated_block() -> None:
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.62 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.62 "), "the generator must start at the section heading"


def test_the_published_recording_still_shows_the_null() -> None:
    """The comparison arm: the old file must not have started scrolling behind the section."""
    published = _payload()["published"]
    assert published["share_of_pairs_best_shift_zero"] == 1.0
    assert published["pairs_where_picture_shifted"] == 0
    assert published["mean_abs_dx_px"] > 1.0, (
        "the null is only interesting because the player really moved; a still recording proves "
        "nothing about the alignment"
    )


def test_the_new_recording_discriminates() -> None:
    payload = _payload()
    verdict = payload["verdict"]
    coup = payload["camera_coupling"]
    assert verdict["discriminates"]
    assert verdict["level_picture_shifts_when_the_camera_does"] > 0.5
    assert verdict["level_picture_holds_when_the_camera_does_not"] > 0.9
    assert coup["pairs_where_camera_moved"] > 500, (
        "the split is the finding; a handful of camera-moving pairs is not a discrimination"
    )


def test_the_detector_control_still_recovers_a_synthetic_scroll() -> None:
    """The positive control, re-read from both arms of the artifact."""
    payload = _payload()
    for arm in ("published", "level"):
        control = payload[arm]["detector_control"]
        for key, truth in (("true_shift_1px", 1), ("true_shift_2px", 2), ("true_shift_3px", 3)):
            row = control[key]
            assert row["recovered_mode"] == truth, f"{arm}: {key} recovered {row['recovered_mode']}"
            assert row["share_recovering_exactly"] == 1.0


def test_the_new_frames_are_full_width_pictures() -> None:
    """10.60's decoder fix, asserted on the data rather than on a unit-test buffer."""
    data = _new_recording()
    frames = np.asarray(data["frames"], dtype=np.uint8)
    assert frames.shape[1:] == (112, 128, 3)
    right = frames[:, :, 96:, :].reshape(frames.shape[0], -1).sum(axis=1)
    assert float(np.mean(right > 0)) > 0.9, (
        "the right quarter of most frames is black, which is what a half-width decode looks like"
    )


def test_the_recording_carries_the_channels_the_study_reads() -> None:
    data = _new_recording()
    for key in ("frames", "states", "actions", "held", "camera"):
        assert key in data, f"10.62 reads a {key} channel"
    payload = _payload()
    assert int(np.asarray(data["held"]).sum()) == payload["camera_coupling"]["held_frames"]
    assert data["frames"].shape[0] == payload["camera_coupling"]["frames"]
    assert data["camera"].shape[0] == data["frames"].shape[0]


def test_the_published_pixel_recording_was_not_touched() -> None:
    """A new recording is a new file; the one §10.31's numbers were measured on stays as it was."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", str(Path(DATASET_PIXEL).relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "", f"the published pixel dataset is modified: {proc.stdout}"
