r"""Gates for README 10.63: the held-player split, the parity of the hygiene arm, and the recorder.

The section makes three claims that could each rot silently: that the committed recording's held
frames are the death apex rather than a boundary or a box; that turning the hygiene on changed every
simulation channel of the new recording not at all; and that the recorder's default still reproduces
the published file. The first two are re-derived from the arrays here, the third from git, and the
helper that counts frames inside a hold gets a positive and a negative control because a metric that
cannot tell a 3-frame blip from a 40-frame freeze is the kind of number this repository keeps having
to take back.

CPU-only, numpy-only; skipped when the arms have not been recorded.

Run:  pytest tests/test_pause_hygiene_study.py -q
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.boundary_channel_study import EXACT_TOLERANCE_SUBPIXELS, load_boundary_recording
from src.evaluation.pause_hygiene_study import (
    DATASET_BOUNDARY,
    DATASET_LEVEL_CLEAR,
    DATASET_LEVEL_PLAIN,
    DEATH_APEX_VY,
    frames_in_holds,
    render_section,
)
from src.utils.paths import REPO_ROOT

ARTIFACT = REPO_ROOT / "results" / "pause_hygiene_metrics.json"
CHANNEL_ARTIFACT = REPO_ROOT / "results" / "boundary_channel_metrics.json"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the pause-hygiene study has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _arms(payload=None):
    payload = payload or _payload()
    return {a["arm"]: a for a in payload["arms"]}


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_10_63_is_the_generated_block() -> None:
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.63 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.63 "), "the generator must start at the section heading"


def test_frames_in_holds_counts_only_runs_long_enough_to_be_a_freeze() -> None:
    """Positive and negative control for the volume metric."""
    blips = np.array([True] * 3 + [False] * 5 + [True] * 3, dtype=bool)
    assert frames_in_holds(blips, minimum_run=10) == 0
    one_long = np.array([False] * 4 + [True] * 12, dtype=bool)
    assert frames_in_holds(one_long, minimum_run=10) == 12
    two_runs = np.concatenate([one_long, np.array([False, True, True, True, True, True, True])])
    assert frames_in_holds(two_runs, minimum_run=10) == 12, "the 6-frame run is a blip, not a hold"


def test_the_published_arms_rates_match_the_file_and_the_other_study() -> None:
    """Same file, two modules: 10.59's study and 10.63 must not disagree about it."""
    published = _arms()["published_forced_mode"]
    rec = load_boundary_recording(DATASET_BOUNDARY)
    exact = np.abs(rec.residue) < EXACT_TOLERANCE_SUBPIXELS
    assert published["transitions"] == rec.n
    assert published["exact_share"] == pytest.approx(float(np.mean(exact)), abs=1e-9)
    assert published["repeats"] == int(np.sum(rec.repeat))
    if CHANNEL_ARTIFACT.is_file():
        channel = json.loads(CHANNEL_ARTIFACT.read_text(encoding="utf-8"))
        liveness = channel["summary"]["liveness"]
        assert liveness["repeat_frames"] == published["repeats"], (
            "10.59 and 10.63 count the same held frames on the same file differently"
        )
        assert liveness["repeat_share"] == pytest.approx(published["repeat_share"], abs=1e-6)


def test_the_held_frames_are_the_death_apex_as_the_section_says() -> None:
    payload = _payload()
    for arm in payload["arms"]:
        rec = load_boundary_recording(str(REPO_ROOT / arm["dataset"]))
        repeat = rec.repeat
        vy = rec.state[:, 3]
        share = float(np.mean(np.abs(vy[repeat] - DEATH_APEX_VY) <= 1e-6)) if repeat.any() else 0.0
        assert share == pytest.approx(arm["death_apex_share_of_repeats"] or 0.0, abs=1e-9)
        assert share > 0.95, (
            f"{arm['arm']}: only {share:.4f} of the holds freeze at the death apex, so the section "
            "cannot say the held player is mostly the death animation"
        )


def test_turning_the_hygiene_on_changed_no_channel_of_the_simulation() -> None:
    payload = _payload()
    parity = payload["arm_parity"]
    assert parity["simulation_channels_identical"]
    assert all(parity["channels"].values())
    assert parity["crc_frames_differing"] > 0, (
        "if the CRC agrees too, the A pulses never reached the console and the parity claim is "
        "about two runs of the same inputs, not about an operator press that did nothing"
    )


def test_dropping_the_holds_is_where_the_identity_gain_comes_from() -> None:
    clear = _arms()["level_entry_hygiene"]
    gain = _payload()["verdict"]["exact_rate_gain_from_dropping_holds_pp"]
    assert gain == pytest.approx(
        100 * (clear["exact_share_excluding_repeats"] - clear["exact_share"]), abs=1e-9
    )
    assert 1.0 < gain < 10.0, "the section calls this a few points, not a revolution"
    assert clear["non_repeat_exception_share"] < 0.01, (
        "the residue 10.59 could not explain is the interesting part only while it stays small"
    )


def test_the_two_level_arms_are_new_files_and_the_published_one_is_untouched() -> None:
    assert Path(DATASET_LEVEL_PLAIN).is_file() and Path(DATASET_LEVEL_CLEAR).is_file()
    assert Path(DATASET_LEVEL_PLAIN) != Path(DATASET_BOUNDARY)
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", str(Path(DATASET_BOUNDARY).relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "", (
        f"the recording 10.59 measured is modified ({proc.stdout.strip()}); a new arm is a new file"
    )
