r"""Gates for README 10.61: the generated section, and the three-way terrain comparison.

The section separates two explanations that a single comparison cannot: "the recording is of another
stage" and "the sampling instrument is broken". The gate therefore re-derives the recording arm from
the committed file without an emulator, refuses the section if the instrument stops reproducing the
session the file came from, and refuses it if the two live walks ever agree on the terrain - because
then the open question 10.36.3 left would no longer be open, and the prose would have to change.

Run:  pytest tests/test_stage_identity_study.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.stage_identity_study import (
    AGREEMENT_TOLERANCE_ROWS,
    compare,
    profile_from_frames,
    profile_from_npz,
    render_section,
)
from src.utils.paths import DATASET_TILEMAP

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "stage_identity_metrics.json"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the stage-identity study has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_readme_10_61_is_the_generated_block() -> None:
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.61 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.61 "), "the generator must start at the section heading"


def test_the_recording_arm_is_re_derivable_without_an_emulator() -> None:
    """The committed file is the only source the recording side reads, and the artifact must match."""
    payload = _payload()
    fresh = profile_from_npz(DATASET_TILEMAP)
    stored = payload["profiles"]["recording"]
    assert fresh["n_columns"] == stored["n_columns"]
    assert fresh["x_range_px"] == pytest.approx(stored["x_range_px"], rel=1e-9)
    assert fresh["observations"] == stored["observations"]


def test_the_instrument_reproduces_the_session_the_file_came_from() -> None:
    """If this stops holding, the terrain comparison below measures the sampler, not the level."""
    payload = _payload()
    verdict = payload["verdict"]
    assert verdict["recording_faithful_to_its_session"]
    assert verdict["recording_matches_its_own_session"] >= 0.9
    assert payload["comparisons"]["recording_vs_forced_mode"]["columns_compared"] > 40, (
        "the two sides barely overlap, so the reproduction rate is not evidence of anything"
    )


def test_the_two_live_walks_do_not_agree_on_the_terrain() -> None:
    """The finding, re-derived: same walk rule, two sessions, the buffers differ."""
    payload = _payload()
    live = payload["comparisons"]["forced_mode_vs_level"]
    own = payload["comparisons"]["recording_vs_forced_mode"]
    assert payload["verdict"]["terrain_differs"]
    assert live["share_agreeing"] < own["share_agreeing"], (
        "the live-vs-live comparison is the one that isolates terrain; if it ever stops being the "
        "worse of the two, the section's reading has to be re-argued, not kept"
    )


def test_both_walks_had_to_clear_the_message_box() -> None:
    """10.60's pause is not an artefact of one savestate: the scripted walk meets it in both."""
    profiles = _payload()["profiles"]
    assert profiles["forced_mode"]["message_boxes_cleared"] >= 1
    assert profiles["level_entry"]["message_boxes_cleared"] >= 1


def test_the_superseded_scratch_figure_is_not_left_quoted() -> None:
    """10.36.3's 32.65% had no artifact behind it: it may survive only as a retraction.

    The retraction log in section 12 is allowed - indeed required - to name a withdrawn number; the
    section that presented it as a current measurement is not.
    """
    text = _readme()
    entry = text[text.index("#### 10.36.3") : text.index("### 10.37")]
    assert "32.65" not in entry, (
        "the paragraph still prints an unreproducible number - point it at 10.61 instead"
    )
    assert "10.61" in entry, "the paragraph that raised the question must name the answer"
    elsewhere = text.replace(text[text.index("## 12.") :], "")
    assert "32.65" not in elsewhere, "a withdrawn figure belongs in the retraction log, not in prose"


def test_compare_scores_only_the_columns_both_sides_see() -> None:
    """Negative controls for the scorer itself."""
    a = {"profile": {"0": 0, "1": 2, "2": 1}}
    same = {"profile": {"0": 0, "1": 2, "2": 1, "7": 0}}
    moved = {"profile": {"0": 2, "1": 0, "2": 3}}
    hit = compare(a, same, tolerance=AGREEMENT_TOLERANCE_ROWS)
    assert hit["columns_compared"] == 3 and hit["share_agreeing"] == 1.0
    miss = compare(a, moved, tolerance=AGREEMENT_TOLERANCE_ROWS)
    assert miss["columns_compared"] == 3 and miss["share_agreeing"] == 0.0
    assert miss["disagreeing_columns"] == [[0, 0, 2], [1, 2, 0], [2, 1, 3]]


def test_the_profile_reads_downward_from_marios_own_row() -> None:
    """The one geometry decision the instrument makes: depth is measured from the centre row."""
    radius = 3
    patch = np.zeros((1, 2 * radius + 1, 2 * radius + 1), dtype=np.int64)
    patch[0, radius + 2, :] = 1  # ground two rows below Mario's tile, all columns
    out = profile_from_frames(np.array([16.0 * 10 + 8.0]), patch)
    assert set(int(k) for k in out["profile"]) == set(range(10 - radius, 10 + radius + 1))
    assert all(v == 2 for v in out["profile"].values())
