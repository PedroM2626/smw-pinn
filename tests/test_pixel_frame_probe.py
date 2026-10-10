r"""Gates for the 10.31.1 pixel-frame audit: the generated paragraph, and the detector behind it.

The claim is a negative one about an artifact this repository published eleven sections ago, so both
halves need a check: the alignment must be able to see a scroll (otherwise "it found none" is
meaningless), and the README must carry the sentence that says the frames are not the level (otherwise
the correction can be edited away while the artifact still reports it).

CPU-only, reads the committed npz; skipped if the probe has not been run.

Run:  pytest tests/test_pixel_frame_probe.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.pixel_frame_probe import BAND_TOP_PX, best_shift, render_probe

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "pixel_frame_probe_metrics.json"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the pixel frame probe has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme():
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_the_readme_paragraph_is_the_generated_one() -> None:
    lines = render_probe(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.31.1 disagrees with the probe:\n" + "\n".join(missing[:2])


def test_the_probe_reports_no_scroll_and_the_control_proves_it_could() -> None:
    measured = _payload()["probe"]
    assert measured["pairs_tested"] > 2000
    assert measured["mean_abs_dx_px"] > 1.0, (
        "the state has to actually move for a null result in the imagery to mean anything"
    )
    assert measured["share_of_pairs_best_shift_zero"] == 1.0
    assert measured["pairs_where_picture_shifted"] == 0
    for truth in ("1", "2", "3"):
        control = measured["detector_control"][f"true_shift_{truth}px"]
        assert control["recovered_mode"] == int(truth), (
            f"a {truth}-px synthetic scroll was not recovered: the alignment is the instrument here, "
            "and if it cannot see a shift the null result is about the instrument"
        )
        assert control["share_recovering_exactly"] > 0.9


def test_the_alignment_function_recovers_a_shift_it_is_handed() -> None:
    """The unit behind the control, on a synthetic frame with a recognisable pattern."""
    base = np.tile(
        np.array([[c * 7 % 251 for c in range(96)]], dtype=np.float32), (BAND_TOP_PX + 40, 1)
    )
    # A camera moving right by 3 px draws the scene shifted left by 3 px.
    shifted = np.roll(base, -3, axis=1).astype(np.float32)
    shift, score, zero = best_shift(base, shifted, max_shift=5)
    assert shift == 3, f"the alignment read a 3 px scroll as {shift} px"
    assert score < zero


def test_the_correction_is_written_where_readers_will_look() -> None:
    text = _readme()
    section_12 = text.split("## 12. Scientific Integrity Statement")[1]
    assert "pixel_frame_probe" in section_12, (
        "section 12 is where this repository records a published reading it has withdrawn"
    )
    subsection = text.split("#### 10.31.1")[1].split("### 10.32")[0]
    assert "An audit of the frames themselves" in subsection, (
        "the audit belongs inside 10.31.1, beside the table whose reading it corrects"
    )
    manifest = (REPO / "results" / "MANIFEST.md").read_text(encoding="utf-8")
    assert "`pixel_frame_probe_metrics.json`" in manifest
