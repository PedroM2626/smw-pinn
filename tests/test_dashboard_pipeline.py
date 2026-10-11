r"""Gates for README 10.60: the hero block, the generated section, and the recording behind both.

The dashboard is the first artifact in this repository whose purpose is to be looked at, so the
gates refuse the easy failure modes: the hero caption and the section are compared line by line
against their generators (a typed number in a caption is exactly the defect this repository has
been fixing), the video files must exist and stay inside what a README can carry, and the claims the
prose makes about the episode - that the held frames are the residue exceptions, that the operator
pressed a button no policy has, that the selected arm is the one that traversed - are re-derived
from the recording rather than believed.

CPU-only, numpy-only; skipped when the dashboard pipeline has not been run.

Run:  pytest tests/test_dashboard_pipeline.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.dashboard_study import render_caption, render_section
from src.planning.mpc_planner import ACTION_BUTTONS

REPO = Path(__file__).resolve().parents[1]
STUDY_ARTIFACT = REPO / "results" / "dashboard_study_metrics.json"
RUN_ARTIFACT = REPO / "results" / "dashboard_run_metrics.json"
SELECTION_ARTIFACT = REPO / "results" / "dashboard_controller_selection.json"
RECORDING = REPO / "data" / "raw" / "smw_dashboard_run.npz"


def _payload():
    if not STUDY_ARTIFACT.is_file():
        pytest.skip("the dashboard study has not been run")
    return json.loads(STUDY_ARTIFACT.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def _recording():
    if not RECORDING.is_file():
        pytest.skip("the dashboard episode has not been recorded")
    with np.load(RECORDING, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def test_readme_10_60_is_the_generated_block() -> None:
    """Not one number in the section is typed."""
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.60 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.60 "), "the generator must start at the section heading"


def test_readme_hero_is_the_generated_caption_at_the_top() -> None:
    """The caption a reader meets first is the one a gate can check, and it is above the abstract."""
    lines = render_caption(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "the README hero disagrees with the generator:\n" + "\n".join(missing[:4])
    hero = text.index(lines[0])
    assert hero < text.index("## 🏆 Direct Answer"), "the hero belongs at the top of the file"


def test_the_hero_media_exists_and_is_a_readme_can_carry() -> None:
    """A GIF that is too big for the page is the same failure as a missing one."""
    payload = _payload()
    for path in (payload["video"]["mp4"], payload["gif"]["gif"]):
        file = Path(path)
        if not file.is_absolute():
            file = REPO / path
        assert file.is_file(), f"{file} is referenced by the README and not committed"
        assert file.stat().st_size < 10_000_000, f"{file} is over GitHub's per-file comfort limit"
    rel = payload["gif"]["gif"].replace("\\", "/")
    assert f"]({rel})" in _readme(), "the hero must embed the GIF, not merely mention it"


def test_the_recording_carries_every_channel_the_dashboard_draws() -> None:
    data = _recording()
    for key in (
        "states",
        "button_probs",
        "executed",
        "values",
        "camera",
        "patches",
        "rgb",
        "live",
        "residue_subpixels",
        "operator",
        "mode",
    ):
        assert key in data, f"the panel set of 10.60 needs a {key} channel"
    n = data["states"].shape[0]
    assert data["states"].shape == (n, 12)
    assert data["button_probs"].shape == (n, len(ACTION_BUTTONS))
    assert data["rgb"].shape[1:] == (224, 256, 3), "the frame must be full width, not half"
    assert data["patches"].shape[1:] == (33, 33)
    # A button-headed actor emits six independent sigmoids, so the channel that must be normalised
    # is a primitive head's, and this recording has one only when the PPO arm was the one recorded.
    probs = np.asarray(data["button_probs"], dtype=np.float64)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert (probs.max(axis=1) >= 0.5).all(), "the caption claims no frame had no likely button"
    if "primitive_probs" in data:
        np.testing.assert_allclose(
            np.asarray(data["primitive_probs"], dtype=np.float64).sum(axis=1), 1.0, atol=1e-4
        )
    assert set(np.unique(data["executed"]).tolist()) <= {0.0, 1.0}


def test_the_selected_arm_is_the_one_that_traversed() -> None:
    """The rule is stated before the run; this re-runs it on the published probe."""
    selection = json.loads(SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    arms = selection["arms"]
    best = max(arms, key=lambda a: a["max_progress_px"])
    assert selection["selected"] == best["arm"] == "dagger"
    ppo = next(a for a in arms if a["arm"] == "ppo")
    assert ppo["progress_px"] < 0.0, "the section's claim is that the PPO arm walks backwards"
    assert ppo["mean_top_probability"] < 0.5, "and that its distribution has not decided anything"
    control = selection["override_control"][0]
    assert control["published_overrides"] is False
    assert control["progress_px"] > best["max_progress_px"], (
        "10.60 says the published override layer costs the actor progress; if a re-run flips that "
        "sign the sentence, not the number, has to change"
    )


def test_held_frames_are_the_residue_exceptions() -> None:
    """The section's central reading, re-derived: on a held frame the identity reads minus the velocity.

    Two edges keep this from being a set equality. The frame a hold releases on steps and moves, so
    its residue is not minus the velocity it held; and 10.59's whole-pixel repositions stay among
    the frames where the player did move.
    """
    data = _recording()
    held = data["live"] < 0.5
    residue = np.asarray(data["residue_subpixels"], dtype=np.float64)
    vx = np.asarray(data["states"], dtype=np.float64)[:, 2]
    disagree = held & (np.abs(residue + vx) > 1e-4)
    assert int(disagree.sum()) <= 1, (
        f"{int(disagree.sum())} held frames do not read minus the velocity; the most one can is the "
        "frame a hold releases on"
    )
    live_exceptions = (~held) & (np.abs(residue) > 1e-6)
    share = float(live_exceptions.sum()) / float((~held).sum())
    assert share < 0.01, f"{share:.4f} of moving frames break the identity, which is not a rounding"


def test_the_operator_pressed_a_button_no_policy_has() -> None:
    """A is outside the action set, which is the whole point of the operator column."""
    assert "A" not in ACTION_BUTTONS
    data = _recording()
    operator = np.asarray(data["operator"], dtype=np.float64)
    held = data["live"] < 0.5
    assert operator.sum() > 0.0, "the published episode has a takeover in it"
    assert np.all(held[operator > 0.5]), "an operator frame must be a held frame"
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    assert run["run"]["frames_under_operator_A"] == int(operator.sum())
    for event in run["pauses"]:
        assert event["resumed"], (
            "10.60 describes every pause as cleared; an uncleared one rewrites it"
        )


def test_the_video_numbers_are_the_recording_numbers() -> None:
    """The caption quotes the render, the render quotes the recording; both must agree."""
    payload = _payload()
    data = _recording()
    states = np.asarray(data["states"], dtype=np.float64)
    progress = states[:, 0] - states[0, 0]
    caption = payload["caption"]
    assert caption["episode_frames"] == states.shape[0]
    assert caption["progress_px"] == pytest.approx(float(progress[-1]), abs=1e-3)
    assert caption["residue_exact_share"] == pytest.approx(
        float(np.mean(np.abs(data["residue_subpixels"]) < 1e-6)), abs=1e-9
    )
    assert caption["operator_A_frames"] == int(np.asarray(data["operator"]).sum())
    assert caption["subscreens_reached"] == int(progress[-1] // 256) + 1


def test_the_frame_buffer_defect_is_recorded_as_a_fact() -> None:
    """10.60's second instrument: the core's request and the buffer's format must be on record."""
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    capture = run["protocol"]["frame_capture"]
    assert capture["requested_pixel_format"] == 2
    assert capture["decoded_as_pixel_format"] == 1
    assert capture["requested_pixel_format"] != capture["decoded_as_pixel_format"], (
        "if the core ever starts honouring its own request, the section's sentence about it is stale"
    )
