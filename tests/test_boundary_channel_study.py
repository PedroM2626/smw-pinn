r"""Gates for README 10.59: the generated section, and the claims its prose makes.

The section reports a refutation, so the gates cut both ways. The channel's failure is re-derived
from the artifact's own sweep (no margin may come close to closing the residue, or the prose that
says it failed is wrong); the pause explanation is re-derived from the recording rather than
believed - that every repeating frame is an exception, that the WRAM CRC moved on every one of them,
and that dropping them improves the identity on all seven recordings. The camera address the WRAM
map now publishes is checked against the scan artifact that identified it, because an address in a
memory map with no evidence behind it is the mistake this section exists to avoid.

CPU-only, numpy-only; skipped when the study or its recording has not been produced.

Run:  pytest tests/test_boundary_channel_study.py -q
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.environment import wram
from src.evaluation.boundary_channel_study import (
    CLASSES,
    EXACT_TOLERANCE_SUBPIXELS,
    MARGINS_PX,
    committed_repeat_report,
    render_section,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "boundary_channel_metrics.json"
SCAN_ARTIFACT = REPO / "results" / "scroll_address_scan_metrics.json"
RESIDUE_ARTIFACT = REPO / "results" / "residue_process_metrics.json"
RECORDING = REPO / "data" / "raw" / "smw_boundary_dataset.npz"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the 10.59 boundary-channel study has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme():
    return (REPO / "README.md").read_text(encoding="utf-8")


def _recording():
    if not RECORDING.is_file():
        pytest.skip("the boundary recording has not been captured")
    with np.load(RECORDING, allow_pickle=False) as blob:
        return {key: np.asarray(blob[key], dtype=np.float64) for key in blob.files} | {
            "episodes": np.asarray(blob["episodes"])
        }


def test_readme_10_59_is_the_generated_block() -> None:
    """Not one number in the section is typed."""
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.59 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.59 "), "the generator must start at the section heading"


def test_the_channel_does_not_close_the_residue() -> None:
    """The refutation, re-derived: no margin comes near, and the identity barely moves."""
    summary = _payload()["summary"]
    sweep = summary["sweep"]
    assert [row["margin_px"] for row in sweep] == [float(m) for m in MARGINS_PX]
    assert max(row["share_of_all_exceptions"] or 0.0 for row in sweep) < 0.05, (
        "the prose says the boundary channel explains almost none of the exceptions; if a margin "
        "starts covering real shares of them, the prediction has to be re-scored, not the sentence kept"
    )
    verdict = _payload()["verdict"]
    assert not verdict["prediction_holds"]
    gain = verdict["exact_gain_from_the_channel"]
    assert 0.0 <= gain < 0.01, (
        f"the channel moved the identity by {gain:.4f}, which is not a refutation"
    )
    assert verdict["exact_gain_from_the_pause"] > 20 * max(gain, 1e-9), (
        "the pause explanation is only a finding if it dominates the channel's contribution"
    )


def test_every_exception_frame_is_labelled_exactly_once() -> None:
    """The accounting is exhaustive over the recording, recomputed here rather than trusted."""
    data = _recording()
    state, nxt = data["states"], data["next_states"]
    episodes = data["episodes"]
    adjacent = np.zeros(state.shape[0], dtype=bool)
    adjacent[:-1] = episodes[:-1] == episodes[1:]
    residue = 16.0 * (nxt[:, 0] - state[:, 0]) - state[:, 2]
    exc = adjacent & (np.abs(residue) >= EXACT_TOLERANCE_SUBPIXELS)
    acc = _payload()["summary"]["accounting"]
    assert acc["n_exceptions"] == int(exc.sum())
    assert sum(acc["classes"].values()) == acc["n_exceptions"]
    assert acc["classes_sum_equals_exceptions"]
    assert tuple(acc["classes"]) == CLASSES, (
        "the table's row order is the order the labels are tested in"
    )


def test_a_repeating_player_record_is_always_an_exception_and_the_console_always_moved() -> None:
    """The two halves of the mechanism claim, straight from the recorded channels."""
    data = _recording()
    state, nxt = data["states"], data["next_states"]
    episodes = data["episodes"]
    adjacent = np.zeros(state.shape[0], dtype=bool)
    adjacent[:-1] = episodes[:-1] == episodes[1:]
    residue = 16.0 * (nxt[:, 0] - state[:, 0]) - state[:, 2]
    exc = adjacent & (np.abs(residue) >= EXACT_TOLERANCE_SUBPIXELS)
    repeat = adjacent & np.all(np.abs(nxt - state) < 1e-9, axis=1)
    assert repeat.any()
    assert float(exc[repeat].mean()) == 1.0, (
        "the section says a repeat *is* an exception, because a body whose velocity byte holds while "
        "its position does not move violates §4.1 by construction"
    )
    moved = data["wram_crc"] != data["next_wram_crc"]
    live = _payload()["summary"]["liveness"]
    assert live["wram_identical_frames"] == int((repeat & ~moved).sum())
    assert live["share_of_repeats_where_wram_changed"] == 1.0, (
        "if the CRC ever stands still, some of these frames are the harness sampling one emulated "
        "frame twice, and the pause is partly an artifact of the recorder rather than of the engine"
    )
    assert set(live["modes_seen"]) == {0x14}, (
        "the repeats are claimed to happen inside interactive mode; a mode change would be a "
        "different mechanism and the limitations paragraph would have to say so"
    )


def test_dropping_the_repeats_improves_the_identity_on_every_recording() -> None:
    """The correction to 10.58 is measured on the six recordings too, not just the new one."""
    committed = committed_repeat_report()
    assert len(committed) == 6, "the section quotes all six committed recordings"
    residue = (
        json.loads(RESIDUE_ARTIFACT.read_text(encoding="utf-8"))
        if RESIDUE_ARTIFACT.is_file()
        else {}
    )
    for name, row in committed.items():
        assert row["exact_rate_off_repeats"] > row["exact_rate_all"], name
        assert row["share_of_exceptions_repeating"] > 0.5, name
    if residue:
        published = {
            name: block["lattice"]["rate_exactly_zero"]
            for name, block in residue["per_recording"].items()
        }
        for name, value in published.items():
            assert abs(committed[name]["exact_rate_all"] - value) < 1e-3, (
                f"{name}: 10.58's lattice rate and this section's identity disagree ({value} vs "
                f"{committed[name]['exact_rate_all']}), so one of the two is measuring a different pair"
            )


def test_the_camera_address_in_the_map_is_the_one_the_scan_chose() -> None:
    """`ADDR_CAMERA_X` is a published claim about the engine, so it is gated to its evidence."""
    if not SCAN_ARTIFACT.is_file():
        pytest.skip("the scroll scan has not been run")
    scan = json.loads(SCAN_ARTIFACT.read_text(encoding="utf-8"))
    assert scan["chosen"] == f"$7E:{wram.ADDR_CAMERA_X:04X}", (
        "the WRAM map and the artifact that identified the address disagree; the map is what every "
        "recorder reads, so this is the one place a silent mismatch would corrupt future data"
    )
    assert scan["parallax_sibling"]["addr_int"] == wram.ADDR_CAMERA_X + 4, (
        "the section says the word four bytes away holds exactly half the camera; if the map's "
        "address moves and the sibling's does not, one of the two is wrong"
    )
    emulator = (REPO / "src" / "environment" / "snes_emulator.py").read_text(encoding="utf-8")
    body = emulator.split("def get_camera_x")[1].split("def get_active_sprites")[0]
    assert "ADDR_CAMERA_X" in body, "get_camera_x must read the gated constant, not a literal"
    passing = {entry["addr"] for entry in scan["candidates"] if entry["passes"]}
    assert scan["chosen"] in passing
    assert all(
        entry["axioms"]["never_ahead_of_mario"] for entry in scan["candidates"] if entry["passes"]
    )


def test_the_terrain_label_is_quoted_with_its_fire_rate() -> None:
    """Coverage without the base rate is the defect this section is warning about."""
    acc = _payload()["summary"]["accounting"]
    assert acc["fire_rates"]["wall_ahead"] > 0.3, (
        "the prose calls the terrain label a non-detector because it fires on most frames; if that "
        "stops being true the label deserves a second look rather than the same sentence"
    )
    text = _readme()
    assert "| The label fires on |" in text, (
        "the accounting table must carry the base rate beside coverage"
    )


def test_section_12_and_the_manifest_record_the_new_artifacts() -> None:
    """A study that is not indexed is a study nobody can find again."""
    text = _readme()
    manifest = (REPO / "results" / "MANIFEST.md").read_text(encoding="utf-8")
    for artifact in (ARTIFACT.name, SCAN_ARTIFACT.name):
        assert f"`{artifact}`" in manifest, f"results/MANIFEST.md does not list {artifact}"
    assert "10.59" in text.split("## 12. Scientific Integrity Statement")[1], (
        "section 12 carries the repository's corrections, and this one is a withdrawn interpretation"
    )


def test_the_recording_is_not_newer_than_the_study_that_quotes_it() -> None:
    """The freshness rule of MANIFEST.md, applied to a dataset instead of a checkpoint."""
    if not (REPO / ".git").is_dir():
        pytest.skip("no git history to order")

    def stamp(path: str) -> int:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", path],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        return int(proc.stdout.strip()) if proc.stdout.strip() else -1

    dataset = stamp("data/raw/smw_boundary_dataset.npz")
    artifact = stamp("results/boundary_channel_metrics.json")
    if dataset < 0 or artifact < 0:
        pytest.skip("the recording or its artifact is not committed yet")
    assert dataset <= artifact, (
        "the recording was re-captured after the study ran, so README 10.59 quotes an artifact that "
        "describes the previous file - re-run `python -m src.evaluation.boundary_channel_study`"
    )
