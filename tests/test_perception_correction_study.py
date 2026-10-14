r"""Gates for README 10.68: the measured behaviour tables and the stage finished on screen.

The section's three claims that a careless test would survive: that the baked tables are the
probe artifacts (a runtime table that drifted from the measurement is a guess with provenance),
that the anchors of the measurement protocol held when the probe ran, and that the stage the
section says was finished was finished by the engine's own mode byte - not by a distance proxy
someone moved.

CPU-only, json only; skipped when the probes have not been run.

Run:  pytest tests/test_perception_correction_study.py -q
"""

import json
from pathlib import Path

import pytest

from src.environment import wram
from src.evaluation.perception_correction_study import gather, render_section

REPO = Path(__file__).resolve().parents[1]
TILE_PROBE = REPO / "results" / "tile_behavior_probe_metrics.json"
SPRITE_PROBE = REPO / "results" / "sprite_behavior_probe_metrics.json"
STAGES = REPO / "results" / "level_dashboards_cleared.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"{path.name} has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_readme_10_68_is_the_generated_block() -> None:
    """Not one number in the section is typed."""
    lines = render_section(gather())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.68 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.68 "), "the generator must start at the section heading"


def test_the_probe_anchors_held_when_the_table_was_measured() -> None:
    """The sweep only trusts itself when live play agrees about its three anchor tiles."""
    payload = _load(TILE_PROBE)
    anchors = payload["protocol"]["anchors"]
    assert anchors["0x0025"] == "decor", "the blank tile must let the drop through"
    assert anchors["0x0100"] == "platform", "the grass top must hold from above"
    assert anchors["0x0124"] == "solid", "the wall the published episode brushed must block"


def test_the_measured_table_covers_every_standard_id() -> None:
    payload = _load(TILE_PROBE)
    tiles = payload["tiles"]
    assert len(tiles) == 0x200
    assert payload["protocol"]["ids_probed"] == 0x200
    assert sum(payload["class_counts"].values()) == 0x200


def test_the_old_guesses_are_named_as_corrected() -> None:
    """The record of what the pre-10.68 classifier called wrong, gated so it stays honest."""
    payload = _load(TILE_PROBE)
    table = {t["tile"]: t["class"] for t in payload["tiles"]}
    assert table["0x003f"] == "decor", "the earth fill was called solid; the engine ignores it"
    assert table["0x0073"] == "decor", "the bush was called a pipe"
    assert table["0x008c"] == "decor", "the old spike guess is not the spikes"
    assert table["0x013f"] == "harmful", "the measured spikes"
    for ramp in ("0x0173", "0x0174", "0x0175", "0x0176"):
        assert table[ramp] == "slope", "the ids behind the old 'pipe' guess are the ramps"


def test_every_measured_sprite_class_is_reported() -> None:
    payload = _load(SPRITE_PROBE)
    counts = payload["class_counts"]
    assert counts.get("stompable", 0) >= 1, "the Rex is measured stompable"
    for record in payload["sprites"]:
        if record["sprite"] == "0x0005":
            assert record["class"] == "stompable", "the Rex is the headline stompable id"
        if record["sprite"] == "0x008a":
            assert record["class"] == "neutral", "the chimney smoke is the headline neutral id"


def test_the_finished_stage_was_finished_by_the_engine() -> None:
    """The 0x2A episode's verdict is the mode byte, and the byte is on the recording."""
    stages = _load(STAGES)
    yoshi = next(s for s in stages["stages"] if str(s["identity"]["translevel"]).lower() == "0x2a")
    assert yoshi["episode"]["controller"] == "mpc"
    assert yoshi["episode"]["planner_seed"] == 7, "the 10.66 clearance draw itself"
    assert yoshi["clear_evidence"]["concluded"]
    assert yoshi["clear_evidence"]["exit_mode"] == "0x0C", "0x14 -> 0x0C with no 0x0B on the way"
    assert not yoshi["clear_evidence"]["lives_lost"], "a clear does not take a life"


def test_every_session_ends_with_the_engines_own_verdict() -> None:
    """Each of the four sessions is recorded to the verdict the engine itself delivered.

    The verdict is two bytes now: the mode byte and the lives counter. The house's walk out the
    door passes through the death animation's own byte on its way to the map, so the byte alone
    would call an exit a death - the lives counter settles it, and the two measured-dead
    sessions show the counter doing the other job.
    """
    stages = _load(STAGES)
    by_id = {str(s["identity"]["translevel"]).lower(): s for s in stages["stages"]}
    assert len(by_id) == 4
    house = by_id["0x28"]
    assert house["episode"]["controller"] == "terrain-runner", "the house is walked out, not flown"
    assert house["clear_evidence"]["concluded"], house["clear_evidence"]["reason"]
    assert house["clear_evidence"]["reached_map"], "the walk out the door ends on the map"
    assert house["episode"]["lives_before"] == house["episode"]["lives_after"], (
        "the exit marches through 0x0B, the death animation's own byte - a life that stays put is "
        "what separates that march from a death"
    )
    assert wram.GAME_MODE_WORLD_MAP in house["episode"]["mode_bytes_seen"]
    for dead in ("0x27", "0x29"):
        session = by_id[dead]
        assert not session["clear_evidence"]["concluded"]
        assert session["episode"]["lives_after"] != session["episode"]["lives_before"], (
            f"{dead}: the measured death must show the engine taking a life, or the verdict is a "
            "guess about the byte it never read"
        )
        assert session["clear_evidence"]["reason"]


def test_the_conclusion_reel_exists_and_is_carried() -> None:
    """The single video with every session: on disk, inside what a README can carry, referenced."""
    payload = _load(REPO / "results" / "level_dashboard_study_metrics.json")
    reel = payload.get("conclusion_video") or {}
    assert reel.get("mp4"), "the study was gathered without stitching the reel"
    file = REPO / reel["mp4"]
    assert file.is_file(), f"{file} is quoted by the README and not committed"
    assert file.stat().st_size < 10_000_000, "the reel is over GitHub's per-file comfort limit"
    assert len(reel["segments"]) == 4, "one segment per session the map hands over"
    assert reel["mp4"] in _readme(), "the README must link the reel, not merely own it"
