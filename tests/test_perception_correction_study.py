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
    assert yoshi["clear_evidence"]["cleared"]
    assert yoshi["clear_evidence"]["exit_mode"] == "0x0C", "0x14 -> 0x0C with no 0x0B on the way"


def test_the_unplayable_sessions_are_reported_with_evidence() -> None:
    """Every other session is measured unplayable, and its record says why."""
    stages = _load(STAGES)
    others = [s for s in stages["stages"] if str(s["identity"]["translevel"]).lower() != "0x2a"]
    assert len(others) == 3
    for stage in others:
        assert not stage["clear_evidence"]["cleared"]
        assert stage["clear_evidence"]["reason"], "a no without a measured reason is an opinion"
        assert stage["episode"]["frames"] <= 1268, (
            "the sessions end by themselves; an episode that ran its whole budget did not measure "
            "the engine's verdict"
        )
