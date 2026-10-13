r"""Gates for the measured Map16 behaviour table (README 10.68).

The table in `src.environment.tile_behavior_table.py` is not a list of guesses: every entry is
what the engine itself did when `scripts/probe_tile_behaviors` wrote that id into the level buffer
and read the console's own collision, death, coin and freeze responses. These gates hold the
pipeline together: the baked module may not drift from the measured artifact, the classes the
section quotes may not drift from the measurements, and the guesses the old hardcoded classifier
made - bushes as pipes, ramps as air, half of every tile number - stay named as corrected.

CPU-only, no emulator: the probe itself is hardware-gated elsewhere.

Run:  pytest tests/test_tile_behavior.py -q
"""

import json
from pathlib import Path

from src.environment import tile_behavior
from src.environment.tile_behavior_table import BEHAVIOR

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "tile_behavior_probe_metrics.json"


def _artifact():
    if not ARTIFACT.is_file():
        raise AssertionError("results/tile_behavior_probe_metrics.json has not been generated")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_baked_table_is_the_probe_artifact() -> None:
    """Every class the runtime classifies through is the class the probe measured."""
    measured = {int(r["tile"], 16): r["class"] for r in _artifact()["tiles"]}
    assert len(measured) == 0x200, "the probe covers all 512 standard tile ids"
    assert BEHAVIOR == measured, (
        "src/environment/tile_behavior_table.py disagrees with the artifact: re-run "
        "`python -m scripts.probe_tile_behaviors --emit-table`"
    )


def test_the_classes_the_section_quotes() -> None:
    """The headline ids of 10.68, gated so the prose cannot drift from the measurement."""
    assert tile_behavior.behavior_of(0x0025) == "decor", "the blank filler"
    assert tile_behavior.behavior_of(0x0100) == "platform", (
        "the grass top: holds, never blocks a side"
    )
    assert tile_behavior.behavior_of(0x0124) == "solid", "the wall the published episode brushed"
    assert tile_behavior.behavior_of(0x0173) == "slope", "a walkable 45-degree ramp"
    assert tile_behavior.behavior_of(0x013F) == "harmful", "the spikes that kill a small Mario"
    assert tile_behavior.behavior_of(0x002B) == "coin", "the coin the counter paid"


def test_the_old_hardcoded_guesses_were_wrong() -> None:
    """The corrected record: what the pre-10.68 id lists called these tiles, and what they are.

    The old `TILE_SOLID` list read only the low byte of every tile number, so it named the fill
    0x003F and "the pipes 0x73-0x76". Measured: the fill is decoration the engine ignores, those
    low bytes belong to the bush pieces, and the ramps - whose ids the old list could not see -
    are 0x0173-0x0176.
    """
    assert tile_behavior.behavior_of(0x003F) == "decor", "the earth fill: no collision at all"
    assert tile_behavior.behavior_of(0x0073) == "decor", "the bush the old list called a pipe"
    assert tile_behavior.behavior_of(0x0074) == "decor"
    for ramp in (0x0173, 0x0174, 0x0175, 0x0176):
        assert tile_behavior.behavior_of(ramp) == "slope", (
            "the ids behind the old 'pipe' guess are the walkable ramps"
        )


def test_the_patch_vocabulary_folds_the_measured_classes() -> None:
    """The level models' four classes, derived rather than hand-listed."""
    assert tile_behavior.patch_class(0x0025) == tile_behavior.PATCH_AIR
    assert tile_behavior.patch_class(0x0049) == tile_behavior.PATCH_AIR, (
        "the bush the old map painted solid"
    )
    assert tile_behavior.patch_class(0x002B) == tile_behavior.PATCH_AIR, (
        "a coin is passable terrain"
    )
    assert tile_behavior.patch_class(0x0100) == tile_behavior.PATCH_SOLID, (
        "a platform holds from above"
    )
    assert tile_behavior.patch_class(0x0124) == tile_behavior.PATCH_SOLID, "a wall blocks the side"
    assert tile_behavior.patch_class(0x013F) == tile_behavior.PATCH_HAZARD, "the spikes"
    assert tile_behavior.patch_class(0x0173) == tile_behavior.PATCH_SLOPE


def test_an_id_the_table_does_not_know_is_reported_solid() -> None:
    """Per-level ids at 0x200 and above are unmeasured, and unmeasured is not 'air'."""
    assert tile_behavior.behavior_of(0x0233) == "solid"
    assert tile_behavior.patch_class(0x0233) == tile_behavior.PATCH_SOLID


def test_every_measured_class_is_represented() -> None:
    """The probe saw all six non-blank classes; a class that stops being measured stops existing."""
    counts = _artifact()["class_counts"]
    assert set(counts) == {"decor", "coin", "solid", "platform", "slope", "harmful"}
    assert counts["harmful"] >= 1, "the spikes are the one hazard class the panel draws red"
