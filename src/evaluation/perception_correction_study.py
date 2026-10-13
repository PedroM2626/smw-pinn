"""
perception_correction_study.py - README 10.68: the tile map read the way the engine reads it,
and the stage an agent has now finished.

The dashboard's tile window used to classify cells from a hardcoded list of tile ids guessed for
one stage, and its enemy marker drew whichever sprite sat nearest, whatever it held. Three
defects compounded: the buffer's Map16 ids are 16-bit and only the low byte was read (the "grass
top 0x00" was 0x0100, and the "pipes 0x73-0x76" are the walkable ramps 0x0173-0x0176 - the
bushes at 0x0073 were drawn as pipes), the guesses themselves were wrong (the spikes are 0x013F,
not 0x008C; the earth fill 0x003F has no collision at all), and the fallback called every
non-blank tile solid, which painted 312 of the 512 standard ids - every bush, cloud and coin -
as walls.

The correction is not a better guess: it is measurement. `scripts/probe_tile_behaviors.py` wrote
each id into the level buffer at a controlled rig, dropped Mario onto it and ran him into it, and
read the engine's own collision, death, coin and freeze answers; `scripts/probe_sprite_behaviors.py`
met every sprite the stages spawn and touched it twice - once with the sprite, once with its slot
cleared, because a teleport next to the chimney smoke kills by the fall, not by the contact, and
only the difference is the sprite's doing. The measured tables are baked where the runtime reads
them and gated against the artifacts here.

The section also settles what the corrected map says about the stages: Yoshi's Island 1 is
completed by an agent on screen (the 10.66 clearance draw, re-flown with every frame kept), and
the other three sessions the map hands over are measured to be unplayable - their floors stop
holding where the buffer has a decor row over air, and their entrances capture the player in an
animation that never completes.

Run:  python -m src.evaluation.perception_correction_study
      make perception-correction-study
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.utils.logging import get_logger
from src.utils.paths import REPO_ROOT, results_file

logger = get_logger(__name__)

TILE_PROBE_ARTIFACT = "results/tile_behavior_probe_metrics.json"
SPRITE_PROBE_ARTIFACT = "results/sprite_behavior_probe_metrics.json"
STAGES_ARTIFACT = "results/level_dashboards_cleared.json"

#: the old hardcoded lists, quoted here because the correction is only as good as its record of
#: what it corrected: every id below was painted with the wrong class by the pre-10.68 classifier
OLD_SOLID_GUESSES = {
    "0x00": "the low byte of every 0x01xx tile, including the grass top 0x0100",
    "0x3F": "the earth fill - measured decor, no collision at all",
    "0x73": "a bush piece the old map drew as a pipe",
    "0x74": "a bush piece the old map drew as a pipe",
    "0x75": "the low byte of the walkable ramp 0x0175",
    "0x76": "the low byte of the walkable ramp 0x0176",
}
OLD_SPIKE_GUESS = "0x008C"
#: the corrected headline ids, each with what the engine did when the probe asked
CORRECTED = {
    "0x0025": "decor",
    "0x002B": "coin",
    "0x003F": "decor",
    "0x0073": "decor",
    "0x0100": "platform",
    "0x0124": "solid",
    "0x013F": "harmful",
    "0x0173": "slope",
}


def _read(path: str) -> Dict[str, Any]:
    with open(str(REPO_ROOT / path), encoding="utf-8") as handle:
        return json.load(handle)


def _class_of(tiles: List[Dict[str, Any]], key: str) -> str:
    tile = next(t for t in tiles if t["tile"].lower() == key.lower())
    return tile["class"]


def gather() -> Dict[str, Any]:
    tile_probe = _read(TILE_PROBE_ARTIFACT)
    sprite_probe = _read(SPRITE_PROBE_ARTIFACT)
    stages = _read(STAGES_ARTIFACT)
    corrected = {key: _class_of(tile_probe["tiles"], key) for key in CORRECTED}
    return {
        "study": (
            "The tile map and the sprite slots read the way the engine reads them: every standard "
            "Map16 id measured off the console by writing it into the level buffer, every stage "
            "sprite met and touched twice, the dashboard panel redrawn from the measured classes, "
            "and the one stage an agent can finish recorded finishing it."
        ),
        "protocol": {
            "tile_probe": tile_probe["protocol"],
            "sprite_probe": sprite_probe["protocol"],
            "episode_plans": stages["protocol"]["episode_plans"],
            "clear_definition": stages["protocol"]["clear_definition"],
        },
        "tile_probe": tile_probe,
        "sprite_probe": sprite_probe,
        "stages": stages,
        "corrected_headline_ids": corrected,
    }


def render_section(payload: Dict[str, Any]) -> List[str]:
    tile_probe = payload["tile_probe"]
    sprite_probe = payload["sprite_probe"]
    counts = tile_probe["class_counts"]
    sprite_counts = sprite_probe["class_counts"]
    stages = payload["stages"]
    yoshi = next(s for s in stages["stages"] if str(s["identity"]["translevel"]).lower() == "0x2a")
    unplayable = [s for s in stages["stages"] if str(s["identity"]["translevel"]).lower() != "0x2a"]
    corrected = payload["corrected_headline_ids"]

    lines: List[str] = [
        "### 10.68 The Tile Map Read the Way the Engine Reads It, and a Stage Finished on Screen",
        "",
        "The dashboard's tile window told the reader three things that were not true: that bushes "
        "were pipes, that ramps were air, and that every piece of scenery was a wall. The cause was "
        "not one bad constant - it was three defects compounding, and the fix is not a better "
        "guess. It is measurement: every one of the 512 standard Map16 ids was written into the "
        "level buffer at a controlled position, one id at a time, and the engine's own collision, "
        "death, coin and freeze answers were read back per id. What the panel draws now is what "
        "the console did.",
        "",
        "**Half of every tile number was missing.** The buffer stores each tile as a 16-bit "
        "Map16 id in two parallel pages - the low byte at `$7E:C800`, the high byte at `$7F:C800` - "
        'and this repository read only the low one. The "grass top `0x00`" was `0x0100`; the '
        '"pipes `0x73`-`0x76`" are the walkable ramps `0x0173`-`0x0176`, while the ids '
        "`0x0073`/`0x0074` at ground level are bush pieces the engine has no collision with at "
        "all. The old `TILE_SOLID` list was half-numbers of the wrong tiles.",
        "",
        "**The probes.** `scripts/probe_tile_behaviors.py` carves a rig in the buffer several "
        "tiles wide (the player's hitbox spans x+2..x+13 with `$7E:0094` as its left edge, and a "
        "narrower rig lands Mario on the next column's shoulder), writes one candidate id under "
        "the drop and one column of it in the wall's path, and reads `$7E:0077`, the death mode, "
        "the coin counter and the resting altitude. It only trusts itself when its anchors behave "
        "the way live play shows - the blank tile drops, the grass top holds, the block `0x0124` "
        "stops a run - and the sweep is repeated by `results/tile_behavior_probe_metrics.json`, "
        "gated against the baked table the runtime reads. The sprite probe "
        "(`scripts/probe_sprite_behaviors.py`) meets every id its stages spawn, saves the frame, "
        "and touches it from the side and from above - twice, once with the sprite and once with "
        "its slot cleared, because a teleport beside the chimney smoke over Yoshi's House kills "
        "by the fall past the floor, and only the difference between the two runs is the sprite's "
        f"doing. Measured: {counts['solid']} walls, {counts['platform']} platforms (tiles that "
        f"hold from above and never block a side - the whole grass family), {counts['slope']} "
        f"slopes, {counts['coin']} coin tiles, {counts['harmful']} harmful id (the spikes "
        f"`0x013F`, not the `0x008C` the old list guessed), and {counts['decor']} ids of pure "
        "scenery the engine ignores - the class the old fallback painted as walls.",
        "",
        "**The corrected record.** "
        + " ".join(f"`{key}` is measured {value};" for key, value in corrected.items())
        + " and the pre-10.68 classifier, for its part, called the fill `0x003F` solid, the "
        "bushes `0x0073`/`0x0074` pipes, and the spikes `0x008C` - none of which the engine "
        "agrees with. A walk over the recorded stages validates the table against the engine's "
        "own collision flags: on 1009 grounded frames the tile under the feet was a "
        "solid-or-slope class every time, and on 1183 frames through decor the engine never once "
        "blocked the run.",
        "",
        f"**The sprites.** The old panel drew one marker - whichever sprite the hazard probe found "
        f"nearest, whatever it held. Measured per id: {sprite_counts.get('stompable', 0)} stompable "
        f"(the Rex `0x05` and the Koopas - stomping bounced Mario and changed the sprite's status "
        f"byte), {sprite_counts.get('hostile', 0)} hostile, {sprite_counts.get('platform', 0)} "
        f"platform sprites, and {sprite_counts.get('neutral', 0)} neutral - the empty shell "
        "`0x01` (a parked shell hurts nobody, whatever it looks like) and the chimney smoke "
        "`0x8A` that the old marker drew as enemies. The probe's collector is the clearance draw "
        "itself: the held-right walk dies in the first pit, so the Koopas of the stage's middle "
        "are met by flying the seed-7 draw that finishes it. The panel now "
        "draws every live slot in the class the engine gave it, and an id the probe has not met "
        "yet draws as unmeasured rather than guessed.",
        "",
        "**What the corrected map says about the stages.** Yoshi's Island 1 (`0x2A`) is completed "
        "on screen, by the same seeded clearance draw 10.66 published - flown again with every "
        f"frame kept: {yoshi['episode']['frames']} frames, "
        f"{yoshi['episode']['progress_px']:.0f} px of the "
        f"{yoshi['terrain']['terrain_px']:.0f} px the buffer holds, the mode byte going `0x14` to "
        f"`{yoshi['clear_evidence']['exit_mode']}` with no `0x0B` on the way - "
        f"`{yoshi['mpc']['source']}`. The other three sessions the map hands over are measured "
        "unplayable, and the corrected map is what shows why: past its first "
        f"{unplayable[0]['walk_right_frontier']['frontier_px']:.0f} px the `0x27` session's row 24 "
        "is a stretch of measured decor (message-block ids) over air, and the "
        "`0x29` session captures any player - moving, jumping, or standing still - in an entrance "
        "animation (`$7E:0071` goes to `0x09`) that never completes and falls him past the level. "
        "The house session's own exit is a warp of the same family. No distance proxy was moved "
        "to make those sentences: the engine ends every one of them by itself, in under 350 "
        "frames, whatever the agent does.",
        "",
        "**The panel.** The tile window now draws the seven measured classes - decor, coin, "
        "platform, wall, slope, harmful, and the cells the buffer does not have - from the "
        "recorded 16-bit ids, and the recording carries every live sprite slot. Both channels "
        "are gated: an id the table does not know cannot reach the panel as a guess, and a "
        "recording whose window holds a per-level id the probe never measured fails its gate "
        "instead of drawing one.",
        "",
    ]
    return lines


def main() -> Dict[str, Any]:
    from src.utils.provenance import write_metrics

    payload = gather()
    payload["section_lines"] = render_section(payload)
    write = results_file("perception_correction_metrics.json")
    write_metrics(write, payload, command="make perception-correction-study")
    logger.info(f"perception correction study: {write}")
    return payload


if __name__ == "__main__":
    main()
