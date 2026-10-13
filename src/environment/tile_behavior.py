"""
tile_behavior.py - classify a Map16 tile the way the engine was measured to treat it.

The behaviour of every standard tile id (0x000-0x1FF) was measured off the engine itself by
`scripts/probe_tile_behaviors`, which writes each id into the level buffer at a controlled rig
position and reads the engine's own collision, death, coin and freeze responses
(`results/tile_behavior_probe_metrics.json`). The measured table is baked into
`tile_behavior_table.py` and gated against the artifact by `tests/test_tile_behavior.py`, so
this module never guesses: what it says about a tile is what the console did.

Two vocabularies read the table:

* `behavior_of` - the measured classes themselves (``decor``, ``coin``, ``platform``, ``solid``,
  ``slope``, ``harmful``, ``interact``), which the dashboard panel draws one colour each;
* `patch_class` - the four-way terrain vocabulary the level-conditioned models were trained on
  (air / solid / hazard / slope), which the old hardcoded id lists used to fill with wrong
  guesses (bushes painted as pipes and walls, ramps painted as air).

A tile id this table does not know - a per-level id at 0x200 or above - is classified solid:
the recordings are gated to levels whose ids the table covers, so an unknown id at runtime is
an unmeasured cell, and the honest thing to show a model is the conservative class rather than
a confidently wrong one.
"""

from __future__ import annotations

from src.environment.tile_behavior_table import BEHAVIOR

#: the measured classes, in the order the artifact's protocol defines them
DECOR = "decor"
COIN = "coin"
PLATFORM = "platform"
SOLID = "solid"
SLOPE = "slope"
HARMFUL = "harmful"
INTERACT = "interact"

#: the four-way patch vocabulary the level models read: air, solid, hazard, slope
PATCH_AIR = 0
PATCH_SOLID = 1
PATCH_HAZARD = 2
PATCH_SLOPE = 3

#: measured classes the patch vocabulary folds into its air class: passable without danger
PATCH_AIR_CLASSES = frozenset({DECOR, COIN, INTERACT})
#: classes that hold the player up from above or block a run from the side
PATCH_SOLID_CLASSES = frozenset({SOLID, PLATFORM})

#: the blank tile: the level buffer's filler, measured decor
TILE_BLANK = 0x0025


def behavior_of(tile_id: int) -> str:
    """The measured behaviour class of one Map16 tile id.

    An id the table does not know (0x200+, or a hole in a probe run) is reported solid: the
    recordings are gated to ids the table covers, so anything else is unmeasured and gets the
    conservative class rather than a guess.
    """
    known = BEHAVIOR.get(int(tile_id) & 0x1FF if int(tile_id) < 0x200 else -1)
    return known if known is not None else SOLID


def patch_class(tile_id: int) -> int:
    """The terrain class of one tile in the level models' four-way vocabulary."""
    kind = behavior_of(tile_id)
    if kind is HARMFUL:
        return PATCH_HAZARD
    if kind is SLOPE:
        return PATCH_SLOPE
    if kind in PATCH_SOLID_CLASSES:
        return PATCH_SOLID
    return PATCH_AIR


def tile_is_blank(tile_id: int) -> bool:
    """True for the buffer's own filler tile - the cell the level data leaves empty."""
    return int(tile_id) == TILE_BLANK
