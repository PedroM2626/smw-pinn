"""
sprite_behavior.py - classify a sprite id the way the engine was measured to treat it.

`scripts.probe_sprite_behaviors` meets every sprite id the recorded stages spawn, saves the
frame, and touches it from the side and from above - twice, once with the sprite and once with
its slot cleared, because a fall past the floor kills both runs and only the difference is the
sprite's doing. The measured classes are baked into `sprite_behavior_table.py` and gated against
`results/sprite_behavior_probe_metrics.json` by `tests/test_sprite_behavior.py`.

The classes:

* ``hostile`` - contact killed a small Mario that the same run without the sprite survived;
* ``stompable`` - a stomp bounced Mario or changed the sprite's status byte (the Rexes and
  Koopas the stages are made of);
* ``platform`` - a stomp held Mario up where the cleared run fell;
* ``neutral`` - the engine ignored the contact (the chimney smoke over Yoshi's House, the
  bouncing blocks: the sprites the old panel drew as enemies);
* ``unprobed`` - an id the probe has not measured yet. It is drawn as unknown and no rule treats
  it as an enemy, because a guessed class is exactly what 10.68 exists to remove.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.environment import wram
from src.environment.sprite_behavior_table import BEHAVIOR

#: the measured classes, in the order the artifact's protocol defines them
HOSTILE = "hostile"
STOMPABLE = "stompable"
PLATFORM = "platform"
COLLECTIBLE = "collectible"
NEUTRAL = "neutral"
UNPROBED = "unprobed"

#: the ids whose contact hurts, for the reflex layers and the runner
HURTING = frozenset({HOSTILE, STOMPABLE})


def behavior_of(sprite_id: int) -> str:
    """The measured behaviour class of one sprite id, or ``unprobed``."""
    return BEHAVIOR.get(int(sprite_id), UNPROBED)


def live_slots(emu: Any) -> List[Dict[str, Any]]:
    """Every live sprite slot with a settled position, in slot order.

    A slot's status byte is written before its position settles, so a freshly loaded slot can
    report coordinates far outside any level; those are dropped here - a probe, a runner or a
    panel that trusted them would measure the fall out of the world instead of a sprite.
    """
    out: List[Dict[str, Any]] = []
    for slot in range(wram.NUM_SPRITE_SLOTS):
        status = emu.read_wram_u8(wram.ADDR_SPRITE_STATUS + slot)
        if status == 0:
            continue
        sprite_id = emu.read_wram_u8(wram.ADDR_SPRITE_ID + slot)
        x = emu.read_wram_u16_le(wram.ADDR_SPRITE_X_LOW + slot) | (
            emu.read_wram_u8(wram.ADDR_SPRITE_X_HIGH + slot) << 8
        )
        y = emu.read_wram_u16_le(wram.ADDR_SPRITE_Y_LOW + slot) | (
            emu.read_wram_u8(wram.ADDR_SPRITE_Y_HIGH + slot) << 8
        )
        if not (0 <= x <= 4096 and 0 <= y <= 448):
            continue
        out.append({"slot": slot, "id": sprite_id, "status": status, "x": x, "y": y})
    return out
