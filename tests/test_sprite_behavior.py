r"""Gates for the measured sprite behaviour table (README 10.68).

The old dashboard drew whichever sprite sat nearest, whatever it was: a bush, a bouncing block
and a power-up all read as "enemy", and the four hundred sprites that were not the nearest were
invisible. The classes here are what the engine itself did when the probe met each id in its own
stage and touched it - twice, with the sprite present and with its slot cleared, because a fall
past the floor kills both runs and only the difference is the sprite's doing.

CPU-only, no emulator: the probe itself is hardware-gated.

Run:  pytest tests/test_sprite_behavior.py -q
"""

import json
from pathlib import Path

import numpy as np

from src.environment import sprite_behavior, wram
from src.environment.sprite_behavior_table import BEHAVIOR

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "sprite_behavior_probe_metrics.json"


def _artifact():
    if not ARTIFACT.is_file():
        raise AssertionError("results/sprite_behavior_probe_metrics.json has not been generated")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_baked_table_is_the_probe_artifact() -> None:
    """Every class the runtime classifies through is the class the probe measured."""
    measured = {int(r["sprite"], 16): r["class"] for r in _artifact()["sprites"]}
    assert BEHAVIOR == measured, (
        "src/environment/sprite_behavior_table.py disagrees with the artifact: re-run "
        "`python -m scripts.probe_sprite_behaviors --emit-table`"
    )


def test_the_probe_covers_every_active_id_in_the_dashboard_recordings() -> None:
    """A brown 'unprobed' cell must not hide an entity the committed episodes already met."""
    recordings = [REPO / "data" / "raw" / "smw_dashboard_run.npz"]
    recordings.extend(sorted((REPO / "data" / "raw").glob("smw_dashboard_run_smw_*.npz")))
    seen = set()
    for path in recordings:
        with np.load(path, allow_pickle=False) as data:
            sprites = np.asarray(data["sprites"])
            active = sprites[:, :, 3] >= sprite_behavior.SPRITE_ACTIVE
            seen.update(int(value) for value in sprites[:, :, 2][active])
    missing = seen - set(BEHAVIOR)
    assert not missing, f"active sprite ids absent from the measured table: {sorted(missing)}"


def test_the_classes_the_section_quotes() -> None:
    """The headline ids of 10.68, gated so the prose cannot drift from the measurement."""
    assert sprite_behavior.behavior_of(0x05) == sprite_behavior.STOMPABLE, (
        "the Rex: stomped, the status byte moves inside the running band"
    )
    assert sprite_behavior.behavior_of(0x04) == sprite_behavior.STOMPABLE, (
        "the Koopa the clear draw passes on screen"
    )
    for shell_less_koopa in (0x00, 0x02):
        assert sprite_behavior.behavior_of(shell_less_koopa) == sprite_behavior.STOMPABLE, (
            "side contact pauses Mario, but the stomp makes the running sprite disappear"
        )
    assert sprite_behavior.behavior_of(0x9F) == sprite_behavior.STOMPABLE
    assert sprite_behavior.behavior_of(0x91) == sprite_behavior.INTERACT, (
        "the message block: touching it opened the engine's own box and froze the player - "
        "with the sprite present and not with its slot cleared, which is the counterfactual "
        "that separates a box from a bad teleport"
    )
    assert sprite_behavior.behavior_of(0x8A) == sprite_behavior.NEUTRAL, (
        "the chimney smoke the old panel drew as an enemy"
    )
    assert sprite_behavior.behavior_of(0x01) == sprite_behavior.NEUTRAL, (
        "the parked shell: touching it hurts nobody, whatever it looks like"
    )


def test_the_objects_the_panel_once_painted_red_are_not_enemies() -> None:
    """The first sweep's wake-up artifact, corrected and gated: shells are not enemies.

    The probe met these ids in dormant slots (status 1), the teleport woke them (1 -> 8), and
    the wake-up read as a stomp - which painted the empty shells and the message block as
    enemies. Only running slots are met now, and these are what the engine actually does with
    them.
    """
    for shell in (0x01, 0x4D, 0x4E, 0xB9):
        kind = sprite_behavior.behavior_of(shell)
        assert kind not in (sprite_behavior.STOMPABLE, sprite_behavior.HOSTILE), (
            f"{shell:#04x} is measured {kind}: a parked shell is an object, not an enemy"
        )


def test_an_id_the_probe_has_not_met_is_reported_unprobed() -> None:
    """Unmeasured is not 'enemy' - that guess is exactly what this section removes."""
    assert sprite_behavior.behavior_of(0xFE) == sprite_behavior.UNPROBED


def test_live_slots_combines_each_slots_own_position_bytes() -> None:
    """A sprite table is structure-of-arrays: the next byte belongs to the next slot."""

    class FakeEmu:
        def __init__(self) -> None:
            self.memory = {
                wram.ADDR_SPRITE_STATUS: 8,
                wram.ADDR_SPRITE_ID: 0x05,
                wram.ADDR_SPRITE_X_LOW: 0x92,
                wram.ADDR_SPRITE_X_LOW + 1: 0xAB,
                wram.ADDR_SPRITE_X_HIGH: 0x13,
                wram.ADDR_SPRITE_Y_LOW: 0x56,
                wram.ADDR_SPRITE_Y_LOW + 1: 0xCD,
                wram.ADDR_SPRITE_Y_HIGH: 0x01,
            }

        def read_wram_u8(self, addr: int) -> int:
            return self.memory.get(addr, 0)

        def read_wram_u16_le(self, addr: int) -> int:
            return self.read_wram_u8(addr) | (self.read_wram_u8(addr + 1) << 8)

    slots = sprite_behavior.live_slots(FakeEmu())
    assert slots == [{"slot": 0, "id": 0x05, "status": 8, "x": 0x1392, "y": 0x156}]


def test_hurting_sprites_are_hostile_or_stompable() -> None:
    """The two measured classes a runner must react to, and nothing else."""
    hurting = {sprite_behavior.HOSTILE, sprite_behavior.STOMPABLE}
    assert sprite_behavior.HURTING == hurting


def test_the_probe_ran_every_touch_twice() -> None:
    """The counterfactual is the whole argument: each record carries both runs or it proves nothing."""
    for record in _artifact()["sprites"]:
        assert "contact_without_sprite" in record and "stomp_without_sprite" in record, (
            f"{record['sprite']} was probed without its counterfactual run"
        )
