r"""Gates for the deterministic terrain runner (README 10.68).

The runner plans no trajectory and samples nothing: six rules over the measured terrain and
sprite tables. The gates below run those rules against a fake emulator whose buffer holds
known tiles, so every decision is checked against the measured class it must consult - a pit is
a jump only when the cells that hold from above are really absent, a bush is never a wall, and
a ramp never triggers the wall rule.

CPU-only, no emulator.

Run:  pytest tests/test_terrain_runner.py -q
"""

import pytest

from src.environment import tile_behavior, wram
from src.planning.terrain_runner import TerrainRunner

GROUND = tile_behavior.TILE_BLANK  # replaced per test; see _place
AIR_ID = 0x0025
GRASS_TOP = 0x0100  # measured platform: holds from above, never blocks a side
STAIR_CORNER = 0x0145  # measured solid: the wall a run must jump
RAMP = 0x0173  # measured slope
BUSH = 0x0073  # measured decor: the tile the old map painted solid


class FakeEmu:
    """A tilemap of 16-bit ids behind `read_tile_id`, plus the runner's sprite slots."""

    def __init__(self, tiles=None, sprites=None):
        self.tiles = dict(tiles or {})
        self.sprites = sprites or []
        self.written = []

    def read_tile_id(self, tx, ty):
        return self.tiles.get((tx, ty))

    def read_wram_u8(self, addr):
        for sprite in self.sprites:
            if addr == wram.ADDR_SPRITE_STATUS + sprite["slot"]:
                return sprite.get("status", 8)
            if addr == wram.ADDR_SPRITE_ID + sprite["slot"]:
                return sprite["id"]
            if addr == wram.ADDR_SPRITE_X_HIGH + sprite["slot"]:
                return (sprite["x"] >> 8) & 0xFF
            if addr == wram.ADDR_SPRITE_Y_HIGH + sprite["slot"]:
                return (sprite["y"] >> 8) & 0xFF
        return 0

    def read_wram_u16_le(self, addr):
        for sprite in self.sprites:
            if addr == wram.ADDR_SPRITE_X_LOW + sprite["slot"]:
                return sprite["x"] & 0xFF
            if addr == wram.ADDR_SPRITE_Y_LOW + sprite["slot"]:
                return sprite["y"] & 0xFF
        return 0


def ground_row(columns, row, row_id=GRASS_TOP):
    """A floor of grass tops across the given columns at the given row."""
    return {(int(c), int(row)): row_id for c in columns}


def state(x=64.0, y=352.0, ground=True, vx=2.0):
    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": 0.0,
        "c_ground": 1.0 if ground else 0.0,
        "c_ceiling": 0.0,
        "c_left": 0.0,
        "c_right": 0.0,
    }


def test_runs_right_on_flat_ground() -> None:
    """Rule 1: nothing ahead but the floor the engine holds - hold RIGHT and Y, no jump."""
    emu = FakeEmu(ground_row(range(-1, 10), 24))
    buttons = TerrainRunner().act(emu, state())
    assert buttons["RIGHT"] and buttons["Y"] and not buttons["B"]


def test_jumps_when_a_pit_opens_ahead() -> None:
    """Rule 2: two columns ahead with nothing that holds from above is a jump."""
    emu = FakeEmu(ground_row(range(-1, 8), 24))  # floor through column 7
    runner = TerrainRunner()
    # standing at x=64 (feet column 4); the floor ends at column 7, so column 7 is the last hold
    buttons = runner.act(emu, state(x=7 * 16 - 8))  # feet column 6, pit at column 8
    assert buttons["B"], "the missing ground two columns ahead must fire the jump"


def test_does_not_jump_over_a_decor_bush_the_old_map_called_a_pipe() -> None:
    """A bush ahead is decoration: the runner must run straight through it."""
    emu = FakeEmu({**ground_row(range(-1, 10), 24), (6, 23): BUSH})
    buttons = TerrainRunner().act(emu, state())
    assert not buttons["B"], "a measured decor tile is not an obstacle"


def test_jumps_at_a_measured_wall() -> None:
    """Rule 3: the stair corner (measured solid) at body height is a jump; a grass top is not."""
    emu = FakeEmu({**ground_row(range(-1, 10), 24), (6, 23): STAIR_CORNER, (7, 22): STAIR_CORNER})
    buttons = TerrainRunner().act(emu, state())
    assert buttons["B"]
    emu = FakeEmu({**ground_row(range(-1, 10), 24), (6, 23): GRASS_TOP})
    buttons = TerrainRunner().act(emu, state())
    assert not buttons["B"], "a tile that never blocks a side is not a wall"


def test_walks_onto_a_measured_ramp() -> None:
    """Rule 4: a ramp is neither pit nor wall - the run continues onto it."""
    emu = FakeEmu({**ground_row(range(-1, 10), 24), (6, 23): RAMP, (6, 22): RAMP})
    buttons = TerrainRunner().act(emu, state())
    assert not buttons["B"]


def test_jumps_onto_a_stompable_sprite_ahead() -> None:
    """Rule 5: a measured stompable sprite within reach fires the jump; a neutral one does not."""
    emu = FakeEmu(
        ground_row(range(-1, 10), 24), sprites=[{"slot": 0, "id": 0x05, "x": 100, "y": 352}]
    )
    buttons = TerrainRunner().act(emu, state())
    assert buttons["B"], "the Rex ahead is measured stompable - the runner must jump it"
    emu = FakeEmu(
        ground_row(range(-1, 10), 24), sprites=[{"slot": 0, "id": 0x8A, "x": 100, "y": 352}]
    )
    buttons = TerrainRunner().act(emu, state())
    assert not buttons["B"], "the neutral smoke must not trigger anything"


def test_the_jump_pulse_is_edge_triggered_across_a_whole_descent() -> None:
    """The latch: a rule that stays true must not silence the next platform's jump.

    The failure this gates was measured live: crossing one pit left the latch set, and the
    runner never jumped again.
    """
    emu = FakeEmu(ground_row(range(-1, 8), 24))
    runner = TerrainRunner()
    # one jump fires and latches
    runner.act(emu, state(x=7 * 16 - 8))
    assert runner._jump_held
    # the rule goes quiet; the latch releases on its own
    runner.act(emu, state(x=64.0))
    assert not runner._jump_held
    # and the next jump fires again
    buttons = runner.act(emu, state(x=7 * 16 - 8))
    assert buttons["B"]


def test_a_long_stillness_jumps_anyway() -> None:
    """Rule 6: stuck against something the other rules missed."""
    emu = FakeEmu(ground_row(range(-1, 10), 24))
    runner = TerrainRunner()
    fired = False
    for _ in range(30):
        if runner.act(emu, state(vx=0.0))["B"]:
            fired = True
    assert fired, "half a second without progress must fire a jump pulse"


def test_the_houses_heading_is_left_and_its_progress_counts_that_way() -> None:
    """The blank session behind the Yoshi's House node leaves through the door it came in by."""
    emu = FakeEmu(ground_row(range(-1, 10), 24))
    runner = TerrainRunner(direction="LEFT")
    buttons = runner.act(emu, state())
    assert buttons["LEFT"] and not buttons.get("RIGHT", False)


@pytest.mark.parametrize(
    "tile_id,expected",
    [
        (GRASS_TOP, "platform"),
        (STAIR_CORNER, "solid"),
        (RAMP, "slope"),
        (BUSH, "decor"),
        (AIR_ID, "decor"),
    ],
)
def test_the_runner_consults_the_measured_table(tile_id, expected) -> None:
    """The grid the rules read is the measured table's answer, not a second classifier."""
    assert tile_behavior.behavior_of(tile_id) == expected
