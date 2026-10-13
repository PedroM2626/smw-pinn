"""
terrain_runner.py - a deterministic agent that finishes stages on the engine's own terrain.

The clearance controller this repository had was a sampled planner over a learned world model:
strong when the model generalized, 1 of 6 seeded draws on Yoshi's Island 1 and 0 of 6 on every
other stage, and its runs ended in pits and on walls it could not see. This agent plans no
trajectory and samples nothing: every frame it reads the level buffer the way 10.68 re-taught
this repository to read it - full 16-bit Map16 ids classified by the measured behaviour table -
plus the twelve sprite slots classified the same way, and applies six rules a reader can hold in
their head:

1. run right;
2. a pit opens ahead - no tile that holds from above within two columns of the feet - so jump;
3. a wall stands ahead - a side-blocking tile at body height within two columns - so jump;
4. a walkable ramp or a platform edge ahead - keep running, the terrain carries the player;
5. a hostile or stompable sprite is close ahead - jump onto or over it;
6. stuck against something the rules missed - jump, and keep the jump pulse edge-triggered as the
   published recorders do, because a held B never re-triggers a jump.

Every decision is a function of the measured terrain and sprite classes, so the agent inherits
the probe's guarantees: it cannot mistake a bush for a pipe or paint a ramp as air, because it
never sees tile ids at all - only what the engine was measured to do with them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.environment import sprite_behavior, tile_behavior, wram
from src.environment.episode_hygiene import PlayerHold

#: how many tile columns ahead of the player's feet the ground rules look
LOOKAHEAD_COLUMNS = 2
#: how many rows below the feet line a tile still counts as ground to land on
GROUND_DEPTH_ROWS = 3
#: the rows above the feet whose side-blocking tiles count as walls at body height
WALL_ROWS_ABOVE = 2
#: how far ahead a hostile sprite triggers the jump, in pixels
SPRITE_REACH_PX = 48.0
#: how far vertically a sprite still counts as in the way, in pixels
SPRITE_REACH_Y = 48.0
#: frames without forward progress after which the agent jumps anyway
STUCK_FRAMES = 24
#: the eight channels `PlayerHold` watches, in the order the recorders use
HOLD_KEYS = ("x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right")


class TerrainRunner:
    """A rule-based agent over the measured terrain and sprite tables. Deterministic.

    `direction` is the run's heading: every stage this repository has surveyed is completed by
    running right, except the blank session behind the Yoshi's House node, whose only way out is
    back through the door it came in by. The rules read the terrain the same way either way;
    only the held heading changes.
    """

    name = "terrain-runner"

    def __init__(self, direction: str = "RIGHT") -> None:
        assert direction in ("RIGHT", "LEFT")
        self._heading = direction
        self._jump_held = False
        self._best_x = 0.0
        self._stalled = 0
        self._hold = PlayerHold(limit=240)

    @property
    def operator_frames(self) -> int:
        """Frames the operator pressed A on the engine's own message boxes."""
        return self._hold.operator_frames

    def grid(
        self, emu: Any, mario_x: float, mario_y: float
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """The terrain classes and the live sprites in the window the rules reason about.

        The grid is [rows, columns] of `tile_behavior` classes around the player's tile cell,
        with `None` where the buffer has no cell; the sprite list carries each slot's measured
        class and its offset from the player.
        """
        feet_col = int(mario_x + 8) // wram.TILE_SIZE_PX
        feet_row = int(mario_y + 32) // wram.TILE_SIZE_PX
        step = 1 if self._heading == "RIGHT" else -1
        columns = np.arange(feet_col - step, feet_col + step * (LOOKAHEAD_COLUMNS + 2), step)
        rows = np.arange(feet_row - WALL_ROWS_ABOVE, feet_row + GROUND_DEPTH_ROWS + 1)
        grid = np.empty((len(rows), len(columns)), dtype=object)
        for i, ty in enumerate(rows):
            for j, tx in enumerate(columns):
                tile_id = emu.read_tile_id(int(tx), int(ty))
                grid[i, j] = None if tile_id is None else tile_behavior.behavior_of(tile_id)

        sprites: List[Dict[str, Any]] = []
        for sprite in sprite_behavior.live_slots(emu):
            sprites.append(
                {
                    **sprite,
                    "dx": step * (sprite["x"] - mario_x),
                    "dy": sprite["y"] - mario_y,
                    "kind": sprite_behavior.behavior_of(sprite["id"]),
                }
            )
        return grid, sprites

    def act(self, emu: Any, state: Dict[str, float]) -> Dict[str, bool]:
        """One frame's buttons, from the measured terrain and sprite tables alone."""
        mario_x, mario_y = float(state["x"]), float(state["y"])
        grid, sprites = self.grid(emu, mario_x, mario_y)

        buttons: Dict[str, bool] = {self._heading: True, "Y": True, "B": False, "A": False}
        jump = False

        # Rule 2: a pit opens ahead - no tile that holds from above within the look-ahead.
        for column in range(1, LOOKAHEAD_COLUMNS + 2):
            holds = any(
                grid[i, column]
                in (tile_behavior.SOLID, tile_behavior.PLATFORM, tile_behavior.SLOPE)
                for i in range(WALL_ROWS_ABOVE, grid.shape[0])
            )
            if not holds:
                jump = True
                break

        # Rule 3: a wall at body height within the look-ahead.
        if not jump:
            for column in range(1, LOOKAHEAD_COLUMNS + 2):
                for row in range(0, WALL_ROWS_ABOVE + 1):
                    if grid[row, column] == tile_behavior.SOLID:
                        jump = True
                        break
                if jump:
                    break

        # Rule 5: a hostile or stompable sprite close ahead at the player's height.
        if not jump:
            for sprite in sprites:
                if sprite["kind"] not in (sprite_behavior.HOSTILE, sprite_behavior.STOMPABLE):
                    continue
                if 0.0 < sprite["dx"] <= SPRITE_REACH_PX and abs(sprite["dy"]) <= SPRITE_REACH_Y:
                    jump = True
                    break

        # Rule 6: stuck against something the rules above missed.
        progress = mario_x - self._best_x if self._heading == "RIGHT" else self._best_x - mario_x
        self._best_x = max(self._best_x, mario_x)
        self._stalled = self._stalled + 1 if progress <= 0.5 else 0
        if self._stalled >= STUCK_FRAMES:
            jump = True

        # The published recorders pulse B because a held B never re-triggers a jump; the pulse is
        # one frame, and the latch below releases the frame after whichever way `jump` went, so a
        # rule that stays true across a whole descent cannot silence the next platform's jump.
        grounded = float(state["c_ground"]) > 0.5
        if jump and grounded and not self._jump_held:
            buttons["B"] = True
        self._jump_held = bool(buttons["B"])

        # The engine's own message boxes suspend the player until A arrives (10.60); the operator
        # belongs to the agent just as it belongs to the recorders, or a message block is a wall
        # no terrain rule can see.
        self._hold.observe(np.asarray([state[k] for k in HOLD_KEYS], dtype=np.float64))
        return self._hold.buttons(buttons)
