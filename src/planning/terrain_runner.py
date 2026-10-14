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
#: frames a fired jump holds B: SMW scales the jump with the hold, and a 1-frame tap is a
#: minimum-height hop - too short for the wide pits and the tall pipes of the later level
JUMP_HOLD_FRAMES = 12
#: an enemy on the ground needs a tap, not an arc: the tap clears the enemy and lands where
#: the runner already was, while anything taller also clears the terrain behind it - on this
#: level that is the raised plateau, whose far end is flush with a pipe mouth the core
#: auto-captures on contact (the same capture that hangs the 0x29 session)
ENEMY_HOP_HOLD = 1
#: the pole-climb: the engine carries the player up with x frozen and vx at zero while the
#: rise speed decays from -112 toward zero at the top ($7E:007D reads -112, -106, -100, -94,
#: ... frame by frame). Jumping on a pole lets go of it, so the stuck rule must not fire
#: mid-climb: the whole rise is the engine making progress of its own kind, and only the
#: frozen position with zero horizontal speed tells it from a jump - a jump always advances x
POLE_CLIMB_VY = -4.0
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
        self._jump_hold = 0
        self._best_x = 0.0
        self._stalled = 0
        self._last_x = float("inf")
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
        jump_hold = JUMP_HOLD_FRAMES

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

        # Rule 5: a hostile or stompable sprite close ahead at the player's height. The hop is
        # deliberately short: the enemy sits on the ground the runner is already running on,
        # and a full arc would clear it *and* the terrain behind it - on this level that is
        # the raised plateau, whose far end is flush with a pipe mouth the core captures on
        # contact. The hop clears the enemy and lands where the runner was.
        if not jump:
            for sprite in sprites:
                if sprite["kind"] not in (sprite_behavior.HOSTILE, sprite_behavior.STOMPABLE):
                    continue
                if 0.0 < sprite["dx"] <= SPRITE_REACH_PX and abs(sprite["dy"]) <= SPRITE_REACH_Y:
                    jump = True
                    jump_hold = ENEMY_HOP_HOLD
                    break

        # The climbable pole: the engine carries the player up at a constant speed with the
        # collision flags reading "grounded", which fools every rule into thinking a wall has
        # stalled the run - and rule 6's jump would let go of the pole over whatever lies
        # below it. On the climb the only honest action is to keep heading and let the
        # engine finish its own animation: at the top the flip deposits the player on the
        # other side, and the rules resume on their own.
        on_pole = (
            abs(float(state["vx"])) < 0.5
            and float(state["vy"]) <= POLE_CLIMB_VY
            and abs(mario_x - self._last_x) < 0.5
        )
        self._last_x = mario_x
        if on_pole:
            self._stalled = 0
            self._jump_hold = 0
            buttons = {self._heading: True, "Y": True, "B": False, "A": False}
            self._jump_held = False
            self._hold.observe(np.asarray([state[k] for k in HOLD_KEYS], dtype=np.float64))
            return self._hold.buttons(buttons)

        # Rule 6: stuck against something the rules above missed.
        progress = mario_x - self._best_x if self._heading == "RIGHT" else self._best_x - mario_x
        self._best_x = max(self._best_x, mario_x)
        self._stalled = self._stalled + 1 if progress <= 0.5 else 0
        if self._stalled >= STUCK_FRAMES:
            jump = True

        # A fired jump holds B: SMW scales the jump's height with the hold, so a one-frame
        # pulse is a hop and the wide pits need the full arc. The hold releases on its own
        # after JUMP_HOLD_FRAMES, and the edge latch keeps a rule that stays true from
        # re-triggering before the landing.
        grounded = float(state["c_ground"]) > 0.5
        if jump and grounded and not self._jump_held:
            buttons["B"] = True
            self._jump_hold = jump_hold
        if self._jump_hold > 0:
            buttons["B"] = True
            self._jump_hold -= 1
        self._jump_held = bool(buttons["B"])

        # The engine's own message boxes suspend the player until A arrives (10.60); the operator
        # belongs to the agent just as it belongs to the recorders, or a message block is a wall
        # no terrain rule can see.
        self._hold.observe(np.asarray([state[k] for k in HOLD_KEYS], dtype=np.float64))
        return self._hold.buttons(buttons)
