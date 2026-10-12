r"""Gates for the pure instruments behind README 10.66's headline numbers.

`terrain_width` is the function that said Yoshi's Island 1 is 5120 px wide, which is what turned
"nothing clears this stage" into "the frame budget was too small". It reads WRAM through the emulator,
but the counting is pure, and the docstring names a failure mode - a stage with a genuinely empty
subscreen in the middle reads shorter than it is - that no live run would ever show. A function whose
failure mode is only described in prose is a function that will be quoted wrong.

The rest of the file pins the two places where a bookkeeping key can become a false provenance claim:
the candidates a recording run chooses, and the context a render is allowed to see.

CPU-only, no emulator, no artifacts.

Run:  pytest tests/test_level_instruments.py -q
"""

import numpy as np

from scripts.record_level_dashboards import candidates
from scripts.survey_world_map import terrain_width
from src.environment import wram
from src.evaluation.level_dashboard_study import attempt_cleared, bottom_line, drawing_context

CELLS = wram.TILEMAP_ROWS_PER_SUBSCREEN * wram.TILEMAP_COLUMNS_PER_SUBSCREEN


class FakeBuffer:
    """Just enough emulator for a pure read: WRAM as a byte array behind `read_wram_u8`."""

    def __init__(self, wram_bytes: np.ndarray) -> None:
        self._wram = wram_bytes

    def read_wram_u8(self, addr: int) -> int:
        return int(self._wram[addr])


def buffer_with(subscreens_with_terrain: int, empty_at: int = None) -> FakeBuffer:
    memory = np.full(wram.WRAM_SIZE_BYTES, wram.TILE_AIR, dtype=np.uint8)
    for sub in range(subscreens_with_terrain):
        if sub == empty_at:
            continue
        base = wram.ADDR_TILEMAP_BASE + sub * wram.TILEMAP_SUBSCREEN_STRIDE
        memory[base] = 0x00  # one solid tile is enough to say "this subscreen has terrain"
    return FakeBuffer(memory)


def test_terrain_width_counts_the_subscreens_that_hold_terrain() -> None:
    """The number the whole clearance question turned on, checked against a buffer of known width."""
    read = terrain_width(buffer_with(20))
    assert read["terrain_subscreens"] == 20
    assert read["terrain_px"] == 20 * 256
    assert read["non_air_cells_by_subscreen"][:20] == [1] * 20
    assert set(read["non_air_cells_by_subscreen"][20:]) == {0}


def test_terrain_width_stops_at_the_first_empty_subscreen_and_says_so() -> None:
    """The documented failure mode, gated rather than left in the docstring.

    A stage whose third subscreen happens to be all air is a stage this reads as two subscreens
    wide. The profile is what lets a reader see that the count stopped early rather than the level
    being short, so both halves are pinned here.
    """
    read = terrain_width(buffer_with(6, empty_at=2))
    assert read["terrain_subscreens"] == 2, "the leading run is what the count means"
    assert read["non_air_cells_by_subscreen"][2] == 0
    assert read["non_air_cells_by_subscreen"][3] == 1, "and the profile still shows what is beyond it"


def test_terrain_width_never_reads_past_the_wram_window() -> None:
    """A scan that ran off the end of the window would read the wrong thing, not fail loudly."""
    read = terrain_width(buffer_with(3), subscreens=4000)
    assert len(read["non_air_cells_by_subscreen"]) <= (
        wram.WRAM_SIZE_BYTES - wram.ADDR_TILEMAP_BASE
    ) // CELLS


def test_a_cleared_attempt_is_the_one_that_never_saw_the_death_mode() -> None:
    """Both spellings of "reached the map" must count, or the table denies what the prose claims."""
    cleared = {"after_exit": {"on_map": True}, "died_on_the_way": False}
    recorded = {"reached_the_map": True, "died_on_the_way": False}
    game_over = {"after_exit": {"on_map": True}, "died_on_the_way": True}
    still_running = {"after_exit": {"on_map": False}, "died_on_the_way": False}
    assert attempt_cleared(cleared) and attempt_cleared(recorded)
    assert not attempt_cleared(game_over), "a game over also ends on the map"
    assert not attempt_cleared(still_running)


def test_the_render_context_cannot_carry_a_second_dataset() -> None:
    """The defect this prevents: `protocol.context.dataset` naming another file than the render."""
    full = {
        "title": "t",
        "level_label": "Yoshi's Island 1",
        "level_index": 1,
        "level_count": 4,
        "clear_line": "x",
        "goal_px": 5120.0,
        "dataset": "data/raw/smw_dashboard_run_smw_map_node_152_136_t2a.npz",
        "output_stem": "dashboard_level_1",
        "progress_px": 1573.125,
    }
    drawn = drawing_context(full)
    assert "dataset" not in drawn and "output_stem" not in drawn and "progress_px" not in drawn
    assert drawn["level_count"] == 4 and drawn["goal_px"] == 5120.0


def test_the_picture_line_carries_both_of_the_numbers_it_claims() -> None:
    stage = {
        "episode": {"progress_px": 1573.125},
        "terrain": {"terrain_px": 5120},
        "mpc": {"clears": 1, "tries": 6},
    }
    assert bottom_line(stage) == "cleared in 1 of 6 attempts · this arm 1573 / 5120 px"
    line = bottom_line({**stage, "mpc": {"clears": 0, "tries": 6}})
    assert line == "0 of 6 attempts cleared it · this arm 1573 / 5120 px", line


def _stage_json(tmp_path, nodes, rounds=None):
    import json

    survey = tmp_path / "survey.json"
    survey.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"rounds": rounds or []}), encoding="utf-8")
    return str(survey), str(campaign)


def test_two_nodes_of_one_level_are_recorded_as_one_stage(tmp_path) -> None:
    """The survey finds Yoshi's Island 1 at two cursor positions; the dashboard has one of each."""
    nodes = [
        {
            "node": [152, 136],
            "translevel": "0x2A",
            "capture": {"saved": "a.state", "right_response_px": 39.0},
        },
        {
            "node": [141, 135],
            "translevel": "0x2A",
            "capture": {"saved": "b.state", "right_response_px": 39.0},
        },
        {
            "node": [68, 135],
            "translevel": "0x29",
            "capture": {"saved": "c.state", "right_response_px": 15.0},
        },
    ]
    rounds = [
        {"round": 1, "node": [152, 104], "translevel": "0x27", "entered": True, "savestate": "d.state"},
        {"round": 2, "node": [9, 9], "translevel": "0x2A", "entered": True, "savestate": "e.state"},
    ]
    survey, campaign = _stage_json(tmp_path, nodes, rounds)
    picked = candidates(survey, campaign)
    assert [p["translevel"] for p in picked] == ["0x2A", "0x29", "0x27"]
    assert picked[0]["state"] == "a.state", "the first node seen for a level is the one recorded"
    assert picked[-1]["capture"]["source"] == "the clearance campaign"


def test_a_stage_that_needs_no_camera_still_gets_a_candidate(tmp_path) -> None:
    """The fallback keeps the recorder runnable against a survey that captured nothing."""
    survey, campaign = _stage_json(tmp_path, [])
    picked = candidates(survey, campaign)
    assert [p["state"] for p in picked] == ["smw_yoshi_island_1_level.state"]
