r"""Gates for README 10.64: the buffer identification, and the two instruments that must agree.

The section's claim is a negative and a positive at once - the forced-mode buffer is the title
screen's own, and it is not the entered level's - and the interesting failure mode is the metric that
could say both are true of any two buffers. Most of a tile column is empty rows, so cell equality
stays high between unrelated levels; the claim therefore has to rest on whole-column identity, and
these gates refuse the section if that distinction ever stops being visible in the numbers. The
class profile is cross-checked against 10.61's independent measurement of the same two sessions, and
the raw-ID reader against the classifier it is supposed to be a sharper version of.

Run:  pytest tests/test_forced_mode_buffer_study.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.environment import wram
from src.evaluation.forced_mode_buffer_study import column_ids, render_section
from src.utils.paths import CORE_PATH, ROM_PATH, STATE_YOSHI_ISLAND_1_LEVEL
from tests.conftest import requires_emulator

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "forced_mode_buffer_metrics.json"
STAGE_ARTIFACT = REPO / "results" / "stage_identity_metrics.json"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the forced-mode buffer study has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _arms(payload=None):
    payload = payload or _payload()
    return {a["arm"]: a for a in payload["arms"]}


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_readme_10_64_is_the_generated_block() -> None:
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.64 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.64 "), "the generator must start at the section heading"


def test_the_forced_mode_buffer_is_the_title_screens_own() -> None:
    comp = _payload()["comparisons"]
    same = comp["forced_mode_preamble_vs_attract_demo"]
    assert same["share_of_columns_identical"] > 0.9, (
        "the two sessions no longer read the same buffer, so the identification has to be re-argued"
    )
    assert same["share_of_cells_equal"] > 0.99


def test_the_entered_level_is_a_different_buffer_even_though_the_cells_say_otherwise() -> None:
    """The trap this section names: empty rows make unrelated buffers look 84% alike."""
    comp = _payload()["comparisons"]
    other = comp["forced_mode_preamble_vs_entered_level"]
    assert other["share_of_columns_identical"] < 0.1
    assert other["share_of_cells_equal"] > 0.75, (
        "the caveat sentence exists because cell equality is inflated by empty rows; if that stops "
        "being true the sentence is wrong, even though the conclusion still is"
    )
    assert other["share_of_cells_equal"] - other["share_of_columns_identical"] > 0.5, (
        "the gap between the two metrics is the whole argument"
    )


def test_the_translevel_bytes_are_the_ones_the_section_prints() -> None:
    arms = _arms()
    assert arms["attract_demo"]["translevel_byte"] == 0x00
    assert arms["forced_mode_preamble"]["translevel_byte"] == 0x00
    assert arms["entered_level"]["translevel_byte"] == 0x2A
    assert arms["attract_demo"]["mode_at_start"] == 0x07, (
        "the section says the boot is on the title screen when the preamble writes 0x14; if the mode "
        "at frame 410 is something else, that sentence is no longer what the console shows"
    )


def test_the_class_profile_agrees_with_10_61s_independent_measurement() -> None:
    """Two studies, one comparison: they may not drift into contradiction about the same sessions."""
    if not STAGE_ARTIFACT.is_file():
        pytest.skip("10.61 has not been run")
    stage = json.loads(STAGE_ARTIFACT.read_text(encoding="utf-8"))
    here = _payload()["comparisons"]["forced_mode_preamble_vs_entered_level"]["class_profile"]
    there = stage["comparisons"]["forced_mode_vs_level"]
    assert abs(here["share_agreeing"] - there["share_agreeing"]) < 0.10, (
        f"10.61 measured {there['share_agreeing']:.4f} and 10.64 measures "
        f"{here['share_agreeing']:.4f} on the same two sessions"
    )


@requires_emulator
def test_the_raw_id_reader_and_the_classifier_describe_the_same_tiles() -> None:
    """The new fingerprint must be the old instrument's input, not a second opinion about it.

    The invariant is the one that cannot be re-derived wrong: a cell the classifier calls air is a
    cell whose ID is the air tile, and a cell it calls terrain has a real ID behind it. Repeating the
    classifier's table here would only prove that the same author wrote both.
    """
    from src.environment.snes_emulator import SnesLibretroEmulator

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    with open(STATE_YOSHI_ISLAND_1_LEVEL, "rb") as fh:
        emu.start_episode(fh.read())
    try:
        radius = 3
        for _ in range(120):
            emu.set_input({"RIGHT": True, "Y": True})
            emu.step_frame()
            state = emu.get_smw_state()
            centre_row = int(state["y"]) // wram.TILE_SIZE_PX
            ids = emu.read_tile_ids(state["x"], state["y"], radius=radius)
            patch = emu.get_local_tilemap_patch(state["x"], state["y"], radius=radius)
            in_range = ids >= 0
            assert np.array_equal((patch == 0) & in_range, (ids == wram.TILE_AIR) & in_range), (
                "air means something different to the two instruments"
            )
            assert np.all((patch > 0) == (in_range & (ids != wram.TILE_AIR)))
            column = column_ids(emu, int(state["x"]) // wram.TILE_SIZE_PX)
            assert column.shape == (wram.TILEMAP_ROWS_PER_SUBSCREEN,)
            if 0 <= centre_row < column.shape[0]:
                assert column[centre_row] == ids[radius, radius]
    finally:
        emu.close()
