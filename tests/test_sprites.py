"""
test_sprites.py
Unit tests for WRAM sprite parsing, relative hazard distance calculation,
and 12-dimensional extended state assembly.
"""


from src.environment.snes_emulator import SnesLibretroEmulator
from tests.conftest import CORE_PATH, ROM_PATH, STATE_PATH, requires_emulator


@requires_emulator
def test_wram_sprite_extraction():
    core_path = CORE_PATH
    rom_path = ROM_PATH
    state_path = STATE_PATH

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        emu.load_state(f.read())
    emu.wram_buffer[0x0100] = 0x14

    # Advance 60 frames into the level
    for _ in range(60):
        emu.step_frame()

    sprites = emu.get_active_sprites()
    # There must be at least one active sprite (Rex at slot 9) in Yoshi's Island 1 start
    assert len(sprites) >= 1
    rex_found = any(s["id"] == 0x05 for s in sprites)
    assert rex_found, "Rex (Sprite ID 0x05) was not found in active sprite table"

    mario_state = emu.get_smw_state()
    hazard = emu.get_nearest_hazard(mario_state["x"], mario_state["y"])

    assert hazard["hazard_active"] == 1.0
    assert hazard["hazard_id"] == 0x05
    assert hazard["delta_x_enemy"] > 0.0  # Rex is ahead of Mario

    ext_state = emu.get_smw_extended_state()
    assert "delta_x_enemy" in ext_state
    assert "delta_y_enemy" in ext_state
    assert "vx_enemy" in ext_state
    assert "hazard_active" in ext_state

    emu.close()
