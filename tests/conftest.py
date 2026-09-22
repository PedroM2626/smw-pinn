"""Shared pytest fixtures and platform guards.

The Libretro core shipped in ``src/environment/bin`` is a Windows DLL, so any
test that boots the real emulator can only run on Windows with the ROM +
savestates present (CI on ubuntu-latest must skip them instead of erroring).
"""

from __future__ import annotations

import ctypes
import os

import pytest

from src.utils.paths import CORE_PATH, ROM_PATH
from src.utils.paths import STATE_YOSHI_ISLAND_1 as STATE_PATH


def emulator_available(
    core_path: str = CORE_PATH,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_PATH,
) -> bool:
    """True only if the native core exists *and* loads in this process."""
    if not (os.path.exists(core_path) and os.path.exists(rom_path) and os.path.exists(state_path)):
        return False
    try:
        ctypes.CDLL(core_path)
    except OSError:
        return False
    return True


requires_emulator = pytest.mark.skipif(
    not emulator_available(),
    reason="Real SNES emulator core not loadable on this platform (needs Windows + ROM).",
)
