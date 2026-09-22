"""Tests for the emulator platform guard (tests/conftest.py).

These are the tests that keep CI green on Linux: emulator-dependent tests
must skip instead of erroring when the Windows Libretro core cannot load.
"""

import ctypes

from tests.conftest import emulator_available


def test_guard_passes_on_capable_platform():
    # This dev machine (Windows + ROM + DLL) can boot the real emulator.
    assert emulator_available() is True


def test_guard_fails_on_missing_files(tmp_path):
    assert (
        emulator_available(
            core_path=str(tmp_path / "missing.dll"),
            rom_path=str(tmp_path / "missing.sfc"),
            state_path=str(tmp_path / "missing.state"),
        )
        is False
    )


def test_guard_fails_when_cdll_raises(monkeypatch):
    def _boom(path):
        raise OSError("cannot load library")

    monkeypatch.setattr(ctypes, "CDLL", _boom)
    # Paths exist here, so only the CDLL failure drives the result.
    assert emulator_available() is False
