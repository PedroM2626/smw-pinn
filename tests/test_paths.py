"""Tests for the centralized hardware-asset path resolution (src/utils/paths.py).

These are CI-safe: nothing here boots the emulator, and the ROM/core lookups are
exercised against temporary files so the tests pass without the Git-LFS assets.
"""

from pathlib import Path

import pytest

from src.utils import paths
from src.utils.paths import (
    HardwareUnavailableError,
    check_rom_integrity,
    resolve_core_path,
    verify_rom_sha1,
)


def test_repo_root_points_at_the_repository():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / "src").is_dir()


def test_resolved_defaults_are_absolute_and_cwd_independent(tmp_path, monkeypatch):
    # Whatever the working directory, defaults must resolve under the repo root.
    monkeypatch.chdir(tmp_path)
    assert Path(paths.ROM_PATH).is_absolute()
    assert Path(paths.DATASET_GAMEPLAY).parent == paths.data_dir()
    assert Path(paths.STATE_YOSHI_ISLAND_1).name == "smw_yoshi_island_1.state"


def test_environment_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv("SMW_ROM", str(tmp_path / "my_dump.sfc"))
    monkeypatch.setenv("SMW_DATA_DIR", str(tmp_path / "assets"))
    assert paths.rom_path() == str(tmp_path / "my_dump.sfc")
    assert paths.data_dir() == tmp_path / "assets"
    # Savestate/dataset helpers anchor to the (overridden) data directory.
    assert paths.state_path("x.state") == str(tmp_path / "assets" / "x.state")
    assert paths.dataset_path("y.npz") == str(tmp_path / "assets" / "y.npz")


def test_resolve_core_path_finds_the_platform_suffix(tmp_path):
    """A stem (or wrong suffix) still resolves when the real core exists."""
    import platform

    suffix = paths._CORE_SUFFIXES.get(platform.system(), ".so")
    real = tmp_path / f"snes9x_libretro{suffix}"
    real.write_bytes(b"MZ")
    # Passing the suffix-less stem must find the existing shared library.
    assert resolve_core_path(str(tmp_path / "snes9x_libretro")) == str(real)
    # A wrong suffix is forgiven when the platform-suffixed sibling exists.
    assert resolve_core_path(str(tmp_path / "snes9x_libretro.wrong")) == str(real)
    # Nothing on disk -> the requested name is returned so the error stays precise.
    ghost = tmp_path / "absent_core"
    assert resolve_core_path(str(ghost)) == str(ghost)


def test_require_rom_raises_with_acquisition_guidance(tmp_path, monkeypatch):
    missing = tmp_path / "absent.sfc"
    monkeypatch.setenv("SMW_ROM", str(missing))
    with pytest.raises(HardwareUnavailableError) as exc:
        paths.require_rom()
    message = str(exc.value)
    # The point of the guard: tell the user *how* to obtain the dump.
    assert "does not distribute the commercial ROM" in message
    assert paths.ROM_SHA1_USA in message
    assert "SMW_ROM" in message


def test_rom_sha1_verification(tmp_path):
    dump = tmp_path / "fake.sfc"
    dump.write_bytes(b"super mario world")
    digest = verify_rom_sha1(dump)
    assert len(digest) == 40 and digest == digest.lower()
    assert check_rom_integrity(dump, expected=digest) is True
    assert check_rom_integrity(dump) is False
    # Unreadable files degrade to "" instead of raising.
    assert verify_rom_sha1(tmp_path / "nope.sfc") == ""


def test_require_core_message_mentions_lfs(tmp_path, monkeypatch):
    monkeypatch.setenv("SMW_CORE", str(tmp_path / "missing_core.dll"))
    with pytest.raises(HardwareUnavailableError) as exc:
        paths.require_core()
    assert "git lfs pull" in str(exc.value)


def test_results_file_creates_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("SMW_RESULTS_DIR", str(tmp_path / "out"))
    written = paths.results_file("nested/metric.json")
    assert Path(written).parent.is_dir()


def test_wram_map_matches_the_reverse_engineered_registers():
    """Guard the addresses README section 3.3 documents as reverse-engineered."""
    from src.environment import wram

    assert wram.ADDR_GAME_MODE == 0x0100
    assert wram.GAME_MODE_INTERACTIVE == 0x14
    assert (wram.ADDR_PLAYER_X, wram.ADDR_PLAYER_Y) == (0x0094, 0x0096)
    assert (wram.ADDR_VX, wram.ADDR_VY) == (0x007B, 0x007D)
    assert wram.ADDR_COLLISION == 0x0077
    assert wram.ADDR_TILEMAP_BASE == 0xC800
    # Collision bits are documented as mutually exclusive single flags.
    assert {
        wram.COLLISION_RIGHT,
        wram.COLLISION_LEFT,
        wram.COLLISION_GROUND,
        wram.COLLISION_CEILING,
    } == {
        0x01,
        0x02,
        0x04,
        0x08,
    }
    assert wram.SUBPIXELS_PER_PIXEL == 16.0


def test_emulator_exposes_the_shared_episode_helpers():
    """The duplicated per-script preamble must be reachable from one place."""
    from src.environment.snes_emulator import SnesLibretroEmulator

    for name in ("get_game_mode", "in_level", "enable_gameplay_mode", "start_episode"):
        assert callable(getattr(SnesLibretroEmulator, name)), name
