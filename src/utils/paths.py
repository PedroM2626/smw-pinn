"""
paths.py
Repo-root-anchored, environment-overridable locations for hardware assets.

Historically every evaluation module repeated the literals
``"data/raw/smw_usa.sfc"`` and ``"src/environment/bin/snes9x_libretro.dll"``,
which broke whenever a script was launched from another working directory and
made it impossible to point the suite at a different dump or core build.

Resolution order (first hit wins):
1. an explicit environment variable (``SMW_ROM``, ``SMW_CORE``, ``SMW_DATA_DIR``,
   ``SMW_RESULTS_DIR``),
2. the path relative to the *repository root* (never the current directory),
3. for the Libretro core, the platform-correct shared-library suffix.

The commercial ROM is deliberately not distributed with the repository: each
user supplies their own dump and can verify it with :func:`verify_rom_sha1`.
"""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

from src.utils.logging import get_logger

# src/utils/paths.py -> repository root is two levels up from src/.
REPO_ROOT = Path(__file__).resolve().parents[2]

# SHA-1 of the *Super Mario World (USA)* dump the published results were
# recorded with (README section 11.2). Verification is a warning-level gate:
# a mismatch is reported loudly but does not abort exploratory runs.
ROM_SHA1_USA = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"

_CORE_SUFFIXES = {
    "Windows": ".dll",
    "Darwin": ".dylib",
    "Linux": ".so",
    "FreeBSD": ".so",
}


class HardwareUnavailableError(RuntimeError):
    """Raised when a required emulator asset (core/ROM/savestate) is missing."""


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def data_dir() -> Path:
    return _env_path("SMW_DATA_DIR", REPO_ROOT / "data" / "raw")


def results_dir() -> Path:
    return _env_path("SMW_RESULTS_DIR", REPO_ROOT / "results")


def checkpoint_dir() -> Path:
    return results_dir() / "checkpoints"


def figures_dir() -> Path:
    return results_dir() / "figures"


def default_core_stem() -> Path:
    """Base (suffix-less) path of the bundled Snes9x Libretro core."""
    return _env_path("SMW_CORE", REPO_ROOT / "src" / "environment" / "bin" / "snes9x_libretro")


def resolve_core_path(core_path: str | os.PathLike[str] | None = None) -> str:
    """Return an existing core file, trying the platform suffix if needed.

    ``core_path`` may be a full file name, a suffix-less stem, or ``None`` to
    use the bundled default (or ``$SMW_CORE``). On Linux/macOS a user-supplied
    ``snes9x_libretro.so``/``.dylib`` is found without code changes.
    """
    base = Path(core_path).expanduser() if core_path else default_core_stem()
    if not base.is_absolute():
        base = REPO_ROOT / base

    candidates: list[Path] = [base]
    suffix = _CORE_SUFFIXES.get(platform.system(), ".so")
    if base.suffix != suffix:
        candidates.append(base.with_suffix(suffix))
        candidates.append(Path(f"{base}{suffix}"))  # stem given without a dot
    if base.suffix == ".zip":  # the shipped core archive
        candidates.append(base.with_suffix(""))
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return str(candidates[0])


def rom_path() -> str:
    return str(_env_path("SMW_ROM", data_dir() / "smw_usa.sfc"))


def state_path(name: str = "smw_yoshi_island_1.state") -> str:
    """Absolute path of a savestate inside the data directory."""
    p = Path(name)
    return str(p if p.is_absolute() else data_dir() / name)


def dataset_path(name: str) -> str:
    p = Path(name)
    return str(p if p.is_absolute() else data_dir() / name)


def results_file(name: str) -> str:
    """Absolute path for a result artifact under ``results/`` (created on demand)."""
    path = results_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def checkpoint_file(name: str) -> str:
    """Absolute path for a model checkpoint under ``results/checkpoints/``."""
    path = checkpoint_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def figure_file(name: str) -> str:
    """Absolute path for a plot/video asset under ``results/figures/``."""
    path = figures_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def verify_rom_sha1(path: str | os.PathLike[str], expected: str = ROM_SHA1_USA) -> str:
    """Return the lowercase SHA-1 of a ROM dump (``""`` when it cannot be read)."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()
    except OSError:
        return ""


def check_rom_integrity(path: str | os.PathLike[str], expected: str = ROM_SHA1_USA) -> bool:
    """True when the dump matches the SHA-1 the published results were made with."""
    return verify_rom_sha1(path, expected) == expected.lower()


def _missing(asset: str, path: str, hint: str) -> HardwareUnavailableError:
    return HardwareUnavailableError(
        f"{asset} not found at {path!r}.\n"
        f"  {hint}\n"
        f"  Override the location with an environment variable "
        f"(SMW_ROM / SMW_CORE / SMW_DATA_DIR) and re-run."
    )


def require_core(core_path: str | os.PathLike[str] | None = None) -> str:
    """Validate the Libretro core, raising with LFS/setup instructions when absent."""
    resolved = resolve_core_path(core_path)
    if not Path(resolved).is_file():
        raise _missing(
            "Libretro core",
            resolved,
            "Fetch the large binaries with `git lfs install && git lfs pull`, "
            "or drop a Snes9x Libretro build (.dll/.so/.dylib) into src/environment/bin/.",
        )
    return resolved


def require_rom(rom: str | os.PathLike[str] | None = None) -> str:
    """Validate a ROM dump, raising with acquisition instructions when absent.

    A SHA-1 mismatch is reported as a warning (exploratory runs with a
    different regional dump stay possible) instead of aborting the process.
    """
    path = Path(rom) if rom else Path(rom_path())
    if not path.is_file():
        raise _missing(
            "Super Mario World ROM",
            str(path),
            "This repository does not distribute the commercial ROM. Supply your own "
            f"*Super Mario World (USA)* dump (expected SHA-1 {ROM_SHA1_USA}) and point "
            "SMW_ROM at it.",
        )
    if not check_rom_integrity(path):
        get_logger(__name__).warning(
            "%s has SHA-1 %s, expected %s: published numbers were recorded with the "
            "verified dump, so treat these results as non-comparable.",
            path,
            verify_rom_sha1(path) or "unreadable",
            ROM_SHA1_USA,
        )
    return str(path)


def require_state(name: str = "smw_yoshi_island_1.state") -> str:
    path = Path(name)
    if not path.is_absolute():
        path = data_dir() / name
    if not path.is_file():
        raise _missing(
            "Savestate",
            str(path),
            "Capture it with `python -m scripts.navigate_to_level --level 1` "
            "(the repository ships the Yoshi's Island 1 and Yoshi's House states via Git-LFS).",
        )
    return str(path)


def hardware_present(core_path: str | None = None, rom: str | None = None) -> bool:
    """True when both the Libretro core and a ROM dump exist on this machine.

    Deliberately shallow (file existence only): entry points use this to decide
    whether to attempt a closed-loop run, while `tests/conftest.py` additionally
    probes whether the native library actually loads in this process.
    """
    return Path(resolve_core_path(core_path)).is_file() and Path(rom or rom_path()).is_file()


# Convenience alias used by the evaluation entry points.
hardware_available = hardware_present


# --- Ready-to-use defaults -----------------------------------------------------
# Import-time constants for argument defaults and module-level configuration.
# They resolve against the repository root (or the SMW_* environment overrides),
# so entry points behave identically from any working directory. Use the
# functions above when the environment may change after import.
CORE_PATH: str = resolve_core_path()
ROM_PATH: str = rom_path()
RESULTS_DIR: str = str(results_dir())
CHECKPOINTS_DIR: str = str(checkpoint_dir())
FIGURES_DIR: str = str(figures_dir())
DATASET_GAMEPLAY: str = dataset_path("smw_gameplay_dataset.npz")
DATASET_MULTI_ENTITY: str = dataset_path("smw_multi_entity_dataset.npz")
DATASET_SET_MULTI_ENTITY: str = dataset_path("smw_set_multi_entity_dataset.npz")
DATASET_TILEMAP: str = dataset_path("smw_tilemap_dataset.npz")
DATASET_PIXEL: str = dataset_path("smw_pixel_dataset.npz")
STATE_YOSHI_ISLAND_1: str = state_path("smw_yoshi_island_1.state")
STATE_YOSHI_ISLAND_2: str = state_path("smw_yoshi_island_2.state")
STATE_YOSHI_HOUSE: str = state_path("smw_yoshi_house.state")

# Checkpoints referenced by more than one entry point.
PINN_HARD_CKPT: str = str(checkpoint_dir() / "pinn_hard_best.pt")
PINN_MULTI_ENTITY_CKPT: str = str(checkpoint_dir() / "pinn_multi_entity_best.pt")
