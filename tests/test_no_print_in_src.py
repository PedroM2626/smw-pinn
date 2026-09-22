"""Gate: no ``print()`` in ``src/`` outside documented CLI exceptions.

CONTRIBUTING.md asks library code to log through ``src.utils.logging.get_logger`` and
reserves ``print()`` for user-facing tools, because stdout in a library module is a
side channel that no log handler, severity filter or captured-stdout test controls.
Enforcement is by AST, not substring, so the word ``print`` inside a docstring, a
comment or a log message is not a false positive - only a real call to ``print(...)`` is.

Two modules are allowlisted, each with a reason; new exceptions must earn their place in
this list, never silently add a ``print`` to the library.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# file (relative to src/) -> why a console print is legitimate there.
ALLOWED_FILES: dict[str, str] = {
    # The ``smw-pinn`` console runner writes human-readable diagnostics to stdout by
    # design (the Makefile mirror); it never runs inside the library import graph.
    "cli.py": "cross-platform console entry point (Makefile replacement)",
    # Emits the reference-baseline metrics as a pipeable JSON document to stdout
    # (`python -m src.evaluation.analytical_baselines | jq`); logging would corrupt it.
    "evaluation/analytical_baselines.py": "machine-readable JSON stdout contract",
}


def _print_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                lines.append(node.lineno)
    return sorted(lines)


def test_no_undocumented_print_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC).as_posix()
        calls = _print_calls(path)
        if not calls:
            continue
        if relative not in ALLOWED_FILES:
            offenders.append(f"{relative}: print() at lines {calls}")
    assert not offenders, (
        "print() in src/ bypasses the logging convention (see CONTRIBUTING.md); "
        "use src.utils.logging.get_logger, or add a documented exception to "
        f"ALLOWED_FILES. Offenders: {offenders}"
    )


def test_allowlist_has_no_dead_entries() -> None:
    """An allowlisted file that no longer prints is stale and must be removed."""
    stale = []
    for relative in ALLOWED_FILES:
        path = SRC / relative
        if not path.is_file() or not _print_calls(path):
            stale.append(relative)
    assert not stale, f"remove stale print allowlist entries: {stale}"
