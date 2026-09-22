"""
provenance.py
Machine-readable "where did this number come from?" metadata.

Every published metric in this repository lives in a flat JSON under `results/`,
but the files themselves carried no record of the seed, commit, library versions
or wall-clock time that produced them - which weakens the reproducibility claim
in README section 12. New artifacts written through :func:`write_metrics` embed a
`_meta` block; :func:`read_metrics` tolerates the legacy files that lack one.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.paths import REPO_ROOT


def git_sha() -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> Optional[bool]:
    """True/False when the working tree state is knowable, None otherwise."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> Dict[str, Any]:
    """Library/hardware versions that materially change the numbers."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:  # pragma: no cover - torch is a hard dependency
        info["torch"] = "missing"
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def build_provenance(
    seed: int | None = None,
    command: str | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the `_meta` block embedded in result artifacts."""
    meta: Dict[str, Any] = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        **environment_info(),
    }
    if seed is not None:
        meta["seed"] = int(seed)
    if command:
        meta["command"] = command
    if extra:
        meta.update(extra)
    return meta


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null so the artifact stays strict JSON.

    `json.dump` happily writes bare `NaN`/`Infinity`, which is not valid JSON: a
    single degenerate R-squared (constant ground truth) would otherwise make the
    artifact unreadable by every non-Python consumer and by `jq` in the CI logs.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_metrics(
    path: str | os.PathLike[str],
    payload: Dict[str, Any],
    seed: int | None = None,
    command: str | None = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write `payload` as JSON with an embedded `_meta` provenance block."""
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)

    document = dict(payload)
    meta = document.pop("_meta", {})
    meta.update(build_provenance(seed=seed, command=command, extra=extra_meta))
    document = {"_meta": _json_safe(meta), **_json_safe(document)}
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, sort_keys=False, default=str, allow_nan=False)
    return str(target)


def read_metrics(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a result artifact (legacy files without `_meta` load unchanged)."""
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    with open(target, encoding="utf-8") as fh:
        return json.load(fh)
