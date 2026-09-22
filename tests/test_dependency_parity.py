"""Manifest parity: requirements.txt must not drift from pyproject.toml.

Root cause of a real gap found in this repository: `requirements.txt` (used by
the Dockerfile and README section 11.2) omitted `ruff`, so an environment built
from it could not run `make lint`. This test makes that class of drift fail
loudly instead of surfacing as a confusing tooling error.

Parsed with regexes rather than `tomllib` because the project supports
Python 3.10 (where `tomllib` does not exist yet) and adds no new dependency.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIREMENT_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\])?\s*([<>=!~]=?.*)?$")


def _requirements_txt() -> dict[str, str]:
    """name (lowercase, normalized) -> specifier string from requirements.txt."""
    found: dict[str, str] = {}
    for raw in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):  # skip comments and --flag lines
            continue
        match = REQUIREMENT_LINE.match(line)
        if match:
            found[match.group(1).lower().replace("_", "-")] = (match.group(3) or "").strip()
    return found


def _pyproject_array(text: str, key: str) -> dict[str, str]:
    """Parse an inline array such as ``dev = [ "pytest>=7.0.0", ... ]``."""
    block = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", text, re.M | re.S)
    if not block:
        return {}
    out: dict[str, str] = {}
    for spec in re.findall(r'"([^"]+)"', block.group(1)):
        match = REQUIREMENT_LINE.match(spec.strip())
        if match:
            out[match.group(1).lower().replace("_", "-")] = (match.group(3) or "").strip()
    return out


def test_every_runtime_dependency_is_declared_in_both_manifests():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runtime = _pyproject_array(pyproject, "dependencies")
    assert runtime, "could not parse [project].dependencies"
    missing = sorted(set(runtime) - set(_requirements_txt()))
    assert not missing, f"in pyproject but missing from requirements.txt: {missing}"


def test_dev_extras_are_declared_in_requirements_txt():
    """`pip install -r requirements.txt` must be able to run the quality gates."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev = _pyproject_array(pyproject, "dev")
    assert "ruff" in dev and "mypy" in dev, "pyproject [dev] lost a linter"
    reqs = _requirements_txt()
    missing = sorted(set(dev) - set(reqs))
    assert not missing, f"dev extras missing from requirements.txt: {missing}"


def test_version_specifiers_match():
    """Same package in both files means the *same* bound, not just presence."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = {**_pyproject_array(pyproject, "dependencies"), **_pyproject_array(pyproject, "dev")}
    reqs = _requirements_txt()
    mismatched = {
        name: (reqs[name], spec)
        for name, spec in declared.items()
        if name in reqs and re.sub(r"\s+", "", spec) != reqs[name]
    }
    assert not mismatched, f"requirements.txt vs pyproject spec drift: {mismatched}"


def test_torch_envelope_is_still_capped():
    """The validated CUDA envelope must not be silently widened (README 11.6)."""
    reqs = _requirements_txt()
    assert "<2.7" in reqs["torch"], "torch upper cap changed: re-validate on RTX hardware first"
    assert "<0.22" in reqs["torchvision"], "torchvision upper cap changed"
    assert "<2.1" in reqs["numpy"], "numpy upper cap changed"
