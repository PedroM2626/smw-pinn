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


def _tuple_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _satisfies(version: str, spec: str) -> bool:
    """True when one version falls inside a ``>=a,<b`` specifier."""
    have = _tuple_version(version)
    for clause in re.split(r"\s*,\s*", spec.strip()):
        match = re.match(r"(>=|<=|==|!=|>|<)\s*([\d.]+)", clause)
        if not match:
            continue
        op, want = match.group(1), _tuple_version(match.group(2))
        ok = {
            ">=": have >= want,
            ">": have > want,
            "<=": have <= want,
            "<": have < want,
            "==": have == want,
            "!=": have != want,
        }[op]
        if not ok:
            return False
    return True


def test_the_ruff_version_is_one_version_in_every_place_that_names_it():
    """The formatter that writes a file must be the version that checks it.

    Four files name ruff: the band in `requirements.txt` and in `pyproject.toml`, the `rev:`
    of the pre-commit hook, and the pin in `requirements.lock`. The band is what CI installs
    and the rev is what a local commit runs, so a widening that moves one and not the other
    reproduces the defect this repository already documents - a tree formatted by one version
    and checked by another, where every file looks dirty. Dependabot PR #9 was exactly that
    shape: it widened the band and could not see the hook.
    """
    reqs = _requirements_txt()
    spec = reqs["ruff"]
    assert spec, "requirements.txt declares ruff with no version bound"

    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    block = re.search(
        r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*(\S+)", config
    )
    assert block, "no ruff-pre-commit rev to compare against the band"
    rev = block.group(1).lstrip("v")

    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    pinned = re.search(r"^ruff==(\S+)", lock, re.M)
    assert pinned, "requirements.lock does not pin ruff"

    assert _satisfies(rev, spec), f"pre-commit rev v{rev} is outside the band {spec}"
    assert _satisfies(pinned.group(1), spec), f"lock pin {pinned.group(1)} is outside {spec}"
    # The band is only the envelope; the two exact statements are what decide which binary
    # formats a file and which judges it, and a partial bump that moves one of them is the
    # desync the comment in requirements.txt warns about.
    assert _tuple_version(rev) == _tuple_version(pinned.group(1)), (
        f"the hook runs ruff {rev} while the lock pins {pinned.group(1)}: one version must "
        "format the tree and another would check it"
    )

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for extra in ("dev", "all"):
        declared = _pyproject_array(pyproject, extra).get("ruff", "")
        assert re.sub(r"\s+", "", declared) == spec, (
            f"pyproject [{extra}] says {declared!r} but requirements.txt says {spec!r}"
        )
