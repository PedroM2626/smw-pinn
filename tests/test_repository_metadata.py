r"""Offline gates on the metadata files that name the repository and its version.

The repository was renamed, and the rename left nine hard-coded URLs pointing at the old path -
reachable only because GitHub redirects, which is the kind of thing that quietly outlives its
welcome. A tag then made two CHANGELOG links resolve and a citation version become checkable.
These tests are offline by design: reachability is a network fact and this suite must not depend
on it, so what is gated here is internal agreement, which is where the drift actually happened.

Run:  pytest tests/test_repository_metadata.py -q
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: files that name the repository or its version, and are meant to agree
SOURCES = (
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    ".pre-commit-config.yaml",
    ".github/ISSUE_TEMPLATE/config.yml",
    "CHANGELOG.md",
    "README.md",
)


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def test_no_published_url_still_names_the_old_repository_path() -> None:
    """A renamed repository must not keep pointing its citation and metadata at the old path."""
    offenders = {
        name: [line.strip() for line in _read(name).splitlines() if "PedroM2626/smw-pinn" in line]
        for name in SOURCES
    }
    stale = {name: lines for name, lines in offenders.items() if lines}
    assert not stale, "URLs still naming the pre-rename repository: " + str(stale)


def _declared_version(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.M)
    assert match, f"no version matched by {pattern}"
    return match.group(1)


def test_the_declared_version_is_one_version_everywhere_it_is_written() -> None:
    """The citation, the package and the changelog links must describe the same release."""
    citation = _declared_version(_read("CITATION.cff"), r"^version:\s*([\d.]+)\s*$")
    package = _declared_version(_read("pyproject.toml"), r'^version\s*=\s*"([\d.]+)"')
    assert citation == package, f"CITATION.cff declares {citation} but the package is {package}"

    changelog = _read("CHANGELOG.md")
    links = re.findall(r"^\[([\d.]+)\]:\s*(\S+)", changelog, flags=re.M)
    tagged = dict(links)
    assert citation in tagged, (
        f"the changelog has no release link for the declared version {citation}: {sorted(tagged)}"
    )
    assert f"/releases/tag/v{citation}" in tagged[citation], (
        f"the changelog's v{citation} link does not name the tag it claims: {tagged[citation]}"
    )
    assert f"v{citation}...HEAD" in changelog, (
        "the changelog's Unreleased comparison must span from the declared release to HEAD"
    )


def test_the_metadata_names_exactly_one_repository() -> None:
    """Every GitHub address written in the metadata must point at this repository.

    The check is on the owner/repo pair rather than on a fixed string, so it can tell a stale
    path from a deliberate link to someone else's work - the upstream tool repositories the
    comments cite are allowed, this project's own citation and metadata are not allowed to
    disagree with each other.
    """
    pairs = set()
    for name in SOURCES:
        pairs.update(re.findall(r"github\.com/([\w.-]+)/([\w.-]+)", _read(name)))
    ours = {(owner, repo) for owner, repo in pairs if owner == "PedroM2626"}
    assert ours == {("PedroM2626", "smw-sciml")}, (
        f"the metadata points at {sorted(ours)}; a renamed repository must not leave a second "
        "address of its own behind"
    )
    for required in ("CITATION.cff", "pyproject.toml", "CHANGELOG.md"):
        assert "PedroM2626/smw-sciml" in _read(required), f"{required} no longer names the repo"
