"""Gate: the repository is English-only.

Reviewers, the paper and every downstream reader work in English, yet the
repository had accumulated Portuguese prose in scripts, comments and docs - which
only becomes visible when someone searches for a concept and finds half of it under
a different language.

Non-ASCII *symbols* stay allowed (the README is full of `$7E:0094`, $v_x$, arrows and
math signs): accented Latin letters and Cyrillic are the fingerprint of prose written
in another language, and unambiguous Portuguese words are caught even when the file
is otherwise ASCII.
"""

import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_DIRS = ["src", "tests", "scripts", "configs", ".github"]
SCANNED_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "Makefile",
    "Dockerfile",
    "pyproject.toml",
    "requirements.txt",
    "results/MANIFEST.md",
    "CITATION.cff",
]

# Latin-1 supplement + Latin extended A/B (minus the multiplication/division signs,
# which are math, not language) and Cyrillic.
ACCENTED_OR_CYRILLIC = re.compile(
    "[À-ÖØ-öø-ƿ"  # U+00C0-U+00D6, U+00D8-U+00F6, U+00F8-U+017F
    "\u0180-\u024f"  # Latin extended B
    "\u0400-\u04ff"  # Cyrillic
    "]"
)
# Portuguese function words that cannot appear in English prose.
PORTUGUESE_WORDS = re.compile(
    r"\b(nao|está|esta\b|estão|você|vosso|também|tambem|nós|nosso|ação|arquivo|"
    r"código|função|método|resultado|tabela|verificar|execução|caminho|pasta|"
    r"usuário|pré-?treino|pós-?treino)\b",
    re.IGNORECASE,
)


# This file necessarily contains the patterns it looks for, so it cannot police itself.
SELF = Path(__file__).name


def _scanned_files() -> list[Path]:
    out: list[Path] = []
    for folder in SCANNED_DIRS:
        out += [
            p
            for p in (REPO_ROOT / folder).rglob("*")
            if p.is_file()
            and p.name != SELF
            and "__pycache__" not in p.parts
            and p.suffix in {".py", ".md", ".yaml", ".yml"}
        ]
    out += [REPO_ROOT / name for name in SCANNED_FILES if (REPO_ROOT / name).is_file()]
    return sorted(set(out))


FILES = _scanned_files()


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_file_is_english_only(path: Path) -> None:
    relative = path.relative_to(REPO_ROOT)
    text = path.read_text(encoding="utf-8")  # a decode error is itself a failure
    for lineno, line in enumerate(text.splitlines(), start=1):
        letters = ACCENTED_OR_CYRILLIC.findall(line)
        assert not letters, (
            f"{relative}:{lineno} uses non-English characters "
            f"({', '.join(sorted({unicodedata.name(c, repr(c)) for c in letters}))}): "
            f"{line.strip()[:120]}"
        )
        portuguese = PORTUGUESE_WORDS.search(line)
        assert not portuguese, (
            f"{relative}:{lineno} contains the Portuguese word "
            f"{portuguese.group(0)!r}: {line.strip()[:120]}"
        )


def test_the_guard_scans_the_repository() -> None:
    """A guard that silently scans nothing is worse than no guard."""
    assert len(FILES) > 120, f"only {len(FILES)} files are covered"
    covered = {p.relative_to(REPO_ROOT).parts[0] for p in FILES}
    for folder in SCANNED_DIRS:
        assert folder in covered, f"{folder}/ is not covered by the English-only guard"
