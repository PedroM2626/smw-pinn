r"""
typography.py
One rule for how numbers and signs are written in the README.

GitHub renders at most ~1,368 math expressions in a README and then fails every remaining
one with the generic "Unable to render expression." - measured on this repository's own
page, where the first 1,368 came out as formulas and the next 300 did not, however valid
their LaTeX. A 400-page research README therefore cannot put *everything* in math: the
budget has to be spent on the expressions that need it.

`demath_typographic` converts the spans that were never mathematics:

* a lone operator - `$\pm$`, `$\times$` - and a unit or a superscript that has a code
  point of its own (`$R^2$`, `$\mu\text{s}$`);
* a span whose whole content is a number (`$72$`, `$-2.80$`, `$24.8%$`).

Symbols, subscripts and formulas keep their math. The function is applied by the renderers
that emit README rows and by the gates that check them, so the document and its checks
agree on presentation from one definition rather than by coincidence of typed strings.

Run:  python -c "from src.utils.typography import demath_typographic as d; print(d('$72$'))"
"""

import re
from typing import Optional

# Typographic spans: content that a Unicode code point or a bare number already expresses.
_SWAP: dict = {
    r"\pm": "\u00b1",  # ±
    r"\times": "\u00d7",  # ×
    r"R^2": "R\u00b2",  # R²
    r"\mu\text{s}": "\u00b5s",  # µs
}
_NUMBER = re.compile(r"^[-+]?[0-9][0-9,]*(\.[0-9]+)?%?$")
_CODE_SPAN = re.compile(r"`[^`]*`")

__all__ = ["demath_typographic", "replacement_for"]


def replacement_for(content: str) -> Optional[str]:
    """The plain-text form of a span's content, or None when the span is mathematics."""
    if content in _SWAP:
        return _SWAP[content]
    if _NUMBER.match(content):
        return content
    return None


def demath_typographic(text: str) -> str:
    """Rewrite every typographic `$…$` span of `text` as plain text.

    Inline code spans are left alone: a `$` inside one is literal text for GitHub, and
    rewriting it would corrupt the address or the command the span documents.
    """
    mask = [True] * len(text)
    for code in _CODE_SPAN.finditer(text):
        for k in range(code.start(), code.end()):
            mask[k] = False
    positions = [k for k, char in enumerate(text) if char == "$" and mask[k]]
    pieces, cursor, changed = [], 0, False
    for a, b in zip(positions[0::2], positions[1::2]):
        repl = replacement_for(text[a + 1 : b])
        if repl is None:
            continue
        pieces.append(text[cursor:a])
        pieces.append(repl)
        cursor = b + 1
        changed = True
    if not changed:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)
