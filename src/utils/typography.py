r"""
typography.py
One rule for how numbers and signs are written in the README.

GitHub renders at most ~1,368 math expressions in a README and then fails every remaining
one with the generic "Unable to render expression." - measured on this repository's own
page, where the first 1,368 came out as formulas and the next 300 did not, however valid
their LaTeX. A 400-page research README therefore cannot put *everything* in math: the
budget has to be spent on the expressions that need it.

`demath_typographic` converts the spans that were never mathematics:

* a lone operator or unit - `$\pm$`, `$\times$`, `$\approx$`, `$\sim$`, `$\mu$`, `$^2$` -
  and a unit or a superscript that has a code point of its own (`$R^2$`, `$\mu\text{s}$`);
* a span whose whole content is a number, optionally trailed by the one sign that made it
  a span (`$72$`, `$-2.80$`, `$24.8%$`, `$5.2\times$`, `$\pm 0$`).

That second job is not cosmetic. GitHub also refuses to *form* a math span whose delimiter
touches a word character - `344.73 $\mu$s` and `rank-$p$` print as raw LaTeX with no error
box at all - and converting the typographic ones removes them from that hazard, which is why
the same function is applied by the renderers and by the gates.

Symbols, subscripts and formulas keep their math. The function is applied by the renderers
that emit README rows and by the gates that check them, so the document and its checks
agree on presentation from one definition rather than by coincidence of typed strings.

Run:  python -c "from src.utils.typography import demath_typographic as d; print(d('$72$'))"
"""

import re
from typing import Optional

# Typographic spans: content that a Unicode code point or a bare number already expresses.
_SWAP: dict = {
    r"\pm": "±",  # ±
    r"\times": "×",  # ×
    r"\approx": "≈",  # ≈
    r"\sim": "~",  # ~
    r"\mu": "µ",  # µ
    r"^2": "²",  # ²
    r"R^2": "R²",  # R²
    r"\mu\text{s}": "µs",  # µs
}
# A number, optionally trailed by the one sign that made it a span at all. Formulas are not
# numbers: `10^5` and `\sim 3.0` keep their math.
_NUMBER = re.compile(r"^[-+]?[0-9][0-9,]*(\.[0-9]+)?(%|\\\%|\\times|\\approx)?$")
_PM_NUMBER = re.compile(r"^\\pm ([-+]?[0-9][0-9,]*(\.[0-9]+)?)$")
_CODE_SPAN = re.compile(r"`[^`]*`")

__all__ = ["demath_typographic", "replacement_for", "unemphasise_math"]


def replacement_for(content: str) -> Optional[str]:
    """The plain-text form of a span's content, or None when the span is mathematics."""
    if content in _SWAP:
        return _SWAP[content]
    if _NUMBER.match(content):
        return content.replace(r"\%", "%").replace(r"\times", "×").replace(r"\approx", "≈")
    pm = _PM_NUMBER.match(content)
    if pm:
        return f"±{pm.group(1)}"
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


WHOLE_LINE_ITALIC = re.compile(r"^(\s*)\*([^*].*[^*])\*(\s*)$")


def unemphasise_math(text: str) -> str:
    r"""Keep the label of an italic note, take the italics off the rest of it.

    Measured on this repository's own page: it holds 317 emphasis runs and not one of them
    contains a rendered expression - GitHub forms no math inside `<em>`, so a note written as
    `*Measured: exact on 93.82% against $\hat v_{x,t+1}$ …*` shows that formula as raw LaTeX,
    with no error box to say so. The italics stay on the label - the part before the first
    colon, or the first word when the colon is not a label - and the sentence continues plain.
    """
    match = WHOLE_LINE_ITALIC.match(text)
    if not match or "$" not in match.group(2):
        return text
    indent, body = match.group(1), match.group(2)
    cut = body.find(":")
    if cut < 0 or "$" in body[:cut]:
        return text
    if cut > 12:
        head, _, tail = body.partition(" ")
        if not tail:
            return text
        return f"{indent}*{head.strip()}:* {tail}"
    label = body[: cut + 1].strip()
    rest = body[cut + 1 :].strip()
    if not rest:
        return text
    return f"{indent}*{label}* {rest}"
