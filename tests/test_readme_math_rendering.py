r"""
test_readme_math_rendering.py
Pin the README to expressions GitHub can actually render.

`$ … $` is not a matter of taste: GitHub's tokenizer refuses to *open* a span after a
space, and it treats every other `$` as a delimiter candidate. A cell written `$ +15.71$`
therefore does not render as math - its dollars pair with their neighbours and KaTeX is
handed `15.71$, $`, which the renderer reports as "Unable to render expression." The same
mispairing comes from a WRAM address left out of a code span (`($7E:00E4 / 7E:00D8$)`),
and a raw `<` or `>` inside a span is eaten by the HTML pass before KaTeX sees it. None of
that is a LaTeX error - the markup is valid KaTeX and breaks only in the renderer the
README is read in, so no syntax checker outside GitHub itself catches it.

`violations()` re-checks the four properties that make the difference over a whole
document, the first test runs it on the README, and the second proves it fires on the
three defects it exists to catch.

Emulator-free, CPU-only, no artifacts.

Run:  pytest tests/test_readme_math_rendering.py -q
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

README = Path(__file__).resolve().parent.parent / "README.md"

DISPLAY_LINE = re.compile(r"^(\s*)\$\$([^$]*)\$\$(\s*)$")
CODE_SPAN = re.compile(r"`[^`]*`")
ADDRESS = re.compile(r"7[E-F]:[0-9A-Fa-f]{2,4}")


def math_spans(text: str) -> List[Dict[str, object]]:
    """Every `$…$` / `$$…$$` span GitHub would form, with its line number.

    Fenced code blocks are skipped and inline code spans are masked, because a `$`
    inside either one is literal text for GitHub as well as for this scanner. A line
    left with one unpaired delimiter is reported with `unpaired` set: the pairing then
    runs into the next line, which is the defect rather than a scanning artifact.
    """
    spans: List[Dict[str, object]] = []
    in_fence = False
    for number, line in enumerate(text.split("\n"), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        display = DISPLAY_LINE.match(line)
        if display:
            spans.append({"line": number, "content": display.group(2), "unpaired": False})
            continue
        mask = [True] * len(line)
        for code in CODE_SPAN.finditer(line):
            for k in range(code.start(), code.end()):
                mask[k] = False
        positions = [k for k, c in enumerate(line) if c == "$" and mask[k]]
        for a, b in zip(positions[0::2], positions[1::2]):
            spans.append({"line": number, "content": line[a + 1 : b], "unpaired": False})
        if len(positions) % 2:
            spans.append({"line": number, "content": "", "unpaired": True})
    return spans


def violations(text: str) -> List[str]:
    """Every rendering-breaking span in the document, as one report per defect."""
    out: List[str] = []
    for span in math_spans(text):
        line, content = span["line"], str(span["content"])
        if span["unpaired"]:
            out.append(f"line {line}: odd number of $ delimiters")
            continue
        if not content:
            out.append(f"line {line}: empty math span")
        if content != content.strip():
            out.append(f"line {line}: whitespace-delimited span ${content}$")
        if "<" in content or ">" in content:
            out.append(f"line {line}: raw comparison sign in ${content}$")
        if ADDRESS.search(content):
            out.append(f"line {line}: WRAM address inside a math span (${content}$)")
    return out


def test_readme_has_no_expression_github_cannot_render() -> None:
    broken = violations(README.read_text(encoding="utf-8"))
    assert not broken, "README expressions GitHub will not render:\n" + "\n".join(broken)


def test_the_detector_fires_on_each_defect_it_exists_to_catch() -> None:
    """A gate that has never failed is not evidence; corrupt one span of each kind."""
    cases = {
        "line 1: whitespace-delimited span $ +15.71$": "| arm | $ +15.71$ |\n",
        "line 1: raw comparison sign in $p < 0.05$": "the contrast gives $p < 0.05$ here\n",
        "line 1: WRAM address inside a math span ($7E:00E4 / 7E:00D8$)": (
            "tables ($7E:00E4 / 7E:00D8$) here\n"
        ),
        "line 1: odd number of $ delimiters": "mode $7E:0016 alone\n",
    }
    for expected, document in cases.items():
        assert expected in violations(document), f"{expected!r} was not detected"


def test_readme_stays_inside_githubs_math_budget() -> None:
    """GitHub renders a bounded number of math expressions per document, then stops.

    Measured on this repository's own page rather than assumed: the README held 1,759 spans,
    the first 1,368 rendered as formulas and every one of the remaining 300 came back as the
    generic "Unable to render expression." - valid LaTeX, in the right delimiters, unrecovered.
    The last third of the document was therefore unreadable however correct it was, and no
    local check of span *syntax* could have shown it.

    The bound is a budget, not a measurement of the file's quality: math is spent on the
    expressions that need it, and `src/utils/typography.py` converts the ones that were only
    ever typography (a lone `\\pm`, a bare number). 1,300 leaves headroom under the observed
    cap; if a future section needs more, the answer is de-math something else, not to raise
    this number.
    """
    spans = math_spans(README.read_text(encoding="utf-8"))
    assert len(spans) <= 1300, (
        f"README holds {len(spans)} math expressions; GitHub stops rendering near 1,368 and "
        "everything past that shows as 'Unable to render expression.' Convert typographic "
        "spans with src.utils.typography.demath_typographic instead of raising this bound."
    )


def test_display_math_is_never_a_paragraph_continuation() -> None:
    """A `$$…$$` line that follows text is inline math to GitHub, and `$…$` inside it errors."""
    lines = README.read_text(encoding="utf-8").split("\n")
    in_fence = False
    joined = []
    for number, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if DISPLAY_LINE.match(line) and number > 1 and lines[number - 2].strip():
            joined.append(f"line {number}: display math continues the previous paragraph")
    assert not joined, "\n".join(joined)


def test_no_macro_githubs_katex_build_refuses() -> None:
    """GitHub's KaTeX runs with a macro allowlist, and `\\operatorname` is not on it."""
    forbidden = ("\\operatorname", "\\middle", "\\bigl(", "\\relax")
    found = [
        f"line {span['line']}: {token}"
        for span in math_spans(README.read_text(encoding="utf-8"))
        for token in forbidden
        if token in str(span["content"])
    ]
    assert not found, "math macros GitHub will not render:\n" + "\n".join(found)


def test_the_detector_leaves_correct_markup_alone() -> None:
    """The repairs must not be over-eager: these forms are what the README uses."""
    good = (
        "| arm | 635.61 $\\pm$ 14.18 | $+15.71$, $0.3125$, $\\lt 10^{-3}$, $+0.50$ |\n"
        "addresses (`$7E:C800`, status $\\ge 8$) and mode `$7E:0100 = 0x08`\n"
        "$$\\hat{X}_{t+1} = X_t + \\frac{v_{x, t}}{16.0}$$\n"
        "```\nshell $ NOT a delimiter\n```\n"
    )
    assert not violations(good)


HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(heading: str) -> str:
    """GitHub's heading slug: markdown stripped, punctuation dropped, spaces to dashes.

    Punctuation is deleted rather than replaced, so `Overview & Abstract` keeps the two
    spaces around the ampersand and becomes `overview--abstract`.
    """
    text = re.sub(r"^#+\s*", "", heading.strip())
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("*", "").lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.strip().replace(" ", "-")


def test_every_table_of_contents_link_lands_on_a_heading() -> None:
    """A link that points at nothing is a broken expression of a different kind.

    Headings carrying math are excluded from the exact check, not because they can be
    computed but because their slug is taken from the *rendered* text and one renderer's
    `$\\to$` contributes a space while another contributes the command's letters. Their
    link is still required to start with the heading's math-free prefix, so a link that
    was retargeted at a renamed section is caught.
    """
    lines = README.read_text(encoding="utf-8").split("\n")
    in_fence = False
    slugs, math_prefixes = set(), set()
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = HEADING.match(line)
        if not heading:
            continue
        title = heading.group(2)
        if "$" in title:
            prefix = _slug(title.split("$")[0])
            if prefix:
                math_prefixes.add(prefix)
        else:
            slugs.add(_slug(title))
    broken = []
    for link in re.findall(r"\]\(#([a-z0-9\-]+)\)", "\n".join(lines)):
        if link in slugs:
            continue
        if any(link.startswith(prefix) for prefix in math_prefixes if prefix):
            continue
        broken.append(link)
    assert not broken, f"table-of-contents links with no heading to land on: {broken}"


def test_the_readme_ranges_its_own_study_sections_correctly() -> None:
    """Section 1 says "Sections 10.37-10.56 train and score ...": a range that stops short of the
    section it is printed in is a claim about a document that has already grown. This is the same
    class of error as the retired weight counts - not a wrong digit, an unstopped sentence.
    """
    text = README.read_text(encoding="utf-8")
    sections = [int(m.group(1)) for m in re.finditer(r"^### 10\.(\d+)\b", text, re.M)]
    assert sections, "the README has no 10.x sections to range over"
    last = max(sections)
    ranges = re.findall(r"Sections 10\.(\d+)-10\.(\d+) train and score", text)
    assert ranges, "section 1 no longer states the range of the study sections"
    assert len(ranges) == 1, f"more than one section claims the study range: {ranges}"
    start, end = (int(a) for a in ranges[0])
    assert (start, end) == (37, last), (
        f"the range is stated as 10.{start}-10.{end} but the study sections run "
        f"10.{min(sections)}-10.{last}"
    )


def test_every_study_section_has_a_table_of_contents_entry() -> None:
    """The other direction of the anchor check: a heading with no link is invisible.

    A 10.57 ToC entry meant for the list was pasted onto the prose of Section 12 instead, which
    left Section 12 with a stray bullet and the table of contents without a section. A link that
    lands on nothing is caught by the anchor test; a heading nobody links to, and an entry that
    sits in the body rather than in the contents, were caught by nothing at all.
    """
    text = README.read_text(encoding="utf-8")
    toc_end = text.index("## 1. Project Overview")
    in_fence = False
    offset = 0
    headings: List[str] = []
    entries: List[Tuple[str, bool]] = []
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            heading = HEADING.match(line)
            if heading and heading.group(1) == "###" and "$" not in heading.group(2):
                title = heading.group(2)
                if re.match(r"10\.\d+\s", title):
                    headings.append(_slug(title))
            entry = re.fullmatch(r"\s+\* \[[^\]]+\]\(#([a-z0-9\-]+)\)", line)
            if entry:
                entries.append((entry.group(1), offset < toc_end))
        offset += len(line) + 1
    assert len(headings) > 20, (
        f"only {len(headings)} study headings parsed: the check is not running"
    )
    linked = {anchor for anchor, in_toc in entries if in_toc}
    missing = [h for h in headings if h not in linked]
    assert not missing, f"study sections with no table-of-contents entry: {missing[:6]}"
    stray = [anchor for anchor, in_toc in entries if not in_toc]
    assert not stray, f"table-of-contents entries pasted into the body: {stray[:4]}"


def unformable_spans(text: str) -> List[str]:
    r"""Every span GitHub will not form, for the two rules measured on the rendered page.

    Rule one: a note written as a whole-line emphasis loses every formula in it - the page holds
    317 `<em>` elements and not one contains a rendered expression, so `$\hat v_{x,t+1}$` inside
    `*Measured …*` prints as source with no error box. Rule two: a `$` that touches an
    alphanumeric, or opens after a hyphen or slash, is not a delimiter - the raw set outside
    emphasis was exactly `$\mu$s`, `$\approx$173`, `frame$^2$`, `rank-$p$` and `$x$/$y$`. The
    typographic ones are converted by `src/utils/typography.unemphasise_math` and
    `demath_typographic`; the rest were reworded, which is what the second rule asks of a fix.
    """
    out: List[str] = []
    in_fence = False
    for number, line in enumerate(text.split("\n"), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or DISPLAY_LINE.match(line):
            continue
        masked = [False] * len(line)
        for code in CODE_SPAN.finditer(line):
            for k in range(code.start(), code.end()):
                masked[k] = True
        dollars = [k for k, char in enumerate(line) if char == "$" and not masked[k]]
        if len(dollars) % 2:
            continue  # already reported as an unpaired line
        whole_note = re.match(r"^\s*\*([^*\s].*)\*\s*$", line)
        if whole_note and "$" in whole_note.group(1):
            out.append(f"line {number}: a note in whole-line emphasis cannot hold math")
            continue
        for a, b in zip(dollars[0::2], dollars[1::2]):
            span = line[a : b + 1]
            before = line[a - 1] if a else ""
            after = line[b + 1] if b + 1 < len(line) else ""
            if re.match(r"[A-Za-z0-9\-/]", before) or re.match(r"[A-Za-z0-9/]", after):
                out.append(f"line {number}: delimiter touches a word or a slash ({span[:44]})")
    return out


def test_no_span_is_written_where_github_cannot_form_it() -> None:
    """Raw LaTeX with no error box - the defect neither the budget nor the tokenizer rules saw."""
    defects = unformable_spans(README.read_text(encoding="utf-8"))
    assert not defects, "spans GitHub leaves as raw LaTeX:\n" + "\n".join(defects[:12])
