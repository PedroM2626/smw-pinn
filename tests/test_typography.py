r"""Unit tests for src/utils/typography.py - the README's math budget and delimiter rules.

The function decides what stops being math in the README, so the two directions both
matter: a typographic span that keeps its `$…$` costs GitHub a render slot, and a real
formula that loses them silently changes the document's meaning. These tests pin the
boundary and the idempotence the gates rely on.

They also pin the second job the rendered page forced on it: GitHub does not read a `$`
as a delimiter when it touches a word character, and forms no math inside emphasis at all,
so a span like `$\mu$s` or a formula inside `*Measured …*` costs a render slot *and* prints
as source with no error box. `unemphasise_math` is the fix for the second shape.

CPU-only.

Run:  pytest tests/test_typography.py -q
"""

from src.utils.typography import demath_typographic, replacement_for, unemphasise_math


def test_typographic_spans_become_plain_text() -> None:
    assert demath_typographic("drift of $-4.25$ px") == "drift of -4.25 px"
    assert demath_typographic("mean $\\pm$ std") == "mean \u00b1 std"
    assert demath_typographic("$R^2$ of 0.99") == "R\u00b2 of 0.99"
    assert demath_typographic("in $12\\text{ ms}$") == "in $12\\text{ ms}$"  # not typographic
    assert demath_typographic("$41.89$ px and $\\mu\\text{s}$") == "41.89 px and \u00b5s"


def test_mathematics_keeps_its_delimiters() -> None:
    for kept in (
        r"$v_x$",
        r"$X_{t+1} = X_t + \frac{v_{x, t}}{16.0}$",
        r"$1.6\times10^{3}$",
        r"$p < 0.05$",
        r"$n = 5$",
        r"$2.1x$",
    ):
        assert demath_typographic(kept) == kept, kept
    assert replacement_for(r"\pm") == "\u00b1"
    assert replacement_for("v_x") is None


def test_code_spans_are_left_alone() -> None:
    """A `$` inside a code span is literal text for GitHub, not a delimiter."""
    line = "at `$7E:0094` with $72$ sub-pixels and `$N = 200$`"
    assert demath_typographic(line) == "at `$7E:0094` with 72 sub-pixels and `$N = 200$`"


def test_conversion_is_idempotent_and_leaves_plain_text_untouched() -> None:
    once = demath_typographic("covers $623.06 \\pm 17.55$ px ($+3.16$, $0.8125$)")
    assert once == "covers $623.06 \\pm 17.55$ px (+3.16, 0.8125)"
    assert demath_typographic(once) == once
    plain = "no math here, and 72 is already a number"
    assert demath_typographic(plain) == plain


def test_a_number_with_its_sign_is_typography_too() -> None:
    """The spans that cost GitHub a render slot *and* print raw, because a `$` touching a
    word character is not a delimiter: `$\\mu$s`, `$\\approx$173`, `frame$^2$`."""
    assert demath_typographic("344.73 $\\mu$s / step") == "344.73 \u00b5s / step"
    assert demath_typographic("stalls at $\\approx$173 frames") == "stalls at \u2248173 frames"
    assert demath_typographic("a $122\\times$ acceleration") == "a 122\u00d7 acceleration"
    assert demath_typographic("(subpixels/frame)$^2$") == "(subpixels/frame)\u00b2"
    assert demath_typographic("exact $0.0\\%$ violation") == "exact 0.0% violation"
    assert (
        demath_typographic("the **DAgger** policy at $\\pm 0$")
        == "the **DAgger** policy at \u00b10"
    )
    # and a relation between two quantities is not a number
    assert demath_typographic(r"$\sim 10^{5}$ orders") == r"$\sim 10^{5}$ orders"


def test_a_note_keeps_its_italics_on_the_label_only() -> None:
    """GitHub forms no math inside emphasis, so a whole-line note has to stop emphasising."""
    note = (
        "*Measured on the 6,329 transitions (§10.54): exact on 93.82% against $\\hat v_{x,t+1}$.*"
    )
    assert unemphasise_math(note) == (
        "*Measured:* on the 6,329 transitions (\u00a710.54): exact on 93.82% against "
        "$\\hat v_{x,t+1}$."
    )
    caption = "   *Left: the $|v_x|$ histogram of both recordings.*"
    assert unemphasise_math(caption) == "   *Left:* the $|v_x|$ histogram of both recordings."
    # a note with nothing to render keeps its whole-line italics
    assert unemphasise_math("*No formula in this note.*") == "*No formula in this note.*"
    # a formula before the label's colon cannot be rescued by moving the colon
    stuck = "*Measured $x$ early: the rest of the sentence.*"
    assert unemphasise_math(stuck) == stuck
    # and a list bullet is not a note
    bullet = "* A bullet whose tail ends in emphasis $v_x$*"
    assert unemphasise_math(bullet) == bullet
