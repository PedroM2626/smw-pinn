r"""Unit tests for src/utils/typography.py - the README's math budget rule.

The function decides what stops being math in the README, so the two directions both
matter: a typographic span that keeps its `$…$` costs GitHub a render slot, and a real
formula that loses them silently changes the document's meaning. These tests pin the
boundary and the idempotence the gates rely on.

CPU-only.

Run:  pytest tests/test_typography.py -q
"""

from src.utils.typography import demath_typographic, replacement_for


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
