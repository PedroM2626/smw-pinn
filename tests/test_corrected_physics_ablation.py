r"""Gates for README 10.57: every cell of its tables, and every quantifier in its prose.

Section 10.57 quotes four generated tables and six findings. The tables are compared row by row
against `render_*` output, so a number cannot be retyped by hand; the findings are re-checked as
*claims over the artifact* - the sentence "at 72.0 every arm flags 0.0000" is a quantifier over
sixteen numbers, and a quantifier is exactly what a per-cell check cannot see.

CPU-only, reads `results/corrected_physics_ablation_metrics.json`; skipped when the study has not
been run.

Run:  pytest tests/test_corrected_physics_ablation.py -q
"""

import json
import re
from pathlib import Path

import pytest

from src.evaluation.corrected_physics_ablation import (
    render_convention_table,
    render_ground_rule_table,
    render_replay_table,
    render_ruler_table,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "corrected_physics_ablation_metrics.json"
CHECKPOINTS = REPO / "results" / "checkpoints"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the 10.57 ablation has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme():
    return (REPO / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "renderer",
    [
        render_convention_table,
        render_ground_rule_table,
        render_ruler_table,
        render_replay_table,
    ],
    ids=lambda fn: fn.__name__,
)
def test_every_table_of_10_57_is_the_artifact(renderer) -> None:
    """One row per arm per table, compared as generated - nothing transcribed."""
    rows = renderer(_payload())
    assert rows, f"{renderer.__name__} produced no rows"
    text = _readme()
    missing = [row for row in rows if row not in text]
    assert not missing, (
        f"README 10.57 disagrees with {renderer.__name__} on {len(missing)} rows:\n"
        + "\n".join(missing[:4])
    )


SHELL_PREFIXES = ("Hard PINN/", "engine rules/")
UNCLAMPED = ("MLP/state/soft", "DeepONet/state/soft", "FNO/state/soft")


def _is_shell(arm: str) -> bool:
    return arm.startswith(SHELL_PREFIXES) or "/residual/hard/" in arm


def test_the_ruler_silence_is_a_property_of_the_clamp_not_of_the_model() -> None:
    """Finding 5's two halves: the clamped arms cannot exceed 72.0, the penalised ones do.

    The first version of this section said "at 72.0 every arm flags 0.0000", which the artifact
    contradicts - three of the four unclamped arms exceed the bound. That is the whole point of the
    finding, so both directions are pinned here: the clamp arms are silent by construction, the
    penalty arms are not, and the ranges the README quotes are recomputed from each subset.
    """
    ruler = _payload()["ruler_within_arm"]
    assert ruler, "no arms were scored"
    text = _readme()

    shells = {arm: block for arm, block in ruler.items() if _is_shell(arm)}
    assert len(shells) == 10, sorted(shells)
    silent = [arm for arm, block in shells.items() if block["at_published_bound"] != 0.0]
    assert not silent, f"a clamped arm exceeded the bound it enforces: {silent}"
    shell_moved = sorted(
        block["difference"] for block in shells.values() if block["difference"] > 0.0
    )
    assert f"{shell_moved[0]:.4f}-{shell_moved[-1]:.4f}" in text, (
        f"README's clamped-arm range is not {shell_moved[0]:.4f}-{shell_moved[-1]:.4f}"
    )

    open_arms = {arm: block for arm, block in ruler.items() if arm.startswith(UNCLAMPED)}
    assert len(open_arms) == 6, sorted(open_arms)
    breaching = {
        arm: block["at_published_bound"]
        for arm, block in open_arms.items()
        if block["at_published_bound"] > 0.0
    }  # noqa: E501
    assert len(breaching) == 3, f"the penalty arms that exceed 72.0 changed: {breaching}"
    for arm, rate in breaching.items():
        assert f"{rate:.2%}" in text, f"README no longer quotes {arm}'s {rate:.2%} at 72.0"


def test_the_carried_position_error_is_identical_across_the_families() -> None:
    """The section says one number covers several models; the artifact has to support that."""
    summary = _payload()["summary"]
    carried = {
        arm: cells["x_mae_px"]["mean"]
        for arm, cells in summary.items()
        if arm.endswith("/carried") and "x_mae_px" in cells
    }
    assert len(carried) >= 4, f"too few carried arms to make the claim: {sorted(carried)}"
    assert max(carried.values()) - min(carried.values()) < 1e-9, carried
    assert f"{next(iter(carried.values())):.4f} px" in _readme()


def test_the_parity_control_is_quoted_as_the_artifact_computes_it() -> None:
    """Finding 6: default arms must reproduce 10.47, and the section must not round that up."""
    parity = _payload()["verdict"]["parity_with_10_47"]
    assert parity["available"], "the 10.47 grid artifact was missing, so parity was never tested"
    assert parity["identical"], parity["max_abs_difference_by_metric"]
    text = _readme()
    assert (
        f"{parity['cell_seed_pairs_compared']}\n" in text
        or str(parity["cell_seed_pairs_compared"]) in text
    ), "the number of compared pairs is not the one the artifact holds"


def test_finding_1s_tolerance_range_is_the_shell_arms_only() -> None:
    """The 0.9xxx range must be the next-convention shells, not some other subset of arms."""
    summary = _payload()["summary"]
    shells = {
        arm: cells["kinematic_violation_rate_tight"]["mean"]
        for arm, cells in summary.items()
        if arm.endswith("/next") and ("hard" in arm or arm.startswith("Hard PINN"))
    }
    assert len(shells) == 4, shells
    values = sorted(shells.values())
    assert f"{values[0]:.4f}-{values[-1]:.4f}" in _readme()
    carried_zero = [
        arm
        for arm, cells in summary.items()
        if arm.endswith("/carried") and cells["kinematic_violation_rate_tight"]["mean"] != 0.0
    ]
    assert not carried_zero, f"a carried arm is not exact at 0.002 px: {carried_zero}"


def test_the_study_published_no_checkpoint() -> None:
    """Retraining the flagship must not silently re-date the weights the paper cites.

    The guard is on the filesystem as well as the protocol flag: `DynamicsTrainer` writes
    `<model_type>_best.pt` wherever `save_dir` points, so the only proof that 10.57 left the
    committed lineage alone is that no `corrphys_*` file exists next to them.
    """
    assert _payload()["protocol"]["checkpoints_published"] is False
    strays = [] if not CHECKPOINTS.is_dir() else list(CHECKPOINTS.glob("corrphys_*"))
    assert not strays, f"10.57 wrote committed checkpoints: {strays}"


def test_the_section_names_which_axes_it_changed() -> None:
    """A correction section that does not say what it left alone invites the wrong reading."""
    text = _readme()
    section = text[text.index("### 10.57") : text.index("## 11. ")]
    for phrase in (
        "No closed loop",
        "Five seeds",
        "button latch",
        "`*/soft` cells",
    ):
        assert phrase in section, f"10.57's limitations dropped {phrase!r}"
    assert re.search(r"position_velocity=\"carried\"", section), "the flag name is not stated"
