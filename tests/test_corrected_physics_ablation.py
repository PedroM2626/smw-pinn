r"""Gates for README 10.57 and 10.57.1: every cell, and every quantifier in the prose.

Section 10.57 quotes four generated tables and six findings; 10.57.1 adds the console table,
its four contrasts and five findings. The tables are compared row by row against `render_*`
output, so a number cannot be retyped by hand; the findings are re-checked as *claims over the
artifact* - "the largest $d_z$ in the study belongs to the arm that dies every time" and "two of
the three families flip sign between the rollout and the console" are quantifiers over many
numbers, and a quantifier is exactly what a per-cell check cannot see.

CPU-only, reading `results/corrected_physics_ablation_metrics.json` and
`results/corrected_physics_mpc_metrics.json`; skipped when the study has not been run.

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


def test_the_published_weights_are_exactly_the_arms_the_console_flies() -> None:
    """Weights appear only for flown arms, under a prefix no other study reads.

    Retraining the flagship must not silently re-date the checkpoints the paper cites, so the
    study publishes nothing unless asked (`--save-checkpoints`), and then exactly the eight
    10.57.1 flies. Two directions are checked and neither is the one a smoke run can break: a
    file outside the eight names means an arm wrote over something it does not own, and when a
    run says it published, a missing file means the closed loop would load a checkpoint that
    was never trained.
    """
    from src.evaluation.corrected_physics_ablation import FLYABLE_LABELS, checkpoint_name

    payload = _payload()
    allowed = {checkpoint_name(label) for label in FLYABLE_LABELS}
    on_disk = (
        set() if not CHECKPOINTS.is_dir() else {p.name for p in CHECKPOINTS.glob("corrphys_*")}
    )
    assert on_disk <= allowed, f"weights outside the flown set: {sorted(on_disk - allowed)}"
    if payload["protocol"]["checkpoints_published"]:
        assert on_disk == allowed, f"missing {sorted(allowed - on_disk)}"


def test_no_other_study_weight_was_touched() -> None:
    """The prefix is a promise: `corrphys_` is the only thing this study may write."""
    from src.evaluation.corrected_physics_ablation import checkpoint_name

    names = {checkpoint_name(label) for label in _payload()["arms"]}
    foreign = [n for n in names if not n.startswith("corrphys_")]
    assert not foreign, f"10.57 would write outside its own prefix: {foreign}"


def test_the_section_names_which_axes_it_changed() -> None:
    """A correction section that does not say what it left alone invites the wrong reading."""
    text = _readme()
    section = text[text.index("### 10.57") : text.index("## 11. ")]
    for phrase in (
        "Five seeds",
        "button latch",
        "`*/soft` cells",
    ):
        assert phrase in section, f"10.57's limitations dropped {phrase!r}"
    assert re.search(r"position_velocity=\"carried\"", section), "the flag name is not stated"


MPC_ARTIFACT = REPO / "results" / "corrected_physics_mpc_metrics.json"


def _manifest_text() -> str:
    return (REPO / "results" / "MANIFEST.md").read_text(encoding="utf-8")


def test_the_runner_registers_this_study_under_its_own_artifact_name() -> None:
    """The runner is shared by five studies; writing to another study's file is the failure."""
    from src.evaluation.corrected_physics_ablation import FLYABLE_LABELS
    from src.evaluation.physics_injection_mpc_benchmark import STUDIES

    assert STUDIES["corrected"][1] == MPC_ARTIFACT.name
    assert len(STUDIES["corrected"][0]) == len(FLYABLE_LABELS) == 8
    assert f"| `{MPC_ARTIFACT.name}` |" in _manifest_text(), "the artifact is not indexed"


def test_the_closed_loop_declares_the_weights_it_loaded() -> None:
    """The lineage of a closed-loop row is part of what it proves.

    `test_declared_inputs_are_not_newer_than_the_result` compares commit dates for the pairs
    declared in MANIFEST.md's freshness block; an undeclared dependency is a checkpoint that can
    be re-dated without invalidating anything. All eight `corrphys_` weights the leg loads have to
    be in that row, and the row must not be marked STALE while the artifact postdates them.
    """
    from src.evaluation.corrected_physics_ablation import FLYABLE_LABELS, checkpoint_name

    block = re.search(r"```freshness\n(.*?)\n```", _manifest_text(), re.S)
    assert block, "MANIFEST.md has no freshness block"
    rows = [line.strip() for line in block.group(1).splitlines() if MPC_ARTIFACT.name in line]
    assert len(rows) == 1, rows
    row = rows[0]
    assert not row.startswith("STALE"), "the leg predates a checkpoint it loaded: re-run it"
    expected = {checkpoint_name(label) for label in FLYABLE_LABELS}
    declared = set(row.partition("<-")[2].split())
    assert expected <= declared, f"undeclared weights: {sorted(expected - declared)}"


def _mpc_payload():
    if not MPC_ARTIFACT.is_file():
        pytest.skip("the 10.57.1 closed loop has not been recorded")
    return json.loads(MPC_ARTIFACT.read_text(encoding="utf-8"))


def test_readme_10_57_1_is_the_flown_artifact() -> None:
    """The console rows and the four convention contrasts, compared as generated."""
    from src.evaluation.physics_injection_mpc_benchmark import (
        render_control_table,
        render_convention_contrasts,
    )

    payload = _mpc_payload()
    text = _readme()
    rows = render_control_table(payload) + render_convention_contrasts(payload)
    missing = [row for row in rows if row not in text]
    assert not missing, (
        "README 10.57.1 disagrees with the closed-loop artifact on "
        f"{len(missing)} rows:\n" + "\n".join(missing[:4])
    )
    contrasts = payload["within_study_contrasts_px"]
    assert len(contrasts) == 4, "four families, one next/carried pair each"
    for key in contrasts:
        left, _, right = key.partition(" -> ")
        assert left.endswith("_next") and right.endswith("_carried"), key


def test_the_closed_loop_flew_the_weights_the_ablation_published() -> None:
    """A row is only evidence if it was flown with the weight this study trained.

    The runner silently omits an arm whose checkpoint is missing, so an artifact recorded
    before `--save-checkpoints` existed would still be well-formed - and would be measuring
    something else. Every 10.57 slug has to appear in the flown set.
    """
    from src.evaluation.corrected_physics_ablation import FLYABLE_LABELS, arm_slug

    payload = _mpc_payload()
    flown = set(payload["multi_seed"]["per_model"])
    expected = {arm_slug(label) for label in FLYABLE_LABELS}
    assert expected <= flown, f"not flown: {sorted(expected - flown)}"


DUPLICATE_PAIRS = (
    ("hard_pinn_next", "mlp_residual_hard_next"),
    ("hard_pinn_carried", "mlp_residual_hard_carried"),
    ("deeponet_residual_hard_next", "published_pc_deeponet_10_42"),
)
REFERENCE_ROWS = (
    "established_wram_engine_rules",
    "published_pc_deeponet_10_42",
    "published_hard_pinn_10_27",
)
CELLS = (
    "progress_px_mean",
    "progress_px_std",
    "progress_px_min",
    "progress_px_max",
    "frames_survived_mean",
    "pit_or_death",
)


def test_the_reference_rows_reproduce_the_earlier_closed_loops() -> None:
    """The parity control every closed-loop section of this repository rests on.

    The same weight files, the same five CEM seeds and the same planner were flown in 10.47
    and 10.53; if a reference row moved here, the eight new rows would not be comparable to
    those tables, and the section would be quoting a different instrument.
    """
    mine = _mpc_payload()["multi_seed"]["per_model"]
    for artifact in ("physics_injection_mpc_metrics.json", "effective_velocity_mpc_metrics.json"):
        path = REPO / "results" / artifact
        if not path.is_file():
            pytest.skip(f"{artifact} has not been recorded")
        other = json.loads(path.read_text(encoding="utf-8"))["multi_seed"]["per_model"]
        missing = [name for name in REFERENCE_ROWS if name not in other]
        assert not missing, f"{artifact} no longer flies the reference rows: {missing}"
        moved = [
            (name, cell)
            for name in REFERENCE_ROWS
            for cell in CELLS
            if abs(float(mine[name][cell]) - float(other[name][cell])) > 1e-9
        ]
        assert not moved, f"a reference row disagrees with {artifact}: {moved}"


def test_the_rows_the_section_calls_duplicates_are_duplicates_of_other_weight_files() -> None:
    """Identical progress, identical recorded sequence, and not the same checkpoint."""
    payload = _mpc_payload()
    from src.evaluation.physics_injection_mpc_benchmark import (
        CORRECTED_ARMS,
        PUBLISHED_CHECKPOINTS,
    )

    files = {**CORRECTED_ARMS, **PUBLISHED_CHECKPOINTS}
    seeds = sorted(payload["per_seed"])
    for left, right in DUPLICATE_PAIRS:
        same = all(
            payload["per_seed"][s][left]["progress_px"]
            == payload["per_seed"][s][right]["progress_px"]
            and payload["per_seed"][s][left]["frames_survived"]
            == payload["per_seed"][s][right]["frames_survived"]
            for s in seeds
        )  # noqa: E501
        assert same, f"{left} and {right} are not duplicates in the artifact"
        assert files[left][0] != files[right][0], (
            f"{left} and {right} would be the same weight file, so the finding is vacuous"
        )
        assert (
            payload["per_controller"][left]["action_sequence"]
            == payload["per_controller"][right]["action_sequence"]
        ), "the stored sequences differ, so the README claim is wrong"

    signatures = {
        tuple(round(payload["per_seed"][s][name]["progress_px"], 2) for s in seeds)
        for name in payload["multi_seed"]["per_model"]
    }
    programs = {block["action_sequence"] for block in payload["per_controller"].values()}  # noqa: E501
    assert len(signatures) == 8 and len(programs) == 8, (
        f"the section says eleven rows are eight programs; the artifact groups them into "
        f"{len(signatures)} by progress and {len(programs)} by sequence"
    )
    assert "Eleven rows, eight programs" in _readme()


def test_the_pit_statements_are_the_deaths_the_artifact_recorded() -> None:
    """Finding 3: the mean is five deaths in one hole, so every figure in that sentence is one."""
    payload = _mpc_payload()
    seeds = sorted(payload["per_seed"])
    per = payload["multi_seed"]["per_model"]
    deaths = {
        n: per[n]["pit_or_death"]
        for n in ("hard_pinn_carried", "hard_pinn_next", "published_hard_pinn_10_27")
    }  # noqa: E501
    assert deaths == {"hard_pinn_carried": 5, "hard_pinn_next": 1, "published_hard_pinn_10_27": 3}
    carried = [payload["per_seed"][s]["hard_pinn_carried"] for s in seeds]
    progress = sorted(row["progress_px"] for row in carried)
    frames = sorted({row["frames_survived"] for row in carried})
    assert frames == [173, 177], frames
    assert len(progress) == 5 and progress[0] == 112.44 and progress[-1] == 116.44
    assert f"{progress[0]:.2f}-{progress[-1]:.2f}" in _readme()
    assert max(progress) - min(progress) == pytest.approx(4.00, abs=0.005)
    budget_limited = sum(1 for b in per.values() if b["budget_reached_rate"] == 1.0)
    assert budget_limited == 6 and f"{budget_limited} of the 11 rows" in _readme()
    agreement = payload["agreement_with_established_physics"]
    assert (
        agreement["hard_pinn_carried"]["sequence_agreement"],
        agreement["hard_pinn_next"]["sequence_agreement"],
    ) == (0.3410, 0.3467)  # noqa: E501
    for value in ("0.3410", "0.3467"):
        assert value in _readme(), f"the agreement rate {value} is quoted nowhere"


def test_the_study_largest_effect_size_belongs_to_the_arm_that_dies_every_time() -> None:
    """The section's superlatives are recomputed, not remembered: largest dz, smallest p, tightest spread."""
    per = _mpc_payload()["multi_seed"]["per_model"]
    paired = {
        n: b["paired_vs_established_rules"]
        for n, b in per.items()
        if b["paired_vs_established_rules"]
    }
    carried = {"hard_pinn_carried", "mlp_residual_hard_carried"}

    def leaders(value, best) -> set[str]:
        scores = {n: value(b) for n, b in paired.items()}
        top = best(scores.values())
        return {n for n, score in scores.items() if score == top}

    assert leaders(lambda b: abs(b["cohen_dz"]), max) == carried
    assert leaders(lambda b: b["ttest_p"], min) == carried
    assert leaders(lambda b: b["difference_std_px"], min) == carried
    assert paired["hard_pinn_carried"]["cohen_dz"] == pytest.approx(-23.32, abs=0.005)


def test_the_convention_flips_sign_between_the_rollout_and_the_console_for_two_families() -> None:
    """Finding 1's load-bearing claim: the drift column and the planner disagree per family."""
    ablation = _payload()["contrasts"]["drift_multistart_mean_px"]
    mpc = _mpc_payload()["within_study_contrasts_px"]
    drift = {}
    for key, block in ablation.items():
        left, _, _ = key.partition(" -> ")
        if "/residual/hard/" in left or left.startswith("Hard PINN/"):
            family = left.split("/")[0].lower().replace(" pinn", "")
            drift[family] = block["mean_difference"]
    closed = {
        key.split("/")[0].split("_")[0]: block["mean_difference_px"] for key, block in mpc.items()
    }
    assert set(drift) == {"hard", "mlp", "deeponet", "fno"}
    assert set(closed) == {"hard", "mlp", "deeponet", "fno"}, closed
    # the two columns have opposite polarity: less drift is better, more progress is better
    open_gain = {f: -drift[f] for f in ("mlp", "deeponet", "fno")}
    flipped = sorted(f for f in open_gain if open_gain[f] * closed[f] < 0)
    assert flipped == ["fno", "mlp"], (
        f"the README says two of three families flip sign, the artifacts say {flipped}"
    )
    assert drift["mlp"] == pytest.approx(-1.20, abs=0.005)
    assert drift["deeponet"] == pytest.approx(9.51, abs=0.005)
    assert drift["fno"] == pytest.approx(7.02, abs=0.005)
    assert closed["mlp"] == pytest.approx(-319.90, abs=0.005)
    assert closed["fno"] == pytest.approx(28.61, abs=0.005)
    assert closed["deeponet"] == pytest.approx(-5.92, abs=0.005)


def test_the_weights_rival_explanation_is_a_measured_fraction_of_the_convention_gap() -> None:
    """Finding 5: 135.69 px of training-run difference is 42% of the 319.90 px convention gap."""
    per = _mpc_payload()["multi_seed"]["per_model"]
    gap = (
        per["hard_pinn_next"]["progress_px_mean"]
        - per["published_hard_pinn_10_27"]["progress_px_mean"]
    )
    convention = abs(
        _mpc_payload()["within_study_contrasts_px"]["hard_pinn_next -> hard_pinn_carried"][
            "mean_difference_px"
        ]
    )  # noqa: E501
    assert gap == pytest.approx(135.69, abs=0.01)
    assert round(100 * gap / convention) == 42, f"{gap:.2f} is not 42% of {convention:.2f}"
    assert "42% of the" in _readme()
