r"""Unit tests for the section-4 physics audit of README 10.54.

CPU-only and emulator-free. The audit is a documentation test, so what is checked here is
that it cannot silently weaken: every claim must have a satisfaction test, the literals that
make a file an implementation - or a dissenter - must still be in the source it reads, and
the prose quote must still be in the section it is attributed to. A claim added without a
test, or a dissenting line refactored away, fails here rather than turning the audit into a
list of sentences that all say "satisfied".
"""

import pytest

from src.evaluation.physics_claim_audit import (
    CLAIMS,
    SATISFACTION_TESTS,
    _read,
    audit_claims,
    render_audit_table,
    render_section_4_qualifiers,
    telemetry_satisfaction,
)


def test_every_claim_has_a_satisfaction_test() -> None:
    """A claim without a test is a sentence nobody checks."""
    asserts = {spec["asserts"] for spec in CLAIMS.values()}
    assert asserts == set(SATISFACTION_TESTS), "claim tests and audit tests diverged"


def test_every_claim_is_quoted_from_the_section_it_is_attributed_to() -> None:
    report = audit_claims()
    unquoted = [name for name, block in report.items() if not block["claim_quoted_from_the_readme"]]
    assert not unquoted, f"section 4 no longer says: {unquoted}"
    assert report["4.1_integration_identity_carried_velocity"]["convention"] == "carried"


def test_the_declared_sites_and_the_dissenters_are_still_in_the_source() -> None:
    """Both lists are literal-matched, so a refactor has to be recorded rather than missed."""
    report = audit_claims()
    for name, block in report.items():
        assert block["sites_missing"] == [], f"{name} lost an implementation site: {block}"
        assert block["dissenting_sites_missing"] == [], (
            f"{name} dissenter is gone; if the code was fixed, remove it from CLAIMS and "
            f"update README 10.54 and Section 12 in the same change"
        )


def test_the_ground_claim_declares_no_implementation_and_two_dissenters() -> None:
    """The corrected §4.3.5 is documented as unimplemented, and the audit says so out loud."""
    block = audit_claims()["4.3.5_ground_flag_gates_gravity"]
    assert block["declared_sites"] == 0
    assert len(block["dissenting_sites"]) == 2
    files = {site.split(":")[0] for site in block["dissenting_sites"]}
    assert files == {"src/losses/physics_losses.py", "src/models/analytical_kinematics.py"}


def test_the_identity_is_exact_where_no_collision_flag_is_set() -> None:
    """The measurement §4.1 now quotes: exact away from contacts, and broken only by them."""
    telemetry = telemetry_satisfaction()
    every = telemetry["identity_vs_carried_velocity"]
    free = telemetry["identity_vs_carried_velocity_without_collision_flag"]
    following = telemetry["identity_vs_next_velocity"]
    assert every["median_abs_residual_px"] == 0.0
    assert every["exact_to_half_a_subpixel_rate"] < free["exact_to_half_a_subpixel_rate"]
    assert free["exact_to_half_a_subpixel_rate"] > 0.95
    assert following["median_abs_residual_px"] > every["median_abs_residual_px"]


def test_the_ground_stratum_gates_the_increment_and_not_the_velocity() -> None:
    """§4.3.5's correction, as a number: the step stops, the body does not."""
    ground = telemetry_satisfaction()["ground_boundary_condition"]
    assert ground["claimed_rate_step_exactly_zero"] > 0.5
    assert ground["claimed_rate_vy_next_exactly_zero"] < 0.05
    assert ground["claimed_median_abs_vy_next"] > 0.0


def test_the_gravity_tiers_are_scored_per_stratum() -> None:
    """The pooled airborne figure hides the gate, so the table has to keep the strata."""
    strata = telemetry_satisfaction()["gravity_step_by_stratum"]
    assert set(strata) == {"ascent_held", "ascent_released", "descent_held", "descent_released"}
    released_ascent = strata["ascent_released"]
    released_descent = strata["descent_released"]
    assert released_ascent["rate_at_3.0"] > released_ascent["rate_at_6.0"]
    assert released_descent["rate_at_6.0"] > max(
        strata[name]["rate_at_6.0"] for name in ("ascent_held", "ascent_released")
    )


@pytest.mark.parametrize("renderer", ["table", "qualifiers"])
def test_the_renderers_emit_one_block_per_claim(renderer: str) -> None:
    """Both README generators read the artifact, and neither may emit an empty or short block."""
    payload = {"claims": audit_claims(), "telemetry": telemetry_satisfaction()}
    for name, block in payload["claims"].items():
        satisfied, evidence = SATISFACTION_TESTS[block["asserts"]](payload["telemetry"])
        block["satisfied_by_the_telemetry"] = satisfied
        block["telemetry_evidence"] = evidence
    if renderer == "table":
        rows = render_audit_table(payload)
        assert len(rows) == len(CLAIMS)
        assert all(row.startswith("| §4.") and row.endswith("|") for row in rows)
    else:
        lines = render_section_4_qualifiers(payload)
        assert len(lines) == 7
        assert all(line.startswith("*Measured") and line.endswith("*") for line in lines)


def test_the_audit_reads_the_readme_it_documents() -> None:
    """A guard against the audit scoring a copy: the file it quotes is the one in the repo."""
    from src.evaluation.physics_claim_audit import README_PATH

    assert README_PATH.replace("\\", "/").endswith("/README.md")
    assert "### 4.3 Horizontal Dynamics" in _read(README_PATH)
